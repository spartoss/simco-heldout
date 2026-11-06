"""Measure CellOT's checkpoint-selection leak: ``model.pt`` vs ``last.pt``.

Training never backpropagates through the test cells -- the gradients in
cellot/train/train.py:222-240 come from ``iterator.train.*`` only. What the test cells
do is *choose* which snapshot is kept: ``evaluate()`` runs every ``eval_freq`` steps on
a single 256-cell test minibatch (train.py:57-59, dataloader.batch_size 256) and
``model.pt`` is overwritten whenever that metric improves (train.py:252-265). With
``n_iters: 50000`` and ``eval_freq: 250`` that is an argmin over **200 draws**.

``last.pt`` is written every ``cache_freq`` steps unconditionally (train.py:268-269),
so the final one is the step-49999 model with no selection applied. Scoring both on the
same test cells isolates the selection step: the gap is what the argmin bought, and it
is bought on the evaluation set, so it is leak rather than generalisation.

Caveat on the direction of the gap. ``last.pt`` is not merely "the same model without
the leak" -- it is also a *different training step*, and CellOT's losses are not
monotone in the step. A gap therefore bounds the leak from above: it mixes the
selection advantage with whatever genuine progress or divergence happened between the
argmin step and step 49999. The checkpoint's recorded step, emitted as ``ckpt_step``,
is what tells the two apart -- an argmin landing early with a large gap reads very
differently from one landing at 49000.

Two arms, because the two runs differ in more than their data:

  ``--arm stim``  results/cross_validation_original_13_HVPV, 168 pairs.
                  ``metric: median`` -- selection on median_error.
                  Source Unstim, DMSO-prefixed cells only, split grouped by stim.

  ``--arm drug``  results_drug/cross_validation_drug_OG13, 9 drugs x 168 pairs.
                  ``data.metric`` is **absent from all 1513 configs**, so
                  ``metric_type`` falls back to mmd (train.py:190-193): the drug arm
                  selects on MMD, not on median_error. Source is the drug's own DMSO
                  plate control (DMSO1/2/3 per batchcorrection_2.csv's use_DMSO, which
                  matches every dumped config), cells are batch-corrected at load time
                  (cellot/data/cell.py:147-194), the frame is restricted to one stim,
                  and the split is grouped by ``drug``.

Both arms also score ``identity`` as the floor -- the untransported source cells. In
the drug arm that is the null "the drug does nothing", which is much harder to beat
than "the stim does nothing".
"""

import argparse
import gc
import os
from pathlib import Path

import numpy as np
import pandas as pd

from baselines.drug_data import (
    dmso_for,
    list_drugs,
    load_bc,
    load_drug_stims,
    model_dir_for,
    pair_frame,
)
from baselines.metrics import score
from baselines.run_baselines import load_celltype, read_list

CKPTS = {"cellot_model": "model.pt", "cellot_last": "last.pt"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["stim", "drug"], required=True)
    p.add_argument("--celltype", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--pairs-file", required=True)
    p.add_argument("--results", required=True,
                   help="stim arm: the cross_validation_* dir. drug arm: the dir "
                        "holding one subdir per drug.")
    p.add_argument("--outdir", required=True)
    p.add_argument("--source", default="Unstim", help="stim arm only")
    p.add_argument("--drugs", default=None,
                   help="drug arm: comma-separated subset; default is every subdir "
                        "with checkpoints")
    p.add_argument("--batch-correction", default=None,
                   help="drug arm: batchcorrection_2.csv; required")
    p.add_argument("--low-memory", action="store_true",
                   help="drug arm: read the file once per stim instead of once. Needed "
                        "for Granulocytes, which otherwise OOMs in a 32 GB allocation.")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--min-eval-cells", type=int, default=20)
    return p.parse_args()


def load_g(model_dir, ckpt_name, n_markers):
    """Rebuild the transport map from one named checkpoint.

    Mirrors run_baselines.load_cellot_g, but parameterised on the checkpoint file and
    additionally returning what the checkpoint recorded about itself: the step it was
    saved at, and the selection metric value that caused it to be saved (``minmedian``
    in the stim arm, ``minmmd`` in the drug arm; absent on last.pt, which is saved
    unconditionally).
    """
    import torch
    from cellot.models.cellot import load_networks
    from cellot.utils.helpers import load_config

    config = load_config(os.path.join(model_dir, "config.yaml"))
    _, g = load_networks(config, input_dim=n_markers)
    ckpt = torch.load(os.path.join(model_dir, "cache", ckpt_name), map_location="cpu")
    g.load_state_dict(ckpt["g_state"])
    g.eval()
    meta = {k: v for k, v in ckpt.items() if not k.endswith("_state")}
    return g, meta


def transport(g, x):
    import torch

    inputs = torch.tensor(np.asarray(x, dtype=np.float32), requires_grad=True)
    return g.transport(inputs).detach().numpy().astype(np.float64)


# ---------------------------------------------------------------------- arms


def run_stim(args, features, stims, rows):
    cells, test_mask = load_celltype(
        args.data_dir, args.celltype, features, args.test_size, args.split_seed
    )
    patients = sorted({p for (p, s) in cells})
    print(f"[INFO] {args.celltype}: {len(patients)} patients, {len(cells)} groups",
          flush=True)

    for stim in stims:
        present = [p for p in patients
                   if (p, args.source) in cells and (p, stim) in cells]
        if not present:
            print(f"[SKIP] {stim}: no patient has both conditions", flush=True)
            continue

        def ev(key):
            return cells[key][test_mask[key]]

        src = np.concatenate([ev((p, args.source)) for p in present
                              if len(ev((p, args.source)))])
        tgt = np.concatenate([ev((p, stim)) for p in present if len(ev((p, stim)))])
        if len(src) < args.min_eval_cells or len(tgt) < args.min_eval_cells:
            print(f"[SKIP] {stim}: {len(src)}/{len(tgt)} test cells", flush=True)
            continue

        model_dir = os.path.join(args.results, stim, args.celltype,
                                 f"model-{stim}_{args.celltype}")
        emit(rows, args, features, model_dir, stim, None, src, tgt, len(present))


def run_drug(args, features, stims, rows):
    drugs = list_drugs(args.results, args.drugs)
    # One read for everything, or one per stim when the celltype does not fit. Only the
    # resident set differs: every quantity is computed within a (stim, drug) pair, which
    # neither form splits across reads.
    for chunk in ([[s] for s in stims] if args.low_memory else [stims]):
        run_drug_chunk(args, features, chunk, drugs, rows)


def run_drug_chunk(args, features, stims, drugs, rows):
    Xall, obsall = load_drug_stims(args.data_dir, args.celltype, features, stims)
    bc = load_bc(args.batch_correction,
                 pd.unique(obsall["cell_type"].astype(str)), stims)
    stim_col = obsall["stim"].astype(str).to_numpy()
    print(f"[INFO] {args.celltype}: {len(Xall)} cells over stims {stims}, drugs "
          f"{sorted(pd.unique(obsall['drug'].astype(str)))}", flush=True)

    for stim in stims:
        in_stim = stim_col == stim
        X, obs = Xall[in_stim], obsall.loc[in_stim]
        for drug in drugs:
            model_dir = model_dir_for(args.results, drug, stim, args.celltype)
            if not os.path.exists(os.path.join(model_dir, "cache", "model.pt")):
                print(f"[WARN] no checkpoint for {drug}/{stim}/{args.celltype}",
                      flush=True)
                continue
            dmso = dmso_for(args.batch_correction, drug)
            if dmso is None:
                print(f"[WARN] no unique use_DMSO for {drug}", flush=True)
                continue

            Xd, obsd, is_test = pair_frame(X, obs, bc, drug, dmso, stim, features,
                                           args.test_size, args.split_seed)
            if Xd is None:
                continue
            d_col = obsd["drug"].astype(str).to_numpy()
            src = Xd[is_test & (d_col == dmso)].astype(np.float64)
            tgt = Xd[is_test & (d_col == drug)].astype(np.float64)
            if len(src) < args.min_eval_cells or len(tgt) < args.min_eval_cells:
                print(f"[SKIP] {drug}/{stim}: {len(src)}/{len(tgt)} test cells",
                      flush=True)
                del Xd
                continue

            emit(rows, args, features, model_dir, stim, drug, src, tgt,
                 len(pd.unique(obsd["patient"].astype(str))))
            del Xd, src, tgt
            gc.collect()

        del X, obs
        gc.collect()

    del Xall, obsall
    gc.collect()


def emit(rows, args, features, model_dir, stim, drug, src, tgt, n_patients):
    """identity + both checkpoints, scored on the same test cells."""

    def base(method, pred, **extra):
        return {
            "arm": args.arm, "celltype": args.celltype, "stim": stim, "drug": drug,
            "method": method, "n_src_cells": len(src), "n_tgt_cells": len(tgt),
            "n_patients": n_patients, **extra, **score(pred, tgt, features),
        }

    rows.append(base("identity", src, ckpt_step=None, sel_metric=None))
    for method, fname in CKPTS.items():
        g, meta = load_g(model_dir, fname, len(features))
        rows.append(base(
            method, transport(g, src),
            ckpt_step=meta.get("step"),
            # The value evaluate() recorded on its 256-cell minibatch when this
            # checkpoint won. Not comparable to the scored columns -- different cells,
            # different subsample -- but it is the number the selection acted on.
            sel_metric=meta.get("minmedian", meta.get("minmmd")),
        ))
        del g
    tag = f"{drug}/{stim}" if drug else stim
    print(f"[INFO] {args.celltype}/{tag}: scored on {len(src)} src / {len(tgt)} tgt",
          flush=True)


def main():
    args = parse_args()
    if args.arm == "drug" and not args.batch_correction:
        raise SystemExit("--arm drug needs --batch-correction (batchcorrection_2.csv)")

    features = read_list(args.features)
    stims = sorted({line.split("/")[0] for line in read_list(args.pairs_file)
                    if line.split("/")[1] == args.celltype})
    if not stims:
        print(f"[SKIP] no models for celltype {args.celltype}")
        return
    print(f"[INFO] {args.celltype}: arm={args.arm} stims={stims}", flush=True)

    rows = []
    (run_stim if args.arm == "stim" else run_drug)(args, features, stims, rows)
    if not rows:
        print(f"[WARN] {args.celltype}: no rows produced")
        return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{args.celltype}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[WRITE] {out} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
