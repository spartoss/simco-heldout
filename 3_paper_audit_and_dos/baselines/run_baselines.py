"""Benchmark closed-form baselines against CellOT, under either of two protocols.

``--eval-mode pooled`` is CellOT's own, and the one to reach for by default:

  * fit every baseline in baselines.METHODS on the 80% train side of CellOT's cell
    split, pooled over all 19 patients -- exactly the cells CellOT trained on
  * transport the 20% test side and score once against the measured test cells
  * one row per (stim, celltype, method)

The split is over *cells*, grouped by stim (cellot/data/cell.py:384-404), with no
patient dimension, so both halves hold all 19 patients. There is no held-out patient
and none is claimed. What this buys is that the baselines and CellOT see the same
training cells and are scored on the same test cells, so the comparison is like for
like on everything except model selection: CellOT's checkpoint is the argmin over ~200
evaluations on those same test cells (cellot/train/train.py:255-265) and a closed-form
fit has no such step. That residual advantage is CellOT's, and only retraining under a
patient-level split would remove it.

``--eval-mode lopo`` is the older, stricter protocol, kept because the earlier result
dirs were produced under it: refit the baselines 19 times, each time leaving one
patient out, and score each held-out patient separately. It is leave-one-patient-out
for the *baselines only* -- the paper's run has no patients_excluded, so CellOT is one
checkpoint loaded outside the patient loop and applied unchanged, and that loop
partitions the evaluation set rather than cross-validating it. Hence the column name
``cellot_insample``, which is kept in both modes. Two flags control how far the
asymmetry is closed there:

  --fit-patients  loo  baselines fit on the other 18 patients -- out-of-sample, so a
                       baseline win is conservative and a loss is not conclusive.
                  all  baselines fit on all 19, exactly CellOT's training set.
  --weighting     unweighted / cell -- see baselines.py.

``--fit-patients`` has no meaning in pooled mode and is ignored there.

Two arms, ``--arm``:

  ``stim``  Unstim -> stim, results/cross_validation_original_13_HVPV, 168 pairs.
            Data handling mirrors cellot/data/cell.py:read_single_anndata --
            DMSO-prefixed cells only (config.data.drug_used: DMSO), markers subset to
            13features.txt in file order, split grouped by stim.

  ``drug``  DMSO -> drug *within* one stim, results_drug/cross_validation_drug_OG13,
            9 drugs x 168 pairs. Source is the drug's own plate control, cells are
            batch-corrected at load time and the split is grouped by drug; all three
            live in baselines/drug_data.py. Rows carry an extra ``drug`` column, so a
            drug run's output dir is not schema-compatible with a stim one -- keep them
            apart and never point summarize.py at both.

            ``identity`` means something much stronger here: "the drug does nothing".
            The drug run also selects checkpoints on MMD rather than median_error
            (``data.metric`` is absent from all 1512 configs), so the stim arm's sharpest
            finding -- a constant shift beating CellOT on its own selection criterion --
            has no counterpart to look for.

Nothing else differs: the same six baselines, the same metrics, the same two protocols.
"""

import argparse
import gc
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split

from baselines.baselines import METHODS, PATIENT_LEVEL, patient_stats
from baselines.drug_data import ROW_CHUNK, iter_drug_units
from baselines.metrics import score

# Minimum cells required for a patient to take part, as a training patient or as the
# held-out one. Quantile functions estimated from a few dozen cells are mostly noise;
# the observed floor in this cohort is ~29 cells (PV10/PI, pDCs).
MIN_CELLS = 50


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["stim", "drug"], default="stim",
                   help="'stim': Unstim -> stim. 'drug': the drug's DMSO control -> drug, "
                        "within each stim, with runtime batch correction.")
    p.add_argument("--celltype", required=True, help="sanitized celltype, e.g. pDCs")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--pairs-file", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--cellot-results", default=None,
                   help="stim arm: cross_validation_original_13_HVPV; omit to skip the "
                        "CellOT arm. drug arm: cross_validation_drug_OG13, the dir "
                        "holding one subdir per drug -- required, it is also the drug list")
    p.add_argument("--drugs", default=None,
                   help="drug arm: comma-separated subset; default is every subdir "
                        "with checkpoints")
    p.add_argument("--batch-correction", default=None,
                   help="drug arm: batchcorrection_2.csv; required")
    p.add_argument("--low-memory", action="store_true",
                   help="drug arm: read the file once per stim instead of once, holding "
                        "one stim's cells at a time. ~7x less memory for ~7x the I/O, and "
                        "identical results. Needed for Granulocytes, which otherwise "
                        "peaks over 23 GB and is OOM-killed in a 32 GB allocation.")
    p.add_argument("--source", default="Unstim", help="stim arm only; the drug arm reads "
                                                      "each drug's control from use_DMSO")
    p.add_argument("--eval-mode", choices=["lopo", "pooled"], default="lopo",
                   help="'pooled': CellOT's own protocol -- fit once on the 80%% train "
                        "cells of all patients, score once on the 20%% test cells, one "
                        "row per (stim, celltype). No held-out patient. 'lopo': refit "
                        "per held-out patient and score each separately.")
    p.add_argument("--eval-split", choices=["all", "test"], default="all",
                   help="'test' reproduces the training-time 80/20 cell split and uses "
                        "only test cells for evaluation (and only train cells for "
                        "fitting), removing CellOT's cell-level leak")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--min-eval-cells", type=int, default=20,
                   help="floor on the held-out patient's evaluation cells; lower than "
                        "MIN_CELLS because --eval-split test keeps only 20% of them")
    p.add_argument("--fit-patients", choices=["loo", "all", "both"], default="loo",
                   help="'loo': fit baselines on the other 18 patients (out-of-sample). "
                        "'all': fit on all 19, i.e. exactly CellOT's training set, making "
                        "the comparison like for like. 'both' emits both in one pass.")
    p.add_argument("--weighting", choices=["unweighted", "cell", "both"],
                   default="unweighted",
                   help="'unweighted': equal weight per patient. 'cell': pool cells, "
                        "matching CellOT's cell-level sampling. 'both' emits both.")
    return p.parse_args()


def expand(choice, options):
    return list(options) if choice == "both" else [choice]


def read_list(path):
    return Path(path).read_text().split()


def load_celltype(data_dir, celltype, features, test_size, split_seed):
    """Return per-(patient, stim) cell matrices and their training-time test masks.

    Reads row-chunked out of the backed file and keeps only the DMSO rows and the 13
    requested markers, in ``features`` order -- the order training used via
    ``data[:, features]`` in cellot/data/cell.py:63. Granulocytes_HVPV.h5ad is 30 GB
    on disk, so materialising the whole matrix is not an option.
    """
    path = Path(data_dir) / f"{celltype}_HVPV.h5ad"
    adata = ad.read_h5ad(path, backed="r")
    obs = adata.obs.copy()
    mask = obs["drug"].astype(str).str.startswith("DMSO").to_numpy()
    var_idx = np.array([adata.var_names.get_loc(f) for f in features])

    blocks = []
    n_rows = adata.shape[0]
    for start in range(0, n_rows, ROW_CHUNK):
        stop = min(start + ROW_CHUNK, n_rows)
        block_mask = mask[start:stop]
        if not block_mask.any():
            continue
        block = adata.X[start:stop]
        if sparse.issparse(block):
            block = block.toarray()
        blocks.append(np.asarray(block)[block_mask][:, var_idx])
    adata.file.close()

    X = np.concatenate(blocks).astype(np.float64)
    del blocks
    obs = obs.loc[mask]
    assert len(X) == len(obs), (len(X), len(obs))

    out, test_mask = {}, {}
    is_test = reproduce_training_split(obs, test_size, split_seed)
    for (patient, stim), idx in obs.groupby(["patient", "stim"], observed=True).indices.items():
        out[(str(patient), str(stim))] = X[idx]
        test_mask[(str(patient), str(stim))] = is_test[idx]
    del X
    gc.collect()
    return out, test_mask


def reproduce_training_split(obs, test_size, seed):
    """Re-derive the 'test' side of the cell split CellOT trained under.

    The paper's config is ``datasplit: {name: train_test, groupby: stim, test_size: 0.2}``
    with ``random_state`` defaulting to 0 in cellot/data/cell.py:384-404, so the split is
    deterministic and can be recomputed -- nothing was recorded.

    Reproducing it needs the same rows in the same order that read_single_anndata hands
    to split_cell_data: the DMSO-filtered matrix in file order. Each stim is split
    independently with a fresh RandomState(seed), so the Unstim partition is identical
    for every target stim, and slicing this frame per stim gives the same result as
    cellot's per-target frame (which holds Unstim + one target).

    Caveat: this is a faithful re-derivation of a documented deterministic split, not a
    recovered artefact -- no split was ever saved, so it cannot be checked against one.
    """
    is_test = np.zeros(len(obs), dtype=bool)
    positions = np.arange(len(obs))
    for _, idx in obs.groupby("stim", observed=True).indices.items():
        _, test_idx = train_test_split(
            positions[idx], random_state=seed, test_size=test_size
        )
        is_test[test_idx] = True
    return is_test


def load_cellot_g(model_dir, n_markers):
    """Rebuild the trained transport map, mirroring make_prediction/predict_ina_og_csv.py."""
    import torch
    from cellot.models.cellot import load_networks
    from cellot.utils.helpers import load_config

    config = load_config(os.path.join(model_dir, "config.yaml"))
    _, g = load_networks(config, input_dim=n_markers)
    ckpt = torch.load(os.path.join(model_dir, "cache/model.pt"), map_location="cpu")
    g.load_state_dict(ckpt["g_state"])
    g.eval()
    return g


def cellot_transport(g, x):
    import torch

    inputs = torch.tensor(np.asarray(x, dtype=np.float32), requires_grad=True)
    return g.transport(inputs).detach().numpy().astype(np.float64)


def stim_units(args, features, stims):
    """One unit per stim, all sharing a single read of the celltype.

    The stim arm's cells are loaded once and sliced per stim -- ``Unstim`` is the source
    for every one of them, so re-reading per pair would be waste.
    """
    cells, test_mask = load_celltype(
        args.data_dir, args.celltype, features, args.test_size, args.split_seed
    )
    patients = sorted({p for (p, s) in cells})
    print(f"[INFO] {args.celltype}: {len(patients)} patients, {len(cells)} "
          f"(patient, stim) groups", flush=True)
    for stim in stims:
        model_dir = None
        if args.cellot_results:
            model_dir = os.path.join(args.cellot_results, stim, args.celltype,
                                     f"model-{stim}_{args.celltype}")
        yield {
            "stim": stim, "drug": None, "source": args.source, "target": stim,
            "cells": cells, "test_mask": test_mask, "patients": patients,
            "model_dir": model_dir,
        }


def units_for(args, features, stims):
    if args.arm == "stim":
        return stim_units(args, features, stims)
    return iter_drug_units(
        args.data_dir, args.celltype, features, stims, args.cellot_results,
        args.batch_correction, drugs=args.drugs,
        test_size=args.test_size, split_seed=args.split_seed,
        per_stim=args.low_memory,
    )


def pooled_rows(args, source, target, label, present, fit_cells, eval_cells,
                fitted, row, g):
    """One fit on the 80% train cells, one score on the 20% test cells. No patient loop.

    This is CellOT's own protocol, applied to the baselines: the split is over cells and
    both halves hold all 19 patients, so there is no held-out patient and none is
    claimed. Every method -- baselines and CellOT alike -- is fit on exactly the cells
    CellOT trained on and scored once, on exactly the cells CellOT's checkpoint
    selection watched. One row per (stim, celltype, method).

    What this equalises, relative to --eval-mode lopo: the baselines no longer pay a
    penalty for being refit without the evaluation patient. What it still does not
    equalise: CellOT's checkpoint is the argmin over ~200 evaluations on these very
    cells (cellot/train/train.py:255-265) and a closed-form fit has no such step, so
    CellOT keeps a selection advantage that no reweighting here can remove.

    No per-patient cell floor is applied to the pooled fit or the pooled evaluation --
    CellOT trained on every DMSO cell of the pair, so parity means using them all. The
    floor survives only where it has to: in the design matrix of the patient-level
    regressions, where a patient with a handful of cells contributes a meaningless row.
    """
    fit_src = {p: fit_cells((p, source)) for p in present}
    fit_tgt = {p: fit_cells((p, target)) for p in present}
    ev_src = {p: eval_cells((p, source)) for p in present}
    ev_tgt = {p: eval_cells((p, target)) for p in present}

    fit_p = [p for p in present if len(fit_src[p]) and len(fit_tgt[p])]
    ev_p = [p for p in present if len(ev_src[p]) and len(ev_tgt[p])]
    if not fit_p or not ev_p:
        print(f"[SKIP] {label}: nothing on one side of the split", flush=True)
        return []

    src_pool = np.concatenate([ev_src[p] for p in ev_p])
    tgt_pool = np.concatenate([ev_tgt[p] for p in ev_p])
    if len(src_pool) < args.min_eval_cells or len(tgt_pool) < args.min_eval_cells:
        print(f"[SKIP] {label}: {len(src_pool)} src / {len(tgt_pool)} tgt test cells",
              flush=True)
        return []

    stat_src = {p: patient_stats(fit_src[p]) for p in fit_p}
    stat_tgt = {p: patient_stats(fit_tgt[p]) for p in fit_p}
    design = [p for p in fit_p
              if len(fit_src[p]) >= MIN_CELLS and len(fit_tgt[p]) >= MIN_CELLS]

    def fit_one(name, subset, weighting):
        fit_fn, _ = METHODS[name]
        return fit_fn(
            [fit_src[p] for p in subset], [fit_tgt[p] for p in subset],
            weighting=weighting,
            stats={"src": [stat_src[p] for p in subset],
                   "tgt": [stat_tgt[p] for p in subset]},
        )

    def transport(apply_, params):
        # Patient by patient, then concatenated: required by the patient-level methods
        # and identical to a single pooled call for every other one.
        return np.concatenate([apply_(ev_src[p], params) for p in ev_p])

    out = [row("pooled", "identity", "invariant", "invariant",
               src_pool, tgt_pool, src_pool, 0)]
    if g is not None:
        out.append(row("pooled", "cellot_insample", "all", "cell",
                       np.concatenate([cellot_transport(g, ev_src[p]) for p in ev_p]),
                       tgt_pool, src_pool, len(fit_p)))

    for weighting in expand(args.weighting, ["unweighted", "cell"]):
        for name, (_, apply_) in fitted.items():
            patient_level = name in PATIENT_LEVEL
            subset = design if patient_level else fit_p
            if patient_level and len(subset) < 4:
                print(f"[WARN] {label}: only {len(subset)} patients with >={MIN_CELLS} "
                      f"cells, skipping {name}", flush=True)
                continue
            params = fit_one(name, subset, weighting)
            out.append(row("pooled", name, "all", weighting,
                           transport(apply_, params), tgt_pool, src_pool,
                           len(subset), alpha=params.get("alpha")))

    print(f"[INFO] {label}: pooled fit on {len(fit_p)} patients "
          f"({sum(len(fit_src[p]) for p in fit_p)} train cells), scored on "
          f"{len(src_pool)} src / {len(tgt_pool)} tgt test cells", flush=True)
    return out


def main():
    args = parse_args()
    if args.eval_mode == "pooled" and args.eval_split != "test":
        raise SystemExit(
            "--eval-mode pooled scores on the test side of CellOT's own split, so it "
            "needs --eval-split test; with 'all' the fit and evaluation sets are the "
            "same cells and every number is in-sample."
        )
    if args.arm == "drug":
        # Both are structural, not conveniences: without the table there is no way to
        # know which DMSO plate is a given drug's control, and the results root doubles
        # as the list of drugs to run.
        if not args.batch_correction:
            raise SystemExit("--arm drug needs --batch-correction (batchcorrection_2.csv)")
        if not args.cellot_results:
            raise SystemExit("--arm drug needs --cellot-results (cross_validation_drug_OG13)")
    features = read_list(args.features)
    celltype = args.celltype

    stims = sorted(
        line.split("/")[0]
        for line in read_list(args.pairs_file)
        if line.split("/")[1] == celltype
    )
    if not stims:
        print(f"[SKIP] no CellOT models for celltype {celltype}")
        return
    print(f"[INFO] {celltype}: stims {stims}", flush=True)

    fit_patient_modes = expand(args.fit_patients, ["loo", "all"])
    weightings = expand(args.weighting, ["unweighted", "cell"])
    arms = [(fp, w) for fp in fit_patient_modes for w in weightings]
    # identity is invariant to both axes and CellOT is fixed at (all, cell) by
    # construction, so both are fitted/scored once and labelled honestly rather than
    # recomputed per arm; summarize.py broadcasts them into every arm's table.
    fitted = {k: v for k, v in METHODS.items() if k != "identity"}
    if args.eval_mode == "pooled":
        # There is no held-out patient to leave out, so the fit-patients axis collapses:
        # every fit uses all of them, which is what "all" already meant.
        if args.fit_patients != "all":
            print(f"[INFO] --fit-patients {args.fit_patients} ignored in pooled mode; "
                  f"every fit uses all patients", flush=True)
        print(f"[INFO] {celltype}: pooled mode, weightings {weightings}", flush=True)
    else:
        print(f"[INFO] {celltype}: arms {arms}", flush=True)

    rows = []
    for unit in units_for(args, features, stims):
        stim, drug = unit["stim"], unit["drug"]
        source, target = unit["source"], unit["target"]
        cells, test_mask, patients = unit["cells"], unit["test_mask"], unit["patients"]
        label = f"{drug}/{stim}" if drug else stim

        def fit_cells(key):
            """Cells a baseline may fit on -- the train side when matching CellOT's split."""
            x = cells[key]
            return x[~test_mask[key]] if args.eval_split == "test" else x

        def eval_cells(key):
            """Cells every method is scored on."""
            x = cells[key]
            return x[test_mask[key]] if args.eval_split == "test" else x

        present = [p for p in patients
                   if (p, source) in cells and (p, target) in cells]
        if not present:
            print(f"[SKIP] {celltype}/{label}: no patient has both conditions", flush=True)
            continue

        g = None
        if unit["model_dir"]:
            if os.path.exists(os.path.join(unit["model_dir"], "cache/model.pt")):
                g = load_cellot_g(unit["model_dir"], len(features))
            else:
                print(f"[WARN] no checkpoint for {label}/{celltype}", flush=True)

        def row(patient, method, fp, w, pred, tgt_eval, src_eval, n_fit, alpha=None):
            return {
                "celltype": celltype, "stim": stim, "patient": patient,
                # Present in the drug arm only, which is what makes a drug result dir
                # schema-incompatible with a stim one -- deliberately, since a row
                # without it would silently pool nine drugs into one (stim, celltype).
                **({"drug": drug} if drug is not None else {}),
                "method": method, "fit_patients": fp, "weighting": w,
                "eval_split": args.eval_split, "n_fit_patients": n_fit,
                "n_src_cells": len(src_eval), "n_tgt_cells": len(tgt_eval),
                # Only linear_median* set this. Worth keeping: an alpha pinned at the
                # top of RIDGE_ALPHAS means the fit found no cross-marker signal and
                # collapsed to a constant shift, which reads very differently from a
                # genuine tie on the error metrics.
                "alpha": alpha,
                **score(pred, tgt_eval, features),
            }

        if args.eval_mode == "pooled":
            rows.extend(pooled_rows(args, source, target, label, present,
                                    fit_cells, eval_cells, fitted, row, g))
            del g
            gc.collect()
            continue

        usable = [
            p for p in patients
            if p in present
            and len(fit_cells((p, source))) >= MIN_CELLS
            and len(fit_cells((p, target))) >= MIN_CELLS
            and len(eval_cells((p, source))) >= args.min_eval_cells
            and len(eval_cells((p, target))) >= args.min_eval_cells
        ]
        if len(usable) < 3:
            print(f"[SKIP] {celltype}/{label}: only {len(usable)} usable patients",
                  flush=True)
            del g
            continue

        # Hoisted once per (stim, celltype): under --eval-split test, fit_cells copies
        # out the 80% train side, and the leave-one-out loop below would otherwise redo
        # that copy for all 18 patients on every one of the 19 iterations. The
        # per-patient moment summaries are hoisted for the same reason -- they are what
        # the median/Gaussian/linear fits consume, and they do not change between
        # refits.
        fit_src = {p: fit_cells((p, source)) for p in usable}
        fit_tgt = {p: fit_cells((p, target)) for p in usable}
        stat_src = {p: patient_stats(fit_src[p]) for p in usable}
        stat_tgt = {p: patient_stats(fit_tgt[p]) for p in usable}

        def fit_all(subset, weighting):
            return {
                name: f(
                    [fit_src[p] for p in subset],
                    [fit_tgt[p] for p in subset],
                    weighting=weighting,
                    stats={
                        "src": [stat_src[p] for p in subset],
                        "tgt": [stat_tgt[p] for p in subset],
                    },
                )
                for name, (f, _) in fitted.items()
            }

        # An "all"-patient fit does not depend on the held-out patient, so it is done
        # once per (stim, celltype) instead of 19 times.
        all_params = {
            (fp, w): fit_all(usable, w) for fp, w in arms if fp == "all"
        }

        for held_out in usable:
            train = [p for p in usable if p != held_out]
            src_eval = eval_cells((held_out, source))
            tgt_eval = eval_cells((held_out, target))

            # identity: fit-free, so one score covers every arm
            _, identity_apply = METHODS["identity"]
            rows.append(row(held_out, "identity", "invariant", "invariant",
                            identity_apply(src_eval, {}), tgt_eval, src_eval, 0))
            if g is not None:
                rows.append(row(held_out, "cellot_insample", "all", "cell",
                                cellot_transport(g, src_eval), tgt_eval, src_eval,
                                len(usable)))

            for fp, w in arms:
                if fp == "all":
                    params, n_fit = all_params[(fp, w)], len(usable)
                else:
                    params, n_fit = fit_all(train, w), len(train)
                for name, (_, apply_) in fitted.items():
                    fitted_params = params[name]
                    rows.append(row(held_out, name, fp, w,
                                    apply_(src_eval, fitted_params),
                                    tgt_eval, src_eval, n_fit,
                                    alpha=fitted_params.get("alpha")))

            print(f"[INFO] {celltype}/{label}/{held_out} done "
                  f"({len(src_eval)} src, {len(tgt_eval)} tgt)", flush=True)

        del g, all_params, fit_src, fit_tgt, stat_src, stat_tgt
        gc.collect()

    if not rows:
        print(f"[WARN] {celltype}: no rows produced")
        return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{celltype}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[WRITE] {out} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
