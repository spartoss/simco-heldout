"""Train one Meta Flow Matching model for one (repeat, k) of the atlas learning-curve design and predict the test donors.

Data (read-only from atlas_heldout): work/lc/fm/cache_full/<celltype>.npz, DMSO cells, all 38 markers, same cells and
train/val split as the CellOT runs (first 80 % of each block = train side). Design: work/lc/design.json (training
donors = order[:k], test donors = test). Maps = lc.paths.PAIRS (168 stim x celltype).
One GLOBAL model per (repeat, k, arm), as MFM's trellis model (one model for all patients x treatments):
- population = (training donor, celltype); its train-side Unstim cells are the source, embedded by the GCN on all 38
  markers + celltype one-hot; the flow acts on the 13 functional markers, conditioned on [stim one-hot (7), celltype
  one-hot (28)]; target = the same donor's train-side stim cells (independent coupling, as trellis: ot_sample False).
- arms: fm         no population embedding (pooled conditional flow; MFM's FM baseline)
        mfm        embedding of the donor's own celltype Unstim population
        mfm_blood  + embeddings of the donor's dominant celltypes (BLOOD_CTS, ~75 % of cells), concatenated
- training: per step --batch (stim, celltype) pairs drawn uniformly, then a training donor; graph = up to --n-embed
  source cells, flow batch = --n-flow source and --n-flow target cells; odd steps update the GNN only, even steps the
  flow only (TrellisMFM.training_step); Adam 1e-4 both (trellis_mfm.yaml). fm: every step is a flow step.
- prediction: each test donor's Unstim cells (<= --max-test, all of its cached cells: never trained on) are embedded
  (same sampling as training) and integrated (midpoint, --ode-steps); quantiles QS of the 13 markers.
- --select-every N (0 = off, the first wave): every N steps, validation error as CellOT's paper-style selection
  (lc/train.py): on a fixed draw of --n-val-cases (pair, training donor) cases, transport up to --n-val-cells of the
  donor's val-side Unstim cells (20-step ODE) and take mean |median(pred) - median(val-side stim cells)| over the 13
  markers; the best state is used for prediction (training donors only: nothing of the test donors).
Output --out: maps.csv, pred_q.npy (1, M, n_test, n_q, 13) [lc/train.py format, scored by lc.ctx_score after
mfm/merge.py], meta.json, model.pt, log.csv.
    python3 -m mfm.train --repeat 0 --k 15 --arm mfm --out work/runs/mfm/repeat_0/k15
"""
import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lc.paths import CELLTYPES, MARKERS, PAIRS, QS, VAL_FRAC, WORK as LCWORK
from lc.train import MIN_TEST_CELLS, durable_save
from mfm.model import MFM

CACHE = LCWORK / "fm" / "cache_full"
BLOOD_CTS = ["Granulocytes", "CD4Tnaive", "cMCs", "CD8Tnaive"]
STIMS = sorted({s for s, _ in PAIRS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--arm", choices=["fm", "mfm", "mfm_blood"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=60_000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-embed", type=int, default=512)
    ap.add_argument("--n-flow", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--knn-k", type=int, default=100)
    ap.add_argument("--max-test", type=int, default=20_000)
    ap.add_argument("--ode-steps", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=2000)
    ap.add_argument("--celltypes", nargs="*", default=None, help="smoke tests")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--select-every", type=int, default=0)
    ap.add_argument("--n-val-cases", type=int, default=256)
    ap.add_argument("--n-val-cells", type=int, default=500)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); t0 = time.time()
    dz = json.load(open(LCWORK / "design.json"))[a.repeat]; train_d, tests = dz["order"][:a.k], dz["test"]
    pairs = [(s, c) for s, c in PAIRS if not a.celltypes or c in a.celltypes]
    cts = sorted({c for _, c in pairs} | (set(BLOOD_CTS) if a.arm == "mfm_blood" else set()))

    # blocks on the GPU: (celltype, cond, donor) -> (train-side rows, all rows)
    blk, panel = {}, None
    for c in cts:
        z = np.load(CACHE / f"{c}.npz"); panel = list(z["markers"])
        for key in z.files:
            if key == "markers": continue
            cond, d = key.split("|")
            if d not in train_d and not (d in tests and cond == "Unstim"): continue
            x = torch.from_numpy(z[key]).to(dev)
            blk[(c, cond, d)] = (x[:max(1, int(round(len(x) * (1 - VAL_FRAC))))], x)
    fi = torch.tensor([panel.index(m) for m in MARKERS], device=dev)
    oh = lambda i, n: torch.nn.functional.one_hot(torch.tensor(i, device=dev), n).float()
    ct_oh = {c: oh(CELLTYPES.index(c), len(CELLTYPES)) for c in CELLTYPES}
    cond_vec = lambda s, c: torch.cat([oh(STIMS.index(s), len(STIMS)), ct_oh[c]])
    donors_of = {(s, c): [d for d in train_d if (c, "Unstim", d) in blk and (c, s, d) in blk] for s, c in pairs}
    train_pairs = [p for p in pairs if donors_of[p]]
    print(f"repeat {a.repeat} k {a.k} arm {a.arm}: {len(train_pairs)} trainable pairs, {len(blk)} blocks, "
          f"{time.time() - t0:.0f}s", flush=True)

    use_emb = a.arm != "fm"; n_pop = 1 + (len(BLOOD_CTS) if a.arm == "mfm_blood" else 0)
    model = MFM(len(MARKERS), len(panel), len(STIMS) + len(CELLTYPES), len(CELLTYPES), n_pop=n_pop,
                use_embedding=use_emb, knn_k=a.knn_k).to(dev)
    opt_flow = torch.optim.Adam(model.decoder.parameters(), lr=a.lr)
    opt_gnn = torch.optim.Adam(model.gnn.parameters(), lr=a.lr) if use_emb else None
    gen = torch.Generator(device="cpu").manual_seed(a.seed)
    pick = lambda x, n: x[torch.randint(0, len(x), (min(n, len(x)) if n else len(x),), generator=gen).to(dev)] \
        if len(x) > n else x

    def populations(c, d, side):
        """Embedding inputs of donor d for celltype c: own Unstim population (+ dominant celltypes)."""
        pops = [(pick(blk[(c, "Unstim", d)][side], a.n_embed), ct_oh[c])]
        if a.arm == "mfm_blood":
            for bc in BLOOD_CTS:
                x = blk[(bc, "Unstim", d)][side] if (bc, "Unstim", d) in blk else blk[(c, "Unstim", d)][side]
                pops.append((pick(x, a.n_embed), ct_oh[bc]))
        return pops

    log = []
    vgen = torch.Generator(device="cpu").manual_seed(a.seed + 3)
    vcases = []
    for _ in range(a.n_val_cases if a.select_every else 0):
        s, c = train_pairs[torch.randint(0, len(train_pairs), (1,), generator=vgen).item()]
        ds = donors_of[(s, c)]; d = ds[torch.randint(0, len(ds), (1,), generator=vgen).item()]
        vu = blk[(c, "Unstim", d)]; vs = blk[(c, s, d)]
        val = lambda b: b[1][len(b[0]):] if len(b[1]) > len(b[0]) else b[0]      # val side (train side if none)
        vcases.append((s, c, d, val(vu)[:a.n_val_cells], val(vs)))

    @torch.no_grad()
    def val_error():
        model.eval(); e = []
        for s, c, d, xu, xs in vcases:
            z = model.embed(populations(c, d, 0)) if use_emb else None
            y = model.transport(z, xu[:, fi], cond_vec(s, c).expand(len(xu), -1), 20)
            e.append((y.median(0).values - xs[:, fi].median(0).values).abs().mean().item())
        model.train()
        return float(np.mean(e))
    best_err, best_state, best_step = float("inf"), None, 0
    for step in range(1, a.steps + 1):
        gnn_step = use_emb and step % 2 == 0
        losses = []
        for _ in range(a.batch):
            s, c = train_pairs[torch.randint(0, len(train_pairs), (1,), generator=gen).item()]
            ds = donors_of[(s, c)]; d = ds[torch.randint(0, len(ds), (1,), generator=gen).item()]
            x0 = pick(blk[(c, "Unstim", d)][0], a.n_flow)[:, fi]; x1 = pick(blk[(c, s, d)][0], a.n_flow)[:, fi]
            n = min(len(x0), len(x1)); x0, x1 = x0[:n], x1[:n]
            if use_emb:
                with torch.set_grad_enabled(gnn_step):
                    z = model.embed(populations(c, d, 0))
                if not gnn_step: z = z.detach()
            else:
                z = None
            losses.append(model.loss(z, x0, x1, cond_vec(s, c).expand(n, -1)))
        loss = torch.stack(losses).mean()
        opt = opt_gnn if gnn_step else opt_flow
        opt_flow.zero_grad(set_to_none=True)
        if opt_gnn: opt_gnn.zero_grad(set_to_none=True)
        loss.backward(); opt.step()
        if step % a.log_every == 0:
            log.append({"step": step, "loss": loss.item(), "kind": "gnn" if gnn_step else "flow"})
            print(f"step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        if a.select_every and step % a.select_every == 0:
            ve = val_error(); log.append({"step": step, "val_err": ve})
            if ve < best_err:
                best_err, best_step = ve, step
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"[VAL ] step {step} val median err {ve:.4f} (best {best_err:.4f} @ {best_step})", flush=True)
    if best_state is not None: model.load_state_dict(best_state)

    # predict the test donors
    model.eval()
    maps = [(s, c) for s, c in pairs]
    pred = np.full((1, len(maps), len(tests), len(QS), len(MARKERS)), np.nan, np.float32)
    qt = torch.tensor([q / 100 for q in QS], device=dev)
    gen.manual_seed(a.seed + 7)
    for i, (s, c) in enumerate(maps):
        if not donors_of[(s, c)]: continue
        for j, d in enumerate(tests):
            if (c, "Unstim", d) not in blk: continue
            x = blk[(c, "Unstim", d)][1]
            if len(x) < MIN_TEST_CELLS: continue
            with torch.no_grad():
                z = model.embed(populations(c, d, 1)) if use_emb else None
                x0 = x[:a.max_test][:, fi]
                y = torch.cat([model.transport(z, x0[b:b + 20_000], cond_vec(s, c).expand(len(x0[b:b + 20_000]), -1),
                                               a.ode_steps) for b in range(0, len(x0), 20_000)])
            pred[0, i, j] = torch.quantile(y, qt, dim=0).cpu().numpy()
    pd.DataFrame([{"stim": s, "celltype": c, "k": a.k, "cap": 0, "train": " ".join(donors_of[(s, c)])}
                  for s, c in maps]).to_csv(out / "maps.csv", index_label="map")
    durable_save(out / "pred_q.npy", lambda fh: np.save(fh, pred))
    def wmodel(fh):
        buf = io.BytesIO(); torch.save(model.state_dict(), buf); fh.write(buf.getvalue())
    durable_save(out / "model.pt", wmodel)
    pd.DataFrame(log).to_csv(out / "log.csv", index=False)
    json.dump({**vars(a), "tests": tests, "train": train_d, "blood_cts": BLOOD_CTS, "panel": panel,
               "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
               "best_step": best_step, "best_val_err": best_err if best_state is not None else None,
               "hours": (time.time() - t0) / 3600}, open(out / "meta.json", "w"), indent=1)
    print(f"[DONE ] {out} {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
