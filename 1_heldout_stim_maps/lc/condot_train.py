"""CondOT-faithful stim maps on the learning-curve design (2026-10-06): one conditional map shared across stims.

CondOT (Bunne, Krause, Cuturi, NeurIPS 2022) conditions the OT map on a description of the PERTURBATION, not of the
donor. Here, as in its `onehot` experiment:
  --mode celltype : one PICNN pair (f, g) per celltype; context = one-hot of the stim (7 stims). The map for
                    (stim, celltype) = grad_x g(x; onehot(stim)).
  --mode global   : one PICNN pair for all 168 (stim, celltype) pairs; context = one-hot stim (7) + one-hot celltype (28).
Training follows CondOT / CellOT exactly: cells pooled over the k training donors, unpaired; each step draws ONE
condition (CondOT samples one target key per iteration), then 10 g steps on source batches and 1 f step with a target
batch of that condition (batch 256); Adam 1e-4, betas (0.5, 0.9); f's W_z clamped >= 0, g penalised (fnorm 1);
uniform(0, 0.1) kernel init, hidden [64, 64, 64, 64], LeakyReLU(0.2) (configs/condot.yaml). Source = the celltype's
Unstim cells, target = its stim cells, train side of each training donor (same split as lc.train.Cells).
Checkpoint: g with the lowest mean (over conditions) median error on the training donors' val-side cells, evaluated
every --eval-every steps (as lc.train's best-g; no test-donor information).
Prediction: each test donor's cached Unstim cells (<= MAX_TEST_CELLS) through g(.; condition) -> quantiles QS.
Output in lc.train's format so lc.where_skill / load_train read it: maps.csv (stim, celltype, k), pred_q.npy
(1, M, n_test, n_q, 13), meta.json.
    python -m lc.condot_train --repeat 0 --mode celltype --celltypes pDCs --out WORK/condot/celltype/repeat_0/pDCs
    python -m lc.condot_merge ...   (per-celltype outputs -> one dir per repeat)
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lc.paths import CACHE, MARKERS, PAIRS, QS, WORK
from lc.picnn import PICNN
from lc.train import ADAM, BATCH, INNER, MAX_TEST_CELLS, MIN_TEST_CELLS, N_VAL, Cells

D = len(MARKERS)
STIMS = sorted({s for s, _ in PAIRS})
CTS = sorted({c for _, c in PAIRS})


def init_uniform(w):
    return torch.nn.init.uniform_(w, a=0.0, b=0.1)          # configs/condot.yaml kernel_init_fxn {name: uniform, b: 0.1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, required=True)
    ap.add_argument("--mode", choices=["celltype", "global"], required=True)
    ap.add_argument("--celltypes", nargs="*", default=None)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--n-iters", type=int, default=50_000)
    ap.add_argument("--eval-every", type=int, default=2_500)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.repeat); random.seed(a.repeat); np.random.seed(a.repeat)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if (out / "pred_q.npy").exists():
        print(f"[SKIP ] {out} done", flush=True); return

    dz = json.load(open(WORK / "design.json"))[a.repeat]; tests, train = dz["test"], dz["order"][:a.k]
    cts = a.celltypes or CTS
    keys = [(s, c) for s, c in PAIRS if c in cts]
    cells = Cells(sorted(cts), dev)
    X = cells.X

    def rows(c, cond, part):
        idx = []
        for d in train:
            if (c, cond, d) not in cells.blk or (c, "Unstim", d) not in cells.blk: continue
            s, n, ntr = cells.blk[(c, cond, d)]
            if part == "train" or n - ntr == 0: idx.append(torch.arange(s, s + ntr))
            else: idx.append(torch.arange(s + ntr, s + n))
        return torch.cat(idx).to(dev) if idx else None

    keys = [k for k in keys if rows(k[1], k[0], "train") is not None]
    src_tr = {c: rows(c, "Unstim", "train") for c in cts}
    src_va = {c: rows(c, "Unstim", "val") for c in cts}
    tgt_tr = {k: rows(k[1], k[0], "train") for k in keys}
    tgt_va = {k: rows(k[1], k[0], "val") for k in keys}
    cdim = len(STIMS) + (len(CTS) if a.mode == "global" else 0)

    def ctx(k, n):
        v = torch.zeros(cdim, device=dev); v[STIMS.index(k[0])] = 1
        if a.mode == "global": v[len(STIMS) + CTS.index(k[1])] = 1
        return v.expand(n, -1)

    kw = dict(input_dim=D, input_dim_label=cdim, hidden_units=[64, 64, 64, 64], activation="leakyrelu",
              softplus_wz_kernels=False, kernel_init_fxn=init_uniform)
    f = PICNN(fnorm_penalty=0, **kw).to(dev); g = PICNN(fnorm_penalty=1, **kw).to(dev)
    of, og = torch.optim.Adam(f.parameters(), **ADAM), torch.optim.Adam(g.parameters(), **ADAM)
    draw = lambda idx, n: X[idx[torch.randint(len(idx), (n,), device=dev)]]

    gen = torch.Generator(device="cpu"); gen.manual_seed(7)
    VAL = {k: (src_va[k[1]][torch.randint(len(src_va[k[1]]), (N_VAL,), generator=gen).to(dev)],
               tgt_va[k][torch.randint(len(tgt_va[k]), (N_VAL,), generator=gen).to(dev)]) for k in keys}

    def val_err():
        e = []
        for k, (si, ti) in VAL.items():
            x = X[si].clone().requires_grad_(True)
            T = g.transport(x, ctx(k, len(x))).detach()
            e.append((T.median(0).values - X[ti].median(0).values).abs().mean().item())
        return float(np.mean(e))

    best, best_err, log = None, np.inf, []
    t0 = time.time()
    for step in range(a.n_iters):
        k = random.choice(keys)                         # CondOT: one condition per iteration
        c = ctx(k, BATCH)
        for _ in range(INNER):
            x = draw(src_tr[k[1]], BATCH).requires_grad_(True)
            og.zero_grad(set_to_none=True)
            T = g.transport(x, c)
            gl = (f(T, c) - (x * T).sum(-1, keepdim=True)).mean() + g.penalize_w()
            gl.backward(); og.step()
        x = draw(src_tr[k[1]], BATCH).requires_grad_(True); y = draw(tgt_tr[k], BATCH)
        of.zero_grad(set_to_none=True)
        T = g.transport(x, c)
        fl = (-f(T, c) + f(y, c)).mean()
        fl.backward(); of.step(); f.clamp_w()
        if not torch.isfinite(fl): raise FloatingPointError(f"non-finite f loss at step {step}")
        if (step + 1) % a.eval_every == 0:
            e = val_err(); log.append({"step": step + 1, "val_median_err": e})
            if e < best_err: best_err, best = e, {n: p.detach().clone() for n, p in g.state_dict().items()}
            print(f"[EVAL ] step {step + 1}/{a.n_iters} val median err {e:.4f} (best {best_err:.4f}) "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
    g.load_state_dict(best)

    pred = np.full((1, len(keys), len(tests), len(QS), D), np.nan, np.float32)
    qt = torch.tensor([q / 100 for q in QS], device=dev)
    for i, k in enumerate(keys):
        for j, d in enumerate(tests):
            if (k[1], "Unstim", d) not in cells.blk: continue
            s, n, ntr = cells.blk[(k[1], "Unstim", d)]
            if n < MIN_TEST_CELLS: continue
            xs = X[s:s + min(n, MAX_TEST_CELLS)]
            T = []
            for s2 in range(0, len(xs), 20_000):
                x = xs[s2:s2 + 20_000].clone().requires_grad_(True)
                T.append(g.transport(x, ctx(k, len(x))).detach())
            pred[0, i, j] = torch.quantile(torch.cat(T), qt, dim=0).cpu().numpy()
    pd.DataFrame([{"stim": s, "celltype": c, "k": a.k} for s, c in keys]).to_csv(out / "maps.csv", index_label="map")
    pd.DataFrame(log).to_csv(out / "val_log.csv", index=False)
    json.dump({"repeat": a.repeat, "mode": a.mode, "celltypes": cts, "k": a.k, "tests": tests, "n_iters": a.n_iters,
               "best_val_median_err": best_err, "context_dim": cdim, "device": str(dev),
               "hours": (time.time() - t0) / 3600}, open(out / "meta.json", "w"), indent=1)
    np.save(out / "pred_q.npy", pred)
    print(f"[DONE ] {len(keys)} maps, {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
