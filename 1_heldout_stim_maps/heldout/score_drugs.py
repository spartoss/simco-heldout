"""Drug effects on held-out donors: how well does each pipeline predict a new donor's drug effect?

For learning-curve repeat r (4 test donors; maps trained on the other 15), for each test donor, (stim or Unstim,
celltype) of the grid and drug, with q in 10/25/50/75/90 and the 13 markers:
    true   = Q(her drug-well cells) - Q(her control-well cells)            (batch-corrected drug cache)
    noise  = |true on one random half of her cells - true on the other|
Predicted effects:
  step:<X>          drug map X applied to her TRUE control cells: Q(X(ctrl)) - Q(ctrl)   (the drug step alone)
  chain:<S>><X>     her unstimulated cells -> stim map S -> drug map X on cells: Q(X(S(U))) - Q(S(U))
  feature:<S>><X>   her unstimulated cells -> stim map S -> quantiles; per-marker drug map X on those values
Stim maps S: CellOT (the learning curve's k = 15 maps of that repeat, same 15 donors) and gauss_ot. Drug maps X: CellOT
(work/heldout/drugs/maps/r<r>, control -> drug under the stim, 128-128-64-32) and the closed-form maps fitted on the
same 15 donors' cells. Unstim "pairs" have no stim step (S = identity).
    python -m heldout.score_drugs --repeat 0 -> WORK/drugs/scores/r<r>.csv.gz
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from heldout.common import CACHE_DRUG, DRUGS, LC, WORK
from lc.baselines import apply as cf_apply, fit as cf_fit
from lc.batched_icnn import BatchedICNN
from lc.paths import MARKERS, PAIRS
from lc.train import sub

QS = (10, 25, 50, 75, 90)
MAXC = 5000


def load_maps(dirs, hidden, dev, keep=lambda r: True):
    out = {}
    for d in dirs:
        m = pd.read_csv(d / "maps.csv"); st = torch.load(d / "final_maps.pt", map_location=dev)["g_last"]
        g = BatchedICNN(len(m), len(MARKERS), hidden, fnorm_penalty=1).to(dev); g.load_state_dict(st)
        for i, r in m.iterrows():
            if keep(r): out[tuple(r[c] for c in ("stim", "celltype", "drug") if c in m)] = (g, i)
    return out


def transport(entry, x, dev):
    g, i = entry
    xb = torch.from_numpy(x).to(dev).unsqueeze(0).requires_grad_(True)
    return sub(g, torch.tensor([i], device=dev)).transport(xb).detach()[0].cpu().numpy()


def Q(x): return np.percentile(x, QS, axis=0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repeat", type=int, required=True)
    ap.add_argument("--celltypes", nargs="*", default=None); ap.add_argument("--drug-maps", default=None)
    ap.add_argument("--out", default=str(WORK / "drugs/scores")); a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dz = json.load(open(LC / "design.json"))[a.repeat]; tests, train = dz["test"], dz["order"]
    plate = json.load(open(CACHE_DRUG / "plate_controls.json"))
    stim_cellot = load_maps([LC / f"train/repeat_{a.repeat}"], (64, 64, 64, 64), dev, keep=lambda r: r.k == 15)
    ddirs = [a.drug_maps] if a.drug_maps else sorted((WORK / f"drugs/maps/r{a.repeat}").glob("h*"))
    drug_cellot = load_maps([__import__("pathlib").Path(d) for d in ddirs], (128, 128, 64, 32), dev)
    rng = np.random.default_rng(a.repeat)
    rows, t0 = [], time.time()
    for ct in sorted({c for _, c in PAIRS}):
        if a.celltypes and ct not in a.celltypes: continue
        zs = np.load(LC / f"cache/{ct}.npz"); zd = np.load(CACHE_DRUG / f"{ct}.npz")
        conds = ["Unstim"] + sorted({s for s, c in PAIRS if c == ct})
        for s in conds:
            # stim maps S on the training donors (closed form); CellOT from the learning curve
            S = {}
            if s != "Unstim":
                tr_u = [zs[f"Unstim|{p}"] for p in train if f"Unstim|{p}" in zs.files and f"{s}|{p}" in zs.files]
                tr_s = [zs[f"{s}|{p}"] for p in train if f"Unstim|{p}" in zs.files and f"{s}|{p}" in zs.files]
                gp = cf_fit(np.concatenate(tr_u), np.concatenate(tr_s))
                S["gauss_ot"] = lambda x, gp=gp: cf_apply("gauss_ot", x, gp)
                if (s, ct) in stim_cellot: S["CellOT"] = lambda x, e=stim_cellot[(s, ct)]: transport(e, x, dev)
            else:
                S = {"CellOT": lambda x: x, "gauss_ot": lambda x: x}
            for drug in DRUGS:
                ck, dk = f"{s}|{plate[drug]}", f"{s}|{drug}"
                ptr = [p for p in train if f"{ck}|{p}" in zd.files and f"{dk}|{p}" in zd.files]
                if not ptr: continue
                src = np.concatenate([zd[f"{ck}|{p}"] for p in ptr]); tgt = np.concatenate([zd[f"{dk}|{p}"] for p in ptr])
                cp = cf_fit(src, tgt)
                X = {m: (lambda x, m=m, cp=cp: cf_apply(m, x, cp)) for m in ("gauss_ot", "quantile_map", "median_shift")}
                if (s, ct, drug) in drug_cellot: X["CellOT"] = lambda x, e=drug_cellot[(s, ct, drug)]: transport(e, x, dev)
                for d in tests:
                    if f"{ck}|{d}" not in zd.files or f"{dk}|{d}" not in zd.files or f"Unstim|{d}" not in zs.files: continue
                    ctrl, drg, U = zd[f"{ck}|{d}"], zd[f"{dk}|{d}"], zs[f"Unstim|{d}"][:MAXC]
                    if min(len(ctrl), len(drg), len(U)) < 20: continue
                    true = Q(drg) - Q(ctrl)
                    pc, pd_ = rng.permutation(len(ctrl)), rng.permutation(len(drg))
                    half = lambda sel_c, sel_d: Q(drg[sel_d]) - Q(ctrl[sel_c])
                    noise = np.abs(half(pc[:len(ctrl) // 2], pd_[:len(drg) // 2]) - half(pc[len(ctrl) // 2:], pd_[len(drg) // 2:]))
                    pred = {}
                    for xm, f in X.items(): pred[f"step:{xm}"] = Q(f(ctrl)) - Q(ctrl)
                    for sm, fs in S.items():
                        Z = fs(U); qz = Q(Z)
                        if sm in X: pred[f"chain:{sm}>{sm}"] = Q(X[sm](Z)) - qz
                        for xm in ("quantile_map", "median_shift"):
                            pred[f"feature:{sm}>{xm}"] = X[xm](qz) - qz        # monotone per marker: acts on the values
                    for qi, q in enumerate(QS):
                        for mi, m in enumerate(MARKERS):
                            rows.append({"repeat": a.repeat, "donor": d, "celltype": ct, "stim": s, "drug": drug, "q": q,
                                         "marker": m, "true": true[qi, mi], "noise": noise[qi, mi],
                                         **{k: v[qi, mi] for k, v in pred.items()}})
        print(f"[DONE ] {ct} ({time.time() - t0:.0f}s, {len(rows)} rows)", flush=True)
    out = __import__("pathlib").Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"r{a.repeat}.csv.gz", index=False)
    print(f"[DONE ] repeat {a.repeat}: {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
