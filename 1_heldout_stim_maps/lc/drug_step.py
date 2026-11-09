"""Drug arm, "step" setting: predict a new donor's drug effect knowing its MEASURED control (stim + the drug's plate DMSO,
batch-corrected), the same information the CellOT drug maps get in simco/notebooks/drug_map_choice.ipynb (step).

Same LODO design, truths and tables as lc/drug_tab.py: E = D - C (q50 / q75 / q90 x 13 markers), D = Q(stim | drug),
C = Q(stim | plate DMSO). One TabPFN table per (fold, stim, celltype, drug); target E - mean E (leave-that-donor-out
training means); features = the compact Unstim set (C_ms_lin) + effT (mean training effect) + the donor's control:
- E_S:   the row's control deviation (C - mean C) + per-marker mean over q50-q90 and q90 - q50 spread (26)
- E_S65: E_S + the whole control profile, 13 markers x 5 quantiles of C - mean C (65 columns)
Training AND test donors use their measured control. Output per test row: E, noise, E0 (mean training effect), E_S, E_S65.
Writes WORK/drug_tab/step_<tag>/fold_<f>.pkl (--design repeat: steprep<tag>/fold_<repeat>.pkl); checkpoints per stim.
    python -m lc.drug_step --fold 0 --stims IFNa --tag _IFNa [--celltypes CD8Tem --drugs SA]
"""
import argparse
import json
import pickle
import time

import numpy as np
import pandas as pd

from lc import tab_v2 as V
from lc.drug_tab import COLS, CPROF, OUT, QI, QK, STIMS, means, row_take
from lc.paths import MARKERS, PAIRS, QS, WORK

CQ = [f"cq_q{q}_{m}" for q in QS for m in MARKERS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--stims", nargs="*", default=STIMS)
    ap.add_argument("--celltypes", nargs="*", default=None)
    ap.add_argument("--drugs", nargs="*", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--design", choices=["lodo", "repeat"], default="lodo",
                    help="lodo: design_lodo.json fold (18 training donors); repeat: design.json repeat --fold (its 15 training donors)")
    a = ap.parse_args()
    G = pickle.load(open(OUT / "tables.pkl", "rb")); DT, plates = G["T"], G["plates"]
    drugs = a.drugs or sorted(plates)
    if a.design == "repeat":
        dz = [x for x in json.load(open(WORK / "design.json")) if x["repeat"] == a.fold][0]
        train, tests, K = dz["order"][:15], dz["test"], 15
    else:
        dz = json.load(open(WORK / "design_lodo.json"))[a.fold]; train, tests, K = dz["order"], dz["test"], 18
    out = OUT / (f"steprep{a.tag}" if a.design == "repeat" else f"step{a.tag}"); out.mkdir(parents=True, exist_ok=True)
    ck = out / f"partial_fold_{a.fold}.pkl"; done, parts = set(), []
    if ck.exists(): done, parts = pickle.load(open(ck, "rb")); print(f"[RESUM] {sorted(done)}", flush=True)
    for s in a.stims:
        if s in done: continue
        t0 = time.time()
        for c in sorted({c for st, c in PAIRS if st == s}):
            if a.celltypes and c not in a.celltypes: continue
            B = V.build(a.fold, K, s, [c], train, tests, QK)
            if B is None or not B.is_test.any(): continue
            trd = sorted(B.loc[~B.is_test, "donor"].unique())
            for g in drugs:
                p = plates[g]
                ted = [d for d in tests if (c, s, g, d) in DT and (c, s, p, d) in DT and (B.donor == d).any()]
                dD = [x for x in trd if (c, s, g, x) in DT and (c, s, p, x) in DT]
                if len(dD) < 5 or not ted: continue
                Dv = {x: DT[(c, s, g, x)][0] for x in dD + ted}; Cv = {x: DT[(c, s, p, x)][0] for x in dD + ted}
                Ev = {x: Dv[x] - Cv[x] for x in dD + ted}
                mE, mC = means(Ev, dD, ted), means(Cv, dD, ted)
                cd = {x: Cv[x] - mC[x] for x in dD + ted}                       # measured control for everyone
                rE = row_take(mE, B); B["effT"] = rE; B["cdev"] = row_take(cd, B)
                for j, m in enumerate(MARKERS):
                    B[f"cpm_{m}"] = [np.mean(cd[d0][QI, j]) if d0 in cd else np.nan for d0 in B.donor]
                    B[f"cps_{m}"] = [cd[d0][QS.index(90), j] - cd[d0][QS.index(50), j] if d0 in cd else np.nan for d0 in B.donor]
                for qi, q in enumerate(QS):
                    for j, m in enumerate(MARKERS):
                        B[f"cq_q{q}_{m}"] = [cd[d0][qi, j] if d0 in cd else np.nan for d0 in B.donor]
                y = pd.Series(row_take(Ev, B) - rE, index=B.index)
                tr = B[B.donor.isin(dD) & ~B.is_test]; te = B[B.is_test & B.donor.isin(ted)]
                preds = {}
                for name, extra in [("E_S", ["effT", "cdev"] + CPROF), ("E_S65", ["effT", "cdev"] + CPROF + CQ)]:
                    cols = COLS + extra
                    preds[name] = rE[B.index.get_indexer(te.index)] + V.fit_tabpfn(tr[cols], y[tr.index], te[cols], ["marker"])
                for d0 in ted:
                    k = (te.donor == d0).to_numpy(); qi, mi = te.qi.to_numpy()[k], te.mi.to_numpy()[k]
                    Dt, Ct = DT[(c, s, g, d0)], DT[(c, s, p, d0)]
                    parts.append(pd.DataFrame({"fold": a.fold, "donor": d0, "drug": g, "plate": p, "stim": s, "celltype": c,
                                               "q": np.array(QS)[qi], "marker": np.array(MARKERS)[mi],
                                               "E": Dt[0][qi, mi] - Ct[0][qi, mi], "noise_D": Dt[1][qi, mi], "noise_C": Ct[1][qi, mi],
                                               "E0": rE[B.index.get_indexer(te.index)][k], "E_S": preds["E_S"][k],
                                               "E_S65": preds["E_S65"][k]}))
        done.add(s); pickle.dump((done, parts), open(ck, "wb"))
        print(f"[STIM ] fold {a.fold} {s} {time.time() - t0:.0f}s", flush=True)
    R = pd.concat(parts, ignore_index=True); R.to_pickle(out / f"fold_{a.fold}.pkl"); ck.unlink(missing_ok=True)
    print(f"[DONE ] {len(R)} rows -> {out / f'fold_{a.fold}.pkl'}")


if __name__ == "__main__":
    main()
