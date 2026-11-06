"""Drug arm, held-out donors: predict a new donor's drug + stim quantiles and drug effects from its Unstim profile (TabPFN).

Truth (batch-corrected drug cache, lc/drug_tables.py): for a (drug g, stim s, celltype c, donor):
  D = Q(s | g), C = Q(s | plate DMSO of g) (its control), drug effect E = D - C; q50 / q75 / q90 x 13 markers.
Input: the donor's Unstim profile, as the stim models: the compact feature set C_ms_lin of lc/tab_v2.py, built for (s, c)
on the fold's training donors. Leave-that-donor-out training means everywhere, as in lc/tab_v2.py.
Models, one TabPFN table per (fold, stim, celltype, plate / drug):
- control  C:  target (C - mean C) - u_dev, extra feature effC = mean C - mu_u;  C_hat = mean C + u_dev + r_hat
- direct   D:  target (D - mean D) - u_dev, extra feature effD = mean D - mu_u;  D_hat = mean D + u_dev + r_hat  (strategy A)
- effect   E:  target E - mean E, extra features effE = mean E and the donor's control profile: the row's control
               deviation (C - mean C), and per marker its mean over q50-q90 and its q90 - q50 spread. Training donors
               use their measured control, the test donor its predicted control C_hat (strategy B, "two-step").
Predictions kept for the test donor (per drug, stim, celltype, q, marker): truths D, C, E and their split-half noise;
  C0 = mean C + u_dev, C_hat; D0 = mean D + u_dev (drug "shift"), D_hat; E0 = mean E (mean training drug effect),
  E_A = D_hat - C_hat, E_B; D_B = C_hat + E_B; D_C0E = C_hat + E0.
Writes WORK/drug_tab/lodo/fold_<f>.pkl; checkpoints after each stim (rerun resumes).
    python -m lc.drug_tab --fold 0 [--stims IFNa --celltypes CD8Tem --drugs SA --tag _smoke]
"""
import argparse
import json
import pickle
import time

import numpy as np
import pandas as pd

from lc import tab_v2 as V
from lc.paths import MARKERS, PAIRS, QS, WORK

OUT = WORK / "drug_tab"
STIMS = ["GMCSF", "IFNa", "IL246", "LPS", "TNFa"]
QK = (50, 75, 90); QI = [QS.index(q) for q in QK]
COLS = V.LAYERS_CT["C_ms_lin"]
CPROF = [f"cpm_{m}" for m in MARKERS] + [f"cps_{m}" for m in MARKERS]


def means(vals, train_ds, test_ds):
    """donor -> mean of vals over the other training donors (training) or over all training donors (test)."""
    tot = np.sum([vals[x] for x in train_ds], 0); n = len(train_ds)
    out = {x: (tot - vals[x]) / (n - 1) for x in train_ds}
    for x in test_ds: out[x] = tot / n
    return out


def row_take(M, rows):
    """M: donor -> (5, 13); rows: DataFrame with donor, qi, mi -> array of M[donor][qi, mi]."""
    return np.array([M[d][qi, mi] if d in M else np.nan for d, qi, mi in zip(rows.donor, rows.qi, rows.mi)])


def fit_predict(B, train_ds, y, extra):
    tr = B[B.donor.isin(train_ds) & ~B.is_test]; te = B[B.is_test]
    cols = COLS + list(extra)
    X = B[cols]
    p = V.fit_tabpfn(X.loc[tr.index], y[tr.index], X.loc[te.index], ["marker"])
    return pd.Series(p, index=te.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--stims", nargs="*", default=STIMS)
    ap.add_argument("--celltypes", nargs="*", default=None)
    ap.add_argument("--drugs", nargs="*", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    G = pickle.load(open(OUT / "tables.pkl", "rb")); DT, plates = G["T"], G["plates"]
    drugs = a.drugs or sorted(plates)
    dz = json.load(open(WORK / "design_lodo.json"))[a.fold]; train, tests = dz["order"], dz["test"]
    out = OUT / f"lodo{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    ck = out / f"partial_fold_{a.fold}.pkl"; done, parts = set(), []
    if ck.exists(): done, parts = pickle.load(open(ck, "rb")); print(f"[RESUM] {sorted(done)}", flush=True)
    for s in a.stims:
        if s in done: continue
        t0 = time.time()
        for c in sorted({c for st, c in PAIRS if st == s}):
            if a.celltypes and c not in a.celltypes: continue
            B = V.build(a.fold, 18, s, [c], train, tests, QK)
            if B is None or not B.is_test.any(): continue
            trd = sorted(B.loc[~B.is_test, "donor"].unique()); ted = sorted(B.loc[B.is_test, "donor"].unique())
            te = B[B.is_test]
            Chat, Cmean = {}, {}
            for p in sorted({plates[g] for g in drugs}):
                dC = [x for x in trd if (c, s, p, x) in DT]
                if len(dC) < 5: continue
                Cv = {x: DT[(c, s, p, x)][0] for x in dC}; mC = means(Cv, dC, ted)
                rowm = row_take(mC, B); B["effT"] = rowm - B.mu_u
                y = pd.Series(row_take(Cv, B) - rowm - B.u_dev.to_numpy(), index=B.index)
                r = fit_predict(B, dC, y, ["effT"])
                Cmean[p] = rowm[B.index.get_indexer(te.index)]
                Chat[p] = Cmean[p] + te.u_dev.to_numpy() + r.to_numpy()
            for g in drugs:
                p = plates[g]
                if p not in Chat: continue
                dD = [x for x in trd if (c, s, g, x) in DT and (c, s, p, x) in DT]
                if len(dD) < 5: continue
                Dv = {x: DT[(c, s, g, x)][0] for x in dD}; Cv = {x: DT[(c, s, p, x)][0] for x in dD}
                Ev = {x: Dv[x] - Cv[x] for x in dD}
                mD, mE, mC = means(Dv, dD, ted), means(Ev, dD, ted), means(Cv, dD, ted)
                # strategy A: drug + stim directly from Unstim
                rD = row_take(mD, B); B["effT"] = rD - B.mu_u
                yD = pd.Series(row_take(Dv, B) - rD - B.u_dev.to_numpy(), index=B.index)
                Dhat = rD[B.index.get_indexer(te.index)] + te.u_dev.to_numpy() + fit_predict(B, dD, yD, ["effT"]).to_numpy()
                # strategy B: drug effect from Unstim + the control profile (measured for training, predicted for test)
                rE = row_take(mE, B); B["effT"] = rE
                cd = {x: Cv[x] - mC[x] for x in dD}
                tC = te.assign(chat=Chat[p]); Ct = {}
                for d0, h in tC.groupby("donor"):
                    M = np.full((5, 13), np.nan); M[h.qi.to_numpy(), h.mi.to_numpy()] = h.chat.to_numpy(); Ct[d0] = M - mC[d0]
                cd.update(Ct)
                B["cdev"] = row_take(cd, B)
                for j, m in enumerate(MARKERS):
                    B[f"cpm_{m}"] = [np.nanmean(cd[d0][QI, j]) if d0 in cd else np.nan for d0 in B.donor]
                    B[f"cps_{m}"] = [cd[d0][QS.index(90), j] - cd[d0][QS.index(50), j] if d0 in cd else np.nan for d0 in B.donor]
                yE = pd.Series(row_take(Ev, B) - rE, index=B.index)
                EB = rE[B.index.get_indexer(te.index)] + fit_predict(B, dD, yE, ["effT", "cdev"] + CPROF).to_numpy()
                ii = B.index.get_indexer(te.index)
                for d0 in ted:
                    k = (te.donor == d0).to_numpy()
                    Dt = DT.get((c, s, g, d0)); Ct_ = DT.get((c, s, p, d0))
                    qi, mi = te.qi.to_numpy()[k], te.mi.to_numpy()[k]
                    nan = np.full(k.sum(), np.nan)
                    rec = pd.DataFrame({"fold": a.fold, "donor": d0, "drug": g, "plate": p, "stim": s, "celltype": c,
                                        "q": np.array(QS)[qi], "marker": np.array(MARKERS)[mi],
                                        "D": Dt[0][qi, mi] if Dt else nan, "C": Ct_[0][qi, mi] if Ct_ else nan,
                                        "noise_D": Dt[1][qi, mi] if Dt else nan, "noise_C": Ct_[1][qi, mi] if Ct_ else nan,
                                        "C0": Cmean[p][k] + te.u_dev.to_numpy()[k], "C_hat": Chat[p][k],
                                        "D0": rD[ii][k] + te.u_dev.to_numpy()[k], "D_hat": Dhat[k], "E0": rE[ii][k], "E_B": EB[k]})
                    parts.append(rec)
        done.add(s); pickle.dump((done, parts), open(ck, "wb"))
        print(f"[STIM ] fold {a.fold} {s} {time.time() - t0:.0f}s", flush=True)
    R = pd.concat(parts, ignore_index=True)
    R["E"] = R.D - R.C; R["E_A"] = R.D_hat - R.C_hat; R["D_B"] = R.C_hat + R.E_B; R["D_C0E"] = R.C_hat + R.E0
    R.to_pickle(out / f"fold_{a.fold}.pkl"); ck.unlink(missing_ok=True)
    print(f"[DONE ] {len(R)} rows -> {out / f'fold_{a.fold}.pkl'}")


if __name__ == "__main__":
    main()
