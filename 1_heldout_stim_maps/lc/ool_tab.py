"""Apply the best held-out-donor model to the OOL cohort (177 samples), for the Fig 5b Stabl protocol (DOS).

Model: shift + TabPFN residual (lc/tab_v2.py, compact layer C_ms_lin, full r_hat, default v2 checkpoint), trained on
all 19 atlas donors (leave-that-donor-out means for training rows), one table per (stim, celltype) of the 168 pairs,
all 7 stims (PI and IL33 included: Stabl uses them).
OOL inputs: measured Unstim quantiles (simco_work/onset_test/ml_dos/measured_unstim_full.csv, q10-q90 of the 13
functional markers, log10 frequencies). OOL is off the atlas scale (protocol: up to -1.8 arcsinh, atlas_vs_ool.ipynb),
so every OOL Unstim quantile and log frequency is HARMONISED: centred on the OOL cohort mean and moved to the atlas
mean (U~ = U - mean_OOL U + mean_atlas U). Hence u_dev = U - mean_OOL U, mk28 likewise, lineage log-frequencies centred
the same way; hvpv = 1 (pregnant); noise_u = the atlas donors' median. The prediction is
  S_hat = mean_atlas S + u_dev + r_hat
(constants per column are irrelevant to Stabl, which standardises: the new information is r_hat).
--raw (2026-10-23, same footing as CellOT, which was applied to the raw OOL cells): no OOL centring; u_dev, mk28 =
OOL value - the atlas donors' mean (off the atlas scale, as CellOT's inputs), lineage log-frequencies raw.
Writes WORK/ool/pred_q.pkl: DataFrame (sample, stim, celltype, q, marker, shift, rhat).
    python -m lc.ool_tab [--stims IFNa --celltypes CD8Tem]   (resumes per stim)
"""
import argparse
import pickle
import time

import numpy as np
import pandas as pd

from lc import tab_v2 as V
from lc.paths import DONORS, MARKERS, PAIRS, QS, WORK
from lc.tab_features import CTS, tables

MEAS = "/scratch/users/bmerwan/simco_work/onset_test/ml_dos/measured_unstim_full.csv"
OUT = WORK / "ool"
STIMS7 = ["GMCSF", "IFNa", "IL246", "IL33", "LPS", "PI", "TNFa"]
QK = (50, 75, 90); QI = [QS.index(q) for q in QK]
COLS = V.LAYERS_CT["C_ms_lin"]


def ool_inputs(raw=False):
    """OOL deviations from the OOL mean (raw: from the atlas mean): {sample: (28, 5, 13)} (NaN where missing) and lineage log-freq deviations."""
    M = pd.read_csv(MEAS, index_col=0)
    X = np.full((len(M), len(CTS), 5, 13), np.nan)
    for j, c in enumerate(CTS):
        for qi, q in enumerate(QS):
            X[:, j, qi] = M[[f"{c}__{m}__q{q}" for m in MARKERS]].to_numpy(float)
    if raw:
        T = tables(); A = np.full((len(CTS), 5, 13), np.nan)
        for j, c in enumerate(CTS):
            A[j] = np.mean([T[(c, "Unstim", d)][0] for d in DONORS if (c, "Unstim", d) in T], 0)
        dev = X - A
    else:
        dev = X - np.nanmean(X, 0)
    fr = np.power(10.0, M[[f"{c}__freq" for c in CTS]].to_numpy(float))       # log10 share -> share
    fr = pd.DataFrame(np.nan_to_num(fr), index=M.index, columns=CTS)
    flin = pd.DataFrame({f"flin_{g}": np.log(fr[cs].sum(1).clip(lower=1e-6)) for g, cs in V.LINEAGES.items()})
    if raw: return list(M.index), dev, flin - V.donor_level(list(DONORS))[V.FLIN].mean()
    return list(M.index), dev, flin - flin.mean()


def ool_rows(c, stim, samples, dev, flin_dev, B, DL):
    """OOL test rows for (stim, c) with the C_ms_lin columns, on the training rows' feature scale."""
    T = tables(); j = CTS.index(c)
    trd = sorted(B.donor.unique())
    U = [T[(c, "Unstim", d)][0] for d in trd]; S = [T[(c, stim, d)][0] for d in trd]
    mu_u, mu_s, sd_u = np.mean(U, 0), np.mean(S, 0), np.std(U, 0)
    noise = np.median([T[(c, "Unstim", d)][1] for d in trd], 0)
    flin_atlas = DL.loc[trd, V.FLIN].mean()
    sl = lambda A: A[QI].ravel()
    mi = np.tile(np.arange(13), len(QK)); qi = np.repeat(QI, 13)
    out = []
    for i, s in enumerate(samples):
        ud = dev[i, j]
        if np.isnan(ud).all(): continue
        f = {"sample": s, "q": np.repeat(QK, 13), "marker": mi, "qi": qi, "mi": mi,
             "u_dev": sl(ud), "mu_u": sl(mu_u), "eff": sl(mu_s - mu_u), "sd_u": sl(sd_u), "noise_u": sl(noise),
             "shift": sl(mu_s + ud), "hvpv": 1.0}
        f.update({f"om_mean_{m}": ud[:, k].mean() for k, m in enumerate(MARKERS)})
        f.update({f"om_spr_{m}": ud[QS.index(90), k] - ud[QS.index(10), k] for k, m in enumerate(MARKERS)})
        Xd = dev[i][:, qi, mi].T                                                  # rows x 28
        f.update(dict(zip(V.MK, Xd.T)))
        with np.errstate(all="ignore"):
            f["dw_mean"] = np.nanmean(Xd, 1); f["dw_sd"] = np.nanstd(Xd, 1)
        f.update((flin_atlas + flin_dev.loc[s]).to_dict())
        out.append(pd.DataFrame(f))
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stims", nargs="*", default=STIMS7)
    ap.add_argument("--celltypes", nargs="*", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--raw", action="store_true", help="no OOL centring (as CellOT); outputs pred_q_raw<tag>.pkl")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    samples, dev, flin_dev = ool_inputs(a.raw)
    if a.raw: a.tag = "raw" + a.tag
    train = list(DONORS); DL = V.donor_level(train)
    ck = OUT / f"partial{a.tag}.pkl"; done, parts = set(), []
    if ck.exists(): done, parts = pickle.load(open(ck, "rb")); print(f"[RESUM] {sorted(done)}", flush=True)
    for stim in a.stims:
        if stim in done: continue
        t0 = time.time()
        for c in sorted({c for s, c in PAIRS if s == stim}):
            if a.celltypes and c not in a.celltypes: continue
            B = V.build(0, len(train), stim, [c], train, [], QK)
            if B is None: continue
            te = ool_rows(c, stim, samples, dev, flin_dev, B, DL)
            r = V.fit_tabpfn(B[COLS], B.target, te[COLS], ["marker"])
            parts.append(pd.DataFrame({"sample": te["sample"], "stim": stim, "celltype": c, "q": te.q,
                                       "marker": np.array(MARKERS)[te.mi], "shift": te["shift"], "rhat": r}))
        done.add(stim); pickle.dump((done, parts), open(ck, "wb"))
        print(f"[STIM ] {stim} {time.time() - t0:.0f}s", flush=True)
    R = pd.concat(parts, ignore_index=True); R.to_pickle(OUT / f"pred_q{a.tag}.pkl"); ck.unlink(missing_ok=True)
    print(f"[DONE ] {len(R)} rows, {R['sample'].nunique()} samples -> {OUT / f'pred_q{a.tag}.pkl'}")


if __name__ == "__main__":
    main()
