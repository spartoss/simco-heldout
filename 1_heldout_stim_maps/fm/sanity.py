"""Phase 0 go/no-go: does the frozen ImmuneFM checkpoint understand the atlas, and can its tokens see a stim response?

Cells: work/lc/fm/cache_full (the lc cache cells, all 38 markers). Donor split = design.json repeat 0 (15 train / 4 test).
1. recon      masked reconstruction (25 % of the non-zero mapped tokens -> MASK, as in pre-training), on test donors:
              accuracy, +-1 accuracy and CE over tokens 1..10 for pretrained / random init / pretrained with the
              marker columns shuffled / per-marker marginal token frequencies (fitted on the training donors).
2. probes     logistic probes trained on training donors, scored on test donors:
              celltype (28 classes, Unstim), stim vs Unstim AUC per (stim, celltype) for PROBE_CTS,
              donor identity (Unstim, within celltype, cells split at random; chance 1/19).
              Inputs: ImmuneFM embedding (pretrained / random init / pretrained with the 13 functional markers masked)
              vs the raw 38 markers vs the raw 25 phenotypic markers.
3. resolution per (stim, celltype, functional marker, donor): |q50(stim) - q50(Unstim)| vs the width of one token
              bin in that donor's Unstim cells; and whether the median TOKEN moves at all.
Output: WORK/fm/sanity/{recon,probes,resolution}.csv and summary.txt.
    python3 -m fm.sanity [--n-per-block 100]          (GPU: slurm/fm_sanity.sbatch; CPU encodes ~90 cells/s)
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from fm.immunefm import MASK, N_BINS, load
from fm.tokenize import bin_width, columns, tokens
from lc.paths import CELLTYPES, DONORS, MARKERS, PAIRS, WORK
FLOOR = 0.0273   # = lc.score.FLOOR (lc.score imports matplotlib, absent from the torch env)

CACHE = WORK / "fm" / "cache_full"
OUT = WORK / "fm" / "sanity"
PROBE_CTS = ["cMCs", "CD4Tnaive", "Bcells", "Granulocytes", "pDCs"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_cells(n_per_block, seed=0):
    """DataFrame-free table: X (n, 38), celltype, cond, donor for up to n_per_block cells per (celltype, cond, donor)."""
    rng = np.random.default_rng(seed)
    X, meta, panel = [], [], None
    for c in CELLTYPES:
        z = np.load(CACHE / f"{c}.npz")
        panel = list(z["markers"]) if panel is None else panel
        assert list(z["markers"]) == panel
        for key in z.files:
            if key == "markers": continue
            cond, d = key.split("|")
            a = z[key]
            a = a[rng.permutation(len(a))[:n_per_block]]
            X.append(a); meta += [(c, cond, d)] * len(a)
    meta = pd.DataFrame(meta, columns=["celltype", "cond", "donor"])
    return np.concatenate(X), meta, panel


def mask_tokens(t, cols, prob=0.25, seed=0):
    """As pretrain.data_mask: among the non-zero tokens of `cols`, mask ceil(prob * count) per cell."""
    g = np.random.default_rng(seed)
    m = np.zeros_like(t, bool)
    sub = t[:, cols]
    for i in range(len(t)):
        nz = np.nonzero(sub[i])[0]
        if len(nz) == 0: continue
        k = int(np.ceil(prob * len(nz)))
        m[i, cols[g.choice(nz, k, replace=False)]] = True
    tm = t.copy(); tm[m] = MASK
    return tm, m


@torch.no_grad()
def recon_scores(model, tm, t, m, batch=4096):
    acc = near = ce = n = 0.0
    for s in range(0, len(t), batch):
        lg = model(torch.from_numpy(tm[s:s + batch]).to(DEV))[..., 1:N_BINS + 1]        # tokens 1..10
        mm = torch.from_numpy(m[s:s + batch]).to(DEV); y = torch.from_numpy(t[s:s + batch]).to(DEV)[mm] - 1
        lg = lg[mm]
        pred = lg.argmax(-1)
        acc += (pred == y).sum().item(); near += ((pred - y).abs() <= 1).sum().item()
        ce += torch.nn.functional.cross_entropy(lg, y, reduction="sum").item(); n += len(y)
    return acc / n, near / n, ce / n


def marginal_scores(t_tr, t_te, m_te, cols):
    acc = near = ce = n = 0.0
    for c in cols:
        v = t_tr[:, c]; v = v[v > 0]
        p = np.bincount(v, minlength=N_BINS + 1)[1:] + 1.0; p /= p.sum()
        y = t_te[m_te[:, c], c] - 1
        if len(y) == 0: continue
        pred = p.argmax()
        acc += (y == pred).sum(); near += (np.abs(y - pred) <= 1).sum(); ce += -np.log(p[y]).sum(); n += len(y)
    return acc / n, near / n, ce / n


def probe(Xtr, ytr, Xte, yte, binary=False):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xtr), ytr)
    if binary:
        return roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1])
    return (clf.predict(sc.transform(Xte)) == yte).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-block", type=int, default=100)
    a = ap.parse_args()
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    tests = json.load(open(WORK / "design.json"))[0]["test"]
    X, meta, panel = load_cells(a.n_per_block)
    is_te = meta.donor.isin(tests).to_numpy()
    src, cols = columns(panel)
    func = [m for m in MARKERS]
    pheno_idx = [i for i, m in enumerate(panel) if m not in func]
    print(f"{len(X)} cells, panel {len(panel)} ({len(src)} mapped), test donors {tests}", flush=True)

    T = tokens(X, panel)
    models = {"pretrained": load(True).to(DEV), "random": load(False).to(DEV)}

    # 1. reconstruction
    rows = []
    te = np.nonzero(is_te)[0]
    tm, m = mask_tokens(T[te], cols)
    for name, mdl in models.items():
        rows.append((name, *recon_scores(mdl, tm, T[te], m)))
    _, cols_sh = columns(panel, shuffle_seed=1)
    Tsh = tokens(X[te], panel, shuffle_seed=1)
    tm_sh, m_sh = mask_tokens(Tsh, cols_sh)
    rows.append(("pretrained_shuffled_cols", *recon_scores(models["pretrained"], tm_sh, Tsh, m_sh)))
    rows.append(("marginal", *marginal_scores(T[~is_te], T[te], m, cols)))
    rec = pd.DataFrame(rows, columns=["arm", "acc", "acc_pm1", "ce"])
    rec.to_csv(OUT / "recon.csv", index=False); print(rec.round(4).to_string(), flush=True)

    # 2. probes
    func_mask = [m for m in MARKERS if m in panel]
    enc = lambda name, t: models[name].encode(t, cols, device=DEV)
    feats = {"fm_pretrained": enc("pretrained", T), "fm_random": enc("random", T),
             "fm_pretrained_funcmasked": enc("pretrained", tokens(X, panel, mask_cols=func_mask)),
             "raw38": X, "raw_pheno": X[:, pheno_idx]}
    print(f"embeddings done {time.time() - t0:.0f}s", flush=True)
    prow = []
    un = (meta.cond == "Unstim").to_numpy()
    for fname, F in feats.items():
        s = un & ~is_te; q = un & is_te
        prow.append((fname, "celltype", "all", probe(F[s], meta.celltype[s], F[q], meta.celltype[q])))
        for c in PROBE_CTS:
            for st in sorted({s_ for s_, c_ in PAIRS if c_ == c}):
                sel = (meta.celltype == c).to_numpy() & meta.cond.isin(["Unstim", st]).to_numpy()
                y = (meta.cond[sel] == st).to_numpy().astype(int)
                tr, tt = ~is_te[sel], is_te[sel]
                prow.append((fname, f"stim_auc:{st}", c, probe(F[sel][tr], y[tr], F[sel][tt], y[tt], binary=True)))
            sel = (meta.celltype == c).to_numpy() & un
            idx = np.random.default_rng(0).permutation(np.nonzero(sel)[0]); h = len(idx) // 2
            prow.append((fname, "donor", c, probe(F[idx[:h]], meta.donor.iloc[idx[:h]], F[idx[h:]], meta.donor.iloc[idx[h:]])))
        print(f"  probes {fname} {time.time() - t0:.0f}s", flush=True)
    pr = pd.DataFrame(prow, columns=["features", "task", "celltype", "score"])
    pr.to_csv(OUT / "probes.csv", index=False)

    # 3. resolution (all cached cells of each block, not the subsample)
    rrow = []
    for c in CELLTYPES:
        z = np.load(CACHE / f"{c}.npz")
        for d in DONORS:
            if f"Unstim|{d}" not in z.files: continue
            U = z[f"Unstim|{d}"]
            if len(U) < 20: continue
            bw = np.median(bin_width(U, panel)); TU = tokens(U, panel)
            for st in sorted({s_ for s_, c_ in PAIRS if c_ == c}):
                if f"{st}|{d}" not in z.files or len(z[f"{st}|{d}"]) < 20: continue
                S = z[f"{st}|{d}"]; TS = tokens(S, panel)
                for mk in MARKERS:
                    i = panel.index(mk); col = cols[list(src).index(i)]
                    dq = np.median(S[:, i]) - np.median(U[:, i])
                    dt = np.median(TS[:, col]) - np.median(TU[:, col])
                    rrow.append((st, c, mk, d, dq, bw, dt))
    res = pd.DataFrame(rrow, columns=["stim", "celltype", "marker", "donor", "dq50", "bin_width", "dtok50"])
    res.to_csv(OUT / "resolution.csv", index=False)

    live = res[res.dq50.abs() > FLOOR]
    lines = [f"cells {len(X)}, test donors {tests}", "", "1. masked reconstruction (test donors)", rec.round(4).to_string(), "",
             "2. probes (mean over celltypes / stims)",
             pr.assign(task=pr.task.str.split(":").str[0]).groupby(["task", "features"]).score.mean().unstack(0).round(3).to_string(), "",
             "3. resolution (live features |dq50| > FLOOR)",
             f"  features {len(res)}, live {len(live)}; median bin width {res.bin_width.median():.3f} arcsinh",
             f"  live features with |dq50| < 1 bin: {(live.dq50.abs() < live.bin_width).mean():.3f}",
             f"  live features whose median token does not move: {(live.dtok50 == 0).mean():.3f}",
             f"  sign agreement token vs value, where the token moves: "
             f"{(np.sign(live.dtok50) == np.sign(live.dq50))[live.dtok50 != 0].mean():.3f}",
             f"  by stim, share with unmoved median token:",
             live.assign(z=live.dtok50 == 0).groupby("stim").z.mean().round(3).to_string(),
             f"\n{time.time() - t0:.0f}s"]
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
