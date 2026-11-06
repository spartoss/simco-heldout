"""Conditional CellOT on the learning-curve design (patient-covariate experiment, step 2) and the donor-ID oracle (step 3).

Same training as lc/train.py (paper ICNN, 10 g + 1 f steps, batch 256, Adam 1e-4 / (0.5, 0.9), best-g selection
on the training donors' validation cells), with two changes:
- PAIRED sampling: each batch row picks a training donor (weighted by its train-side Unstim cell count, so the source
  marginal is the pooled one), then a source and a target cell from THAT donor, plus that donor's context. The
  networks are lc/cond_icnn.CondBatchedICNN: f(y; c), g(x; c).
- --ctx sets the context:
    none   : c ignored. The control: same sampling and code, no donor information.
    pc     : the donor's same-celltype train-side Unstim 13 marker medians, z-scored and reduced to N_PC principal
             components fitted on that map's training donors only (scores divided by their training sd); test donors
             are projected. Generalises to new donors. Fewer than N_PC + 1 donors -> fewer PCs, zero-padded.
    onehot : donor identity (19-dim). Cannot generalise; only with --oracle.
    ft     : the donor's same-celltype mean Unstim cell embedding from the atlas-fine-tuned ImmuneFM encoder of this
             repeat (WORK/fm/finetune/repeat_<r>/<--ft-init>/donor_emb.npz, fm/embed_donors.py: trained on the
             repeat's 15 training donors only), reduced as pc (--n-pc PCs fitted on the map's training donors).
    pcblood: pc plus the donor's other dominant celltypes (BLOOD_CTS, ~75 % of Unstim cells): covariate = [own 13
             Unstim medians | each BLOOD_CTS celltype's 13 train-side Unstim medians | their 4 Unstim frequencies
             (WORK/fm/counts.csv)], z-scored, each of the 3 blocks scaled by 1/sqrt(block size) so they weigh equally
             in the PCA, then --n-pc PCs on the map's training donors.
    lda    : the donor-signature covariate (heldout/donor_covariate.py, notebooks/donor_covariate.ipynb): regularised LDA
             of the 28 Unstim log frequencies, donors as classes (max between- / within-donor variance over a donor's
             wells), refitted per repeat and per training size k on the first k training donors only; the same vector
             for every celltype. --n-pc = number of axes (<= 3). File heldout/donor_covariate.py --lc-ctx.
    cluster: the cell's PHENOTYPE subcluster (phase 2 of the ImmuneFM plan; closed form in fm/stratified.py). Per map,
             k-means (--K) on the z-scored phenotypic markers (fm cache: the 25 non-functional ones; --stable keeps
             those whose training median moves <= STABLE_THR under the stim) of the training donors' train-side
             Unstim cells. Sampling pairs source and target cells within the same (donor, subcluster) slot (slots
             drawn by source count, so the source marginal is still the pooled one; a slot with < MIN_SLOT target
             cells falls back to all of that donor's target cells), and c = the subcluster one-hot. Test cells are
             mapped with their own subcluster.
- --ctx-wd L2 penalty on the context weights (Bc, Gc) of f and g, per map: shrinks towards the plain ICNN (the
  unregularised pc context memorised the training donors).
- --val-metric (checkpoint selection, best g per map): pooled (default, as before) = |q of the transported val cells -
  q of the target val cells| with both pooled over the training donors (one fixed paired draw of N_VAL cells);
  donor = the same error computed within each training donor (up to --val-cells-per-donor of her val-side source and
  target cells, all of them if fewer), then averaged over donors with equal weight. --val-qs: the quantiles in the
  error, averaged with equal weight (default 50 = the median error); mean |.| over the 13 markers in both cases.
- --arch (2026-10-13): how the context enters the ICNN (lc/cond_icnn.py): gate (default, unchanged), bias, film,
  trans (donor-level translation only), affine (translation + donor-level slope change per marker around the map's
  pooled training Unstim mean).
- --shift-reg L: adds L * mean ||T(x) - x - delta||^2 to the g loss, delta = the map's pooled training mean shift
  (mean target - mean source): pulls the map towards the feature shift (an in-model shrinkage; any --ctx).
- --ctx-drop P: during training each batch row's context is zeroed with probability P (the f and g steps alike), so
  the network must also work without it and the context can only add a correction. Evaluation uses the full context.
- --oracle: the donor-ID oracle in the pooled protocol: every map trains on all 19 donors' train sides, and the
  "test donors" are the same 19 donors scored on their own held-out 20 % (val side) with their own context.
  How much of the response is donor-specific at all, if the map knew the donor perfectly?
Outputs as lc/train.py (maps.csv, pred_q.npy [n_eval + 1, M, n_test, n_q, 13], val_err.npy, meta.json) plus ctx.npy
(M, n_test, cdim) for the test donors. Score with lc/cond_score.py (learning curve) or --oracle scoring there.
    python -m lc.cond_train --repeat 0 --ctx pc --ks 3 5 8 15 --out <dir>
    python -m lc.cond_train --oracle --ctx onehot --out <dir>
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lc.cond_icnn import CondBatchedICNN, loss_f, loss_g, sub
from lc.paths import CACHE, DONORS, MARKERS, PAIRS, QS, VAL_FRAC, WORK
from lc.train import ADAM, BATCH, INNER, MAX_TEST_CELLS, MIN_TEST_CELLS, N_VAL, Cells, durable_save

D = len(MARKERS)
N_PC = 3
STABLE_THR = 0.2
BLOOD_CTS = ["Granulocytes", "CD4Tnaive", "cMCs", "CD8Tnaive"]     # largest Unstim celltypes (mean freq 0.63/.05/.04/.03)
MIN_SLOT = 5
FM_CACHE = WORK / "fm" / "cache_full"
LDA_CTX = WORK.parent / "heldout" / "donor_covariate" / "lc_ctx.json"


def cluster_setup(maps, cells, tests, K, stable, dev, max_test, seed=0):
    """Per-map subclusters (see --ctx cluster). Returns IDX (rows of X, grouped by slot), train / val slot tables for
    PairedSampler (starts into IDX), CT (M, n_slots, K) one-hot contexts, and test labels {(map, test slot j): (n,)}."""
    from sklearn.cluster import KMeans
    X = cells.X
    ph = {}                                     # (celltype, cond, donor) -> phenotype rows, aligned with X
    for c in sorted({m["celltype"] for m in maps}):
        z = np.load(FM_CACHE / f"{c}.npz"); panel = list(z["markers"])
        fi = [panel.index(mk) for mk in MARKERS]; pi = [i for i, mk in enumerate(panel) if mk not in MARKERS]
        for (cc, cond, d), (st, n, ntr) in cells.blk.items():
            if cc != c: continue
            a = z[f"{cond}|{d}"]
            chk = X[st:st + min(n, 64)].cpu().numpy()
            assert np.array_equal(a[:len(chk)][:, fi], chk), f"fm cache rows differ from cache rows: {c} {cond} {d}"
            ph[(c, cond, d)] = torch.from_numpy(a[:, pi]).to(dev)
    idx_parts, pos = [], 0
    tab = {"train": [], "val": []}; CT = []; TL = {}
    nslot = max(len(m["train"]) for m in maps) * K
    for i, m in enumerate(maps):
        c, stim = m["celltype"], m["stim"]
        U = torch.cat([ph[(c, "Unstim", d)][:cells.blk[(c, "Unstim", d)][2]] for d in m["train"]])
        S = torch.cat([ph[(c, stim, d)][:cells.blk[(c, stim, d)][2]] for d in m["train"]])
        keep = torch.ones(U.shape[1], dtype=torch.bool, device=dev)
        if stable: keep = (S.median(0).values - U.median(0).values).abs() <= STABLE_THR
        mu, sd = U[:, keep].mean(0), U[:, keep].std(0) + 1e-6
        g = torch.Generator(device="cpu").manual_seed(seed * 100_003 + i)
        sub = U[torch.randperm(len(U), generator=g)[:20_000].to(dev)][:, keep]
        km = KMeans(min(K, len(sub)), n_init=4, random_state=0).fit(((sub - mu) / sd).cpu().numpy())
        cen = torch.from_numpy(km.cluster_centers_).float().to(dev)
        lab = lambda P: torch.cdist(((P[:, keep] - mu) / sd).float(), cen).argmin(1)
        ctx = np.zeros((nslot, K), np.float32)
        for part in ("train", "val"):
            slots = []
            for d in m["train"]:
                (ss, sn, sntr), (ts, tn, tntr) = cells.blk[(c, "Unstim", d)], cells.blk[(c, stim, d)]
                if part == "train" or sn - sntr == 0 or tn - tntr == 0: sa, sb, ta, tb = 0, sntr, 0, tntr
                else: sa, sb, ta, tb = sntr, sn, tntr, tn
                ls, lt = lab(ph[(c, "Unstim", d)][sa:sb]), lab(ph[(c, stim, d)][ta:tb])
                tall = torch.arange(ts + ta, ts + tb, device=dev)
                for k in range(K):
                    srows = torch.arange(ss + sa, ss + sb, device=dev)[ls == k]; trows = tall[lt == k]
                    if len(trows) < MIN_SLOT: trows = tall
                    idx_parts += [srows, trows]
                    slots.append((pos, len(srows), pos + len(srows), len(trows))); pos += len(srows) + len(trows)
                    ctx[len(slots) - 1, k] = 1
            tab[part].append(slots)
        CT.append(ctx)
        for j, d in enumerate(tests):
            if (c, "Unstim", d) in cells.blk:
                TL[(i, j)] = lab(ph[(c, "Unstim", d)][:max_test])
    IDX = torch.cat(idx_parts)
    return IDX, tab["train"], tab["val"], torch.from_numpy(np.stack(CT)).to(dev), TL


class PairedSampler:
    """Per map: a list of donors with (src start, src count, tgt start, tgt count). draw(B) -> src idx, tgt idx,
    donor slot j, each (M, B). Donor slot drawn with probability proportional to its src count."""

    def __init__(self, blocks, device):
        n = max(len(b) for b in blocks)
        arr = np.zeros((len(blocks), n, 4), np.int64)
        for i, b in enumerate(blocks):
            for j, t in enumerate(b): arr[i, j] = t
        assert (arr[:, :, 1].sum(1) > 0).all(), "a map has no cells"
        self.a = torch.from_numpy(arr).to(device)
        self.cum = torch.cumsum(self.a[:, :, 1], 1); self.tot = self.cum[:, -1:]

    def draw(self, B, gen=None):
        M = self.a.shape[0]; dev = self.a.device
        u = torch.minimum((torch.rand(M, B, device=dev, generator=gen) * self.tot).long(), self.tot - 1)
        j = torch.searchsorted(self.cum, u, right=True)
        g = lambda col: torch.gather(self.a[:, :, col], 1, j)
        su = (torch.rand(M, B, device=dev, generator=gen) * g(1)).long()
        tu = (torch.rand(M, B, device=dev, generator=gen) * g(3)).long()
        return g(0) + torch.minimum(su, g(1) - 1), g(2) + torch.minimum(tu, g(3) - 1), j


def pc_context(train_cov, test_cov, n_pc=N_PC, w=None):
    """z-score + PCA on the training donors' covariates; scores / training sd; zero-padded to n_pc columns.
    w: per-feature weights applied after the z-score (block weighting)."""
    out_tr = np.zeros((len(train_cov), n_pc)); out_te = np.zeros((len(test_cov), n_pc))
    if len(train_cov) < 2: return out_tr, out_te
    mu, sd = train_cov.mean(0), train_cov.std(0) + 1e-9
    w = np.ones(train_cov.shape[1]) if w is None else w
    Z = (train_cov - mu) / sd * w; zc = Z.mean(0)
    k = min(n_pc, len(Z) - 1)
    _, _, Vt = np.linalg.svd(Z - zc, full_matrices=False)
    P = Vt[:k].T
    s_tr = (Z - zc) @ P; s_sd = s_tr.std(0) + 1e-9
    out_tr[:, :k] = s_tr / s_sd
    out_te[:, :k] = (((test_cov - mu) / sd * w - zc) @ P) / s_sd
    return out_tr, out_te


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--design", default=str(WORK / "design.json"), help="design JSON (default: the learning curve)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctx", choices=["none", "pc", "onehot", "ft", "cluster", "pcblood", "lda"], required=True)
    ap.add_argument("--K", type=int, default=4, help="--ctx cluster: subclusters per map")
    ap.add_argument("--stable", action="store_true", help="--ctx cluster: stim-stable phenotypic markers only")
    ap.add_argument("--ft-init", choices=["pretrained", "random"], default="pretrained")
    ap.add_argument("--n-pc", type=int, default=N_PC)
    ap.add_argument("--ctx-wd", type=float, default=0.0)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--ks", nargs="*", type=int, default=[3, 5, 8, 15])
    ap.add_argument("--n-iters", type=int, default=20_000)
    ap.add_argument("--eval-every", type=int, default=2_500)
    ap.add_argument("--pairs", nargs="*", default=None, help="stim:celltype subset")
    ap.add_argument("--celltypes", nargs="*", default=None)
    ap.add_argument("--cache-dir", default=str(CACHE))
    ap.add_argument("--cov-cache-dir", default=None,
                    help="cache the donor covariate (pc / pcblood medians) is computed from; default = --cache-dir. With "
                         "--cache-dir cache_dnorm, pass the raw cache: centred medians are ~0 for every donor")
    ap.add_argument("--clip-test", type=float, default=0.0,
                    help="clamp each test cell, per marker, to the [p, 100 - p] percentiles of the map's pooled training "
                         "Unstim cells before transport (0 = off); e.g. 0.5")
    ap.add_argument("--val-metric", choices=["pooled", "donor"], default="pooled")
    ap.add_argument("--val-qs", nargs="*", type=int, default=[50])
    ap.add_argument("--val-cells-per-donor", type=int, default=1000)
    ap.add_argument("--arch", choices=["gate", "bias", "film", "trans", "affine"], default="gate")
    ap.add_argument("--shift-reg", type=float, default=0.0)
    ap.add_argument("--ctx-drop", type=float, default=0.0)
    a = ap.parse_args()
    assert a.val_metric == "pooled" or a.ctx != "cluster", "--val-metric donor is not implemented for --ctx cluster"
    assert (a.ctx == "onehot") <= a.oracle, "onehot context only makes sense with --oracle"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    pairs = [p for p in PAIRS if (not a.pairs or f"{p[0]}:{p[1]}" in a.pairs) and (not a.celltypes or p[1] in a.celltypes)]
    if a.oracle:
        tests = list(DONORS)
        maps = [{"stim": s, "celltype": c, "k": len(DONORS), "train": list(DONORS)} for s, c in pairs]
    else:
        dz = json.load(open(a.design))[a.repeat]; tests = dz["test"]
        maps = [{"stim": s, "celltype": c, "k": k, "train": dz["order"][:k]} for s, c in pairs for k in a.ks]
    cells = Cells(sorted({m["celltype"] for m in maps}), dev, a.cache_dir)
    X = cells.X
    for m in maps:
        m["train"] = [d for d in m["train"] if (m["celltype"], "Unstim", d) in cells.blk and (m["celltype"], m["stim"], d) in cells.blk]
    maps = [m for m in maps if m["train"]]
    M = len(maps)
    cdim = {"none": 0, "pc": a.n_pc, "ft": a.n_pc, "pcblood": a.n_pc, "lda": a.n_pc, "onehot": len(DONORS), "cluster": a.K}[a.ctx]

    # covariate per (celltype, donor): train-side Unstim medians (of --cov-cache-dir if given: same rows, other values)
    cov_dir = Path(a.cov_cache_dir or a.cache_dir)
    cov, cz = {}, {}
    for (c, cond, d), (s, n, ntr) in cells.blk.items():
        if cond != "Unstim": continue
        if a.cov_cache_dir is None: cov[(c, d)] = X[s:s + ntr].median(0).values.cpu().numpy()
        else:
            if c not in cz: cz[c] = np.load(cov_dir / f"{c}.npz")
            x = cz[c][f"Unstim|{d}"]; assert len(x) == n, f"cov cache rows differ: {c} {d}"
            cov[(c, d)] = torch.from_numpy(x[:ntr]).median(0).values.numpy()
    if a.ctx == "ft":
        assert not a.oracle
        z = np.load(WORK / f"fm/finetune/repeat_{a.repeat}/{a.ft_init}/donor_emb.npz")
        zc, zd = list(z["celltypes"]), list(z["donors"])
        cov = {(c, d): z["finetuned"][zd.index(d), zc.index(c)] for (c, d) in cov
               if not np.isnan(z["finetuned"][zd.index(d), zc.index(c)]).any()}
    wts = None
    if a.ctx == "pcblood":
        freq = pd.read_csv(WORK / "fm" / "counts.csv").set_index(["celltype", "donor"]).freq
        bmed = {}
        for bc in BLOOD_CTS:
            zb = np.load(cov_dir / f"{bc}.npz")
            for d in DONORS:
                if f"Unstim|{d}" in zb.files:
                    x = zb[f"Unstim|{d}"]; bmed[(bc, d)] = np.median(x[:max(1, int(round(len(x) * (1 - VAL_FRAC))))], 0)
        blood = {d: np.concatenate([bmed.get((bc, d), np.full(D, np.nan)) for bc in BLOOD_CTS] +
                                   [np.array([freq.get((bc, d), np.nan) for bc in BLOOD_CTS])]) for d in DONORS}
        bmu = np.nanmean(np.stack(list(blood.values())), 0)
        blood = {d: np.where(np.isnan(v), bmu, v) for d, v in blood.items()}
        cov = {(c, d): np.concatenate([v, blood[d]]) for (c, d), v in cov.items()}
        nb = len(BLOOD_CTS)
        wts = np.concatenate([np.full(D, D ** -0.5), np.full(D * nb, (D * nb) ** -0.5), np.full(nb, nb ** -0.5)])
    if a.ctx == "lda":
        assert not a.oracle and a.n_pc <= 3
        lda = json.load(open(LDA_CTX))
    cdef = np.zeros(128 if a.ctx == "ft" else D + (D * len(BLOOD_CTS) + len(BLOOD_CTS) if a.ctx == "pcblood" else 0))
    nd = max(len(m["train"]) for m in maps)
    ctx_tr = np.zeros((M, nd, max(cdim, 1)), np.float32); ctx_te = np.zeros((M, len(tests), max(cdim, 1)), np.float32)
    for i, m in enumerate(maps):
        if a.ctx in ("pc", "ft", "pcblood"):
            tr = np.stack([cov.get((m["celltype"], d), cdef) for d in m["train"]])
            te = np.stack([cov.get((m["celltype"], d), cdef) for d in tests])
            ctx_tr[i, :len(tr)], ctx_te[i] = pc_context(tr, te, a.n_pc, wts)
        elif a.ctx == "lda":
            v = lda[f"{a.repeat}_{m['k']}"]
            ctx_tr[i, :len(m["train"])] = np.array([v[d][:cdim] for d in m["train"]])
            ctx_te[i] = np.array([v[d][:cdim] for d in tests])
        elif a.ctx == "onehot":
            for j, d in enumerate(m["train"]): ctx_tr[i, j, DONORS.index(d)] = 1
            for j, d in enumerate(tests): ctx_te[i, j, DONORS.index(d)] = 1
    CT = torch.from_numpy(ctx_tr).to(dev)

    def blocks(part):
        res = []
        for m in maps:
            b = []
            for d in m["train"]:
                ss, sn, sntr = cells.blk[(m["celltype"], "Unstim", d)]; ts, tn, tntr = cells.blk[(m["celltype"], m["stim"], d)]
                if part == "train" or sn - sntr == 0 or tn - tntr == 0: b.append((ss, sntr, ts, tntr))
                else: b.append((ss + sntr, sn - sntr, ts + tntr, tn - tntr))
            res.append(b)
        return res
    if a.ctx == "cluster":
        assert not a.oracle
        IDX, btr, bva, CT, TL = cluster_setup(maps, cells, tests, a.K, a.stable, dev, MAX_TEST_CELLS, a.repeat)
        rows = lambda i: X[IDX[i]]
        print(f"[CLUST] {M} maps, {len(IDX)} slot rows", flush=True)
    else:
        btr, bva, rows = blocks("train"), blocks("val"), (lambda i: X[i])
    tr_s = PairedSampler(btr, dev)
    ctx_drop = (lambda c: c * (torch.rand(c.shape[0], c.shape[1], 1, device=c.device) >= a.ctx_drop)) \
        if (a.ctx_drop > 0 and cdim) else (lambda c: c)
    gval = torch.Generator(device=dev); gval.manual_seed(7)
    vs, vt, vj = PairedSampler(bva, dev).draw(N_VAL, gval)
    ctx_of = lambda j: torch.gather(CT, 1, j.unsqueeze(-1).expand(-1, -1, CT.shape[-1])) if cdim else None

    pd.DataFrame([{"stim": m["stim"], "celltype": m["celltype"], "k": m["k"], "cap": 0, "train": " ".join(m["train"])}
                  for m in maps]).to_csv(out / "maps.csv", index_label="map")
    np.save(out / "ctx.npy", ctx_te[:, :, :cdim])
    json.dump({"repeat": a.repeat, "ctx": a.ctx, "K": a.K if a.ctx == "cluster" else None, "stable": a.stable, "ft_init": a.ft_init if a.ctx == "ft" else None, "n_pc": a.n_pc,
               "ctx_wd": a.ctx_wd, "arch": a.arch, "shift_reg": a.shift_reg, "ctx_drop": a.ctx_drop, "oracle": a.oracle, "tests": tests, "test_cells": "val side" if a.oracle
               else "all cached (<= 50k)", "markers": MARKERS, "qs": list(QS), "M": M, "n_iters": a.n_iters,
               "eval_every": a.eval_every, "ks": a.ks, "cache_dir": a.cache_dir, "cov_cache_dir": a.cov_cache_dir, "clip_test": a.clip_test,
               "val_metric": a.val_metric, "val_qs": a.val_qs, "val_cells_per_donor": a.val_cells_per_donor, "device": str(dev),
               "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None}, open(out / "meta.json", "w"), indent=1)

    torch.manual_seed(a.repeat)
    f = CondBatchedICNN(M, D, cdim, fnorm_penalty=0, arch=a.arch).to(dev); g = CondBatchedICNN(M, D, cdim, fnorm_penalty=1, arch=a.arch).to(dev)
    if a.arch == "affine" or a.shift_reg > 0:   # per map pooled training means (train side), as the sampler's marginals
        assert a.ctx != "cluster"
        mu_s = torch.stack([torch.cat([X[ss:ss + sn] for (ss, sn, _, _) in b]).mean(0) for b in btr])
        mu_t = torch.stack([torch.cat([X[ts:ts + tn] for (_, _, ts, tn) in b]).mean(0) for b in btr])
        DELTA = mu_t - mu_s
        if a.arch == "affine": f.center.copy_(mu_s); g.center.copy_(mu_s)
    of, og = torch.optim.Adam(f.parameters(), **ADAM), torch.optim.Adam(g.parameters(), **ADAM)
    best = [p.detach().clone() for p in g.parameters()]
    best_err = torch.full((M,), float("inf"), device=dev)
    n_eval = a.n_iters // a.eval_every
    pred_q = np.full((n_eval + 1, M, len(tests), len(QS), D), np.nan, np.float32)
    val_err = np.full((n_eval, M), np.nan, np.float32)
    step0 = 0
    if (out / "state.pt").exists():
        st = torch.load(out / "state.pt", map_location=dev)
        f.load_state_dict(st["f"]); g.load_state_dict(st["g"]); of.load_state_dict(st["of"]); og.load_state_dict(st["og"])
        best, best_err, step0 = st["best"], st["best_err"], st["step"]
        pred_q = np.load(out / "pred_q.partial.npy"); val_err = np.load(out / "val_err.partial.npy")
        torch.set_rng_state(st["rng_cpu"])
        if dev.type == "cuda": torch.cuda.set_rng_state(st["rng_cuda"])
        print(f"[RESUME] from step {step0}", flush=True)

    # test cells per (celltype, donor): all cached Unstim (<= 50k), or the val side in the oracle protocol
    test_src = {}
    for c in sorted({m["celltype"] for m in maps}):
        for j, d in enumerate(tests):
            if (c, "Unstim", d) not in cells.blk: continue
            s, n, ntr = cells.blk[(c, "Unstim", d)]
            x = X[s + ntr:s + n] if a.oracle else X[s:s + min(n, MAX_TEST_CELLS)]
            if len(x) >= MIN_TEST_CELLS: test_src[(c, j)] = x
    by_ct = {}
    for i, m in enumerate(maps): by_ct.setdefault(m["celltype"], []).append(i)
    LO = HI = None
    if a.clip_test > 0:                     # per map: percentile range of its training donors' train-side Unstim cells
        LO = torch.empty(M, D, device=dev); HI = torch.empty(M, D, device=dev)
        pq = torch.tensor([a.clip_test / 100, 1 - a.clip_test / 100], device=dev)
        for i, m in enumerate(maps):
            U = torch.cat([X[s:s + ntr] for d in m["train"] for (s, n, ntr) in [cells.blk[(m["celltype"], "Unstim", d)]]])
            if len(U) > 200_000: U = U[torch.randperm(len(U), device=dev)[:200_000]]
            LO[i], HI[i] = torch.quantile(U, pq, dim=0)
    qt = torch.tensor([q / 100 for q in QS], device=dev)
    CTe = torch.from_numpy(ctx_te).to(dev)

    def eval_test(net, slot):
        for c, idx in by_ct.items():
            for j in range(len(tests)):
                if (c, j) not in test_src: continue
                x = test_src[(c, j)]
                ch = max(1, min(32, 15_000_000 // (len(x) * D)))
                for s in range(0, len(idx), ch):
                    ii = torch.tensor(idx[s:s + ch], device=dev)
                    cc = CTe[ii, j].unsqueeze(1) if cdim else None
                    per = max(1, 200_000 // len(ii)); T = []
                    for s2 in range(0, len(x), per):
                        xb = x[s2:s2 + per].unsqueeze(0).expand(len(ii), -1, -1).clone()
                        if LO is not None: xb = torch.maximum(torch.minimum(xb, HI[ii].unsqueeze(1)), LO[ii].unsqueeze(1))
                        xb.requires_grad_(True)
                        if a.ctx == "cluster":       # per-cell context: each cell's subcluster under each map
                            lb = torch.stack([TL[(int(m_), j)][s2:s2 + per] for m_ in ii])
                            cc = torch.nn.functional.one_hot(lb, a.K).float()
                        T.append(sub(net, ii).transport(xb, cc).detach())
                    T = torch.cat(T, 1)
                    pred_q[slot, ii.cpu().numpy(), j] = torch.quantile(T, qt, dim=1).permute(1, 0, 2).cpu().numpy()

    vq = torch.tensor([q / 100 for q in a.val_qs], device=dev)
    if a.val_metric == "donor":             # per map and training donor: the first NV val-side rows (the cache is shuffled)
        NV = a.val_cells_per_donor
        ar = torch.arange(NV)
        def pad(col_s, col_n):
            st = torch.tensor([[t[col_s] for t in b] + [0] * (nd - len(b)) for b in bva])
            n = torch.tensor([[t[col_n] for t in b] + [0] * (nd - len(b)) for b in bva]).clamp(max=NV)
            ok = ar[None, None] < n[..., None]                                    # (M, nd, NV)
            return (st[..., None] + torch.where(ok, ar[None, None], 0)).to(dev), ok.to(dev)
        DVs, DVs_ok = pad(0, 1); DVt, DVt_ok = pad(2, 3)
        DVj = torch.arange(nd, device=dev)[None, :, None].expand(M, nd, NV)
        dv_ok = DVs_ok[:, :, 0] & DVt_ok[:, :, 0]                                # donors with val cells
        nanq = lambda Y, ok: torch.nanquantile(torch.where(ok[..., None], Y, Y.new_tensor(float("nan"))), vq, dim=2)   # (n_q, m, nd, D)
        tq = torch.cat([nanq(X[DVt[s:s + 16]], DVt_ok[s:s + 16]) for s in range(0, M, 16)], 1)

    def val_median_error():
        e = torch.empty(M, device=dev)
        if a.val_metric == "pooled":
            for s in range(0, M, 256):
                idx = torch.arange(s, min(M, s + 256), device=dev)
                x = rows(vs[idx]).requires_grad_(True)
                T = sub(g, idx).transport(x, ctx_of(vj[idx]) if cdim else None).detach()
                if a.val_qs == [50]:            # the original metric, exactly (torch median = lower median)
                    e[idx] = (T.median(1).values - rows(vt[idx]).median(1).values).abs().mean(1)
                else:
                    e[idx] = (torch.quantile(T, vq, dim=1) - torch.quantile(rows(vt[idx]), vq, dim=1)).abs().mean((0, 2))
            return e
        ch = 16
        for s in range(0, M, ch):
            idx = torch.arange(s, min(M, s + ch), device=dev); m = len(idx)
            x = X[DVs[idx]].reshape(m, nd * NV, D).requires_grad_(True)
            T = sub(g, idx).transport(x, ctx_of(DVj[idx].reshape(m, -1)) if cdim else None).detach().reshape(m, nd, NV, D)
            err = (nanq(T, DVs_ok[idx]) - tq[:, idx]).abs().mean((0, 3))             # (m, nd)
            ok = dv_ok[idx]
            e[idx] = torch.where(ok, err, err.new_zeros(())).sum(1) / ok.sum(1)
        return e

    t0 = time.time()
    for step in range(step0, a.n_iters):
        for _ in range(INNER):
            si, _, j = tr_s.draw(BATCH)
            src = rows(si).requires_grad_(True)
            og.zero_grad(set_to_none=True)
            cj = ctx_drop(ctx_of(j))
            if a.shift_reg > 0:
                T = g.transport(src, cj)
                lg = (f(T, cj).squeeze(-1) - (src * T).sum(-1)).mean(1) + a.shift_reg * ((T - src - DELTA[:, None, :]) ** 2).sum(-1).mean(1)
            else: lg = loss_g(f, g, src, cj)
            (lg + g.penalize_w() + a.ctx_wd * g.ctx_l2()).sum().backward()
            og.step()
        si, ti, j = tr_s.draw(BATCH)
        src = rows(si).requires_grad_(True)
        of.zero_grad(set_to_none=True)
        lf = loss_f(f, g, src, rows(ti), ctx_drop(ctx_of(j)))
        (lf + a.ctx_wd * f.ctx_l2()).sum().backward()
        of.step(); f.clamp_w()
        if not torch.isfinite(lf).all():
            raise FloatingPointError(f"non-finite f loss at step {step}")
        if (step + 1) % a.eval_every == 0:
            k = (step + 1) // a.eval_every - 1
            e = val_median_error(); val_err[k] = e.cpu().numpy()
            better = e < best_err; best_err = torch.where(better, e, best_err)
            with torch.no_grad():
                for b, p in zip(best, g.parameters()): b[better] = p.detach()[better]
            eval_test(g, k)
            durable_save(out / "pred_q.partial.npy", lambda fh: np.save(fh, pred_q))
            durable_save(out / "val_err.partial.npy", lambda fh: np.save(fh, val_err))
            state = {"f": f.state_dict(), "g": g.state_dict(), "of": of.state_dict(), "og": og.state_dict(),
                     "best": best, "best_err": best_err, "step": step + 1, "rng_cpu": torch.get_rng_state(),
                     "rng_cuda": torch.cuda.get_rng_state() if dev.type == "cuda" else None}
            durable_save(out / "state.pt", lambda fh: torch.save(state, fh))
            el = time.time() - t0
            print(f"[EVAL ] step {step + 1}/{a.n_iters}  val median err median {e.median():.4f}  "
                  f"{el / 60:.1f} min, {(el / (step + 1 - step0)) * 1000:.1f} ms/iter", flush=True)

    gb = CondBatchedICNN(M, D, cdim, fnorm_penalty=1, arch=a.arch).to(dev)
    if a.arch == "affine": gb.center.copy_(g.center)
    with torch.no_grad():
        for p, b in zip(gb.parameters(), best): p.copy_(b)
    eval_test(gb, n_eval)
    durable_save(out / "pred_q.npy", lambda fh: np.save(fh, pred_q))
    durable_save(out / "val_err.npy", lambda fh: np.save(fh, val_err))
    durable_save(out / "final_maps.pt", lambda fh: torch.save({"g_last": g.state_dict(), "g_best": gb.state_dict(),
                                                              "f": f.state_dict(), "best_err": best_err}, fh))
    print(f"[DONE ] {M} maps, ctx {a.ctx}, {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
