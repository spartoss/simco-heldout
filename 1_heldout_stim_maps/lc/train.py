"""Train many CellOT stim maps at once on one GPU, for the donor learning curve.

One map per (stim, celltype) pair and training-set size k. Map (pair, k) is trained on the first k donors of the
repeat's order: training batches are drawn uniformly from the pooled train-side cells of those donors (as the
paper's shuffled DataLoader over pooled cells). Paper settings otherwise (see lc/batched_icnn.py): 10 g steps +
1 f step per iteration, batch 256, Adam(1e-4, betas 0.5/0.9), 50k iterations.

Every --eval-every iterations:
- validation median error on held-out cells of the TRAINING donors, i.e. the paper's selection metric; the best
  g per map is kept ("paper-style" checkpoint, which never sees the test donors);
- the 10/25/50/75/90th percentiles of the transported cells of each TEST donor, per marker.
Outputs in --out: maps.csv, pred_q.npy [n_ckpt, M, n_test, n_q, n_markers] (the last slice is the paper-style best
checkpoint), val_err.npy [n_eval, M], meta.json; state.pt for resuming.

    python -m lc.train --repeat 0 --out <dir>                  # the learning curve, one repeat
    python -m lc.train --bench --out <dir>                     # timing only
    python -m lc.train --parity --out <dir>                    # 4 pairs, all 19 donors, vs the paper's checkpoints
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lc.batched_icnn import BatchedICNN, loss_f, loss_g
from lc.paths import CACHE, DONORS, MARKERS, PAIRS, QS, VAL_FRAC, WORK

D = len(MARKERS)
BATCH, INNER = 256, 10
ADAM = dict(lr=1e-4, betas=(0.5, 0.9), weight_decay=0)
N_VAL = 2000          # validation cells per map (fixed draw)
MIN_TEST_CELLS = 20
MAX_TEST_CELLS = 50_000   # quantiles of a test donor from at most this many cells (a random subset: the cache is shuffled)
PARITY_PAIRS = [("PI", "CD4Tnaive"), ("LPS", "cMCs"), ("IFNa", "Bcells"), ("TNFa", "Granulocytes")]


def durable_save(path, write):
    """Write to a temp file, fsync, then rename: a node crash leaves the previous file intact
    (repeat 3 of the first run lost every file written just before a crash)."""
    import os
    path = Path(path); tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        write(fh); fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, path)


class Cells:
    """All cached cells of the needed celltypes in one GPU tensor, addressed by (celltype, cond, donor) blocks."""

    def __init__(self, celltypes, device, cache_dir=CACHE):
        chunks, self.blk, pos = [], {}, 0
        for c in celltypes:
            z = np.load(Path(cache_dir) / f"{c}.npz")
            for k in z.files:
                cond, d = k.rsplit("|", 1); a = z[k]          # drug cache keys: "<stim>|<treatment>|<donor>"
                self.blk[(c, cond, d)] = (pos, len(a), max(1, int(round(len(a) * (1 - VAL_FRAC)))))
                chunks.append(a); pos += len(a)
        self.X = torch.from_numpy(np.concatenate(chunks)).to(device)

    def table(self, maps_blocks, part, caps=None):
        """Per-map padded (start, count) of the train or val part of each listed block.
        caps[i] > 0 keeps only the first caps[i] train cells of each of map i's donors (the cache is shuffled,
        so that is a random subsample): the "cells per donor" axis."""
        n = max(len(b) for b in maps_blocks)
        st = np.zeros((len(maps_blocks), n), np.int64); ct = np.zeros_like(st)
        for i, blocks in enumerate(maps_blocks):
            for j, key in enumerate(blocks):
                if key not in self.blk: continue
                s, tot, ntr = self.blk[key]
                if part == "train" or tot - ntr == 0:      # a donor too small for a val part: reuse its cells
                    cap = caps[i] if caps is not None and caps[i] > 0 else ntr
                    st[i, j], ct[i, j] = s, min(ntr, cap)
                else:
                    st[i, j], ct[i, j] = s + ntr, tot - ntr
        return st, ct


class Sampler:
    """Uniform draws over the pooled cells of each map's blocks: (M, B) indices into Cells.X."""

    def __init__(self, st, ct, device):
        self.st = torch.from_numpy(st).to(device)
        self.cum = torch.from_numpy(np.cumsum(ct, 1)).to(device)
        self.ct = torch.from_numpy(ct).to(device)
        self.tot = self.cum[:, -1:].clone()
        assert (self.tot > 0).all(), "a map has no cells"

    def draw(self, B, gen=None):
        u = (torch.rand(self.tot.shape[0], B, device=self.tot.device, generator=gen) * self.tot).long()
        u = torch.minimum(u, self.tot - 1)
        j = torch.searchsorted(self.cum, u, right=True)
        before = torch.gather(self.cum - self.ct, 1, j)
        return torch.gather(self.st, 1, j) + (u - before)


def sub(net, idx):
    """A BatchedICNN view restricted to the maps in idx (for evaluation)."""
    v = BatchedICNN.__new__(BatchedICNN); torch.nn.Module.__init__(v)
    v.fnorm_penalty = net.fnorm_penalty
    v.W = [w[idx] for w in net.W]; v.A = [a[idx] for a in net.A]; v.Ab = [b[idx] for b in net.Ab]
    return v


def transport_chunked(g, idx, x, chunk_cells=200_000):
    """T(x) for maps idx on the shared cells x (n, d) -> (len(idx), n, d), chunked to bound memory."""
    out = []
    per = max(1, chunk_cells // max(1, len(idx)))
    for s in range(0, len(x), per):
        xb = x[s:s + per].unsqueeze(0).expand(len(idx), -1, -1).clone().requires_grad_(True)
        out.append(sub(g, idx).transport(xb).detach())
    return torch.cat(out, 1)


def build(args, design):
    if args.spec:
        # custom design (lc/design_donors.py): {"tests": [...], "train_sets": [{"name", "donors"}], "caps": [...],
        # "celltypes": optional subset}. One map per (pair, train set, cap).
        sp = json.load(open(args.spec)); tests = sp["tests"]
        if "maps" in sp:     # explicit source / target conditions (drug maps): one map per (entry, train set)
            ents = [e for e in sp["maps"] if (not args.celltypes or e["celltype"] in args.celltypes)]
            return [{**e, "k": len(ts["donors"]), "cap": cap, "set": ts["name"], "train": ts["donors"]}
                    for e in ents for ts in sp["train_sets"] for cap in sp.get("caps", [0])], tests
        base_pairs = [tuple(p) for p in sp["pairs"]] if "pairs" in sp else PAIRS
        pairs = [p for p in base_pairs if not sp.get("celltypes") or p[1] in sp["celltypes"]]
        if args.pairs: pairs = [p for p in pairs if f"{p[0]}:{p[1]}" in args.pairs]
        if args.celltypes: pairs = [p for p in pairs if p[1] in args.celltypes]
        maps = [{"stim": s, "celltype": c, "k": len(ts["donors"]), "cap": cap, "set": ts["name"], "train": ts["donors"]}
                for s, c in pairs for ts in sp["train_sets"] for cap in sp.get("caps", [0])]
        return maps, tests
    if args.parity:
        pairs, tests, maps = PARITY_PAIRS, [], []
        for p in pairs: maps.append({"stim": p[0], "celltype": p[1], "k": len(DONORS), "cap": 0, "train": DONORS})
    else:
        dz = design[args.repeat]; tests = dz["test"]
        pairs = [p for p in PAIRS if not args.pairs or f"{p[0]}:{p[1]}" in args.pairs]
        ks = args.ks or dz["ks"]
        maps = [{"stim": s, "celltype": c, "k": k, "cap": cap, "train": dz["order"][:k]}
                for s, c in pairs for k in ks for cap in args.caps]
    return maps, tests


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--design", default=str(WORK / "design.json"), help="design JSON (default: the learning curve)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-iters", type=int, default=50_000)
    ap.add_argument("--eval-every", type=int, default=2_500)
    ap.add_argument("--pairs", nargs="*", default=None, help="stim:celltype subset")
    ap.add_argument("--ks", nargs="*", type=int, default=None)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--cache-dir", default=str(CACHE))
    ap.add_argument("--spec", default=None, help="custom design JSON (lc/design_donors.py)")
    ap.add_argument("--celltypes", nargs="*", default=None, help="restrict to these celltypes (split a spec over jobs)")
    ap.add_argument("--max-test-cells", type=int, default=None, help="test-donor cells used for quantiles")
    ap.add_argument("--hidden", nargs="*", type=int, default=[64, 64, 64, 64],
                    help="ICNN hidden units; the paper's drug maps use 128 128 64 32")
    ap.add_argument("--caps", nargs="*", type=int, default=[0],
                    help="training cells per donor (0 = all cached); one map per (pair, k, cap)")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = a.tf32
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    design = json.load(open(a.design))
    if a.bench:
        return bench(a, design, dev, out)

    maps, tests = build(a, design)
    cells = Cells(sorted({m["celltype"] for m in maps}), dev, a.cache_dir)
    for m in maps: m.setdefault("src", "Unstim"); m.setdefault("tgt", m["stim"])
    has = lambda m, cond: any((m["celltype"], cond, d) in cells.blk for d in m["train"])
    dropped = [m for m in maps if not (has(m, m["src"]) and has(m, m["tgt"]))]
    maps = [m for m in maps if m not in dropped]
    if dropped: print(f"[WARN ] {len(dropped)} maps have no training cells and are dropped: {dropped[:3]}")
    M = len(maps)
    src_blocks = [[(m["celltype"], m["src"], d) for d in m["train"]] for m in maps]
    tgt_blocks = [[(m["celltype"], m["tgt"], d) for d in m["train"]] for m in maps]
    caps = [m["cap"] for m in maps]
    s_tr = Sampler(*cells.table(src_blocks, "train", caps), dev)
    t_tr = Sampler(*cells.table(tgt_blocks, "train", caps), dev)
    gval = torch.Generator(device=dev); gval.manual_seed(7)
    vs = Sampler(*cells.table(src_blocks, "val"), dev).draw(N_VAL, gval)
    vt = Sampler(*cells.table(tgt_blocks, "val"), dev).draw(N_VAL, gval)
    n_src = cells.table(src_blocks, "train", caps)[1].sum(1); n_tgt = cells.table(tgt_blocks, "train", caps)[1].sum(1)
    pd.DataFrame([{**{k: v for k, v in m.items() if k != "train"}, "train": " ".join(m["train"]),
                   "n_train_src": int(a_), "n_train_tgt": int(b_)} for m, a_, b_ in zip(maps, n_src, n_tgt)]
                 ).to_csv(out / "maps.csv", index_label="map")
    max_test = a.max_test_cells or MAX_TEST_CELLS
    json.dump({"repeat": a.repeat, "tests": tests, "markers": MARKERS, "qs": list(QS), "M": M,
               "n_iters": a.n_iters, "dropped": len(dropped), "cache": a.cache_dir, "caps": a.caps, "hidden": a.hidden, "eval_every": a.eval_every, "device": str(dev),
               "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None}, open(out / "meta.json", "w"), indent=1)

    torch.manual_seed(a.repeat)
    H = tuple(a.hidden)
    f, g = BatchedICNN(M, D, H, fnorm_penalty=0).to(dev), BatchedICNN(M, D, H, fnorm_penalty=1).to(dev)
    of, og = torch.optim.Adam(f.parameters(), **ADAM), torch.optim.Adam(g.parameters(), **ADAM)
    best = [p.detach().clone() for p in g.parameters()]
    best_err = torch.full((M,), float("inf"), device=dev)
    n_eval = a.n_iters // a.eval_every
    pred_q = np.full((n_eval + 1, M, max(1, len(tests)), len(QS), D), np.nan, np.float32)
    val_err = np.full((n_eval, M), np.nan, np.float32)
    step0 = 0
    if (out / "state.pt").exists():
        st = torch.load(out / "state.pt", map_location=dev)
        f.load_state_dict(st["f"]); g.load_state_dict(st["g"]); of.load_state_dict(st["of"]); og.load_state_dict(st["og"])
        best = st["best"]; best_err = st["best_err"]; step0 = st["step"]
        pred_q = np.load(out / "pred_q.partial.npy"); val_err = np.load(out / "val_err.partial.npy")
        torch.set_rng_state(st["rng_cpu"]);
        if dev.type == "cuda": torch.cuda.set_rng_state(st["rng_cuda"])
        print(f"[RESUME] from step {step0}", flush=True)

    # test donors: shared source cells per (celltype, source condition, donor)
    test_src = {}
    for c, sc in sorted({(m["celltype"], m["src"]) for m in maps}):
        for j, d in enumerate(tests):
            if (c, sc, d) in cells.blk:
                s, n, _ = cells.blk[(c, sc, d)]
                if n >= MIN_TEST_CELLS: test_src[((c, sc), j)] = cells.X[s:s + min(n, max_test)]
    by_ct = {}
    for i, m in enumerate(maps): by_ct.setdefault((m["celltype"], m["src"]), []).append(i)
    qt = torch.tensor([q / 100 for q in QS], device=dev)

    def eval_test(net, slot):
        for c, idx in by_ct.items():
            for j in range(len(tests)):
                if (c, j) not in test_src: continue
                ch = max(1, min(32, 15_000_000 // (len(test_src[(c, j)]) * D)))   # torch.quantile size limit
                for s in range(0, len(idx), ch):
                    ii = idx[s:s + ch]
                    T = transport_chunked(net, torch.tensor(ii, device=dev), test_src[(c, j)])
                    pred_q[slot, ii, j] = torch.quantile(T, qt, dim=1).permute(1, 0, 2).cpu().numpy()

    def val_median_error():
        e = torch.empty(M, device=dev)
        for s in range(0, M, 256):
            idx = torch.arange(s, min(M, s + 256), device=dev)
            x = cells.X[vs[idx]].requires_grad_(True)
            T = sub(g, idx).transport(x).detach()
            e[idx] = (T.median(1).values - cells.X[vt[idx]].median(1).values).abs().mean(1)
        return e

    t0 = time.time()
    for step in range(step0, a.n_iters):
        tgt = cells.X[t_tr.draw(BATCH)]
        for _ in range(INNER):
            src = cells.X[s_tr.draw(BATCH)].requires_grad_(True)
            og.zero_grad(set_to_none=True)
            (loss_g(f, g, src) + g.penalize_w()).sum().backward()
            og.step()
        src = cells.X[s_tr.draw(BATCH)].requires_grad_(True)
        of.zero_grad(set_to_none=True)
        lf = loss_f(f, g, src, tgt)
        lf.sum().backward()
        of.step(); f.clamp_w()
        if not torch.isfinite(lf).all():
            raise FloatingPointError(f"non-finite f loss at step {step} in maps {torch.nonzero(~torch.isfinite(lf)).flatten()[:10].tolist()}")
        if (step + 1) % a.eval_every == 0:
            k = (step + 1) // a.eval_every - 1
            e = val_median_error(); val_err[k] = e.cpu().numpy()
            better = e < best_err; best_err = torch.where(better, e, best_err)
            with torch.no_grad():
                for b, p in zip(best, g.parameters()):
                    b[better] = p.detach()[better]
            if tests: eval_test(g, k)
            durable_save(out / "pred_q.partial.npy", lambda fh: np.save(fh, pred_q))
            durable_save(out / "val_err.partial.npy", lambda fh: np.save(fh, val_err))
            state = {"f": f.state_dict(), "g": g.state_dict(), "of": of.state_dict(), "og": og.state_dict(),
                        "best": best, "best_err": best_err, "step": step + 1, "rng_cpu": torch.get_rng_state(),
                        "rng_cuda": torch.cuda.get_rng_state() if dev.type == "cuda" else None}
            durable_save(out / "state.pt", lambda fh: torch.save(state, fh))
            el = time.time() - t0
            print(f"[EVAL ] step {step + 1}/{a.n_iters}  val median err median {e.median():.4f}  "
                  f"{el / 60:.1f} min elapsed, {(el / (step + 1 - step0)) * 1000:.1f} ms/iter", flush=True)

    gb = BatchedICNN(M, D, H, fnorm_penalty=1).to(dev)
    with torch.no_grad():
        for p, b in zip(gb.parameters(), best): p.copy_(b)
    if tests: eval_test(gb, n_eval)
    durable_save(out / "pred_q.npy", lambda fh: np.save(fh, pred_q))
    durable_save(out / "val_err.npy", lambda fh: np.save(fh, val_err))
    durable_save(out / "final_maps.pt", lambda fh: torch.save({"g_last": g.state_dict(), "g_best": gb.state_dict(),
                                                              "f": f.state_dict(), "best_err": best_err}, fh))
    if a.parity:
        parity(maps, cells, vs, vt, g, gb, dev, out)
    print(f"[DONE ] {M} maps, {(time.time() - t0) / 3600:.2f} h", flush=True)


def parity(maps, cells, vs, vt, g_last, g_best, dev, out):
    """Our all-donor maps vs the paper's model.pt on the same validation cells (median error)."""
    from lc.reference_icnn import ICNN
    R = Path("/home/groups/gbrice/ptb-drugscreen/ot/cellot/results/cross_validation_original_13_HVPV")  # read-only
    rows = []
    for i, m in enumerate(maps):
        ck = torch.load(R / m["stim"] / m["celltype"] / f"model-{m['stim']}_{m['celltype']}/cache/model.pt", map_location=dev)
        ref = ICNN(D, [64] * 4, fnorm_penalty=1).to(dev); ref.load_state_dict(ck["g_state"])
        x = cells.X[vs[i]].clone().requires_grad_(True); y = cells.X[vt[i]]
        e_ref = (ref.transport(x).detach().median(0).values - y.median(0).values).abs().mean().item()
        e_id = (x.detach().median(0).values - y.median(0).values).abs().mean().item()
        r = {"pair": f"{m['stim']}:{m['celltype']}", "identity": e_id, "paper model.pt": e_ref}
        for name, net in [("ours last", g_last), ("ours best", g_best)]:
            xi = x.detach().unsqueeze(0).requires_grad_(True)
            r[name] = (sub(net, torch.tensor([i], device=dev)).transport(xi)[0].detach().median(0).values
                       - y.median(0).values).abs().mean().item()
        rows.append(r)
    t = pd.DataFrame(rows); t.to_csv(out / "parity.csv", index=False); print(t.round(4).to_string(index=False))
    # lenient on purpose: the paper's model.pt trained on part of these cells and was selected on them
    ok = bool(((t["ours best"] <= np.maximum(2.0 * t["paper model.pt"], t["paper model.pt"] + 0.02))
               & (t["ours best"] < t["identity"])).all())
    print(f"[PARITY] {'PASS' if ok else 'FAIL'}")
    if ok: (WORK / "PARITY_PASS").write_text(t.to_csv(index=False))


def bench(a, design, dev, out):
    dz = design[0]
    rows = []
    for M in [168, 840, 1680]:
        pairs = PAIRS * (M // len(PAIRS))
        maps = [{"stim": s, "celltype": c, "train": dz["order"][:8]} for s, c in pairs][:M]
        cells = Cells(sorted({m["celltype"] for m in maps}), dev)
        s_tr = Sampler(*cells.table([[(m["celltype"], "Unstim", d) for d in m["train"]] for m in maps], "train"), dev)
        t_tr = Sampler(*cells.table([[(m["celltype"], m["stim"], d) for d in m["train"]] for m in maps], "train"), dev)
        f, g = BatchedICNN(M, D).to(dev), BatchedICNN(M, D, fnorm_penalty=1).to(dev)
        of, og = torch.optim.Adam(f.parameters(), **ADAM), torch.optim.Adam(g.parameters(), **ADAM)
        for it in range(60):
            if it == 10:
                torch.cuda.synchronize() if dev.type == "cuda" else None; t0 = time.time()
            tgt = cells.X[t_tr.draw(BATCH)]
            for _ in range(INNER):
                src = cells.X[s_tr.draw(BATCH)].requires_grad_(True)
                og.zero_grad(); (loss_g(f, g, src) + g.penalize_w()).sum().backward(); og.step()
            src = cells.X[s_tr.draw(BATCH)].requires_grad_(True)
            of.zero_grad(); loss_f(f, g, src, tgt).sum().backward(); of.step(); f.clamp_w()
        torch.cuda.synchronize() if dev.type == "cuda" else None
        ms = (time.time() - t0) / 50 * 1000
        rows.append({"M": M, "ms_per_iter": ms, "hours_50k": ms * 50_000 / 3.6e6,
                     "gpu_s_per_map": ms * 50_000 / 1000 / M, "mem_GB": torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else None})
        print(rows[-1], flush=True)
        del f, g, of, og, cells; torch.cuda.empty_cache() if dev.type == "cuda" else None
    pd.DataFrame(rows).to_csv(out / "bench.csv", index=False)


if __name__ == "__main__":
    main()
