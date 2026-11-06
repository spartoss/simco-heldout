"""In-sample fit vs held-out generalisation: is a model under-fit, or is it a donor problem?

For each repeat and k, every model predicts the TRAINING donors' stim quantiles from their own val-side Unstim cells
(the last 20 % of each cached block: cells never trained on, donors trained on), scored with the lc skill
(1 - |pred - truth| / |truth - Unstim|, cases with effect > FLOOR; truth / Unstim = val-side quantiles). Same cells,
same quantiles, same scoring for every model. Compare with the same models' test-donor skill (lc.model_zoo):
- in-sample high, test low  -> fits the donors it saw, does not transfer: a donor problem;
- in-sample low too         -> it does not even fit its own donors: under-fit.
Models: CellOT_paired (atlas_heldout cond_train/<r>/none final_maps.pt g_best), fm / mfm / mfm_blood (_sel model.pt);
shift = training mean (train side) + val-side Unstim deviation (the same feature-level baseline, in-sample).
Output work/insample/cases.parquet-like pickle + summary.csv.
    python3 -m mfm.insample --repeats 1 2 3 4 --ks 3 15            (GPU: slurm/insample.sbatch)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lc.cond_icnn import CondBatchedICNN, sub
from lc.paths import CELLTYPES, MARKERS, PAIRS, QS, VAL_FRAC, WORK as LCWORK
from mfm.model import MFM
from mfm.train import BLOOD_CTS, STIMS

CACHE = LCWORK / "fm" / "cache_full"
RUNS = Path(__file__).resolve().parents[1] / "work" / "runs"
OUT = Path(__file__).resolve().parents[1] / "work" / "insample"
FLOOR, MIN_VAL = 0.0273, 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", nargs="*", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--ks", nargs="*", type=int, default=[3, 15])
    ap.add_argument("--max-cells", type=int, default=2000)
    ap.add_argument("--ode-steps", type=int, default=100)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    blk, panel = {}, None
    for c in CELLTYPES:
        z = np.load(CACHE / f"{c}.npz"); panel = list(z["markers"])
        for key in z.files:
            if key == "markers": continue
            cond, d = key.split("|"); x = torch.from_numpy(z[key]).to(dev)
            ntr = max(1, int(round(len(x) * (1 - VAL_FRAC))))
            blk[(c, cond, d)] = (x[:ntr], x[ntr:])
    fi = torch.tensor([panel.index(m) for m in MARKERS], device=dev)
    oh = lambda i, n: torch.nn.functional.one_hot(torch.tensor(i, device=dev), n).float()
    ct_oh = {c: oh(CELLTYPES.index(c), len(CELLTYPES)) for c in CELLTYPES}
    qt = torch.tensor([q / 100 for q in QS], device=dev)
    quant = lambda x: torch.quantile(x, qt, dim=0).cpu().numpy()
    rows = []
    for r in a.repeats:
        dz = json.load(open(LCWORK / "design.json"))[r]
        for k in a.ks:
            train_d = dz["order"][:k]
            # CellOT_paired: the k-maps of this repeat's cond_train none run
            cdir = LCWORK / f"cond_train/repeat_{r}/none"
            cm = pd.read_csv(cdir / "maps.csv"); M = len(cm)
            g = CondBatchedICNN(M, len(MARKERS), 0, fnorm_penalty=1).to(dev)
            g.load_state_dict(torch.load(cdir / "final_maps.pt", map_location=dev, weights_only=False)["g_best"])
            cidx = {(s, c): i for i, (s, c, kk) in enumerate(zip(cm.stim, cm.celltype, cm.k)) if kk == k}
            flows = {}
            for arm in ("fm", "mfm", "mfm_blood"):
                d = RUNS / f"{arm}_sel/repeat_{r}/k{k}"
                if not (d / "model.pt").exists(): continue
                meta = json.load(open(d / "meta.json"))
                n_pop = 1 + (len(BLOOD_CTS) if arm == "mfm_blood" else 0)
                m = MFM(len(MARKERS), len(panel), len(STIMS) + len(CELLTYPES), len(CELLTYPES), n_pop=n_pop,
                        use_embedding=arm != "fm", knn_k=meta["knn_k"]).to(dev)
                m.load_state_dict(torch.load(d / "model.pt", map_location=dev, weights_only=False)); m.eval()
                flows[arm] = m
            gen = torch.Generator(device="cpu").manual_seed(r * 100 + k)
            pick = lambda x, n: x[torch.randperm(len(x), generator=gen)[:n].to(dev)] if len(x) > n else x
            for s, c in PAIRS:
                trd = [d for d in train_d if (c, "Unstim", d) in blk and (c, s, d) in blk]
                if len(trd) < 2: continue
                mu_u = np.mean([quant(blk[(c, "Unstim", d)][0][:, fi]) for d in trd], 0)
                mu_s = np.mean([quant(blk[(c, s, d)][0][:, fi]) for d in trd], 0)
                cond = torch.cat([oh(STIMS.index(s), len(STIMS)), ct_oh[c]])
                for d in trd:
                    xu, xs = blk[(c, "Unstim", d)][1], blk[(c, s, d)][1]
                    if len(xu) < MIN_VAL or len(xs) < MIN_VAL: continue
                    x0 = pick(xu, a.max_cells)[:, fi]
                    u, t = quant(xu[:, fi]), quant(xs[:, fi])
                    preds = {"identity": u, "shift": mu_s + u - mu_u}
                    if (s, c) in cidx:
                        xg = x0.unsqueeze(0).clone().requires_grad_(True)
                        preds["CellOT_paired"] = quant(sub(g, torch.tensor([cidx[(s, c)]], device=dev))
                                                       .transport(xg, None).detach()[0])
                    with torch.no_grad():
                        for arm, m in flows.items():
                            pops = [(pick(blk[(c, "Unstim", d)][0], 512), ct_oh[c])]
                            if arm == "mfm_blood":
                                for bc in BLOOD_CTS:
                                    xb = blk[(bc, "Unstim", d)][0] if (bc, "Unstim", d) in blk else blk[(c, "Unstim", d)][0]
                                    pops.append((pick(xb, 512), ct_oh[bc]))
                            zz = m.embed(pops) if arm != "fm" else None
                            preds[arm] = quant(m.transport(zz, x0, cond.expand(len(x0), -1), a.ode_steps))
                    eff = np.abs(t - u)
                    for name, p in preds.items():
                        rows.append(pd.DataFrame({"repeat": r, "k": k, "stim": s, "celltype": c, "donor": d,
                                                  "method": name, "q": np.repeat(QS, len(MARKERS)),
                                                  "marker": np.tile(MARKERS, len(QS)), "pred": p.ravel(),
                                                  "true": t.ravel(), "eff": eff.ravel()}))
            print(f"[DONE ] repeat {r} k {k} ({len(flows)} flows) {time.time() - t0:.0f}s", flush=True)
    D = pd.concat(rows, ignore_index=True); D = D[D.eff > FLOOR]
    D["skill"] = 1 - (D.pred - D.true).abs() / D.eff
    key = ["repeat", "k", "stim", "celltype", "donor", "q", "marker"]
    n = D.groupby(key).method.transform("nunique"); D = D[n == n.max()]           # common cases
    D.to_pickle(OUT / "cases.pkl")
    S = D.groupby(["method", "k", "repeat"]).skill.median().groupby(["method", "k"]).mean().unstack("k")
    S.round(4).to_csv(OUT / "summary.csv"); print(S.round(3).to_string())


if __name__ == "__main__":
    main()
