"""Donor-level ImmuneFM covariates (phase 1): per (donor, celltype), the mean cell embedding of the donor's Unstim
DMSO cells (first --n cells of the shuffled fm cache; Unstim only, so nothing of any stim response leaks in).
Arms: pretrained, random (same architecture at random init: the ablation), funcmasked (pretrained, the 13
functional markers set to MASK: phenotype only). --ckpt (fm/finetune.py) instead embeds with that checkpoint only:
arms "finetuned" and "finetuned_funcmasked" (quantile tokens, 38 panel positions). A fine-tune has seen its repeat's
training donors only, so its covariates are valid for that repeat's lc15 fold only.
Output WORK/fm/donor_emb.npz: <arm> (n_donors, n_celltypes, 128) float32 (NaN where a donor has < 20 cells),
donors, celltypes. Whole-blood weighting uses WORK/fm/counts.csv (fm/counts.py) at read time.
    python3 -m fm.embed_donors [--n 2000]        (GPU: slurm/fm_embed.sbatch)
"""
import argparse
import os
import time

import numpy as np
import torch

from fm.finetune import load_ft
from fm.immunefm import load
from fm.tokenize import Q_MASK, columns, q_tokens, tokens
from lc.paths import CELLTYPES, DONORS, MARKERS, WORK

CACHE = WORK / "fm" / "cache_full"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=str(WORK / "fm" / "donor_emb.npz"))
    a = ap.parse_args()
    t0 = time.time()
    if a.ckpt:
        ft, ck = load_ft(a.ckpt, DEV)
        fi = [ck["panel"].index(m) for m in MARKERS]; allc = list(range(len(ck["panel"])))
        arms = ["finetuned", "finetuned_funcmasked"]
    else:
        models = {k: v.to(DEV) for k, v in {"pretrained": load(True), "random": load(False)}.items()}
        arms = list(models) + ["funcmasked"]
    E = {k: np.full((len(DONORS), len(CELLTYPES), 128), np.nan, np.float32) for k in arms}
    for j, c in enumerate(CELLTYPES):
        z = np.load(CACHE / f"{c}.npz"); panel = list(z["markers"]); _, cols = columns(panel)
        for i, d in enumerate(DONORS):
            k = f"Unstim|{d}"
            if k not in z.files or len(z[k]) < 20: continue
            X = z[k][:a.n]
            if a.ckpt:
                assert panel == ck["panel"]
                T = q_tokens(X, ck["edges"]).astype(np.int64); Tm = T.copy(); Tm[:, fi] = Q_MASK
                E["finetuned"][i, j] = ft.encode(T, allc, device=DEV, pos=ck["pos"]).mean(0)
                E["finetuned_funcmasked"][i, j] = ft.encode(Tm, allc, device=DEV, pos=ck["pos"]).mean(0)
                continue
            T = tokens(X, panel); Tm = tokens(X, panel, mask_cols=MARKERS)
            for arm, mdl in models.items():
                E[arm][i, j] = mdl.encode(T, cols, device=DEV).mean(0)
            E["funcmasked"][i, j] = models["pretrained"].encode(Tm, cols, device=DEV).mean(0)
        print(f"[DONE ] {c} {time.time() - t0:.0f}s", flush=True)
    tmp = a.out + f".{os.getpid()}.tmp.npz"
    np.savez(tmp, donors=np.array(DONORS), celltypes=np.array(CELLTYPES), **E); os.replace(tmp, a.out)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
