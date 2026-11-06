"""Replicate-well ceiling on the learning-curve cases, with atlas_heldout's skill definition.

Skill definition (atlas_heldout lc/score.py, reimplemented here): one case = (test donor, stim, celltype, quantile
q in QS, marker); truth = q-th percentile of the donor's stimulated cells, identity = the same of her Unstim cells
(both: the three DMSO plate-control wells pooled, <= 10,000 cells, i.e. lc's cache); effect = |truth - identity|;
skill = 1 - |prediction - truth| / effect; only cases with effect > FLOOR (0.0273) are kept; the headline number is
the median over cases within a repeat (its 4 test donors x all 168 pairs), averaged over the 5 repeats of
lc/design.json. Per-stim and per-celltype tables take the median over all cases of all repeats (lc's
curve_by_stim). The case set and the FLOOR filter are always lc's (pooled effect), whatever the variant.

Variants ("prediction" -> "target", identity = the target's identity):
  split_half      first half -> second half of the pooled cells, identity pooled Unstim     (lc's ceiling; validation)
  well            stim well A -> stim well B, identity pooled Unstim         (replicate-well ceiling, lc cell budget)
  pseudo_well     the same with the pooled cells randomly re-split into groups of the well sizes (sampling only)
  well_big / pseudo_well_big   the same with up to 30,000 cells per well (less sampling noise)
  well_src        stim A -> stim B, identity = Unstim well B (the source changes well too: input on B's plate)
  delta_transfer  stim A - Unstim A + Unstim B -> stim B, identity Unstim B (a perfect donor response learned on
                  plate A, applied to the input of plate B)
Ordered well pairs: all (A, B), A != B, among usable wells (>= MIN_CELLS cells); under PI, DMSO1 is excluded
(plate artefact, simco CLAUDE.md) and reported separately as stim "PI (with DMSO1)".
    python -m wc.ceiling
"""
import json

import numpy as np
import pandas as pd

from wc.paths import CELLTYPES, FLOOR, LC_DESIGN, MARKERS, MIN_CELLS, PAIRS, QS, QUANT, SUMMARY, WELLS

def load_cases():
    blocks = []
    for ct in CELLTYPES:
        f = QUANT / f"{ct}.npz"
        if not f.exists():
            print(f"[MISS ] {ct}"); continue
        z = np.load(f); keys = set(z.files)
        get = lambda k: z[k] if k in keys else None
        n = lambda k: int(z[k + "|n"]) if k + "|n" in keys else 0
        donors = sorted({k.split("|")[1] for k in keys})
        for stim in sorted({s for s, c in PAIRS if c == ct}):
            for d in donors:
                S, U = f"{stim}|{d}", f"Unstim|{d}"
                if n(S + "|pool") < MIN_CELLS or n(U + "|pool") < MIN_CELLS: continue
                tr, iden = z[S + "|pool"], z[U + "|pool"]
                eff_pool = np.abs(tr - iden)
                base = dict(celltype=ct, stim=stim, donor=d)
                out = [("split_half", "h1", "h2", z[S + "|h1"], z[S + "|h2"], iden)]
                for with_d1 in ([False, True] if stim == "PI" else [False]):
                    wells = [w for w in WELLS if (with_d1 or not (stim == "PI" and w == "DMSO1"))]
                    lab = "PI (with DMSO1)" if with_d1 else stim
                    for A in wells:
                        for B in wells:
                            if A == B: continue
                            okw = n(f"{S}|w:{A}") >= MIN_CELLS and n(f"{S}|w:{B}") >= MIN_CELLS
                            okW = n(f"{S}|W:{A}") >= MIN_CELLS and n(f"{S}|W:{B}") >= MIN_CELLS
                            okU = n(f"{U}|W:{A}") >= MIN_CELLS and n(f"{U}|W:{B}") >= MIN_CELLS
                            if okw:
                                out.append(("well", A, B, z[f"{S}|w:{A}"], z[f"{S}|w:{B}"], iden, lab))
                                out.append(("pseudo_well", A, B, z[f"{S}|pw:{A}"], z[f"{S}|pw:{B}"], iden, lab))
                            if okW:
                                out.append(("well_big", A, B, z[f"{S}|W:{A}"], z[f"{S}|W:{B}"], iden, lab))
                                out.append(("pseudo_well_big", A, B, z[f"{S}|PW:{A}"], z[f"{S}|PW:{B}"], iden, lab))
                            if okW and okU:
                                uA, uB = z[f"{U}|W:{A}"], z[f"{U}|W:{B}"]
                                out.append(("well_src", A, B, z[f"{S}|W:{A}"], z[f"{S}|W:{B}"], uB, lab))
                                out.append(("delta_transfer", A, B, z[f"{S}|W:{A}"] - uA + uB, z[f"{S}|W:{B}"], uB, lab))
                for o in out:
                    var, A, B, pred, tgt, idn = o[:6]; lab = o[6] if len(o) > 6 else stim
                    blocks.append(((var, lab, ct, d, A, B), np.abs(pred - tgt).ravel(), np.abs(tgt - idn).ravel(),
                                   eff_pool.ravel()))
        print(f"[LOAD ] {ct}", flush=True)
    nq, nm = len(QS), len(MARKERS)
    qcol = np.repeat(np.array(QS), nm); mcol = np.tile(np.array(MARKERS), nq)
    meta = pd.DataFrame([b[0] for b in blocks], columns=["variant", "stim", "celltype", "donor", "src", "tgt"])
    for c in meta.columns: meta[c] = meta[c].astype("category")
    D = meta.loc[meta.index.repeat(nq * nm)].reset_index(drop=True)
    D["q"] = np.tile(qcol, len(blocks)); D["marker"] = pd.Categorical(np.tile(mcol, len(blocks)))
    D["err"] = np.concatenate([b[1] for b in blocks]); D["effect"] = np.concatenate([b[2] for b in blocks])
    D["effect_pool"] = np.concatenate([b[3] for b in blocks])
    D["skill"] = 1 - D.err / D.effect
    return D


def by_repeat(L):
    """Attach repeats: each case once per repeat whose test donors include its donor (lc's aggregation)."""
    design = json.load(open(LC_DESIGN))
    m = pd.DataFrame([(dz["repeat"], d) for dz in design for d in dz["test"]], columns=["repeat", "donor"])
    return L.merge(m, on="donor")


def main():
    SUMMARY.mkdir(parents=True, exist_ok=True)
    D = load_cases()
    L = D[D.effect_pool > FLOOR].copy()                      # lc's live filter, on the pooled effect
    L.to_pickle(SUMMARY / "cases_live.pkl")
    main_stims = ~L.stim.astype(str).str.contains("with DMSO1")
    R = by_repeat(L[main_stims])
    per_rep = R.groupby(["variant", "repeat"], observed=True).skill.median().rename("median_skill").reset_index()
    per_rep.to_csv(SUMMARY / "ceiling_per_repeat.csv", index=False)
    head = per_rep.groupby("variant").median_skill.agg(skill="mean", skill_sd="std").reset_index()
    head["n_rows_per_repeat"] = R.groupby(["variant", "repeat"], observed=True).size().groupby("variant").mean().reindex(head.variant).values
    # all 19 donors, each once
    head["skill_all_donors"] = L[main_stims].groupby("variant", observed=True).skill.median().reindex(head.variant).values
    head.to_csv(SUMMARY / "ceiling_overall.csv", index=False)
    print(head.round(4).to_string(index=False))
    allR = by_repeat(L)
    for by in ["stim", "celltype", "q", "marker"]:
        t = allR.groupby(["variant", by], observed=True).skill.agg(median_skill="median", n="size").reset_index()
        t.to_csv(SUMMARY / f"ceiling_by_{by}.csv", index=False)
    # well-pair table (which plates disagree)
    t = allR.groupby(["variant", "stim", "src", "tgt"], observed=True).skill.agg(median_skill="median", n="size").reset_index()
    t.to_csv(SUMMARY / "ceiling_by_wellpair.csv", index=False)
    print(f"[DONE ] -> {SUMMARY}")


if __name__ == "__main__":
    main()
