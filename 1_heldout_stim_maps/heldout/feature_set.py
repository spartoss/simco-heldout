"""The stim feature set chosen by the user (2026-10-12): a (stim, celltype, marker, quantile) is a feature if, at that
quantile q in (50, 75, 90), the mean over donors of |q(stim) - q(Unstim)| > 0.1 (per donor, cache_stim_all: DMSO
wells pooled, uncorrected, <= 10k cells; statistics from heldout/inclusion_grid.py -> features.csv). PI and IL33 are
not used. A (stim, celltype) pair is in the set if at least one of its features passes.
Writes (git-tracked, small):
    meta/feature_set.csv : one row per feature (stim, celltype, marker, q, mean_effect, in_map_grid)
    meta/feature_pairs.csv : one row per pair (stim, celltype, n_features, quantiles, in_map_grid)
    python -m heldout.feature_set
"""
import pandas as pd

from heldout.common import WORK
from lc.paths import REPO

THR = 0.1
QS = (50, 75, 90)


def main():
    F = pd.read_csv(WORK / "features" / "inclusion_grid" / "features.csv")
    rows = [{"stim": r.stim, "celltype": r.celltype, "marker": r.marker, "q": q, "mean_effect": round(getattr(r, f"mean_q{q}"), 4),
             "in_map_grid": r.in_map_grid}
            for r in F.itertuples() for q in QS if getattr(r, f"mean_q{q}") > THR]
    S = pd.DataFrame(rows).sort_values(["stim", "celltype", "marker", "q"])
    S.to_csv(REPO / "meta" / "feature_set.csv", index=False)
    P = S.groupby(["stim", "celltype"]).agg(n_features=("q", "size"), n_markers=("marker", "nunique"),
                                             quantiles=("q", lambda x: " ".join(map(str, sorted(set(x))))),
                                             in_map_grid=("in_map_grid", "first")).reset_index()
    P.to_csv(REPO / "meta" / "feature_pairs.csv", index=False)
    print(f"{len(S)} features ({S[['stim', 'celltype', 'marker']].drop_duplicates().shape[0]} (stim, celltype, marker)), "
          f"{len(P)} pairs ({P.in_map_grid.sum()} with a map, {(~P.in_map_grid).sum()} without)")
    print("features per q:", S.q.value_counts().sort_index().to_dict())
    print("per stim:", S.groupby("stim").size().to_dict())
    print("pairs without a map:", [f"{s}:{c}" for s, c in P.loc[~P.in_map_grid, ["stim", "celltype"]].values])


if __name__ == "__main__":
    main()
