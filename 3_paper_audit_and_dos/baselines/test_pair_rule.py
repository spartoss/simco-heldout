"""Replay the paper's stated pair-selection rule against the actual 168/28 split.

The stated rule: a (stim, celltype) pair is kept if at least one of the 13 functional
markers shows an arcsinh median difference between stim and Unstim above 0.1 -- except
IL33, where the threshold is 0.01.

The rule leaves two things unspecified, so all combinations are tried and scored:

  aggregation  how per-patient medians become one number per (stim, celltype, marker)
    pooled     mean_p median_p(stim) - mean_p median_p(Unstim)   [median_shift_fit's convention]
    median     median_p [ median_p(stim) - median_p(Unstim) ]
    mean       mean_p   [ median_p(stim) - median_p(Unstim) ]
  sign
    abs        |diff| -- a response in either direction counts
    signed     diff   -- only increases count

Ground truth is valid_jobs.txt (168 lines), which is 1:1 with features_evaluation/.
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

STIMS = ["GMCSF", "IFNa", "IL246", "IL33", "LPS", "PI", "TNFa"]
SOURCE = "Unstim"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--medians", required=True, help="dir of per-celltype medians CSVs")
    p.add_argument("--valid-jobs", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--il33-threshold", type=float, default=0.01)
    return p.parse_args()


def load_medians(indir):
    files = sorted(Path(indir).glob("*.csv"))
    if not files:
        raise SystemExit(f"no medians CSVs in {indir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"loaded {len(files)} celltypes, {len(df):,} (celltype, stim, patient, marker) rows")
    return df


def load_truth(path):
    kept = set()
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        stim, celltype = line.split(",")[:2]
        kept.add((stim, celltype))
    print(f"ground truth: {len(kept)} kept pairs")
    return kept


def marker_diffs(df):
    """Per (celltype, stim, marker) effect size, under each aggregation."""
    wide = df.pivot_table(
        index=["celltype", "patient", "marker"], columns="stim", values="median"
    )
    out = []
    for stim in STIMS:
        if stim not in wide.columns:
            continue
        per_patient = (wide[stim] - wide[SOURCE]).dropna()
        g = per_patient.groupby(level=["celltype", "marker"])
        agg = pd.DataFrame({
            "median": g.median(),
            "mean": g.mean(),
            # mean-of-medians differenced, i.e. the same quantity median_shift_fit uses
            "pooled": (
                wide[stim].groupby(level=["celltype", "marker"]).mean()
                - wide[SOURCE].groupby(level=["celltype", "marker"]).mean()
            ),
            "n_patients": g.size(),
        })
        agg["stim"] = stim
        out.append(agg.reset_index())
    return pd.concat(out, ignore_index=True)


def evaluate_rule(diffs, truth, agg, sign, thr, il33_thr):
    d = diffs.copy()
    d["effect"] = d[agg].abs() if sign == "abs" else d[agg]
    d["thr"] = np.where(d.stim == "IL33", il33_thr, thr)
    d["passes"] = d.effect > d.thr

    pair = d.groupby(["stim", "celltype"]).agg(
        n_passing=("passes", "sum"),
        max_effect=("effect", "max"),
    )
    pair["predicted_keep"] = pair.n_passing >= 1
    pair["actual_keep"] = [idx in truth for idx in pair.index]

    tp = int((pair.predicted_keep & pair.actual_keep).sum())
    fp = int((pair.predicted_keep & ~pair.actual_keep).sum())
    fn = int((~pair.predicted_keep & pair.actual_keep).sum())
    tn = int((~pair.predicted_keep & ~pair.actual_keep).sum())
    return pair, dict(agg=agg, sign=sign, tp=tp, fp=fp, fn=fn, tn=tn,
                      agree=tp + tn, total=len(pair))


def main():
    args = parse_args()
    df = load_medians(args.medians)
    truth = load_truth(args.valid_jobs)
    diffs = marker_diffs(df)

    markers = sorted(df.marker.unique())
    print(f"{len(markers)} markers: {markers}\n")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    diffs.to_csv(outdir / "marker_effects.csv", index=False)

    results, tables = [], {}
    for agg, sign in itertools.product(["pooled", "median", "mean"], ["abs", "signed"]):
        pair, summary = evaluate_rule(
            diffs, truth, agg, sign, args.threshold, args.il33_threshold
        )
        results.append(summary)
        tables[(agg, sign)] = pair

    res = pd.DataFrame(results).sort_values("agree", ascending=False)
    print(f"=== stated rule (>{args.threshold}, IL33 >{args.il33_threshold}) "
          f"vs the actual {len(truth)}/{len(tables[('pooled','abs')])} split ===")
    print(res.to_string(index=False), "\n")

    # NB: .agg on a Series resolves to Series.aggregate, so index by label.
    best = res.iloc[0]
    best_agg, best_sign = best["agg"], best["sign"]
    pair = tables[(best_agg, best_sign)]
    pair.to_csv(outdir / "pair_rule_best.csv")
    print(f"--- best variant: agg={best_agg}, sign={best_sign} "
          f"({best['agree']}/{best['total']} pairs agree) ---\n")

    fp = pair[pair.predicted_keep & ~pair.actual_keep]
    fn = pair[~pair.predicted_keep & pair.actual_keep]
    print(f"rule keeps but paper dropped ({len(fp)}):")
    print(fp[["max_effect", "n_passing"]].to_string(float_format=lambda v: f"{v:8.4f}") or "  none")
    print(f"\npaper kept but rule drops ({len(fn)}):")
    print(fn[["max_effect", "n_passing"]].to_string(float_format=lambda v: f"{v:8.4f}") or "  none")

    # What threshold, if any, would reproduce the split exactly under the best aggregation?
    print("\n=== threshold sweep under the best aggregation (non-IL33 stims) ===")
    d = diffs.copy()
    d["effect"] = d[best_agg].abs() if best_sign == "abs" else d[best_agg]
    mx = d.groupby(["stim", "celltype"])["effect"].max()
    actual = pd.Series({idx: idx in truth for idx in mx.index})
    for t in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        pred = mx > t
        print(f"  thr={t:<5} agree {int((pred == actual).sum()):3d}/{len(mx)}  "
              f"(keeps {int(pred.sum()):3d})")

    print("\n=== per-stim max effect, best aggregation ===")
    print(mx.groupby(level="stim").describe()[["min", "25%", "50%", "max"]]
            .to_string(float_format=lambda v: f"{v:8.4f}"))


if __name__ == "__main__":
    main()
