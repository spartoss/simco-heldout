"""Tables for the model.pt-vs-last.pt comparison produced by selection_leak.py.

Three questions, in order of what they license:

1. Where did the argmin land? ``ckpt_step`` is the step model.pt was saved at. If it
   clusters late, selection had little room to act and the gap below is mostly training
   progress; if it is spread over the whole run, the argmin is picking among snapshots
   that are not ordered by quality, which is the regime where it mines noise.

2. How large is the gap, per metric? Reported as last.pt minus model.pt, so a positive
   number means the selected checkpoint scores better -- which is what selection was
   for, and on these cells it is not evidence of anything else.

3. Does it change the verdict? Only meaningful in the stim arm, where the pooled
   baselines exist: re-run the head-to-heads from summarize.py with CellOT represented
   by last.pt instead of model.pt. If median_shift still wins, the earlier conclusion
   did not rest on the leak.

The live/null split is identity's MMD, as in summarize.py, so the rows quoted here are
the same 127 pairs.
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["mean_error", "median_error", "w1", "mmd", "corr_error"]
LIVE_MMD = 0.005


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--indir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--pooled-dir", default=None,
                   help="stim arm only: simco_work/baselines/pooled, to re-run the "
                        "baseline head-to-heads with last.pt")
    p.add_argument("--live-mmd", type=float, default=LIVE_MMD)
    return p.parse_args()


def load(indir):
    files = sorted(glob.glob(str(Path(indir) / "*.csv")))
    if not files:
        raise SystemExit(f"no CSVs in {indir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["drug"] = df["drug"].fillna("-")
    return df


KEY = ["stim", "celltype", "drug"]


def mark_live(df, thresh):
    """Tag each pair live/null by identity's MMD, the same rule summarize.py uses."""
    ident = (df[df.method == "identity"][KEY + ["mmd"]]
             .rename(columns={"mmd": "identity_mmd"}))
    df = df.merge(ident, on=KEY, how="left", validate="many_to_one")
    df["live"] = df["identity_mmd"] >= thresh
    return df


def main():
    args = parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    df = mark_live(load(args.indir), args.live_mmd)
    arm = df["arm"].iloc[0]
    df.to_csv(out / "all_rows.csv", index=False)

    n_pairs = df[df.method == "identity"].shape[0]
    n_live = int(df[df.method == "identity"]["live"].sum())
    print(f"arm={arm}  pairs={n_pairs}  live={n_live}  null={n_pairs - n_live}")

    wide = df.pivot_table(index=["stim", "celltype", "drug", "live"],
                          columns="method", values=METRICS)

    # --- 1. where the argmin landed
    steps = df[df.method == "cellot_model"]["ckpt_step"]
    q = steps.quantile([0, .25, .5, .75, 1]).round(0)
    print("\nckpt_step of model.pt (n_iters=50000, eval_freq=250 -> 200 candidates)")
    print(f"  min {q[0]:.0f}  q25 {q[.25]:.0f}  median {q[.5]:.0f}  "
          f"q75 {q[.75]:.0f}  max {q[1]:.0f}")
    print(f"  fraction in the last 10% of training (>=45000): "
          f"{(steps >= 45000).mean():.3f}")
    steps.describe().to_csv(out / "ckpt_step.csv")

    # --- 2. the gap, live pairs only
    rows = []
    for m in METRICS:
        sub = wide[m].loc[wide.index.get_level_values("live")]
        gap = sub["cellot_last"] - sub["cellot_model"]
        rows.append({
            "metric": m,
            "model_pt": sub["cellot_model"].mean(),
            "last_pt": sub["cellot_last"].mean(),
            "identity": sub["identity"].mean(),
            "mean_gap": gap.mean(),
            "median_gap": gap.median(),
            # The share of pairs where selection actually bought something on this
            # metric. Only one metric per arm was selected on; the others are along
            # for the ride and a value near 0.5 there is what no effect looks like.
            "frac_model_better": float((gap > 0).mean()),
        })
    gaps = pd.DataFrame(rows).set_index("metric")
    gaps.to_csv(out / "gap_live.csv")
    print(f"\nmodel.pt vs last.pt, {n_live} live pairs "
          f"(gap = last - model, positive means selection helped)")
    print(gaps.round(5).to_string())

    # --- 3. does it change the verdict
    if args.pooled_dir:
        pooled = pd.concat(
            [pd.read_csv(f) for f in sorted(glob.glob(str(Path(args.pooled_dir) / "*.csv")))],
            ignore_index=True,
        )
        pooled = pooled[pooled.weighting.isin(["cell", "invariant"])]
        base = pooled.pivot_table(index=["stim", "celltype"], columns="method",
                                  values=METRICS)
        ours = df[df.live].pivot_table(index=["stim", "celltype"], columns="method",
                                       values=METRICS)
        methods = [c for c in base.columns.get_level_values(1).unique()
                   if c not in ("identity", "cellot_insample")]
        lines = []
        for m in METRICS:
            b, o = base[m].reindex(ours.index), ours[m]
            for cell_ot, label in (("cellot_model", "model.pt"), ("cellot_last", "last.pt")):
                for meth in methods:
                    if meth not in b:
                        continue
                    win = (b[meth] < o[cell_ot]).mean()
                    lines.append({"metric": m, "cellot": label, "baseline": meth,
                                  "baseline_beats_cellot": float(win),
                                  "baseline_mean": b[meth].mean(),
                                  "cellot_mean": o[cell_ot].mean()})
        verdict = pd.DataFrame(lines)
        verdict.to_csv(out / "verdict_vs_baselines.csv", index=False)
        head = verdict[
            ((verdict.metric == "median_error") & (verdict.baseline == "median_shift"))
            | ((verdict.metric == "mean_error") & (verdict.baseline == "gauss_ot"))
            | ((verdict.metric == "corr_error") & (verdict.baseline == "gauss_ot"))
            | ((verdict.metric == "mmd") & (verdict.baseline == "gauss_ot"))
        ]
        print("\nhead-to-heads, live pairs, cell-weighted "
              "(baseline_beats_cellot = share of pairs the baseline wins)")
        print(head.round(4).to_string(index=False))

    # --- per-stim and per-drug breakdown of the gap on the selected metric
    sel_metric = "median_error" if arm == "stim" else "mmd"
    sub = wide[sel_metric].loc[wide.index.get_level_values("live")].reset_index()
    sub["gap"] = sub["cellot_last"] - sub["cellot_model"]
    for by in (["stim"] + ([] if arm == "stim" else ["drug"])):
        tab = sub.groupby(by).agg(
            n=("gap", "size"), model_pt=("cellot_model", "mean"),
            last_pt=("cellot_last", "mean"), mean_gap=("gap", "mean"),
            frac_model_better=("gap", lambda g: float((g > 0).mean())),
        )
        tab.to_csv(out / f"gap_by_{by}.csv")
        print(f"\n{sel_metric} (the arm's selection metric) by {by}")
        print(tab.round(5).to_string())

    # --- drug arm: identity is the null "the drug does nothing"
    if arm == "drug":
        sub = wide["mmd"].loc[wide.index.get_level_values("live")].reset_index()
        tab = pd.DataFrame({
            "model_pt_beats_identity": [(sub.cellot_model < sub.identity).mean()],
            "last_pt_beats_identity": [(sub.cellot_last < sub.identity).mean()],
        })
        tab.to_csv(out / "vs_identity.csv", index=False)
        print("\nvs the 'drug does nothing' null, live pairs, MMD")
        print(tab.round(4).to_string(index=False))

    print(f"\n[WRITE] {out}")


if __name__ == "__main__":
    main()
