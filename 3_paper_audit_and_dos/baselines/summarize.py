"""Pool the per-celltype LOPO results into the CellOT-vs-baselines comparison.

Two views only, both in the metrics' own units:

  1. raw means -- the average error each method makes. Caveat: not comparable
     across celltypes (marker scales differ), so the average leans on the
     celltypes where everyone does badly.
  2. paired win rates -- of the N held-out-patient evaluations, the fraction where
     a baseline's error is smaller than CellOT's on that same
     (celltype, stim, patient). No normalisation, no ratios, nothing to blow up;
     0.50 means a coin flip.

Reminder on what is being compared: the CellOT column is ``cellot_insample`` -- the
paper's models were trained on all 19 patients, so every held-out patient was in
CellOT's training set and not in the baselines'. CellOT is being flattered here.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["mean_error", "median_error", "w1", "mmd", "corr_error"]
CELLOT = "cellot_insample"
KEYS = ["celltype", "stim", "patient"]


def keys_for(df):
    """Pairing keys, extended by ``drug`` on a drug-arm run.

    Without it the nine drugs of one (stim, celltype) collapse into a single pivot cell
    and every win rate is computed against a mean over drugs -- wrong, and silently so.
    A stim-arm dir has no such column and is unaffected.
    """
    if "drug" in df.columns and df["drug"].notna().any():
        return KEYS + ["drug"]
    return KEYS


def pair_keys(df):
    """The unit a stimulus/drug effect is measured on: one pair, pooled over patients."""
    return ["stim", "celltype"] + (["drug"] if "drug" in keys_for(df) else [])


def load(indir):
    files = sorted(Path(indir).glob("*.csv"))
    if not files:
        raise SystemExit(f"no result CSVs in {indir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    keys = keys_for(df)
    print(f"loaded {len(files)} celltypes, {len(df)} rows, "
          f"{df.groupby(keys).ngroups} ({', '.join(keys)}) cells\n")
    return df


def raw_means(df):
    return df.groupby("method")[METRICS].mean()


def win_rates(df):
    """Fraction of held-out patients where each method beats CellOT, metric by metric."""
    wide = df.pivot_table(index=keys_for(df), columns="method", values=METRICS)
    rows = {}
    for method in sorted(df.method.unique()):
        if method == CELLOT:
            continue
        rows[method] = {
            m: float((wide[(m, method)] < wide[(m, CELLOT)]).mean()) for m in METRICS
        }
    return pd.DataFrame(rows).T


def per_stim(df, metric):
    return df.pivot_table(index="stim", columns="method", values=metric, aggfunc="mean")


def split_live_null(df, threshold=0.005):
    """Separate pairs by whether the perturbation does anything.

    ``identity``'s MMD is the distance between a patient's untreated cells and their own
    measured treated cells -- i.e. the size of the real effect. Where it is near zero
    (IL33, much of TNFa) there is no response to predict, and every method is fitting
    noise, so pooling those pairs with the rest hides what the methods do when there is
    actually something to learn.

    On the drug arm the same threshold is being asked a harder question: identity there
    is "the drug does nothing", so a near-null pair means the drug had no measurable
    effect on that celltype under that stim, and far more pairs should land in it.
    """
    keys = pair_keys(df)
    effect = df[df.method == "identity"].groupby(keys)["mmd"].mean()
    live = effect[effect >= threshold].index
    key = pd.MultiIndex.from_frame(df[keys])
    return df[key.isin(live)], df[~key.isin(live)], effect


def select_arm(df, fit_patients, weighting):
    """Keep one (fit_patients, weighting) arm, plus the two methods that have no arm.

    identity is fit-free and CellOT is fixed at (all, cell) by construction, so both are
    emitted once and have to survive every filter -- dropping CellOT would leave
    win_rates with nothing to compare against.
    """
    if fit_patients is None and weighting is None:
        return df
    keep = df.method.isin([CELLOT, "identity"])
    arm = pd.Series(True, index=df.index)
    if fit_patients is not None and "fit_patients" in df.columns:
        arm &= df.fit_patients == fit_patients
    if weighting is not None and "weighting" in df.columns:
        arm &= df.weighting == weighting
    return df[keep | arm]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--indir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--fit-patients", choices=["loo", "all"], default=None,
                   help="restrict to one arm; omit only if the run has a single arm")
    p.add_argument("--weighting", choices=["unweighted", "cell"], default=None)
    args = p.parse_args()

    df = load(args.indir)
    # The single-arm dirs (lopo, lopo_test, lopo_test_v2) predate both columns: they were
    # written before the fit_patients/weighting arms existed, so absence means one arm.
    arm_cols = [c for c in ("fit_patients", "weighting") if c in df.columns]
    arms = df.loc[~df.method.isin([CELLOT, "identity"]), arm_cols]
    n_arms = arms.drop_duplicates().shape[0] if arm_cols else 1
    if n_arms > 1 and args.fit_patients is None and args.weighting is None:
        print(f"WARNING: {n_arms} arms present and no filter given -- every table below "
              f"averages a method over all of them, which is not a quantity anyone "
              f"wants. Pass --fit-patients/--weighting.\n")
    df = select_arm(df, args.fit_patients, args.weighting)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Roughly increasing capacity: fit-free, then diagonal, then marker-mixing, then
    # patient-adaptive, with CellOT last.
    order = ["identity", "median_shift", "quantile_map", "gauss_ot",
             "linear_median", "linear_median_ridge", CELLOT]
    order = [m for m in order if m in set(df.method)]

    live, null, effect = split_live_null(df)
    n_live = effect.ge(0.005).sum()
    unit = ", ".join(pair_keys(df))
    print(f"{n_live} of {len(effect)} ({unit}) pairs have a real response "
          f"(identity MMD >= 0.005); {len(effect) - n_live} are near-null\n")

    tables = {
        "raw_means_all": raw_means(df).reindex(order),
        "raw_means_live": raw_means(live).reindex(order),
        "raw_means_null": raw_means(null).reindex(order),
        "win_rate_vs_cellot_all": win_rates(df),
        "win_rate_vs_cellot_live": win_rates(live),
        "win_rate_vs_cellot_null": win_rates(null),
        "stim_effect_size": effect.groupby("stim").median().to_frame("identity_mmd"),
        "per_stim_mmd": per_stim(df, "mmd"),
        "per_stim_mean_error": per_stim(df, "mean_error"),
        "per_celltype_mmd": df.pivot_table(
            index="celltype", columns="method", values="mmd", aggfunc="mean"
        ).reindex(columns=order),
    }

    # Drug arm only. The per-drug effect size is the one to read first: a drug whose
    # identity MMD is near zero everywhere did not measurably perturb these cells, and
    # nothing downstream of it -- CellOT's map included -- can be learning a real signal.
    if "drug" in keys_for(df):
        tables["drug_effect_size"] = (
            effect.groupby("drug").agg(["median", "max", "count"])
        )
        tables["per_drug_mmd"] = df.pivot_table(
            index="drug", columns="method", values="mmd", aggfunc="mean"
        ).reindex(columns=order)
        by_drug = {d: win_rates(sub) for d, sub in live.groupby("drug")}
        if by_drug:
            tables["win_rate_by_drug"] = pd.concat(by_drug, names=["drug"])

    # Where the ridge penalty lands is a result in its own right: pinned at the top of
    # the path means the patient's baseline profile carried no usable cross-marker
    # signal and the fit degenerated to a constant shift.
    if "alpha" in df.columns and df["alpha"].notna().any():
        tables["ridge_alpha_by_stim"] = (
            df[df.alpha.notna() & (df.alpha > 0)]
            .pivot_table(index="stim", columns="method", values="alpha", aggfunc="median")
        )

    for name, table in tables.items():
        table.to_csv(outdir / f"{name}.csv")
        print(f"=== {name} ===")
        print(table.to_string(float_format=lambda v: f"{v:9.4f}"))
        print()

    df.to_csv(outdir / "all_rows.csv", index=False)
    print(f"[WRITE] {outdir}")


if __name__ == "__main__":
    main()
