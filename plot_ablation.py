#!/usr/bin/env python3
# Plot WER change vs baseline for ADAS2T-XGB and ADAS2T-MLP from ablate_driver outputs

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="ablation_runs", help="Directory with ablate_* outputs")
    ap.add_argument("--save", default="wer_change_vs_baseline.png", help="Where to save the main plot")
    ap.add_argument("--per_dataset", action="store_true", help="Also make per-dataset plots")
    args = ap.parse_args()

    runs = Path(args.runs_dir)
    csv_long = runs / "ablation_summary_long.csv"
    if not csv_long.exists():
        raise SystemExit(f"Not found: {csv_long}. Run ablate_driver.py first.")

    df = pd.read_csv(csv_long)
    routers = df[df["model"].isin(["ADAS2T-XGB", "ADAS2T-MLP"])].copy()

    # Get baseline WER per model & dataset
    baseline = routers[routers["condition"] == "baseline"]
    baseline_map = baseline.set_index(["dataset", "model"])["WER"].to_dict()

    # Compute delta from baseline (positive means worse than baseline)
    routers["delta_WER"] = routers.apply(
        lambda r: r["WER"] - baseline_map.get((r["dataset"], r["model"]), 0),
        axis=1
    )

    # Aggregate mean delta per condition
    cond_pivot = (
        routers.groupby(["condition", "model"])["delta_WER"]
        .mean()
        .unstack("model")
        .sort_index()
    )

    # Remove baseline row (delta=0 always)
    cond_pivot = cond_pivot.drop(index="baseline", errors="ignore")

    # Plot mean delta WER across datasets
    ax = cond_pivot.plot(kind="bar", rot=45)
    ax.set_ylabel("ΔWER vs Baseline (mean across datasets)")
    ax.set_xlabel("Ablation condition")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Change in WER vs baseline (lower is better)")
    ax.legend(title="")
    plt.tight_layout()
    plt.savefig(args.save, dpi=200)
    print(f"Saved: {args.save}")

    # Optional per-dataset delta plots
    if args.per_dataset:
        out_dir = runs / "plots_delta_per_dataset"
        out_dir.mkdir(parents=True, exist_ok=True)
        ds_cond = (
            routers.pivot_table(index=["dataset", "condition"], columns="model", values="delta_WER", aggfunc="mean")
            .reset_index()
        )
        for cond in ds_cond["condition"].unique():
            if cond == "baseline":
                continue
            sub = ds_cond[ds_cond["condition"] == cond].sort_values("dataset")
            fig, ax = plt.subplots()
            ax.plot(sub["dataset"], sub["ADAS2T-XGB"], marker="o", label="ADAS2T-XGB")
            ax.plot(sub["dataset"], sub["ADAS2T-MLP"], marker="o", label="ADAS2T-MLP")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(f"ΔWER vs Baseline — {cond}")
            ax.set_ylabel("ΔWER")
            ax.set_xlabel("Dataset")
            ax.set_xticklabels(sub["dataset"], rotation=45, ha="right")
            ax.legend()
            plt.tight_layout()
            out_path = out_dir / f"deltaWER_by_dataset__{cond}.png"
            plt.savefig(out_path, dpi=200)
            plt.close(fig)
            print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
