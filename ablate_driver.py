#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ablate_driver.py (CV-aware, absolute script + data paths)

Automates paper-aligned ablations with CV-aware training:
  1) Retrain XGB (5-fold) + MLP (5-fold) with selected feature groups dropped
  2) Evaluate with eval_mls_eng.py (auto-detects CV ensembles in CWD)
  3) Aggregate WERs into CSV summaries (long + pivots + deltas vs baseline)

Per-condition artifacts are placed under:
  runs_dir/<condition>/

Expected scripts in the same folder as this file:
  - train_xgb_cls.py  (supports --drop_groups, --cv_folds)
  - train_mlp_cls.py  (supports --drop_groups, --cv_folds)
  - eval_mls_eng.py   (auto-detects CV ensembles)

Outputs:
  - runs_dir/ablate_<cond>.json
  - runs_dir/ablation_summary_long.csv
  - runs_dir/ablation_summary_pivot_router.csv
  - runs_dir/ablation_summary_delta_vs_baseline.csv
"""

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path
import pandas as pd

DEFAULT_CONDITIONS = [
    "baseline",
    # "mfcc",
    # "prosodic_vad",
    # "neural_embeddings",
    # "signal_temporal",


    "mfcc_and_prosodic_vad",
    "mfcc_and_neural_embeddings",
    "mfcc_and_signal_temporal",
    "prosodic_vad_and_neural_embeddings",
    "prosodic_vad_and_signal_temporal",
    "neural_embeddings_and_signal_temporal",

    "mfcc_and_prosodic_vad_and_neural_embeddings",
    "mfcc_and_prosodic_vad_and_signal_temporal",
    "mfcc_and_neural_embeddings_and_signal_temporal",
    "prosodic_vad_and_neural_embeddings_and_signal_temporal",
]

def run(cmd, cwd=None):
    print("\n$ " + " ".join(map(str, cmd)))
    subprocess.check_call(cmd, cwd=cwd)

def resolve_path(base: Path, p: str) -> Path:
    """Return absolute path for p. If p is relative, resolve against base."""
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp).resolve()

def train_for_condition(cond: str, device: str, py: str, work_dir: Path, cv_folds: int,
                        scripts_dir: Path, table_path: Path, labels_path: Path):
    """
    Train XGB+MLP CV models for a given ablation condition into work_dir.
    We keep cwd=work_dir so artifacts land there, but call scripts by absolute path.
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    xgb_script = scripts_dir / "train_xgb_cls.py"
    mlp_script = scripts_dir / "train_mlp_cls.py"
    if not xgb_script.exists():
        raise FileNotFoundError(f"Missing script: {xgb_script}")
    if not mlp_script.exists():
        raise FileNotFoundError(f"Missing script: {mlp_script}")

    drop_args = []
    if cond != "baseline":
        drop_args = ["--drop_groups", cond]

    # XGB (CV)
    run([
        py, str(xgb_script),
        "--device", device,
        "--cv_folds", str(cv_folds),
        "--table", str(table_path),
        "--labels", str(labels_path),
        *drop_args,
        "--model_out", "adas2t_xgb_cls_fold{fold}.json",
        "--labels_out", "label_order_xgb.npy",
        "--meta_out", "adas2t_xgb_meta_cv.json",
        "--imp_out", "adas2t_xgb_feat_importance_gain_cv.csv",
        "--oof_out", "adas2t_xgb_oof.csv",
        "--metrics_out", "adas2t_xgb_cv_metrics.json",
    ], cwd=work_dir)

    # MLP (CV)
    run([
        py, str(mlp_script),
        "--device", device,
        "--cv_folds", str(cv_folds),
        "--table", str(table_path),
        "--labels", str(labels_path),
        *drop_args,
        "--out", "adas2t_mlp_cls_fold{fold}.pth",
        "--meta_out", "adas2t_mlp_meta_cv.json",
        "--oof_out", "adas2t_mlp_oof.csv",
        "--metrics_out", "adas2t_mlp_cv_metrics.json",
    ], cwd=work_dir)

def evaluate_condition(cond: str, max_clips: int, runs_dir: Path, py: str, work_dir: Path, scripts_dir: Path) -> Path:
    """
    Evaluate current artifacts (in work_dir); write JSON into runs_dir/ablate_<cond>.json
    """
    eval_script = scripts_dir / "eval_mls_eng.py"
    if not eval_script.exists():
        raise FileNotFoundError(f"Missing script: {eval_script}")

    out_json = runs_dir / f"ablate_{cond}.json"
    run([
        py, str(eval_script),
        "--max_clips", str(max_clips),
        "--out_json", str(out_json),
        "--no_oracle", "--no_lgbm"
    ], cwd=work_dir)
    return out_json

def load_results(json_path: Path) -> dict:
    with open(json_path, "r") as f:
        return json.load(f)

def collect_long_table(results_by_cond: dict) -> pd.DataFrame:
    """
    Build a tidy long table: [dataset, condition, model, WER]
    Filters out Oracle if present.
    """
    rows = []
    for cond, payload in results_by_cond.items():
        results = payload.get("results", {})
        for dataset, model2wer in results.items():
            for model_name, wer in model2wer.items():
                if model_name == "ADAS2T-Oracle":
                    continue
                rows.append({
                    "dataset": dataset,
                    "condition": cond,
                    "model": model_name,
                    "WER": float(wer),
                })
    return pd.DataFrame(rows)

def make_router_pivots(df_long: pd.DataFrame):
    """
    Create a pivot table just for routers (ADAS2T-*), and a Δ vs baseline.
    """
    routers = df_long[df_long["model"].str.startswith("ADAS2T-")].copy()
    pivot = routers.pivot_table(index="dataset", columns="condition", values="WER", aggfunc="mean")

    # ΔWER vs baseline (negative = better than baseline)
    if "baseline" in pivot.columns:
        deltas = pivot.subtract(pivot["baseline"], axis=0)
    else:
        deltas = pd.DataFrame(index=pivot.index)  # empty if no baseline present
    return pivot, deltas

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="ablation_runs_v2",
                    help="Directory to store per-condition JSONs and CSV summaries")
    ap.add_argument("--max_clips", type=int, default=100,
                    help="Max clips per dataset during eval")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda",
                    help="Device for training (and extractor slice building inside training)")
    ap.add_argument("--cv_folds", type=int, default=5,
                    help="Number of CV folds for both XGB and MLP training")
    ap.add_argument("--conditions", type=str, default=",".join(DEFAULT_CONDITIONS),
                    help="Comma-separated list of ablation conditions")
    ap.add_argument("--skip_train", action="store_true",
                    help="Skip training, only run eval in each condition work dir")
    ap.add_argument("--skip_eval", action="store_true",
                    help="Skip eval, only (re)aggregate CSVs from existing JSONs")

    # NEW: allow explicit data file locations; default to filenames next to this driver
    ap.add_argument("--table", default="training_table_25k_dropped_voxpopuli.csv",
                    help="Path to training features CSV")
    ap.add_argument("--labels", default="clip_labels_25k_dropped_voxpopuli.csv",
                    help="Path to labels CSV")
    return ap.parse_args()

def main():
    args = parse_args()
    py = sys.executable

    # Resolve directories and script/data paths
    scripts_dir = Path(__file__).resolve().parent
    runs_dir = (scripts_dir / args.runs_dir).resolve() if not Path(args.runs_dir).is_absolute() else Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    table_path = resolve_path(scripts_dir, args.table)
    labels_path = resolve_path(scripts_dir, args.labels)

    if not table_path.exists():
        raise FileNotFoundError(f"--table not found: {table_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"--labels not found: {labels_path}")

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    print("Ablation conditions:", conditions)
    print(f"Scripts dir:  {scripts_dir}")
    print(f"Runs dir:     {runs_dir}")
    print(f"Table path:   {table_path}")
    print(f"Labels path:  {labels_path}")

    results_by_cond = {}

    for cond in conditions:
        print("\n" + "=" * 80)
        print(f"[{cond}] Starting condition")

        # Each condition has its own work dir to avoid fold-file collisions
        work_dir = runs_dir / cond
        work_dir.mkdir(parents=True, exist_ok=True)

        if not args.skip_train:
            train_for_condition(cond, args.device, py, work_dir, args.cv_folds,
                                scripts_dir, table_path, labels_path)

        if not args.skip_eval:
            out_json = evaluate_condition(cond, args.max_clips, runs_dir, py, work_dir, scripts_dir)
        else:
            out_json = runs_dir / f"ablate_{cond}.json"
            if not out_json.exists():
                print(f"ERROR: {out_json} not found and --skip_eval used. Exiting.")
                sys.exit(2)

        payload = load_results(out_json)
        results_by_cond[cond] = payload

    # Aggregate CSVs
    print("\nAggregating results → CSVs")
    df_long = collect_long_table(results_by_cond)
    long_csv = runs_dir / "ablation_summary_long.csv"
    df_long.to_csv(long_csv, index=False)

    pivot_router, delta_vs_base = make_router_pivots(df_long)
    pivot_csv = runs_dir / "ablation_summary_pivot_router.csv"
    pivot_router.to_csv(pivot_csv)

    delta_csv = runs_dir / "ablation_summary_delta_vs_baseline.csv"
    delta_vs_base.to_csv(delta_csv)

    print("\nDone.")
    print(f"- Long table:            {long_csv}")
    print(f"- Router pivot:          {pivot_csv}")
    print(f"- ΔWER vs baseline:      {delta_csv}")

if __name__ == "__main__":
    main()
