#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, json, os, re
import numpy as np, pandas as pd
import matplotlib
import matplotlib.pyplot as plt

MODEL_ORDER = [
    "openai/whisper-large-v3",
    "nvidia/parakeet-tdt-0.6b-v2",
    "ADAS2T-XGB",
    "ADAS2T-MLP",
]
EXCLUDE_KEYS = {"ADAS2T-Oracle", "ADAS2T-LGBM"}

# Google-ish palette (high contrast, print-friendly)
COLOR = {
    "openai/whisper-large-v3": "#1a73e8",   # blue
    "nvidia/parakeet-tdt-0.6b-v2": "#34a853",  # green
    "ADAS2T-XGB": "#ea4335",                # red
    "ADAS2T-MLP": "#fbbc04",                # yellow/orange
}

def set_ieee_rc():
    # Try Times New Roman; fallback to Times/DejaVu Serif
    cand = ["Times New Roman", "Times", "DejaVu Serif"]
    for f in cand:
        try:
            matplotlib.font_manager.findfont(f, fallback_to_default=False)
            plt.rcParams["font.family"] = f
            break
        except Exception:
            continue
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9

def clean_name(name):
    # Drop "(N=...)" from dataset display
    return re.sub(r"\s*\(N=\d+\)\s*$", "", name).strip()

def to_percent(x):
    return 100.0 * float(x)

def make_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def main(args):
    set_ieee_rc()
    make_dirs(args.out_tables, args.out_figs)

    with open(args.infile) as f:
        blob = json.load(f)
    results = blob["results"]
    timings = blob["timings"]

    # Main WER table (percentage), filtered models
    ds_rows = []
    for ds, rec in results.items():
        row = {"Dataset": clean_name(ds)}
        for m in MODEL_ORDER:
            if m in rec:
                row[m] = to_percent(rec[m])
        ds_rows.append(row)

    main_df = pd.DataFrame(ds_rows).set_index("Dataset")[MODEL_ORDER]
    # Save CSV + LaTeX
    main_df_rounded = main_df.applymap(lambda v: f"{v:.2f}")
    main_df.to_csv(os.path.join(args.out_tables, "table_main_wer.csv"))
    with open(os.path.join(args.out_tables, "table_main_wer.tex"), "w") as f:
        f.write(main_df_rounded.to_latex(escape=True, column_format="lrrrr"))

    # Plot per-dataset bar charts
    for ds, rec in results.items():
        ds_clean = clean_name(ds)
        # Filter keys
        pairs = [(m, rec[m]) for m in MODEL_ORDER if m in rec and m not in EXCLUDE_KEYS]
        if not pairs: 
            continue

        labels = [lbl.replace("openai/", "").replace("nvidia/", "").replace("-tdt-0.6b-v2","").replace("-large-v3","").replace("ADAS2T-","ADAS2T-") for lbl,_ in pairs]
        vals = [to_percent(v) for _, v in pairs]
        cols = [COLOR[m] for m,_ in pairs]

        fig, ax = plt.subplots(figsize=(4.0, 2.6), dpi=300)
        x = np.arange(len(vals))
        barw = 0.92  # tight bars (little empty space)
        bars = ax.bar(x, vals, width=barw, edgecolor="black", linewidth=0.4, color=cols)

        ax.set_ylabel("WER (%)")
        ax.set_title(ds_clean)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_ylim(0, max(vals)*1.20 if vals else 1)
        ax.legend([b for b in bars], [p[0].split("/")[-1] for p in pairs], loc="upper right", frameon=False)
        ax.margins(x=0.01)
        ax.grid(axis="y", alpha=0.15, linewidth=0.5)
        plt.tight_layout(pad=0.2)
        outp = os.path.join(args.out_figs, f"wer_{re.sub(r'[^A-Za-z0-9]+','_',ds_clean)}.pdf")
        plt.savefig(outp, bbox_inches="tight")
        plt.close(fig)

    # Latency plots (base vs meta total)
    for ds, tm in timings.items():
        ds_clean = clean_name(ds)
        base = tm["base_models_avg_sec"]
        meta = tm["meta_avg_sec"]

        labels, vals, cols = [], [], []
        for m in ["openai/whisper-large-v3", "nvidia/parakeet-tdt-0.6b-v2"]:
            if m in base:
                labels.append(m.split("/")[-1].replace("-large-v3","").replace("-tdt-0.6b-v2",""))
                vals.append(base[m])
                cols.append(COLOR[m])

        for k in ["ADAS2T-XGB", "ADAS2T-MLP"]:
            if k in meta:
                labels.append(k)
                vals.append(meta[k]["total_sec"])
                cols.append(COLOR[k])

        fig, ax = plt.subplots(figsize=(4.0, 2.6), dpi=300)
        x = np.arange(len(vals))
        bars = ax.bar(x, vals, width=0.92, edgecolor="black", linewidth=0.4, color=cols)

        ax.set_ylabel("Avg latency (s)")
        ax.set_title(ds_clean)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_ylim(0, max(vals)*1.20 if vals else 1)
        ax.legend([b for b in bars], labels, loc="upper right", frameon=False)
        ax.margins(x=0.01)
        ax.grid(axis="y", alpha=0.15, linewidth=0.5)
        plt.tight_layout(pad=0.2)
        outp = os.path.join(args.out_figs, f"latency_{re.sub(r'[^A-Za-z0-9]+','_',ds_clean)}.pdf")
        plt.savefig(outp, bbox_inches="tight")
        plt.close(fig)

    print(f"✓ Wrote {os.path.join(args.out_tables, 'table_main_wer.csv')} and LaTeX.")
    print(f"✓ Plots saved under: {args.out_figs}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default="adas2t_eval_results.json")
    ap.add_argument("--out_tables", default="paper_tables")
    ap.add_argument("--out_figs", default="paper_figs")
    args = ap.parse_args()
    main(args)
