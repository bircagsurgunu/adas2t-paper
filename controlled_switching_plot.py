#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot router probabilities & selections from switching_strict_log.csv.

• Shows P(Parakeet) for both routers
• Shades Parakeet-good vs Whisper-good blocks
• Uses IEEE-friendly fonts and saves a vector PDF

Input:
  • switching_strict_log.csv

Output:
  • figure_switching_strict.pdf
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib, matplotlib.pyplot as plt

def set_ieee():
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
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9

def raster(ax, picks, y0):
    for i, m in enumerate(picks):
        mm = (m or "").lower()
        c = "#a5d6a7" if "parakeet" in mm else "#90caf9"
        ax.vlines(i, y0, y0+1, colors=c, linewidth=2.0)

def plot(csv_path, out_pdf):
    set_ieee()
    df = pd.read_csv(csv_path)
    x = np.arange(len(df))

    fig = plt.figure(figsize=(6.4, 3.6), dpi=300)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 0.7, 0.2], hspace=0.25)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    # Probabilities — show P(Parakeet)
    ax1.plot(x, df["p_parakeet_xgb"].values, linewidth=1.2,
            label="ADAS2T-XGB: P(Parakeet)", color="#1a73e8")
    ax1.plot(x, df["p_parakeet_mlp"].values, linewidth=1.2,
            label="ADAS2T-MLP: P(Parakeet)", color="#ea4335")
    ax1.set_ylabel("P(Parakeet)")
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(axis="y", alpha=0.2, linewidth=0.5)

    # Legend inside, lower left, smaller font
    # Legend inside, lower left, small font, white background
    ax1.legend(
        loc="lower left",
        frameon=True,
        bbox_to_anchor=(0.0, 0.05),
        facecolor="white",
        framealpha=0.8,
        ncol=1,
        handlelength=2,
        fontsize=5,
        borderpad=0.3
    )




    # Shading for blocks
    start = 0
    while start < len(df):
        tag_is_w = df.loc[start, "segment"].startswith("Whisper")
        end = start
        while end+1 < len(df) and df.loc[end+1, "segment"] == df.loc[start, "segment"]:
            end += 1
        ax1.axvspan(start-0.5, end+0.5,
                    color=("#e8f5e9" if not tag_is_w else "#e3f2fd"),
                    alpha=0.6, linewidth=0)
        start = end + 1

    # Raster of picks – color by Parakeet vs Whisper
    ax2.set_ylim(0, 2)
    ax2.set_yticks([0.5, 1.5]); ax2.set_yticklabels(["XGB pick", "MLP pick"])
    ax2.set_xlim(-0.5, len(df)-0.5)
    ax2.set_xlabel("Clip index")
    raster(ax2, df["pick_xgb"].values, 0.0)
    raster(ax2, df["pick_mlp"].values, 1.0)

    plt.tight_layout(pad=0.3)
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"✓ wrote {out_pdf}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="switching_log.csv")
    ap.add_argument("--out_pdf", type=str, default="figure_switching_plotted.pdf")
    args = ap.parse_args()
    plot(args.csv, args.out_pdf)
