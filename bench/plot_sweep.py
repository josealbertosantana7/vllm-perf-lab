#!/usr/bin/env python3
"""
Plot throughput / tail-latency vs concurrency from a summary.csv, one line per
run label (e.g. tp1 vs tp2). Reusable across experiments.

  python bench/plot_sweep.py --summary results/exp03_batching_7b/summary.csv \
    --out results/exp03_batching_7b/charts --title "Qwen2.5-7B, TP=2, 2xT4"
"""
import argparse
import os
import matplotlib
matplotlib.use("Agg")            # headless: no display needed
import pandas as pd
import matplotlib.pyplot as plt


def _line(ax, df, ycol, ylabel):
    for label, g in df.groupby("label"):
        g = g.sort_values("concurrency")
        ax.plot(g["concurrency"], g[ycol], marker="o", label=str(label))
    ax.set_xlabel("concurrency (in-flight requests)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="results/summary.csv")
    p.add_argument("--out", default="results/charts")
    p.add_argument("--title", default="")
    a = p.parse_args()

    df = pd.read_csv(a.summary)
    os.makedirs(a.out, exist_ok=True)
    prefix = f"{a.title} — " if a.title else ""

    fig, ax = plt.subplots(figsize=(7, 4))
    _line(ax, df, "throughput_tok_s", "output throughput (tok/s)")
    ax.set_title(prefix + "throughput vs concurrency")
    fig.tight_layout()
    fig.savefig(f"{a.out}/throughput_vs_concurrency.png", dpi=120)

    fig, ax = plt.subplots(figsize=(7, 4))
    _line(ax, df, "ttft_p99_s", "TTFT p99 (s)")
    ax.set_title(prefix + "tail latency vs concurrency")
    fig.tight_layout()
    fig.savefig(f"{a.out}/ttft_p99_vs_concurrency.png", dpi=120)

    print(f"wrote {a.out}/throughput_vs_concurrency.png and ttft_p99_vs_concurrency.png")


if __name__ == "__main__":
    main()
