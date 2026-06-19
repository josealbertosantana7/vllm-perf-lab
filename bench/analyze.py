#!/usr/bin/env python3
"""
Turn results/summary.csv into charts for the README / experiment writeups.

Reads the one-row-per-run summary that benchmark.py appends, and plots
throughput and TTFT-p99 across runs (e.g. tp1 vs tp2, or a concurrency sweep).

  python bench/analyze.py --summary results/summary.csv --out results/charts
"""
import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="results/summary.csv")
    p.add_argument("--out", default="results/charts")
    p.add_argument("--x", default="label", help="column to use on the x axis (e.g. 'label' or 'concurrency')")
    args = p.parse_args()

    df = pd.read_csv(args.summary)
    os.makedirs(args.out, exist_ok=True)

    # Throughput
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df[args.x].astype(str), df["throughput_tok_s"], color="#4C9F70")
    ax.set_ylabel("output throughput (tok/s)")
    ax.set_xlabel(args.x)
    ax.set_title("Throughput by run")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(f"{args.out}/throughput.png", dpi=120)

    # TTFT p99
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df[args.x].astype(str), df["ttft_p99_s"], color="#C46A5B")
    ax.set_ylabel("TTFT p99 (s)")
    ax.set_xlabel(args.x)
    ax.set_title("Tail time-to-first-token by run")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(f"{args.out}/ttft_p99.png", dpi=120)

    print(f"wrote charts to {args.out}/")


if __name__ == "__main__":
    main()
