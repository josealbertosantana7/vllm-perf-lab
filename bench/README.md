# bench/ — load generation & analysis

Runs on your **Mac**. Speaks only HTTP to the vLLM server on the GPU plane.

- **`benchmark.py`** — fires `--num-prompts` requests at `--concurrency`, streaming each so it can
  measure **TTFT** (time to first token) and **TPOT** (time per output token), plus aggregate
  throughput. Writes a per-request CSV and appends a one-line summary to `results/summary.csv`.
- **`analyze.py`** — turns `results/summary.csv` into throughput / TTFT charts under `results/charts/`.

### Typical experiment loop
```bash
# baseline (single GPU), then tensor-parallel, labelled so they line up in the chart
python bench/benchmark.py --base-url $URL --model $MODEL --concurrency 32 --label tp1 --output results/tp1_c32.csv
python bench/benchmark.py --base-url $URL --model $MODEL --concurrency 32 --label tp2 --output results/tp2_c32.csv
python bench/analyze.py --summary results/summary.csv --x label
```

For a **concurrency sweep** (to find the KV-cache saturation point), loop:
```bash
for c in 1 4 8 16 32 64 128; do
  python bench/benchmark.py --base-url $URL --model $MODEL --concurrency $c \
    --num-prompts $((c*4)) --label "c$c" --output results/sweep_c$c.csv
done
python bench/analyze.py --x concurrency
```

> Token counting here is approximate (it counts streamed text chunks). For exact token-level
> accounting, vLLM also reports `usage` in the final stream chunk — `--stream-options include_usage`
> is already requested, so you can extend `benchmark.py` to read it.
