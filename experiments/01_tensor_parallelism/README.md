# Experiment 01 — Tensor Parallelism (TP=1 vs TP=2)

**Concept:** A tensor-parallel group shards each weight matrix across GPUs; every GPU does part of
every layer and they `all-reduce` to combine. It trades NVLink/PCIe communication for more compute
and more aggregate memory bandwidth.

**Hypothesis:** Going `TP=1 → TP=2` on the same 7B model increases output throughput and lets the KV
cache hold more concurrent sequences, at the cost of some per-token latency from cross-GPU comms.

### Run
GPU plane (Kaggle, 2×T4):
```bash
# baseline — pin to one GPU
CUDA_VISIBLE_DEVICES=0 ./serving/launch_vllm.sh serving/configs/tp1.env
# (new session) tensor-parallel across both
./serving/launch_vllm.sh serving/configs/tp2.env
```
Control plane (Mac), same load against each:
```bash
python bench/benchmark.py --base-url $URL --model Qwen/Qwen2.5-7B-Instruct \
  --concurrency 32 --num-prompts 200 --label tp1 --output results/tp1_c32.csv
python bench/benchmark.py --base-url $URL --model Qwen/Qwen2.5-7B-Instruct \
  --concurrency 32 --num-prompts 200 --label tp2 --output results/tp2_c32.csv
python bench/analyze.py --x label
```

### Measure
- `throughput_tok_s`, `ttft_p99_s`, `tpot_p50_s` from the summary.
- In Grafana: `vllm:num_requests_running` (bigger sustainable batch) and `vllm:gpu_cache_usage_perc`.

### Expected / write up here
Throughput up with TP=2; note whether TPOT rises slightly (comms overhead) and how much more KV-cache
headroom you get. Paste `results/charts/throughput.png` and explain the numbers in your own words.
