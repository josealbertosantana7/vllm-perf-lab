# Experiment 03 — Paged Attention & the KV Cache

**Concept:** Paged attention stores the KV cache in fixed-size blocks (like OS virtual-memory pages)
instead of one contiguous buffer per sequence. That kills fragmentation and lets vLLM pack many more
concurrent sequences — and reuse blocks via prefix caching.

**Hypothesis (a):** As concurrency rises, `gpu_cache_usage_perc` climbs; past saturation vLLM queues
and preempts, and TTFT-p99 jumps sharply (a visible "cliff").
**Hypothesis (b):** With shared prompt prefixes, `--enable-prefix-caching` lowers TTFT because the
prefill KV blocks are reused.

### Run — saturation sweep
Launch with a *small* cache so it saturates early:
```bash
# in paged_attn.env, set GPU_MEM_UTIL=0.55, then:
./serving/launch_vllm.sh serving/configs/paged_attn.env
```
```bash
for c in 1 4 8 16 32 64 128; do
  python bench/benchmark.py --base-url $URL --model Qwen/Qwen2.5-7B-Instruct \
    --concurrency $c --num-prompts $((c*4)) --label "c$c" --output results/sweep_c$c.csv
done
python bench/analyze.py --x concurrency
```

### Run — prefix caching on/off
Same load, once with and once without `--enable-prefix-caching`, using prompts that share a long
prefix. Compare `ttft_p50_s`.

### Measure
- Grafana `vllm:gpu_cache_usage_perc` rising to ~1.0; `vllm:num_requests_waiting` growing at the cliff.
- TTFT-p99 vs concurrency (find the knee); TTFT with/without prefix caching.

### Expected / write up here
A throughput plateau + TTFT cliff at the saturation point, and lower TTFT with prefix caching.
This is the most "performance engineering" of the experiments — annotate the cliff on the chart.
