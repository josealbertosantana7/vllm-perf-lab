# Experiment 04 — Hybrid Parallelism (TP + Expert Parallel)

**Concept:** Real MoE deployments combine strategies — tensor-parallel the dense/attention layers,
expert-parallel the MoE layers. On 2 GPUs the combination is modest, but the *mechanics* and the
measurement method are exactly what scale to 8/16-GPU clusters.

**Hypothesis:** On the MoE model, TP+EP together uses both GPUs more fully than either alone, raising
sustainable batch size and throughput versus single-GPU.

### Run
```bash
# moe.env already sets TP=2 and --enable-expert-parallel (that's the hybrid)
./serving/launch_vllm.sh serving/configs/moe.env
```
Compare three points by relaunching with different settings and re-benchmarking:
1. single GPU, no EP (`CUDA_VISIBLE_DEVICES=0`, TP=1, drop `--enable-expert-parallel`)
2. TP=2 only
3. TP=2 + expert parallel (hybrid)
```bash
python bench/benchmark.py --base-url $URL --model allenai/OLMoE-1B-7B-0924-Instruct \
  --concurrency 32 --num-prompts 200 --label hybrid --output results/hybrid_c32.csv
python bench/analyze.py --x label
```

### Measure
- Throughput and scaling efficiency across the three configurations.
- Per-GPU utilization (`nvidia-smi dmon` on the box) to show both GPUs are busy.

### Expected / write up here
Hybrid should be the best of the three on the MoE. Discuss how this generalizes (TP within a node,
EP/pipeline across nodes) — the point Fregly makes about composing parallelism strategies.
