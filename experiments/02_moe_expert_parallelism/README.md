# Experiment 02 — MoE Expert Parallelism

**Concept:** In a Mixture-of-Experts model, each token is routed to only a few experts, so *active*
params ≪ *total* params. Expert parallelism (`--enable-expert-parallel`) places different experts on
different GPUs instead of sharding every matrix the way TP does.

**Hypothesis:** A 7B-total / ~1B-active MoE serves with throughput closer to a 1B dense model than a
7B dense model, because only a fraction of the weights fire per token.

### Run
```bash
./serving/launch_vllm.sh serving/configs/moe.env      # OLMoE-1B-7B + --enable-expert-parallel
```
```bash
python bench/benchmark.py --base-url $URL --model allenai/OLMoE-1B-7B-0924-Instruct \
  --concurrency 32 --num-prompts 200 --label moe_ep --output results/moe_ep_c32.csv
```

### Measure
- Throughput vs the 7B *dense* result from experiment 01 (active-vs-total params is the story).
- GPU memory per device — how the experts spread across the two T4s.

### Expected / write up here
MoE throughput should land well above dense-7B for similar quality. Discuss the active-vs-total
parameter trade-off and when EP beats plain TP for MoE.
