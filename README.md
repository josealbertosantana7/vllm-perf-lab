# vllm-perf-lab

> A hands-on **LLM inference performance lab**: serve models with [vLLM](https://github.com/vllm-project/vllm)
> on cloud GPUs, drive load from a laptop, and measure the behavior of tensor parallelism,
> MoE expert parallelism, paged attention, and the KV cache with **Prometheus + Grafana**.
>
> Built to apply the concepts from Chris Fregly's *AI Systems Performance Engineering* (O'Reilly)
> without owning a data center.

---

## Why this exists (the honest constraint)

I'm a student on a **MacBook Air (Apple M2, 8 GB RAM)**. Three of the four technologies I wanted to
practice have a hard dependency I can't satisfy locally:

- **CUDA is NVIDIA-only** — it cannot run on Apple Silicon.
- **vLLM's real engine targets CUDA GPUs.**
- **Tensor / MoE / hybrid parallelism are multi-GPU techniques** — you can't split a model across
  devices you don't have.

So this project is split into **two planes**, which is also how real inference platforms are built:

```
            ┌──────────────────────────────┐         ┌────────────────────────────────────┐
            │   CONTROL / OBSERVABILITY      │  HTTP   │        GPU / SERVING PLANE          │
            │   (my MacBook, runs anywhere)  │ ──────► │   (Kaggle free 2×T4, or rented)    │
            │                                │ metrics │                                    │
            │  • benchmark/load generator    │ ◄────── │  • vLLM server (CUDA)              │
            │  • Prometheus  (scrapes)       │ tunnel  │  • /metrics Prometheus endpoint    │
            │  • Grafana dashboards          │         │  • Triton custom kernel            │
            │  • analysis + writeups         │         │  • TP=2 / expert-parallel runs     │
            └──────────────────────────────┘         └────────────────────────────────────┘
```

The GPU plane runs on **Kaggle Notebooks**, which give **2× NVIDIA T4 (32 GB total), free, ~30 hrs/week**.
Two GPUs is the unlock: it's enough to genuinely demonstrate **tensor parallelism (`TP=2`)** and
**expert parallelism**, at zero cost.

---

## What each concept maps to (the actual deliverable)

The point isn't "run vLLM once." It's **a set of reproducible experiments where every graph is
explained by a concept from the book.**

| Concept (Fregly)        | Experiment                                              | Metric that proves it                        | Folder |
|-------------------------|--------------------------------------------------------|----------------------------------------------|--------|
| **Tensor parallelism**  | Same 7–8B model at `TP=1` vs `TP=2` on 2×T4            | throughput (tok/s), TTFT, GPU mem per device | [`experiments/01_tensor_parallelism`](experiments/01_tensor_parallelism) |
| **MoE expert parallel** | Serve a small MoE with `--enable-expert-parallel`      | tok/s vs *active* params, per-expert routing | [`experiments/02_moe_expert_parallelism`](experiments/02_moe_expert_parallelism) |
| **Paged attn / KV cache** | Ramp concurrency until KV cache saturates; prefix cache on/off | `gpu_cache_usage_perc`, preemptions, TTFT cliff | [`experiments/03_paged_attention_kv_cache`](experiments/03_paged_attention_kv_cache) |
| **Hybrid parallelism**  | TP + expert parallel together on the MoE               | scaling efficiency vs single GPU             | [`experiments/04_hybrid_parallelism`](experiments/04_hybrid_parallelism) |
| **CUDA literacy**       | Fused softmax kernel in **Triton**, benchmarked vs PyTorch | kernel latency, speedup factor           | [`experiments/05_custom_triton_kernel`](experiments/05_custom_triton_kernel) |

---

## Repo layout

```
vllm-perf-lab/
├── serving/          # vLLM launch script + per-experiment config (.env) files
├── kernels/          # custom Triton kernel + benchmark vs PyTorch   ← the CUDA artifact
├── bench/            # async load generator (TTFT/TPOT/throughput) + analysis plots
├── observability/    # Prometheus config + Grafana dashboards (provisioned as code)
├── experiments/      # one folder per experiment: hypothesis → command → result → explanation
├── notebooks/        # Kaggle setup that launches the GPU plane on 2×T4
├── scripts/          # tunnel helper to expose the cloud /metrics endpoint
└── results/          # CSVs + charts you generate (committed, they ARE the portfolio)
```

---

## Quickstart

### 1. GPU plane — launch vLLM on Kaggle (free 2×T4)
Follow [`notebooks/kaggle_setup.md`](notebooks/kaggle_setup.md). In short:
1. New Kaggle notebook → Settings → Accelerator = **GPU T4 x2**, Internet = **On**.
2. Install vLLM, run `serving/launch_vllm.sh` with a config (e.g. `tp2.env`).
3. Start a `cloudflared` tunnel to expose the server's `/metrics` + API publicly.
4. Copy the public URL it prints.

### 2. Control plane — on your Mac
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-bench.txt          # client + analysis only; NOT vLLM

# Point the load generator at the tunnel URL from step 1:
python bench/benchmark.py \
  --base-url https://YOUR-TUNNEL.trycloudflare.com \
  --model Qwen/Qwen2.5-7B-Instruct \
  --concurrency 32 --num-prompts 200 \
  --output results/tp2_c32.csv
```

### 3. Observability — Grafana on your Mac
```bash
# lightest option for 8 GB RAM: native install (no Docker Desktop)
brew install grafana prometheus
# put the tunnel URL into observability/prometheus.yml, then:
brew services start prometheus
brew services start grafana
# open http://localhost:3000  (admin/admin) and import observability/grafana/dashboards/vllm.json
```
(Or `docker compose -f observability/docker-compose.yml up` if you prefer containers / are on a cloud box.)

---

## Hardware notes that will save you hours

- **T4 is Turing (CC 7.5): no bfloat16.** Always launch vLLM with `--dtype float16`, or it errors / falls back.
- **32 GB across 2 GPUs is tight.** An 8B model in fp16 is ~16 GB; tune `--gpu-memory-utilization`
  (start 0.90) and `--max-model-len` (e.g. 8192) so the KV cache has room.
- **Kaggle has no inbound ports** — the `cloudflared` tunnel (outbound) is how you reach it from your Mac.
  The free tunnel URL changes every run; update `observability/prometheus.yml` each session.

---

## Roadmap (build it in phases — don't try to do everything at once)

- [ ] **Phase 1** — single GPU: serve a 7B model, drive load, see TTFT/tok/s in Grafana.
- [ ] **Phase 2** — flip to `TP=2`, re-run, compare → `experiments/01`.
- [ ] **Phase 3** — serve a small MoE with expert parallel → `experiments/02`.
- [ ] **Phase 4** — saturate the KV cache, toggle prefix caching → `experiments/03`.
- [ ] **Phase 5** — combine TP + EP → `experiments/04`.
- [ ] **Phase 6** — write + benchmark the Triton kernel → `experiments/05`.

---

## Credits
Concepts and motivation from **Chris Fregly, *AI Systems Performance Engineering* (O'Reilly)**.
Tooling: [vLLM](https://github.com/vllm-project/vllm), [Triton](https://github.com/triton-lang/triton),
[Prometheus](https://prometheus.io/), [Grafana](https://grafana.com/).
