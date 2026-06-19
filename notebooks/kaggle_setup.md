# Launching the GPU plane on Kaggle (free 2×T4)

Kaggle gives 2× NVIDIA T4 (16 GB each) for free, ~30 hrs/week. That's enough for real `TP=2` and
expert parallelism. Create a new **Notebook** (not a Script) and follow these cells.

## 0. Notebook settings
- Right sidebar → **Accelerator** → `GPU T4 x2`
- **Internet** → `On` (needed to download weights + run cloudflared)

## 1. Sanity check the GPUs
```python
!nvidia-smi --query-gpu=index,name,memory.total --format=csv
# expect two rows: Tesla T4, 15360 MiB each
```

## 2. Install vLLM
```python
!pip -q install "vllm>=0.6.0"
```

## 3. Pull this repo (so the launch script + configs are available)
```python
!git clone https://github.com/<YOUR_USERNAME>/vllm-perf-lab.git
%cd vllm-perf-lab
!chmod +x serving/launch_vllm.sh
```

## 4. Start the vLLM server in the background
```python
import subprocess, time
# T4 has NO bf16 — the config already forces --dtype float16
server = subprocess.Popen(["bash", "serving/launch_vllm.sh", "serving/configs/tp2.env"])
time.sleep(90)   # give it time to load weights + build CUDA graphs
!curl -s http://localhost:8000/v1/models | head -c 300
```

## 5. Expose it with a cloudflared tunnel
```python
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared
!chmod +x /usr/local/bin/cloudflared
# prints a public https URL like https://xxxx.trycloudflare.com
get_ipython().system_raw("cloudflared tunnel --url http://localhost:8000 > tunnel.log 2>&1 &")
import time; time.sleep(8)
!grep -o 'https://[a-z0-9-]*\\.trycloudflare\\.com' tunnel.log | head -1
```

## 6. Drive it from your Mac
Copy that URL. On your laptop:
```bash
export URL=https://xxxx.trycloudflare.com
python bench/benchmark.py --base-url $URL --model Qwen/Qwen2.5-7B-Instruct \
  --concurrency 32 --num-prompts 200 --label tp2 --output results/tp2_c32.csv
```
And point `observability/prometheus.yml` at `xxxx.trycloudflare.com` to graph it live in Grafana.

---
### Tips
- The trycloudflare URL is **new every session** — update it in `prometheus.yml` each time.
- Loading a 7B model takes a minute or two; watch `tunnel.log` / server logs if curl fails.
- For the MoE / paged-attention experiments, swap the config in step 4 (`moe.env`, `paged_attn.env`).
- Kaggle idle-disconnects — keep the tab active while benchmarking.
