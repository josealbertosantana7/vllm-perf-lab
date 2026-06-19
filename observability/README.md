# observability/ — Prometheus + Grafana

vLLM exposes Prometheus metrics at `/metrics` out of the box. These are the ones the dashboard uses:

| Metric | What it tells you | Concept |
|---|---|---|
| `vllm:num_requests_running` / `_waiting` | live batch size vs queue depth | batching / scheduling |
| `vllm:gpu_cache_usage_perc` | fraction of KV-cache blocks in use | **paged attention / KV cache** |
| `vllm:generation_tokens_total` | decode tokens (rate → tok/s) | **throughput** |
| `vllm:time_to_first_token_seconds` | TTFT histogram | prefill + queueing |
| `vllm:time_per_output_token_seconds` | TPOT histogram | decode efficiency |

### Run it (two options)

**A) Native on the Mac (lightest for 8 GB RAM)**
```bash
brew install grafana prometheus
# edit prometheus.yml -> put the cloudflared host in `targets`
prometheus --config.file=observability/prometheus.yml      # or: brew services start prometheus
brew services start grafana
# open http://localhost:3000  (admin/admin)
#  - add a Prometheus datasource pointing at http://localhost:9090
#  - import observability/grafana/dashboards/vllm.json
```

**B) Containers (portable; also what you'd run on a cloud box)**
```bash
cd observability && docker compose up
# Grafana http://localhost:3000 — datasource + dashboard are auto-provisioned
```

### The tunnel
Kaggle has no inbound ports, so you expose vLLM with an outbound `cloudflared` tunnel
(see [`../scripts/tunnel.sh`](../scripts/tunnel.sh)). Put the printed host (no `https://`) into
`prometheus.yml` under `targets:`. The free URL changes each session — update it each time, or
register a named tunnel for a stable hostname.
