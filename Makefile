# Convenience targets. `make help` lists them.
URL ?= https://CHANGE-ME.trycloudflare.com
MODEL ?= Qwen/Qwen2.5-7B-Instruct
C ?= 32
N ?= 200

.PHONY: help bench analyze obs-up obs-down
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n",$$1,$$2}'

bench: ## Run a benchmark: make bench URL=... LABEL=tp2
	python bench/benchmark.py --base-url $(URL) --model $(MODEL) \
	  --concurrency $(C) --num-prompts $(N) --label $(LABEL) --output results/$(LABEL)_c$(C).csv

analyze: ## Build charts from results/summary.csv
	python bench/analyze.py --x label

obs-up: ## Start Grafana + Prometheus via docker compose
	docker compose -f observability/docker-compose.yml up -d

obs-down: ## Stop the observability stack
	docker compose -f observability/docker-compose.yml down
