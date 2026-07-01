#!/usr/bin/env python3
"""
Async load generator for a vLLM OpenAI-compatible server.

Runs N prompts at a fixed concurrency, streaming each completion so we can
measure the two latencies that matter for serving:

  TTFT  (time to first token)   -> dominated by prefill + queueing
  TPOT  (time per output token) -> dominated by decode throughput / batching

It also reports aggregate output throughput (tokens/sec) for the whole run.
Runs on a laptop; it only speaks HTTP to the GPU plane.

Example:
  python bench/benchmark.py \
    --base-url https://YOUR-TUNNEL.trycloudflare.com \
    --model Qwen/Qwen2.5-7B-Instruct \
    --concurrency 32 --num-prompts 200 --max-tokens 128 \
    --output results/tp2_c32.csv
"""
import argparse
import asyncio
import csv
import time
from dataclasses import dataclass, asdict

import aiohttp

PROMPTS = [
    "Explain tensor parallelism to a new engineer in three sentences.",
    "What is a KV cache and why does it grow with sequence length?",
    "Summarize how paged attention reduces memory fragmentation.",
    "Describe Mixture-of-Experts routing and why only some experts fire.",
    "Why does time-to-first-token differ from time-per-output-token?",
]


@dataclass
class RequestResult:
    ok: bool
    ttft_s: float        # time to first streamed token
    total_s: float       # full request wall time
    output_tokens: int
    tpot_s: float        # (total - ttft) / (output_tokens - 1)


async def one_request(session, base_url, model, prompt, max_tokens) -> RequestResult:
    url = f"{base_url.rstrip('/')}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    ttft = None
    out_tokens = 0
    try:
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            async for raw in resp.content:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                if ttft is None:
                    ttft = time.perf_counter() - start
                # token accounting: count streamed chunks that carry text
                if '"text":' in data and '""' not in data.split('"text":')[1][:4]:
                    out_tokens += 1
    except Exception:
        return RequestResult(False, 0.0, 0.0, 0, 0.0)

    total = time.perf_counter() - start
    ttft = ttft if ttft is not None else total
    tpot = (total - ttft) / max(out_tokens - 1, 1)
    return RequestResult(True, ttft, total, out_tokens, tpot)


async def run(args):
    sem = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    results: list[RequestResult] = []

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def guarded(i):
            async with sem:
                prompt = PROMPTS[i % len(PROMPTS)]
                results.append(
                    await one_request(session, args.base_url, args.model, prompt, args.max_tokens)
                )

        wall_start = time.perf_counter()
        await asyncio.gather(*(guarded(i) for i in range(args.num_prompts)))
        wall = time.perf_counter() - wall_start

    ok = [r for r in results if r.ok]
    if not ok:
        print("All requests failed — is the tunnel URL correct and the server up?")
        return

    def pct(values, p):
        s = sorted(values)
        return s[min(int(len(s) * p), len(s) - 1)]

    ttfts = [r.ttft_s for r in ok]
    tpots = [r.tpot_s for r in ok]
    total_out = sum(r.output_tokens for r in ok)

    summary = {
        "concurrency": args.concurrency,
        "num_prompts": args.num_prompts,
        "ok": len(ok),
        "failed": len(results) - len(ok),
        "wall_s": round(wall, 2),
        "throughput_tok_s": round(total_out / wall, 1),
        "ttft_p50_s": round(pct(ttfts, 0.50), 3),
        "ttft_p99_s": round(pct(ttfts, 0.99), 3),
        "tpot_p50_s": round(pct(tpots, 0.50), 4),
    }

    print("\n=== run summary ===")
    for k, v in summary.items():
        print(f"  {k:18}: {v}")

    if args.output:
        import os
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(ok[0]).keys()))
            w.writeheader()
            for r in ok:
                w.writerow(asdict(r))
        # append a one-line summary row to results/summary.csv for cross-run plots
        summ_path = os.path.join(os.path.dirname(args.output) or ".", "summary.csv")
        new = not os.path.exists(summ_path)
        with open(summ_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["label", *summary.keys()])
            if new:
                w.writeheader()
            w.writerow({"label": args.label or os.path.basename(args.output), **summary})
        print(f"\nwrote {args.output} and appended summary to {summ_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", required=True, help="tunnel URL of the vLLM server")
    p.add_argument("--model", required=True)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--num-prompts", type=int, default=100)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--output", default=None, help="per-request CSV path, e.g. results/tp2_c32.csv")
    p.add_argument("--label", default=None, help="label for the summary row (e.g. 'tp2')")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
