# Kaggle quickstart — ONE cell (self-contained, no clone)

Your only manual steps:

1. Go to <https://www.kaggle.com/code> → **New Notebook**.
2. Right sidebar → **Accelerator** = `GPU T4 x2`, **Internet** = `On`.
   (Internet requires a phone-verified Kaggle account — free.)
3. Paste the cell below and **Run** it.
4. Wait ~3–5 min. It prints `=== GIVE THIS URL TO CLAUDE ===` with a URL. Copy it.
5. Paste the URL back into Claude Code. Claude drives all benchmarks from there.

> This cell is **self-contained** — it does NOT clone this repo, so it works whether the repo is
> private or public. The benchmark client runs on your Mac (Claude's side), not on Kaggle.

```python
# vllm-perf-lab — self-contained GPU bootstrap (no git clone). Single Kaggle cell, GPU T4 x2 + Internet ON.
import subprocess, sys, time, re, urllib.request

# --- experiment knobs: change these to switch experiments ---
MODEL      = "Qwen/Qwen2.5-7B-Instruct"   # MoE run: "allenai/OLMoE-1B-7B-0924-Instruct"
TP         = 2                             # tensor-parallel size: 2, or 1 for the baseline
EXTRA_ARGS = []                            # e.g. ["--enable-expert-parallel"] or ["--enable-prefix-caching"]

print("[0/4] stopping any previous vLLM/tunnel in this session (lets you re-run safely)...")
subprocess.run("pkill -f 'vllm serve'; pkill -f cloudflared; sleep 3", shell=True)

print("[1/4] installing vLLM (a few min; the red pip 'conflict' warnings are harmless)...")
subprocess.run([sys.executable,"-m","pip","install","-q","vllm>=0.6.0"], check=True)

print(f"[2/4] launching vLLM: {MODEL}  TP={TP}  {EXTRA_ARGS}")
cmd = ["vllm","serve",MODEL,"--tensor-parallel-size",str(TP),"--dtype","float16",
       "--gpu-memory-utilization","0.90","--max-model-len","8192",
       "--port","8000","--host","0.0.0.0",*EXTRA_ARGS]
subprocess.Popen(cmd, stdout=open("server.log","w"), stderr=subprocess.STDOUT)

print("[3/4] cloudflared + waiting for server (up to ~8 min)...")
subprocess.run("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/"
               "cloudflared-linux-amd64 -O /usr/local/bin/cloudflared && "
               "chmod +x /usr/local/bin/cloudflared", shell=True, check=True)
ready=False
for _ in range(48):
    try: urllib.request.urlopen("http://localhost:8000/v1/models",timeout=2); ready=True; break
    except Exception: time.sleep(10)
print("    server ready:", ready)
if not ready:
    print("---- last lines of server.log ----"); print(open("server.log").read()[-2000:])

print("[4/4] opening tunnel...")
subprocess.Popen(["cloudflared","tunnel","--url","http://localhost:8000"],
                 stdout=open("tunnel.log","w"), stderr=subprocess.STDOUT)
url=None
for _ in range(20):
    time.sleep(3)
    try:
        m=re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", open("tunnel.log").read())
        if m: url=m.group(0); break
    except FileNotFoundError: pass
print("\n"+"="*44+"\n=== GIVE THIS URL TO CLAUDE ===\n"+(url or "re-run this cell in a moment")+"\n"+"="*44)
```

### Switching experiments
Change the knobs at the top and re-run the cell:

| Experiment | MODEL | TP | EXTRA_ARGS |
|---|---|---|---|
| 01 baseline | `Qwen/Qwen2.5-7B-Instruct` | `1` | `[]` |
| 01 tensor-parallel | `Qwen/Qwen2.5-7B-Instruct` | `2` | `[]` |
| 02 MoE expert-parallel | `allenai/OLMoE-1B-7B-0924-Instruct` | `2` | `["--enable-expert-parallel"]` |
| 03 prefix caching | `Qwen/Qwen2.5-7B-Instruct` | `2` | `["--enable-prefix-caching"]` |

> Re-running restarts the tunnel, so the URL changes each time — paste the new one to Claude.
> `server ready: False` prints the tail of `server.log` so you can see the cause (usually still loading, or OOM).
