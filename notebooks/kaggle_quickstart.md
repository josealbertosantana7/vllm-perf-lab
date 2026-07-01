# Kaggle quickstart — ONE cell

Your only manual steps:

1. Go to <https://www.kaggle.com/code> → **New Notebook**.
2. Right sidebar → **Accelerator** = `GPU T4 x2`, **Internet** = `On`.
   (Internet requires a phone-verified Kaggle account — free.)
3. Paste the cell below into the notebook and **Run** it.
4. Wait ~3–5 min. It prints a line like `=== GIVE THIS URL TO CLAUDE ===`. Copy that URL.
5. Paste the URL back into Claude Code. That's it — Claude drives the benchmarks from there.

```python
# vllm-perf-lab — one-shot GPU-plane bootstrap. Run in a single Kaggle cell (GPU T4 x2 + Internet ON).
import subprocess, sys, os, time, re, urllib.request

CONFIG = "serving/configs/tp2.env"   # swap later: tp1.env / moe.env / paged_attn.env

print("[1/5] installing vLLM (a few minutes the first time)...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm>=0.6.0"], check=True)

print("[2/5] cloning repo...")
if not os.path.isdir("vllm-perf-lab"):
    subprocess.run(["git", "clone", "-q",
                    "https://github.com/josealbertosantana7/vllm-perf-lab.git"], check=True)
os.chdir("vllm-perf-lab")

print(f"[3/5] launching vLLM ({CONFIG}) in the background...")
subprocess.Popen(["bash", "serving/launch_vllm.sh", CONFIG],
                 stdout=open("server.log", "w"), stderr=subprocess.STDOUT)

print("[4/5] installing cloudflared + waiting for the server (up to ~8 min)...")
subprocess.run("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/"
               "cloudflared-linux-amd64 -O /usr/local/bin/cloudflared && "
               "chmod +x /usr/local/bin/cloudflared", shell=True, check=True)
ready = False
for _ in range(48):
    try:
        urllib.request.urlopen("http://localhost:8000/v1/models", timeout=2); ready = True; break
    except Exception:
        time.sleep(10)
print("    server ready:", ready, "(if False, check server.log)")

print("[5/5] opening public tunnel...")
subprocess.Popen(["cloudflared", "tunnel", "--url", "http://localhost:8000"],
                 stdout=open("tunnel.log", "w"), stderr=subprocess.STDOUT)
url = None
for _ in range(20):
    time.sleep(3)
    try:
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", open("tunnel.log").read())
        if m: url = m.group(0); break
    except FileNotFoundError:
        pass

print("\n" + "=" * 44)
print("=== GIVE THIS URL TO CLAUDE ===")
print(url or "no URL yet — re-run this line in a moment; check tunnel.log")
print("=" * 44)
print("Keep this notebook tab OPEN while Claude benchmarks. To switch experiments,")
print("change CONFIG at the top and re-run the cell.")
```

> If `server ready: False`, the model is still loading or OOM'd — open `server.log` in the file browser.
> On 2×T4, `tp2.env` (Qwen2.5-7B, float16) fits comfortably.
