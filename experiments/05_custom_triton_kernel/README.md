# Experiment 05 — Custom Triton Kernel (CUDA literacy)

**Concept:** Softmax is memory-bound — naive implementations make several passes over global memory.
A fused kernel loads each row into on-chip SRAM once, does all the math there, and writes once.
This is the kernel-level half of "AI systems performance engineering."

**Hypothesis:** A fused Triton softmax matches `torch.softmax` numerically and beats (or matches) it in
latency, because it cuts global-memory traffic.

### Run (GPU plane)
```bash
pip install -r requirements-gpu.txt
python kernels/fused_softmax_triton.py
```

### Measure
- `max abs error` vs `torch.softmax` (should be ~1e-7).
- ms/iter for torch vs triton, and the speedup factor.
- Optional: profile with `ncu`/`nsys` and report achieved memory bandwidth.

### Expected / write up here
Correct to ~1e-7 and competitive/faster latency. Record your T4 numbers and explain *why* fusion helps
for a memory-bound op. Stretch goal: extend to a fused attention kernel and connect it to paged attention.
