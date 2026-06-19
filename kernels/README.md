# kernels/ — the CUDA artifact

You can't compile CUDA on the Mac, so the GPU-programming piece lives here and runs on the
GPU plane (Kaggle/Colab T4). **Triton** is the right tool: it generates the same PTX/CUDA that
vLLM's own kernels use, but you write the program in Python — and Fregly's book uses it heavily.

### `fused_softmax_triton.py`
A numerically-stable softmax that loads each matrix row into on-chip SRAM **once**, does max/exp/sum/divide
there, and writes **once**. That fusion is the whole point: softmax is memory-bound, so cutting the number
of global-memory passes is what buys the speedup.

```bash
pip install -r requirements-gpu.txt   # on the GPU box; pulls torch+triton
python kernels/fused_softmax_triton.py
```

Expected output: `max abs error` ~1e-7 (matches PyTorch) and a speedup factor over `torch.softmax`.
Record the numbers in [`experiments/05_custom_triton_kernel`](../experiments/05_custom_triton_kernel).

### Where to take it next
- Add a **fused attention** (softmax(QKᵀ/√d)·V) kernel — the conceptual core of paged attention.
- Profile with `nsys`/`ncu` on the cloud box and tie the numbers to the kernel's memory traffic.
