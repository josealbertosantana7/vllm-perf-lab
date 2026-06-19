#!/usr/bin/env python3
"""
A fused softmax written in Triton — your "CUDA literacy" artifact.

Triton compiles to the same CUDA/PTX that vLLM's kernels do, but lets you write
the GPU program in Python. A naive softmax reads/writes the input several times
(max, exp, sum, divide). This version loads each row ONCE into fast SRAM, does
all the math there, and writes once — the classic fusion win Fregly discusses
when talking about memory-bound kernels.

Runs on the GPU plane (Kaggle/Colab T4). Compares against torch.softmax for
correctness and speed.

  python kernels/fused_softmax_triton.py
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    in_row_stride, out_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # one program instance handles one row of the matrix
    row_idx = tl.program_id(0)
    row_start = in_ptr + row_idx * in_row_stride
    col_off = tl.arange(0, BLOCK_SIZE)
    mask = col_off < n_cols

    # load the whole row into SRAM (out-of-range lanes -> -inf so they don't win the max)
    row = tl.load(row_start + col_off, mask=mask, other=-float("inf"))

    # numerically-stable softmax, all in registers/SRAM
    row = row - tl.max(row, axis=0)
    numerator = tl.exp(row)
    denominator = tl.sum(numerator, axis=0)
    softmax = numerator / denominator

    out_start = out_ptr + row_idx * out_row_stride
    tl.store(out_start + col_off, softmax, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2, "expects a 2D (rows, cols) tensor"
    n_rows, n_cols = x.shape
    block_size = triton.next_power_of_2(n_cols)
    out = torch.empty_like(x)
    _softmax_kernel[(n_rows,)](
        out, x,
        x.stride(0), out.stride(0),
        n_cols,
        BLOCK_SIZE=block_size,
    )
    return out


def main():
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found. Run this on the GPU plane (Kaggle/Colab), not the Mac.")

    torch.manual_seed(0)
    x = torch.randn(4096, 2048, device="cuda", dtype=torch.float32)

    # correctness
    ref = torch.softmax(x, dim=1)
    got = triton_softmax(x)
    max_err = (ref - got).abs().max().item()
    print(f"max abs error vs torch.softmax: {max_err:.3e}  ({'OK' if max_err < 1e-5 else 'FAIL'})")

    # speed
    def bench(fn):
        for _ in range(10):  # warmup
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / 100  # ms/iter

    t_torch = bench(lambda: torch.softmax(x, dim=1))
    t_triton = bench(lambda: triton_softmax(x))
    print(f"torch.softmax : {t_torch:.3f} ms/iter")
    print(f"triton fused  : {t_triton:.3f} ms/iter")
    print(f"speedup       : {t_torch / t_triton:.2f}x")


if __name__ == "__main__":
    main()
