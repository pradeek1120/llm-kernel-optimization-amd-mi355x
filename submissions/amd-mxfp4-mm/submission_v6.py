#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Hybrid `mxfp4-mm` follow-up submission.

`submission_v5.py` improved the small benchmark shapes by forcing the observed
32x128 asm kernel, but it regressed larger shapes. This version keeps that
small-shape fast path and falls back to the default `aiter.gemm_a4w4(...)`
dispatch for larger `m`, which should preserve the baseline behavior there.
"""
from __future__ import annotations

from dataclasses import dataclass

from task import input_t, output_t

_BACKEND = None
_OUT_CACHE: dict[tuple[int, int, int], object] = {}

_KERNEL_SMALL_M = "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E"


@dataclass
class _Backend:
    aiter: object
    torch: object
    dtypes: object
    dynamic_mxfp4_quant: object
    e8m0_shuffle: object
    gemm_a4w4_asm: object


def _load_backend() -> _Backend:
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    try:
        import aiter
        import torch
        from aiter import dtypes
        from aiter.ops.gemm_op_a4w4 import gemm_a4w4_asm
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.utility.fp4_utils import e8m0_shuffle
    except ImportError as exc:
        raise RuntimeError(
            "custom_kernel requires the AMD ROCm/aiter runtime. "
            "Use Popcorn remote execution or run this on an AMD ROCm machine."
        ) from exc

    _BACKEND = _Backend(
        aiter=aiter,
        torch=torch,
        dtypes=dtypes,
        dynamic_mxfp4_quant=dynamic_mxfp4_quant,
        e8m0_shuffle=e8m0_shuffle,
        gemm_a4w4_asm=gemm_a4w4_asm,
    )
    return _BACKEND


def _quant_mxfp4(backend: _Backend, x):
    try:
        x_fp4, bs_e8m0 = backend.dynamic_mxfp4_quant(x, shuffle=True)
    except TypeError:
        x_fp4, bs_e8m0 = backend.dynamic_mxfp4_quant(x)
        bs_e8m0 = backend.e8m0_shuffle(bs_e8m0)
    return x_fp4.view(backend.dtypes.fp4x2), bs_e8m0.view(backend.dtypes.fp8_e8m0)


def _get_out(backend: _Backend, device, m: int, n: int):
    device_index = device.index if getattr(device, "index", None) is not None else -1
    key = (device_index, ((m + 31) // 32) * 32, n)
    out = _OUT_CACHE.get(key)
    if out is None:
        out = backend.torch.empty((key[1], n), dtype=backend.dtypes.bf16, device=device)
        _OUT_CACHE[key] = out
    return out


def custom_kernel(data: input_t) -> output_t:
    backend = _load_backend()
    A, _, _, B_shuffle, B_scale_sh = data

    if not A.is_contiguous():
        A = A.contiguous()

    m, _ = A.shape
    A_q, A_scale_sh = _quant_mxfp4(backend, A)

    if m <= 32:
        n = B_shuffle.shape[0]
        out = _get_out(backend, A.device, m, n)
        backend.gemm_a4w4_asm(
            A_q,
            B_shuffle,
            A_scale_sh,
            B_scale_sh,
            out,
            _KERNEL_SMALL_M,
            bpreshuffle=True,
            log2_k_split=0,
        )
        return out[:m]

    return backend.aiter.gemm_a4w4(
        A_q,
        B_shuffle,
        A_scale_sh,
        B_scale_sh,
        dtype=backend.dtypes.bf16,
        bpreshuffle=True,
    )
