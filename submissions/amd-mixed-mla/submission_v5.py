#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X

"""
MLA decode split sweep follow-up.

Same path as `submission_v2.py`, but lowers persistent split count to 8.
"""
from __future__ import annotations

import torch

from task import input_t, output_t

from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.mla import mla_decode_fwd

NUM_HEADS = 16
NUM_KV_HEADS = 1
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
V_HEAD_DIM = KV_LORA_RANK
SM_SCALE = 1.0 / (QK_HEAD_DIM**0.5)

PAGE_SIZE = 1
NUM_KV_SPLITS = 8
FP8_DTYPE = aiter_dtypes.fp8
Q_DTYPE = "fp8"
KV_DTYPE = "fp8"

_META_CACHE: dict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]] = {}


def quantize_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    finfo = torch.finfo(FP8_DTYPE)
    amax = tensor.abs().amax().clamp(min=1e-12)
    scale = amax / finfo.max
    fp8_tensor = (tensor / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)
    return fp8_tensor, scale.to(torch.float32).reshape(1)


def _make_mla_decode_metadata(
    batch_size: int,
    max_q_len: int,
    nhead: int,
    nhead_kv: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    num_kv_splits: int = NUM_KV_SPLITS,
) -> dict[str, torch.Tensor]:
    info = get_mla_metadata_info_v1(
        batch_size,
        max_q_len,
        nhead,
        q_dtype,
        kv_dtype,
        is_sparse=False,
        fast_mode=False,
        num_kv_splits=num_kv_splits,
        intra_batch_mode=True,
    )
    work = [torch.empty(shape, dtype=dtype, device="cuda") for shape, dtype in info]
    (
        work_metadata,
        work_indptr,
        work_info_set,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
    ) = work

    get_mla_metadata_v1(
        qo_indptr,
        kv_indptr,
        kv_last_page_len,
        nhead // nhead_kv,
        nhead_kv,
        True,
        work_metadata,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        page_size=PAGE_SIZE,
        kv_granularity=max(PAGE_SIZE, 16),
        max_seqlen_qo=max_q_len,
        uni_seqlen_qo=max_q_len,
        fast_mode=False,
        max_split_per_batch=num_kv_splits,
        intra_batch_mode=True,
        dtype_q=q_dtype,
        dtype_kv=kv_dtype,
    )

    return {
        "work_meta_data": work_metadata,
        "work_indptr": work_indptr,
        "work_info_set": work_info_set,
        "reduce_indptr": reduce_indptr,
        "reduce_final_map": reduce_final_map,
        "reduce_partial_map": reduce_partial_map,
    }


def _get_cached_decode_setup(
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    batch_size: int,
    max_q_len: int,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    nhead: int,
    nhead_kv: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    total_kv = int(kv_indptr[-1].item())
    key = (
        int(qo_indptr.data_ptr()),
        int(kv_indptr.data_ptr()),
        q_dtype,
        kv_dtype,
        batch_size,
        max_q_len,
        total_kv,
        nhead,
        nhead_kv,
    )
    cached = _META_CACHE.get(key)
    if cached is not None:
        return cached

    kv_indices = torch.arange(total_kv, dtype=torch.int32, device="cuda")
    kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)
    meta = _make_mla_decode_metadata(
        batch_size,
        max_q_len,
        nhead,
        nhead_kv,
        q_dtype,
        kv_dtype,
        qo_indptr,
        kv_indptr,
        kv_last_page_len,
        NUM_KV_SPLITS,
    )
    cached = (kv_indices, kv_last_page_len, meta)
    _META_CACHE[key] = cached
    return cached


def _aiter_mla_decode(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    config: dict,
    q_scale: torch.Tensor | None = None,
    kv_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    nq = config["num_heads"]
    nkv = config["num_kv_heads"]
    dq = config["qk_head_dim"]
    dv = config["v_head_dim"]
    max_q_len = config["q_seq_len"]

    kv_indices, kv_last_page_len, meta = _get_cached_decode_setup(
        q.dtype,
        kv_buffer.dtype,
        config["batch_size"],
        max_q_len,
        qo_indptr,
        kv_indptr,
        nq,
        nkv,
    )

    kv_buffer_4d = kv_buffer.view(kv_buffer.shape[0], PAGE_SIZE, nkv, kv_buffer.shape[-1])
    out = torch.empty((q.shape[0], nq, dv), dtype=torch.bfloat16, device="cuda")
    mla_decode_fwd(
        q.view(-1, nq, dq),
        kv_buffer_4d,
        out,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        max_q_len,
        page_size=PAGE_SIZE,
        nhead_kv=nkv,
        sm_scale=SM_SCALE,
        logit_cap=0.0,
        num_kv_splits=NUM_KV_SPLITS,
        q_scale=q_scale,
        kv_scale=kv_scale,
        intra_batch_mode=True,
        **meta,
    )
    return out


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data

    if Q_DTYPE == "fp8":
        q_input, q_scale = quantize_fp8(q)
    else:
        q_input, q_scale = q, None

    if KV_DTYPE == "fp8":
        kv_input, kv_scale = kv_data["fp8"]
    else:
        kv_input, kv_scale = kv_data["bf16"], None

    return _aiter_mla_decode(
        q_input,
        kv_input,
        qo_indptr,
        kv_indptr,
        config,
        q_scale=q_scale,
        kv_scale=kv_scale,
    )
