# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

import vllm._custom_ops as ops
from vllm.platforms import current_platform
from vllm.attention.ops.triton_decode_attention import decode_attention_fwd

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="Cutlass MLA Requires compute capability of 10 or above.",
        allow_module_level=True)


def ref_mla(
        out: Tensor,  # (bs, num_heads, v_head_dim)
        query: Tensor,  # (bs, num_heads, head_dim)
        kv_cache: Tensor,  # (num_blocks, block_size, head_dim)
        scale: float,
        block_tables: Tensor,  # (bs, max_num_blocks)
        seq_lens: Tensor,  # (bs,)
):
    bs, num_heads, v_head_dim = out.shape
    head_dim = query.shape[2]

    for i in range(bs):
        # gather and flatten KV-cache
        kv = kv_cache[
            block_tables[i]]  # (max_num_blocks, block_size, head_dim)
        kv = kv.view(1, -1,
                     head_dim)[:, :seq_lens[i]]  # (1, seq_len, head_dim)
        v = kv[:, :, :v_head_dim]

        q = query[i].view(num_heads, 1, head_dim)
        o = F.scaled_dot_product_attention(q,
                                           kv,
                                           v,
                                           scale=scale,
                                           enable_gqa=True)
        out[i] = o.view(num_heads, v_head_dim)

    return out


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mean_seq_len", [128, 256, 512, 1024, 4096])
@pytest.mark.parametrize("bs", [1, 2, 4])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("block_size", [1, 16, 64, 128])
@pytest.mark.parametrize("use_cutlass", [True, False])
def test_cutlass_mla_decode(dtype: torch.dtype, mean_seq_len: int, bs: int,
                            varlen: bool, block_size: int, use_cutlass: bool):
    torch.set_default_dtype(dtype)
    torch.set_default_device('cuda')
    torch.manual_seed(42)

    d = 576
    h_q = 128
    dv = 512

    q_nope_dim = 128
    q_pe_dim = 64
    scale = (q_nope_dim + q_pe_dim)**(-0.5)
    if varlen:
        seq_lens = torch.empty(bs).normal_(mean_seq_len, mean_seq_len / 2)
        seq_lens = seq_lens.clip(2).to(torch.int32)
    else:
        seq_lens = torch.full((bs, ), mean_seq_len, dtype=torch.int32)
    max_seq_len = seq_lens.max().item()
    block_num = (max_seq_len + block_size - 1) // block_size

    # Pad block_num so that small blocks can be packed into full 128-sized
    # CUTLASS tiles. One 128-wide tile can hold (128 // block_size) small
    # blocks.
    pack_factor = 128 // block_size
    block_num = ((block_num + pack_factor - 1) // pack_factor) * pack_factor

    q = torch.randn(bs, h_q, d) * 10
    block_table = torch.randint(0,
                                bs * block_num, (bs, block_num),
                                dtype=torch.int32)

    kv_cache = torch.randn(block_table.numel(), block_size, d)

    out_ref = q.new_zeros(bs, h_q, dv)
    ref_mla(out_ref, q, kv_cache, scale, block_table, seq_lens)
    out_ans = torch.zeros_like(out_ref)
    q_nope = q[:, :, :dv].clone()
    q_pe = q[:, :, dv:].clone()
    print(f"{use_cutlass=}")
    if use_cutlass:
        ops.cutlass_mla_decode(out_ans, q_nope, q_pe, kv_cache, seq_lens,
                                block_table, scale)
    else:
        q = torch.cat([q_nope, q_pe], dim=-1)
        kv_c_and_k_pe_cache = kv_cache.unsqueeze(2)
        kv_c_cache = kv_c_and_k_pe_cache[..., :dv]
        PAGE_SIZE = kv_c_and_k_pe_cache.size(1)
        num_kv_splits = 4  # TODO: heuristic
        attn_logits = torch.empty((bs, h_q, num_kv_splits, dv + 1),
                                  dtype=torch.float32,
                                  device=q.device)
        decode_attention_fwd(q, kv_c_and_k_pe_cache, kv_c_cache, out_ans,
                             block_table, seq_lens, attn_logits, num_kv_splits,
                             scale, PAGE_SIZE)

    torch.testing.assert_close(out_ans, out_ref, atol=1e-2, rtol=1e-2)
