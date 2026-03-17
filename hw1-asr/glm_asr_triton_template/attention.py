"""
Triton Multi-Head Attention Implementation
End-to-end implementation using Triton kernels

*** STUDENT ASSIGNMENT ***
Fill in the TODO sections to implement attention using Triton kernels
"""

import numpy as np
import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


def get_stream():
    """Get current CUDA stream pointer."""
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None


# ============================================================================
# Triton Kernels for Attention
# ============================================================================

@triton.jit
def attention_scores_kernel(
    q_ptr,
    k_ptr,
    scores_ptr,
    scale,
    seq_k,
    head_dim,
    stride_q0,
    stride_q1,
    stride_q2,
    stride_k0,
    stride_k1,
    stride_k2,
    stride_s0,
    stride_s1,
    stride_s2,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Compute scaled attention scores for a single query position.
    Grid: (batch_heads, seq_q)

    *** TODO: Implement this kernel ***
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    # ============================================================================
    # TODO: Implement attention score computation
    # ============================================================================
    #
    # Step 1: Load query vector for this position
    # Step 2: Load all keys for this batch_head
    # Step 3: Compute dot-product scores and scale
    # Step 4: Store scores

    # YOUR CODE HERE
    pid_q = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)

    q = tl.load(
        q_ptr + pid_bh * stride_q0 + pid_q * stride_q1 + offs_d * stride_q2,
        mask=offs_d < head_dim,
        other=0.0,
    ).to(tl.float32)

    k = tl.load(
        k_ptr
        + pid_bh * stride_k0
        + offs_k[:, None] * stride_k1
        + offs_d[None, :] * stride_k2,
        mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim),
        other=0.0,
    ).to(tl.float32)

    scores = tl.sum(k * q[None, :], axis=1) * scale

    tl.store(
        scores_ptr
        + pid_bh * stride_s0
        + pid_q * stride_s1
        + offs_k * stride_s2,
        scores,
        mask=offs_k < seq_k,
    )
    #pass


@triton.jit
def softmax_inplace_kernel(scores_ptr, stride_s, seq_k, BLOCK_SIZE: tl.constexpr):
    """
    Apply softmax along the last dimension (seq_k).
    Grid: (batch_heads * seq_q,)
    """
    row = tl.program_id(0)

    # ============================================================================
    # TODO: Implement softmax
    # ============================================================================
    #
    # Step 1: Load scores row with masking
    # Step 2: Subtract max for stability
    # Step 3: Compute exp and normalize
    # Step 4: Store back

    # YOUR CODE HERE
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < seq_k

    scores = tl.load(
        scores_ptr + row * stride_s + offs,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)

    scores = scores - tl.max(scores, axis=0)
    exp_scores = tl.exp(scores)
    denom = tl.sum(exp_scores, axis=0)
    probs = exp_scores / denom

    tl.store(scores_ptr + row * stride_s + offs, probs, mask=mask)
    #pass


@triton.jit
def attention_output_kernel(
    attn_ptr,
    v_ptr,
    output_ptr,
    seq_k,
    head_dim,
    stride_w0,
    stride_w1,
    stride_w2,
    stride_v0,
    stride_v1,
    stride_v2,
    stride_o0,
    stride_o1,
    stride_o2,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Compute attention output: attn_weights @ V
    Grid: (batch_heads, seq_q)
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    # ============================================================================
    # TODO: Implement attention output computation
    # ============================================================================
    #
    # Step 1: Load attention weights for this query
    # Step 2: Load all values for this batch_head
    # Step 3: Compute weighted sum
    # Step 4: Store output

    # YOUR CODE HERE
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)

    attn = tl.load(
        attn_ptr
        + pid_bh * stride_w0
        + pid_q * stride_w1
        + offs_k * stride_w2,
        mask=offs_k < seq_k,
        other=0.0,
    ).to(tl.float32)

    v = tl.load(
        v_ptr
        + pid_bh * stride_v0
        + offs_k[:, None] * stride_v1
        + offs_d[None, :] * stride_v2,
        mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim),
        other=0.0,
    ).to(tl.float32)

    out = tl.sum(v * attn[:, None], axis=0)

    tl.store(
        output_ptr
        + pid_bh * stride_o0
        + pid_q * stride_o1
        + offs_d * stride_o2,
        out,
        mask=offs_d < head_dim,
    )
    #pass


@triton.jit
def causal_mask_kernel(
    scores_ptr,
    seq_k,
    offset,
    stride_s0,
    stride_s1,
    stride_s2,
    BLOCK_K: tl.constexpr,
):
    """
    Apply causal mask to attention scores.
    Grid: (batch_heads, seq_q)
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < seq_k
    scores = tl.load(
        scores_ptr
        + pid_bh * stride_s0
        + pid_q * stride_s1
        + offs_k * stride_s2,
        mask=mask,
        other=-1e9,
    )
    current_pos = pid_q + offset
    scores = tl.where(offs_k > current_pos, -1e9, scores)
    tl.store(
        scores_ptr
        + pid_bh * stride_s0
        + pid_q * stride_s1
        + offs_k * stride_s2,
        scores,
        mask=mask,
    )


# ============================================================================
# Attention Classes
# ============================================================================

class MultiHeadAttention:
    """Multi-head attention using Triton kernels."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.scale = 1.0 / np.sqrt(self.head_dim)

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Compute multi-head attention.

        Args:
            q: Query (batch, num_heads, seq_q, head_dim)
            k: Key (batch, num_kv_heads, seq_k, head_dim)
            v: Value (batch, num_kv_heads, seq_k, head_dim)
            attention_mask: Optional mask (batch, 1, seq_q, seq_k)
            is_causal: Whether to apply causal masking

        Returns:
            Output (batch, num_heads, seq_q, head_dim)
        """
        batch, num_heads, seq_q, head_dim = q.shape
        _, num_kv_heads, seq_k, _ = k.shape

        if num_kv_heads != num_heads:
            k = self._expand_kv(k, self.num_queries_per_kv)
            v = self._expand_kv(v, self.num_queries_per_kv)

        return scaled_dot_product_attention(
            q, k, v, attention_mask, is_causal, self.scale
        )

    def _expand_kv(self, x: torch.Tensor, num_repeats: int) -> torch.Tensor:
        """Expand KV heads for GQA using broadcast (zero-copy)."""
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x_expanded = x[:, :, None, :, :].expand(
            batch, num_kv_heads, num_repeats, seq_len, head_dim
        )
        return x_expanded.reshape(batch, num_kv_heads * num_repeats, seq_len, head_dim)


def next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    return 1 << (x - 1).bit_length() if x > 0 else 1


MAX_ATTENTION_DIM = 256
FLASH_ATTENTION_CONFIGS = {
    64: {"block_m": 64, "block_n": 64, "num_warps": 4, "num_stages": 2},
    128: {"block_m": 32, "block_n": 64, "num_warps": 8, "num_stages": 2},
}


@triton.jit
def flash_attention_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    output_ptr,
    scale,
    seq_q,
    seq_k,
    stride_q0,
    stride_q1,
    stride_q2,
    stride_k0,
    stride_k1,
    stride_k2,
    stride_v0,
    stride_v1,
    stride_v2,
    stride_o0,
    stride_o1,
    stride_o2,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    FlashAttention-like forward kernel for the Phase 1 minimal path.
    Grid: (batch_heads, ceil_div(seq_q, BLOCK_M))
    """
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    mask_q = offs_m < seq_q
    mask_qd = mask_q[:, None] & (offs_d[None, :] < HEAD_DIM)

    q = tl.load(
        q_ptr
        + pid_bh * stride_q0
        + offs_m[:, None] * stride_q1
        + offs_d[None, :] * stride_q2,
        mask=mask_qd,
        other=0.0,
    ).to(tl.float32)

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    for start_n in tl.range(0, seq_k, BLOCK_N):
        k_offsets = start_n + offs_n
        mask_k = k_offsets < seq_k
        mask_kd = mask_k[:, None] & (offs_d[None, :] < HEAD_DIM)

        k = tl.load(
            k_ptr
            + pid_bh * stride_k0
            + k_offsets[:, None] * stride_k1
            + offs_d[None, :] * stride_k2,
            mask=mask_kd,
            other=0.0,
        ).to(tl.float32)

        v = tl.load(
            v_ptr
            + pid_bh * stride_v0
            + k_offsets[:, None] * stride_v1
            + offs_d[None, :] * stride_v2,
            mask=mask_kd,
            other=0.0,
        ).to(tl.float32)

        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        scores = tl.where(mask_k[None, :], scores, -float("inf"))

        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
        m_i = m_new

    output = acc / l_i[:, None]

    tl.store(
        output_ptr
        + pid_bh * stride_o0
        + offs_m[:, None] * stride_o1
        + offs_d[None, :] * stride_o2,
        output,
        mask=mask_qd,
    )


def _expand_kv_for_gqa(x: torch.Tensor, num_repeats: int) -> torch.Tensor:
    """Expand KV heads for GQA using the same broadcast pattern as the main path."""
    batch, num_kv_heads, seq_len, head_dim = x.shape
    x_expanded = x[:, :, None, :, :].expand(
        batch, num_kv_heads, num_repeats, seq_len, head_dim
    )
    return x_expanded.reshape(batch, num_kv_heads * num_repeats, seq_len, head_dim)


def torch_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Pure Torch reference implementation for correctness checks."""
    _, _, seq_q, head_dim = q.shape
    _, _, seq_k, _ = k.shape

    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)

    scores = torch.einsum("bnqd,bnkd->bnqk", q.to(torch.float32), k.to(torch.float32))
    scores = scores * float(scale)

    if is_causal:
        causal_mask = torch.triu(
            torch.ones((seq_q, seq_k), dtype=torch.float32, device=q.device),
            diagonal=1,
        ) * -1e9
        scores = scores + causal_mask[None, None, :, :]

    if attention_mask is not None:
        scores = scores + attention_mask.to(torch.float32)

    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.einsum("bnqk,bnkd->bnqd", attn_weights, v.to(torch.float32))
    return output.to(q.dtype)


def _assert_attention_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float = 1e-3,
    rtol: float = 1e-3,
):
    """Raise a readable error if the optimized path diverges from the Torch reference."""
    actual_f = actual.detach().to(torch.float32).cpu()
    expected_f = expected.detach().to(torch.float32).cpu()

    max_abs = float((actual_f - expected_f).abs().max())
    torch.testing.assert_close(actual_f, expected_f, atol=atol, rtol=rtol)
    print(f"  {name}: PASS (max_abs_diff={max_abs:.6f})")


def _can_use_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    is_causal: bool,
) -> bool:
    """Return whether the minimal Phase 1 flash path can handle this call."""
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        return False
    if attention_mask is not None or is_causal:
        return False
    if q.shape[-1] not in FLASH_ATTENTION_CONFIGS:
        return False
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        return False
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        return False
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        return False
    if q.shape[-2] == 0 or k.shape[-2] == 0:
        return False
    return True


def _flash_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Launch the minimal FlashAttention-like forward kernel."""
    batch, num_heads, seq_q, head_dim = q.shape
    _, _, seq_k, _ = k.shape
    config = FLASH_ATTENTION_CONFIGS[head_dim]

    q_flat = (
        q.reshape(batch * num_heads, seq_q, head_dim).to(torch.float32).contiguous()
    )
    k_flat = (
        k.reshape(batch * num_heads, seq_k, head_dim).to(torch.float32).contiguous()
    )
    v_flat = (
        v.reshape(batch * num_heads, seq_k, head_dim).to(torch.float32).contiguous()
    )
    output = torch.empty_like(q_flat, dtype=torch.float32)

    grid = (batch * num_heads, triton.cdiv(seq_q, config["block_m"]))
    flash_attention_fwd_kernel[grid](
        q_flat,
        k_flat,
        v_flat,
        output,
        float(scale),
        seq_q,
        seq_k,
        q_flat.stride(0),
        q_flat.stride(1),
        q_flat.stride(2),
        k_flat.stride(0),
        k_flat.stride(1),
        k_flat.stride(2),
        v_flat.stride(0),
        v_flat.stride(1),
        v_flat.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        HEAD_DIM=head_dim,
        BLOCK_M=config["block_m"],
        BLOCK_N=config["block_n"],
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )

    return output.reshape(batch, num_heads, seq_q, head_dim).to(q.dtype)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Scaled dot-product attention using Triton kernels.
    """
    batch, num_heads, seq_q, head_dim = q.shape
    _, _, seq_k, _ = k.shape

    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)

    if _can_use_flash_attention(q, k, v, attention_mask, is_causal):
        return _flash_scaled_dot_product_attention(q, k, v, float(scale))

    seq_k_padded = next_power_of_two(seq_k)
    head_dim_padded = next_power_of_two(head_dim)

    use_triton = (
        q.is_cuda
        and seq_k_padded <= MAX_ATTENTION_DIM
        and head_dim_padded <= MAX_ATTENTION_DIM
    )

    if use_triton:
        q_flat = q.reshape(batch * num_heads, seq_q, head_dim).to(torch.float32)
        k_flat = k.reshape(batch * num_heads, seq_k, head_dim).to(torch.float32)
        v_flat = v.reshape(batch * num_heads, seq_k, head_dim).to(torch.float32)

        if seq_k_padded != seq_k or head_dim_padded != head_dim:
            k_padded = torch.zeros(
                (batch * num_heads, seq_k_padded, head_dim_padded),
                dtype=torch.float32,
                device=q.device,
            )
            v_padded = torch.zeros_like(k_padded)
            q_padded = torch.zeros(
                (batch * num_heads, seq_q, head_dim_padded),
                dtype=torch.float32,
                device=q.device,
            )
            k_padded[:, :seq_k, :head_dim] = k_flat
            v_padded[:, :seq_k, :head_dim] = v_flat
            q_padded[:, :, :head_dim] = q_flat
            k_flat = k_padded
            v_flat = v_padded
            q_flat = q_padded

        scores = torch.empty(
            (batch * num_heads, seq_q, seq_k_padded),
            dtype=torch.float32,
            device=q.device,
        )
        output = torch.empty(
            (batch * num_heads, seq_q, head_dim_padded),
            dtype=torch.float32,
            device=q.device,
        )

        grid = (batch * num_heads, seq_q)
        attention_scores_kernel[grid](
            q_flat,
            k_flat,
            scores,
            float(scale),
            seq_k_padded,
            head_dim_padded,
            q_flat.stride(0),
            q_flat.stride(1),
            q_flat.stride(2),
            k_flat.stride(0),
            k_flat.stride(1),
            k_flat.stride(2),
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            BLOCK_K=seq_k_padded,
            BLOCK_D=head_dim_padded,
        )

        if seq_k_padded != seq_k:
            scores[:, :, seq_k:] = -1e9

        if is_causal:
            mask = torch.triu(
                torch.ones((seq_q, seq_k_padded), dtype=torch.float32, device=q.device),
                diagonal=1,
            ) * -1e9
            scores = scores + mask[None, :, :]

        if attention_mask is not None:
            if attention_mask.ndim == 4:
                attention_mask = attention_mask.reshape(
                    batch * num_heads, seq_q, seq_k
                )
            if seq_k_padded != seq_k:
                mask_padded = torch.zeros(
                    (batch * num_heads, seq_q, seq_k_padded),
                    dtype=torch.float32,
                    device=q.device,
                )
                mask_padded[:, :, :seq_k] = attention_mask
                mask_padded[:, :, seq_k:] = -1e9
                attention_mask = mask_padded
            scores = scores + attention_mask

        scores_2d = scores.reshape(batch * num_heads * seq_q, seq_k_padded)
        block = seq_k_padded
        softmax_inplace_kernel[(scores_2d.shape[0],)](
            scores_2d, scores_2d.stride(0), seq_k_padded, BLOCK_SIZE=block
        )
        scores = scores_2d.reshape(batch * num_heads, seq_q, seq_k_padded)

        attention_output_kernel[grid](
            scores,
            v_flat,
            output,
            seq_k_padded,
            head_dim_padded,
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            v_flat.stride(0),
            v_flat.stride(1),
            v_flat.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            BLOCK_K=seq_k_padded,
            BLOCK_D=head_dim_padded,
        )

        if head_dim_padded != head_dim:
            output = output[:, :, :head_dim]

        return output.reshape(batch, num_heads, seq_q, head_dim).to(q.dtype)

    scores = torch.einsum("bnqd,bnkd->bnqk", q, k) * scale

    if is_causal:
        mask = torch.triu(
            torch.ones((seq_q, seq_k), dtype=torch.float32, device=q.device),
            diagonal=1,
        ) * -1e9
        scores = scores + mask[None, None, :, :]

    if attention_mask is not None:
        scores = scores + attention_mask

    scores = scores - torch.max(scores, dim=-1, keepdim=True).values
    attn_weights = torch.exp(scores)
    attn_weights = attn_weights / torch.sum(attn_weights, dim=-1, keepdim=True)
    output = torch.einsum("bnqk,bnkd->bnqd", attn_weights, v)

    return output.to(q.dtype)


if __name__ == "__main__":
    print("Testing Triton Attention...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"  Device: {device}")
    print(f"  Seed:   {seed}")

    batch_size = 2
    num_heads = 4
    head_dim = 64

    print("\nNumerical comparison against Torch reference:")

    # Basic self-attention
    q = torch.randn(batch_size, num_heads, 19, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, 19, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, 19, head_dim, device=device)
    actual = scaled_dot_product_attention(q, k, v)
    expected = torch_attention_reference(q, k, v)
    _assert_attention_close("basic", actual, expected)

    # Causal self-attention
    q = torch.randn(batch_size, num_heads, 21, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, 21, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, 21, head_dim, device=device)
    actual = scaled_dot_product_attention(q, k, v, is_causal=True)
    expected = torch_attention_reference(q, k, v, is_causal=True)
    _assert_attention_close("causal", actual, expected)

    # External attention mask
    q = torch.randn(batch_size, num_heads, 17, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, 17, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, 17, head_dim, device=device)
    mask = torch.zeros(
        (batch_size, num_heads, 17, 17), dtype=torch.float32, device=device
    )
    mask[:, :, :, 17 // 2 :] = -1e9
    actual = scaled_dot_product_attention(q, k, v, attention_mask=mask)
    expected = torch_attention_reference(q, k, v, attention_mask=mask)
    _assert_attention_close("masked", actual, expected)

    # Grouped Query Attention path through MultiHeadAttention
    num_kv_heads = 2
    q = torch.randn(batch_size, num_heads, 15, head_dim, device=device)
    k_gqa = torch.randn(batch_size, num_kv_heads, 15, head_dim, device=device)
    v_gqa = torch.randn(batch_size, num_kv_heads, 15, head_dim, device=device)
    attn = MultiHeadAttention(
        hidden_size=num_heads * head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    actual = attn(q, k_gqa, v_gqa)
    k_ref = _expand_kv_for_gqa(k_gqa, num_heads // num_kv_heads)
    v_ref = _expand_kv_for_gqa(v_gqa, num_heads // num_kv_heads)
    expected = torch_attention_reference(q, k_ref, v_ref, scale=attn.scale)
    _assert_attention_close("gqa", actual, expected)

    # Decode-like path: q_len < k_len
    q = torch.randn(batch_size, num_heads, 3, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, 11, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, 11, head_dim, device=device)
    actual = scaled_dot_product_attention(q, k, v)
    expected = torch_attention_reference(q, k, v)
    _assert_attention_close("q_len_lt_k_len", actual, expected)

    print("\nAll Triton attention reference checks passed.")

    print("\nTriton Attention working!")
