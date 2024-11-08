import torch


def my_attn_QKV(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Calculate scaled dot product attention.

    Args:
        Q (Tensor): Query tensor of shape (..., seq_len_N, d_k).
        K (Tensor): Key tensor of shape (..., seq_len_M, d_k).
        V (Tensor): Value tensor of shape (..., seq_len_M, d_v).
        mask (Tensor, optional): Optional mask tensor of shape (..., seq_len_N, seq_len_M).
            If a boolean tensor, False elements are replaced with -inf and True
            with 0 before addition to the similarity.  If not boolean, it's
            directly added to the similarity.

    Returns:
        Tensor: Output tensor of shape (..., seq_N, d_v).
    """
    assert Q.size(-1) == K.size(-1)
    assert K.size(-2) == V.size(-2)

    if mask is None:
        mask = torch.zeros(Q.size(-2), K.size(-2), dtype=Q.dtype, device=Q.device)
    if mask.dtype == torch.bool:
        # Convert boolean mask to float with -inf elements where False and 0 where True
        mask = (~mask).float().masked_fill(~mask, float("-inf"))

    similarity = torch.matmul(Q, K.transpose(-1, -2)) / K.size(-1) ** 0.5
    similarity += mask
    weights = similarity.softmax(dim=-1)
    output = torch.matmul(weights, V)

    return output


def my_attn_QKV_multihead(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
    num_heads: int = 1,
) -> torch.Tensor:
    """Implement scaled dot attention with multiple heads.

    This function splits the input tensors Q, K, and V into multiple heads and
    applies the scaled dot product attention separately for each head. It then
    combines the attention results from individual heads to produce the final
    output tensor. See also the doc of `my_attn_QKV`.

    Args:
        Q (Tensor): See `my_attn_QKV` for details.
        K (Tensor): See `my_attn_QKV` for details.
        V (Tensor): See `my_attn_QKV` for details.
        mask (Tensor, optional): See `my_attn_QKV` for details.
        num_heads (int): Number of attention heads to use.

    Returns:
        Tensor: The output tensor. Refer to `my_attn_QKV` for detailed shape information.
    """
    assert Q.size(-1) % num_heads == 0
    assert K.size(-1) % num_heads == 0
    assert V.size(-1) % num_heads == 0

    # Split Q, K, V tensors into multiple heads
    Q_split = Q.view(*Q.shape[:-1], num_heads, -1).transpose(-3, -2)
    K_split = K.view(*K.shape[:-1], num_heads, -1).transpose(-3, -2)
    V_split = V.view(*V.shape[:-1], num_heads, -1).transpose(-3, -2)
    if mask is not None:
        # split mask too
        mask = mask.unsqueeze(-3)

    # Compute attention for each head
    attn = my_attn_QKV(Q_split, K_split, V_split, mask=mask)

    # Combine attention results from all heads
    out = attn.transpose(-3, -2).reshape(*Q.shape[:-1], V.size(-1))

    if True:
        # before split: [..., seq_len, feature_dim]
        # after split: [..., seq_len, num_heads, feature_dim // num_heads]
        # after transpose: [..., num_heads, seq_len, feature_dim // num_heads]
        assert Q_split.shape == tuple(
            (*Q.shape[:-2], num_heads, Q.size(-2), Q.size(-1) // num_heads)
        )
        assert K_split.shape == tuple(
            (*K.shape[:-2], num_heads, K.size(-2), K.size(-1) // num_heads)
        )
        assert V_split.shape == tuple(
            (*V.shape[:-2], num_heads, V.size(-2), V.size(-1) // num_heads)
        )

        # attn shape: [..., num_heads, seq_len_N, d_v // num_heads]
        assert attn.shape[-3:] == tuple(
            (num_heads, Q.size(-2), V.size(-1) // num_heads)
        )

        # after transpose: [..., seq_len_N, num_heads, d_v // num_heads]
        # after reshape: [..., seq_len_N, d_v]
        assert out.shape == tuple((*out.shape[:-2], Q.size(-2), V.size(-1)))

        # mask shape: [..., 1, seq_len_N, seq_len_N]
        if mask is not None:
            assert mask.shape[-2:] == tuple((Q.size(-2), Q.size(-2)))

    return out
