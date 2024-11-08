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
