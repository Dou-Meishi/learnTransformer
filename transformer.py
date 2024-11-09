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
            assert mask.shape[-2:] == tuple((Q.size(-2), K.size(-2)))

    return out


class MyMultiheadAttention(torch.nn.Module):
    """Implement multihead attention layer."""

    def __init__(
        self,
        xq_dim: int,
        num_heads: int,
        xk_dim: int | None = None,
        xv_dim: int | None = None,
        bias: bool = False,
        batch_first: bool = True,
    ):
        """Initialize the multihead attention layer.

        Args:
            xq_dim (int): Dimension of the Query tensor.
            num_heads (int): Number of attention heads to use.
            xk_dim (int, optional): Dimension of the Key tensor (default is equal to xq_dim).
            xv_dim (int, optional): Dimension of the Value tensor (default is equal to xq_dim).
            bias (bool): Whether to include bias terms (not implemented yet).
            batch_first (bool): Specify whether the input is batch-first (not implemented yet).
        """
        super().__init__()

        assert xq_dim % num_heads == 0

        if xk_dim is None:
            xk_dim = xq_dim
        if xv_dim is None:
            xv_dim = xq_dim
        if bias is True:
            raise NotImplementedError
        if batch_first is False:
            raise NotImplementedError

        self.num_heads = num_heads

        self.linear_Q = torch.nn.Linear(xq_dim, xq_dim, bias=bias)
        self.linear_K = torch.nn.Linear(xk_dim, xq_dim, bias=bias)
        self.linear_V = torch.nn.Linear(xv_dim, xq_dim, bias=bias)
        self.final_linear = torch.nn.Linear(xq_dim, xq_dim, bias=bias)

        # reinitialize parameters with xavier_normal if necessary
        # see https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#MultiheadAttention

    def load_from_pytorch_module(self, model: torch.nn.MultiheadAttention):
        """Load weights from a PyTorch model"""
        if model._qkv_same_embed_dim:
            Wq, Wk, Wv = torch.split(model.in_proj_weight, 3 * [model.embed_dim])
            self.load_state_dict(
                {
                    "linear_Q.weight": Wq,
                    "linear_K.weight": Wk,
                    "linear_V.weight": Wv,
                    "final_linear.weight": model.out_proj.weight,
                }
            )
        else:
            self.load_state_dict(
                {
                    "linear_Q.weight": model.q_proj_weight,
                    "linear_K.weight": model.k_proj_weight,
                    "linear_V.weight": model.v_proj_weight,
                    "final_linear.weight": model.out_proj.weight,
                }
            )

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perform forward pass for multihead attention.

        Args:
            xq (Tensor): Query tensor of shape (batch_size, seq_len_N, xq_dim).
            xk (Tensor): Key tensor of shape (batch_size, seq_len_M, xk_dim).
            xv (Tensor): Value tensor of shape (batch_size, seq_len_M, xv_dim).
            mask (Tensor, optional): Optional mask tensor of shape (...,
                seq_len_N, seq_len_M).  If a boolean tensor, True elements are
                replaced with -inf and False with 0 before addition to the
                similarity.  If not boolean, it's directly added to the
                similarity.

        Note:
        The mask here differs from `my_attn_QKV`. If mask is boolean, ~mask
        is passed to `my_attn_QKV`. If not boolean, it's passed directly.

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len_N, xq_dim).
        """
        Q = self.linear_Q(xq)
        K = self.linear_K(xk)
        V = self.linear_V(xv)

        if mask is not None and mask.dtype == torch.bool:
            # see the docstring for explanation.
            mask = ~mask
        attn = my_attn_QKV_multihead(Q, K, V, mask=mask, num_heads=self.num_heads)

        return self.final_linear(attn)


class MyTransformerEncoderLayer(torch.nn.Module):
    """Implement TransformerEncoderLayer."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 1,
        fc_hidden_dim: int = 2048,
        dropout: float = 0.0,
        norm_first: bool = True,
        batch_first: bool = True,
        bias: bool = False,
    ):
        """Initialize the transformer encoder layer.

        Args:
            d_model (int): The input and output feature dimension.
            num_heads (int): The number of attention heads. Default is 1.
            fc_hidden_dim (int): The hidden dimension for the feedforward neural
                network. Default is 2048.
            dropout (float): Not implemented in this version. Default is 0.0.
            norm_first (bool): Whether to apply layer normalization first. Default is True.
            batch_first (bool): Whether input and output tensors are batch first. Default is True.
            bias (bool): Whether to include bias in linear layers. Default is False.

        Note:
            Dropout is not implemented in this version.
            Layer normalization is applied first.
            Input and output are assumed to be in batch first format.
        """
        super().__init__()

        if dropout > 0.0:
            raise NotImplementedError
        if norm_first is not True:
            raise NotImplementedError
        if batch_first is not True:
            raise NotImplementedError
        if bias is not False:
            raise NotImplementedError

        self.self_attn = MyMultiheadAttention(
            d_model, num_heads, bias=bias, batch_first=batch_first
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_model, fc_hidden_dim, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Linear(fc_hidden_dim, d_model, bias=bias),
        )
        self.norm1 = torch.nn.LayerNorm(d_model, bias=bias)
        self.norm2 = torch.nn.LayerNorm(d_model, bias=bias)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass of the Transformer Encoder Layer.

        Args:
        - x (torch.Tensor): Input tensor with shape (batch_size, seq_len, d_model).
        - mask (torch.Tensor, optional): Mask tensor with shape (seq_len, seq_len) or None.

        Returns:
        - torch.Tensor: Transformed output tensor with shape (batch_size, seq_len, d_model).
        """
        y = self.norm1(x)
        y = self.self_attn(y, y, y, mask=mask)
        x = x + y

        y = self.norm2(x)
        y = self.mlp(y)
        x = x + y

        return x

    def load_from_pytorch_module(self, model: torch.nn.TransformerEncoderLayer):
        """Load weights from a PyTorch model"""
        self.self_attn.load_from_pytorch_module(model.self_attn)
        self.mlp.load_state_dict(
            {
                "0.weight": model.linear1.weight,
                "2.weight": model.linear2.weight,
            }
        )
        self.norm1.load_state_dict(model.norm1.state_dict())
        self.norm2.load_state_dict(model.norm2.state_dict())
