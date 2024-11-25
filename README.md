# What happens in A Transformer Layer

A simple implementation of PyTorch's `TransformerEncoderLayer`. See my [post](https://dou-meishi.github.io/org-blog/2024-10-30-LearnTransformer/notes.html) for a brief introduction.

This repo contains equivalent implementations of

- `nn.functional.scaled_dot_product_attention` (my version: `my_attn_QKV` in `./transformers.py`);

- `nn.MultiheadAttention` (my version: `MyMultiheadAttention` in `./transformers.py`);

- `nn.TransformerEncoderLayer` (my version: `MyTransformerEncoderLayer` in `./transformers.py`);

- `nn.BatchNorm1d` (my version: `MyBatchNorm1d` in `./normalization.py`);

- `nn.BatchNorm2d` (my version: `MyBatchNorm2d` in `./normalization.py`);

- `nn.LayerNorm` (my version: `MyLayerNorm` in `./normalization.py`).

Run the script `./test.py` to see the benchmark results.
