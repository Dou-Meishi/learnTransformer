# ---
# jupyter:
#   jupytext:
#     split_at_heading: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ## Benchmark scaled dot attention

# %%
from transformer import my_attn_QKV
print("="*60)
print("Benchmark scaled dot attention")

# %%
print("\nwithout mask case")

Q = torch.randn(32, 10, 3)
K = torch.randn(32, 20, 3)
V = torch.randn(32, 20, 5)

print("Input shape")
print(f"\tQ: {Q.shape}")
print(f"\tK: {K.shape}")
print(f"\tV: {V.shape}")

attn = my_attn_QKV(Q, K, V)
expected_attn = F.scaled_dot_product_attention(Q, K, V)

print("Output shape")
print(f"\t   {attn.shape}")
print(f"Matched expected shape: {attn.shape == expected_attn.shape}")

print("Error of output", (attn - expected_attn).abs().max().item())
print(f"Matched expected output: {torch.allclose(attn, expected_attn, atol=1e-6)}")

# %%
print("\nwith mask case")

Q = torch.randn(32, 10, 3)
K = torch.randn(32, 20, 3)
V = torch.randn(32, 20, 5)
# generate a random boolean matrix as the mask
mask = torch.randn(10, 20) > 0


print("Input shape")
print(f"\tQ: {Q.shape}")
print(f"\tK: {K.shape}")
print(f"\tV: {V.shape}")

attn = my_attn_QKV(Q, K, V, mask=mask)
expected_attn = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)

print("Output shape")
print(f"\t   {attn.shape}")
print(f"Matched expected shape: {attn.shape == expected_attn.shape}")

print("Error of output", (attn - expected_attn).abs().max().item())
print(f"Matched expected output: {torch.allclose(attn, expected_attn, atol=1e-6)}")
