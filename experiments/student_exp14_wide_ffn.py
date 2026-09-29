"""Post-norm RoPE GPT with an expanded SwiGLU feed-forward layer."""

import torch
from torch import nn
from torch.nn import functional as F

from model import GPT


class SwiGLU(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.gate = nn.Linear(width, hidden)
        self.value = nn.Linear(width, hidden)
        self.output = nn.Linear(hidden, width)

    def forward(self, x):
        return self.output(F.silu(self.gate(x)) * self.value(x))


class PostNormRoPEBlock(nn.Module):
    def __init__(self, original, context, head_dim):
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        self.heads = original.heads
        self.norm1, self.norm2 = original.norm1, original.norm2
        self.qkv, self.proj = original.qkv, original.proj
        self.mlp = original.mlp

        positions = torch.arange(context, dtype=torch.float32)
        inv_freq = 10000.0 ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        angles = positions[:, None] * inv_freq[None, :]
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def rotate(self, x):
        length = x.shape[-2]
        cos = self.cos[:length][None, None].to(dtype=x.dtype)
        sin = self.sin[:length][None, None].to(dtype=x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos), dim=-1
        ).flatten(-2)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).view(
            batch, length, 3, self.heads, width // self.heads
        ).permute(2, 0, 3, 1, 4)
        q, k = self.rotate(q), self.rotate(k)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = self.norm1(
            x + self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        )
        return self.norm2(x + self.mlp(x))


class PostNormRoPEGPT(GPT):
    def __init__(self, config):
        super().__init__(config)
        self.pos = None
        head_dim = config["width"] // config["heads"]
        ffn_hidden = config.get("ffn_hidden", round(8 * config["width"] / 3))
        self.blocks = nn.ModuleList(
            [PostNormRoPEBlock(block, self.context, head_dim) for block in self.blocks]
        )
        for block in self.blocks:
            block.mlp = SwiGLU(config["width"], ffn_hidden)
            block.mlp.apply(GPT.initialize)

    def features(self, ids):
        x = self.token(ids)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


def build_model(config):
    return PostNormRoPEGPT(config)
