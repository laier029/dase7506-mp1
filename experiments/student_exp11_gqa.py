"""Post-norm RoPE-SwiGLU GPT with grouped-query attention (GQA)."""

import torch
from torch import nn
from torch.nn import functional as F

from model import GPT


class SwiGLU(nn.Module):
    def __init__(self, width):
        super().__init__()
        hidden = round(8 * width / 3)
        self.gate = nn.Linear(width, hidden)
        self.value = nn.Linear(width, hidden)
        self.output = nn.Linear(hidden, width)

    def forward(self, x):
        return self.output(F.silu(self.gate(x)) * self.value(x))


class GQAPostNormRoPEBlock(nn.Module):
    def __init__(self, original, context, width, query_heads, kv_heads):
        super().__init__()
        if width % query_heads:
            raise ValueError("width must be divisible by query_heads")
        if query_heads % kv_heads:
            raise ValueError("query_heads must be divisible by kv_heads")

        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.head_dim = width // query_heads
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")

        self.norm1, self.norm2 = original.norm1, original.norm2
        self.proj = original.proj

        kv_width = kv_heads * self.head_dim
        self.qkv = nn.Linear(width, width + 2 * kv_width)
        self.qkv.apply(GPT.initialize)

        self.mlp = SwiGLU(width)
        self.mlp.apply(GPT.initialize)

        positions = torch.arange(context, dtype=torch.float32)
        inv_freq = 10000.0 ** (
            -torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim
        )
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
        kv_width = self.kv_heads * self.head_dim
        q, k, v = self.qkv(x).split((width, kv_width, kv_width), dim=-1)

        q = q.view(batch, length, self.query_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.rotate(q), self.rotate(k)

        # PyTorch CPU SDPA does not provide a portable GQA kernel, so expand each
        # KV head only for the attention calculation. The learned KV projections
        # and checkpoint still contain kv_heads rather than query_heads.
        repeats = self.query_heads // self.kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        x = self.norm1(x + self.proj(attended))
        return self.norm2(x + self.mlp(x))


class GQAPostNormRoPEGPT(GPT):
    def __init__(self, config):
        super().__init__(config)
        self.pos = None
        width = config["width"]
        query_heads = config["heads"]
        kv_heads = config.get("kv_heads", query_heads)
        self.blocks = nn.ModuleList(
            [
                GQAPostNormRoPEBlock(
                    block, self.context, width, query_heads, kv_heads
                )
                for block in self.blocks
            ]
        )

    def features(self, ids):
        x = self.token(ids)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


def build_model(config):
    return GQAPostNormRoPEGPT(config)
