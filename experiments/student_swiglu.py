"""Baseline GPT with a parameter-matched SwiGLU MLP in each block."""

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


def build_model(config):
    model = GPT(config)
    for block in model.blocks:
        block.mlp = SwiGLU(config["width"])
        block.mlp.apply(GPT.initialize)
    return model
