import torch

from torch import nn
import math


class Embedding(nn.Module):

    def __init__(self, L, homogeneous_projection=False) -> None:
        super().__init__()

        self.L = L

        self.homogeneous_projection = homogeneous_projection

        coeff = [2**i * math.pi for i in range(L)]

        coeff = torch.tensor(coeff, dtype=torch.float32)

        self.register_buffer("coeff", coeff)

    def forward(self, x):

        if self.homogeneous_projection:

            x = torch.cat((x, torch.ones_like(x[..., 0:1])), dim=-1)

            x = x/torch.linalg.norm(x, dim=-1, keepdim=True)

        x = x.unsqueeze(-1)

        x = torch.repeat_interleave(x, self.L, dim=-1)

        x = self.coeff*x

        sin_x = torch.sin(x)

        cos_x = torch.cos(x)

        x = torch.cat((sin_x, cos_x), dim=-1)

        return x.view(*x.shape[0:-2], -1)
