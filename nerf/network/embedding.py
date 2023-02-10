import torch

from torch import nn
import math

class Embedding(nn.Module):

    def __init__(self, L) -> None:
        super().__init__()

        self.L = L

        coeff = [2**i * math.pi for i in range(L)]

        coeff = torch.tensor(coeff, dtype=torch.float32)

        self.register_buffer("coeff", coeff)
        
    def forward(self, x):

        x = x.unsqueeze(-1)

        x = torch.repeat_interleave(x, self.L, dim=-1)

        x = self.coeff*x

        sin_x = torch.sin(x)

        cos_x = torch.cos(x)

        x = torch.cat((sin_x, cos_x), dim=-1)

        B, N, d, L = x.shape
        
        return x.view(B,N,d*L)

