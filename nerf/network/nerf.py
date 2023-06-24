import torch

from torch import jit, nn
from typing import Dict, Optional
from nerf.network.noise_layer import Noise


class Nerf(nn.Module):

    def __init__(self, Lp, Ld, homogeneous_projection=False) -> None:
        super().__init__()

        pos_dim = 3

        if homogeneous_projection:
            pos_dim += 1

        self.dnn1 = nn.Sequential(
            nn.Linear(2*pos_dim*Lp, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.dnn2 = nn.Sequential(
            nn.Linear(2*pos_dim*Lp+256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.density = nn.Sequential(
            nn.Linear(256, 1),
            Noise(1e0),
            nn.ReLU()
        )

        self.color = nn.Sequential(
            nn.Linear(256+2*3*Ld, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )

    def forward(self, x, d):

        F = self.dnn1(x)

        F = torch.cat((F, x), dim=-1)

        F = self.dnn2(F)

        sigma = self.density(F)

        F = torch.cat((F, d), dim=-1)

        color = self.color(F)

        return sigma, color
