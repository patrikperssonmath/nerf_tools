import torch

from torch import nn

class Nerf(nn.Module):

    def __init__(self, Lp, Ld) -> None:
        super().__init__()

        self.dnn1 = nn.Sequential(
            nn.Linear(2*3*Lp, 256),
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
            nn.Linear(2*3*Lp+256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256+1),
            nn.ReLU()
        )

        self.density = nn.Sequential(
            nn.Linear(256, 256+1),
            nn.ReLU()
        )

        self.color = nn.Sequential(
            nn.Linear(256+2*3*Ld, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )
        

    def forward(self, x, d, data=None):

        B,N,_ = x.shape

        x = x.reshape(B*N,-1)
        d = d.reshape(B*N,-1)

        F = self.dnn1(x)

        F = torch.cat((F, x), dim=-1)

        F = self.dnn2(F)

        sigma, F = torch.split(F, [1, 256], dim=-1)

        F = torch.cat((F, d), dim=-1)

        color = self.color(F)

        return sigma.view(B,N,-1), color.view(B,N,-1)

