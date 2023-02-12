import torch

from torch import jit, nn
from typing import Dict, Optional


class NerfRender(nn.Module):

    def __init__(self, pos_embedding, direction_embedding, nerf) -> None:
        super().__init__()

        self.pose_embedding = pos_embedding
        self.direction_embedding = direction_embedding
        self.nerf = nerf

    def forward(self, ray, t, data: Optional[Dict[str, torch.Tensor]] = None, sorted: bool = False):

        if not sorted:
            t, _ = torch.sort(t, dim=-2)

        o, d = torch.split(ray.unsqueeze(-2), [3, 3], dim=-1)

        x = o + t[..., 1:, :]*d

        x = self.pose_embedding(x)

        _, N, _ = x.shape

        d = self.direction_embedding(d).expand(-1, N, -1)

        sigma, color = self.nerf(x, d, data)

        return self.integrate(t, sigma, color)

    def integrate(self, t, sigma, c):

        dt = t[..., 1:, :] - t[..., 0:-1, :]

        sdt = sigma*dt

        Ti = torch.exp(-torch.cumsum(sdt, dim=-2))[..., 0:-1, :]

        Ti = torch.cat((torch.ones_like(Ti[..., 0:1, :]), Ti), dim=-2)

        alpha = (1.0 - torch.exp(-sdt))

        wi = Ti*alpha

        return (wi*c).sum(dim=-2), (wi*t[..., 1:, :]).sum(dim=-2), wi
