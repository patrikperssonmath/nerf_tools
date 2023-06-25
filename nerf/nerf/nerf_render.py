import torch

from torch import jit, nn
from typing import Dict, Optional
from nerf.nerf import NerfDensity
from nerf.nerf import NerfColor


def integrate_ray(t: torch.Tensor, sigma, c, infinite: bool = False, normalize: bool = False):

    dt = t[..., 1:, :] - t[..., :-1, :]

    # In the original imp the last distance is infinity.
    # Is this really correct since the integration is between
    # tn and tf where tf is not necessarily inf
    # practical consequence: at least the last color point will
    # receive a high weight even if the last sigma is only slightly positive.

    if infinite:
        dt = torch.cat((dt, 1e10*torch.ones_like(dt[..., 0:1, :])), dim=-2)
    else:
        dt = torch.cat((dt, torch.zeros_like(dt[..., 0:1, :])), dim=-2)

    sdt = sigma*dt

    Ti = torch.exp(-torch.cumsum(sdt, dim=-2))[..., 0:-1, :]

    Ti = torch.cat((torch.ones_like(Ti[..., 0:1, :]), Ti), dim=-2)

    alpha = (1.0 - torch.exp(-sdt))

    wi = Ti*alpha

    if normalize:

        C = wi.sum(dim=-2, keepdim=True)

        C = C.where(C > 0, torch.ones_like(C))

        wi = wi/C

    return (wi*c).sum(dim=-2), (wi*t).sum(dim=-2), wi


class NerfRender(nn.Module):

    def __init__(self, Lp, Ld, homogeneous_projection) -> None:
        super().__init__()

        self.nerf_density = NerfDensity(Lp, homogeneous_projection)

        self.nerf_color = NerfColor(Ld)

    def forward(self, ray, t):

        t, _ = torch.sort(t, dim=-2)

        o, d = torch.split(ray.unsqueeze(-2), [3, 3], dim=-1)

        x = o + t*d

        sigma, F = self.nerf_density(x)

        color = self.nerf_color(F, d)

        return integrate_ray(t, sigma, color)

    def evaluate(self, x, max_chunk=2048):

        sigma_list = []

        for pos in torch.split(x, max_chunk):

            sigma = self.evaluate_density(pos)

            sigma_list.append(sigma.cpu())

        return torch.cat(sigma_list, dim=0)

    def evaluate_density(self, x):

        return self.nerf_density(x)[0]
