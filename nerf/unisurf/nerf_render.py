import torch

from torch import nn
from nerf.unisurf import NerfDensity
from nerf.unisurf import NerfColor


class DensityActivation(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        return self.sigmoid(-10.0*x)


def integrate_ray_old(t, sigma, color):

    alpha = torch.cat((torch.ones_like(sigma[..., 0:1, :]), 1.0-sigma), dim=-2)

    wi = sigma*torch.cumprod(alpha, dim=-2)[..., :-1, :]

    return (wi*color).sum(dim=-2), (wi*t).sum(dim=-2), wi


def integrate_ray(t, sigma, color):

    alpha_log = torch.cat(
        (torch.ones_like(sigma[..., 0:1, :]), 1.0-sigma+1e-6), dim=-2).log()

    wi = torch.exp((sigma+1e-6).log() + torch.cumsum(alpha_log, dim=-2)[..., :-1, :])

    return (wi*color).sum(dim=-2), (wi*t).sum(dim=-2), wi


class NerfRender(nn.Module):

    def __init__(self, Lp, Ld, homogeneous_projection) -> None:
        super().__init__()

        self.nerf_density = NerfDensity(Lp, homogeneous_projection)

        self.nerf_color = NerfColor(Ld)

        self.density_activation = DensityActivation()

    def forward(self, ray, t):

        t, _ = torch.sort(t, dim=-2)

        o, d = torch.split(ray.unsqueeze(-2), [3, 3], dim=-1)

        x = o + t*d

        sigma, h, n = self.evaluate_density_gradient(x)

        color = self.evaluate_color(x, n, h, d)

        return integrate_ray(t, sigma, color)

    def evaluate_density_gradient(self, p):

        with torch.enable_grad():
            p.requires_grad_(True)
            y, h = self.nerf_density(p)
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            n = torch.autograd.grad(
                outputs=y,
                inputs=p,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
                allow_unused=True)[0]

        n = n / (torch.norm(n, dim=-1, keepdim=True)+1e-6)

        return self.density_activation(y), h, n

    def evaluate_color(self, x, n, h, d):

        return self.nerf_color(x, n, h, d)

    def evaluate(self, x, max_chunk=2048):

        sigma_list = []

        for pos in torch.split(x, max_chunk):

            sigma = self.evaluate_density(pos)

            sigma_list.append(sigma.cpu())

        return torch.cat(sigma_list, dim=0)

    def evaluate_density(self, x):

        return self.density_activation(self.nerf_density(x)[0])

    def evaluate_ray_density(self, ray, t):

        o, d = torch.split(ray.unsqueeze(-2), [3, 3], dim=-1)

        x = o + t*d

        return self.evaluate_density(x)

    def evaluate_normal(self, x):

        _, _, n = self.evaluate_density_gradient(x)

        return n
