import torch

from torch import jit, nn
from typing import Dict, Optional


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


class NerfRender(jit.ScriptModule):

    def __init__(self, pos_embedding, direction_embedding, nerf, max_batch=2**14) -> None:
        super().__init__()

        self.pose_embedding = pos_embedding
        self.direction_embedding = direction_embedding
        self.nerf = nerf
        self.max_batch = max_batch

    @jit.script_method
    def forward(self, ray, t, data: Optional[Dict[str, torch.Tensor]] = None, sorted_t: bool = True):

        if not sorted_t:
            t, _ = torch.sort(t, dim=-2)

        o, d = torch.split(ray.unsqueeze(-2), [3, 3], dim=-1)

        x = o + t*d

        x = self.pose_embedding(x)

        _, N, _ = x.shape

        d = self.direction_embedding(d).expand(-1, N, -1)

        sigma, color = self.nerf(x, d, data)

        return integrate_ray(t, sigma, color)

    @jit.script_method
    def evaluate(self, x, d, data: Optional[Dict[str, torch.Tensor]] = None):

        v = torch.cat((x, d), dim=-1)

        B = v.shape[0]

        N = B // self.max_batch

        rem = B % N

        color_list = []

        sigma_list = []

        for i in range(N):

            sigma, color = self.evaluate_batch(
                v[i*self.max_batch:(i+1)*self.max_batch], data)

            color_list.append(color.cpu())
            sigma_list.append(sigma.cpu())

        if rem > 0:

            sigma, color = self.evaluate_batch(
                v[N*self.max_batch:], data)

            color_list.append(color.cpu())
            sigma_list.append(sigma.cpu())

        return torch.cat(sigma_list, dim=0), torch.cat(color_list, dim=0)

    def evaluate_batch(self, pos_dir, data: Optional[Dict[str, torch.Tensor]] = None):

        x, d = torch.split(pos_dir, [3, 3], dim=-1)

        x = self.pose_embedding(x)
        d = self.direction_embedding(d)

        return self.nerf(x, d, data)
