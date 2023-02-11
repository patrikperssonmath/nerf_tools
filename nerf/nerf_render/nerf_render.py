import torch

from torch import nn

class NerfRender(nn.Module):

    def __init__(self, pos_embedding, direction_embedding, nerf) -> None:
        super().__init__()

        self.pose_embedding = pos_embedding
        self.direction_embedding = direction_embedding
        self.nerf = nerf

    def forward(self, ray, t, data=None, sorted=False):

        o, d = torch.split(ray.unsqueeze(-2), [3, 3], dim=-1)
        
        x = o + t[...,1:,:]*d
        
        x = self.pose_embedding(x)

        _, N, _ = x.shape

        d = self.direction_embedding(d).expand(-1, N, -1)

        sigma, color = self.nerf(x, d, data)

        return self.integrate(t, sigma, color, sorted)

    def integrate(self, t, sigma, c, sorted = False):

        if not sorted:
            t, _ = torch.sort(t, dim=-2)

        dt = t[..., 1:, :] - t[..., 0:-1, :]

        sdt = sigma*dt

        Ti = torch.exp(-torch.cumsum(sdt, dim = -2))[..., 0:-1, :]

        Ti = torch.cat((torch.ones_like(Ti[...,0:1, :]), Ti), dim=-2)

        alpha = (1.0 - torch.exp(-sdt))

        wi = Ti*alpha
        
        return (wi*c).sum(dim=-2), wi

