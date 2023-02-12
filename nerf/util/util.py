
import torch


@torch.jit.script
def uniform_sample(tn, tf, N: int):

    dt = (tf - tn)/N

    dt = dt.unsqueeze(-1)

    i = torch.arange(0, N, device=tf.device,
                     dtype=tf.dtype).unsqueeze(0).unsqueeze(-1)

    u = torch.rand_like(i)

    t = tn.unsqueeze(-1) + dt*i + dt*u

    return t


def generate_grid(H, W, **kwargs):

    grid_x, grid_y = torch.meshgrid(torch.arange(0, W, **kwargs),
                                    torch.arange(0, H, **kwargs),
                                    indexing='xy')

    return torch.stack((grid_x, grid_y)).unsqueeze(0)


def generate_rays(T, intrinsics, H, W):

    grid = generate_grid(H, W, device=intrinsics.device,
                         dtype=intrinsics.dtype)

    R = T[:, 0:3, 0:3]
    t = T[:, 0:3, 3:]

    focal, prinicpal = torch.split(intrinsics.view(-1, 4, 1, 1), [2, 2], dim=1)

    d = (grid - prinicpal)/focal

    d = torch.cat((d, torch.ones_like(d[:, 0:1])), dim=1)

    B, C, H, W = d.shape

    d = (torch.transpose(R, -2, -1) @ d.view(B, C, H*W)).view(B, C, H, W)

    d = d / torch.linalg.norm(d, dim=1, keepdim=True)

    o = -torch.transpose(R, -2, -1) @ t

    o = o.view(-1, 3, 1, 1).expand(-1, -1, H, W)

    ray = torch.cat((o, d), dim=1)

    return ray
