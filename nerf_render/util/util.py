
import torch


def uniform(a, b):

    return a + (b-a)*torch.rand_like(a)


def uniform_sample(tn, tf, N: int):

    dt = (tf - tn).unsqueeze(-1)/N
    tn = tn.unsqueeze(-1)

    i = torch.arange(1, N+1, device=tn.device,
                     dtype=tn.dtype).unsqueeze(0).unsqueeze(-1)

    a = tn + (i-1)*dt
    b = tn + i*dt

    return uniform(a, b)


def resample(w, t, N: int, R: int):

    w = torch.nn.functional.interpolate(
        w.permute(0, 2, 1), R, mode="linear", align_corners=True).permute(0, 2, 1)
    t = torch.nn.functional.interpolate(
        t.permute(0, 2, 1), R, mode="linear", align_corners=True).permute(0, 2, 1)

    w_cdf = torch.cumsum(w, dim=1)

    C = w_cdf[:, -1:]

    w_cdf = w_cdf / C.where(C > 0, torch.ones_like(C))

    B = w.shape[0]

    u = torch.rand((B, N), device=w.device, dtype=w.dtype)

    idx = torch.searchsorted(w_cdf.squeeze(-1), u,
                             right=True).unsqueeze(-1).clamp(0, R-1)

    return torch.gather(t, 1, idx)


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
