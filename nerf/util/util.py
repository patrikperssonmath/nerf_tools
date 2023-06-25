
import torch

def ray_to_points(ray, t):

    o, d = torch.split(ray.unsqueeze(-2), [3, 3], dim=-1)

    x = o + t*d

    return x, d

def where(mask, x, y):

    mask = mask.float()

    return mask*x + (1.0-mask)*y


def linspace(tn, tf, N):

    dt = (tf - tn).unsqueeze(-1)
    tn = tn.unsqueeze(1)

    i = torch.linspace(0, 1, N,
                       device=tn.device,
                       dtype=tn.dtype).unsqueeze(0).unsqueeze(-1)

    return tn + dt*i


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


def resample(w, t, N: int):

    t1 = t[:, :-1]
    t2 = t[:, 1:]

    w1 = w[:, :-1]
    w2 = w[:, 1:]

    delta_t = (t2-t1)

    k = (w2-w1)/delta_t.where(delta_t > 0, torch.ones_like(delta_t))
    m = w1 - k*t1

    c = 0.5*k*delta_t**2 + m*delta_t

    # can become negative due to numerical errors. Must be positive.
    c = c.abs()

    c = torch.cat((torch.zeros_like(c[:, 0:1]), c), dim=1)

    w_cdf = torch.cumsum(c, dim=1)

    C = w_cdf[:, -1:]

    w_cdf = w_cdf / C.where(C > 0, torch.ones_like(C))

    B, S = w_cdf.shape[0:2]

    u = torch.rand((B, N), device=w.device, dtype=w.dtype)

    idx = torch.searchsorted(w_cdf.squeeze(-1), u,
                             right=True).unsqueeze(-1).clamp(0, S-1)

    u = u.unsqueeze(-1)

    w1 = torch.gather(w_cdf, 1, (idx - 1).clamp(0, S-1))
    t1 = torch.gather(t, 1, (idx - 1).clamp(0, S-1))

    w2 = torch.gather(w_cdf, 1, idx)
    t2 = torch.gather(t, 1, idx)

    # k = ((w2-w1)/(t2-t1))
    # m = w1 - k*t1

    # w = k*tu + m

    # tu = (w-m)/k

    delta_t = (t2-t1)
    delta_t = delta_t.where(delta_t > 0, torch.ones_like(delta_t))

    k = (w2-w1)/delta_t
    m = w1 - k*t1
    tu = (u-m)/k.where(k > 0, torch.ones_like(delta_t))

    return tu.clamp(t1, t2)


def resample_old(w, t, N: int, R: int):

    w = torch.nn.functional.interpolate(
        w.permute(0, 2, 1), R, mode="linear", align_corners=True).permute(0, 2, 1)

    t = torch.nn.functional.interpolate(
        t.permute(0, 2, 1), R, mode="linear", align_corners=True).permute(0, 2, 1)

    w_cdf = torch.cumsum(w, dim=1)

    C = w_cdf[:, -1:]

    w_cdf = w_cdf / C.where(C > 0, torch.ones_like(C))

    B, S = w.shape[0:2]

    u = torch.rand((B, N), device=w.device, dtype=w.dtype)

    idx = torch.searchsorted(w_cdf.squeeze(-1), u,
                             right=True).unsqueeze(-1).clamp(0, S-1)

    u = u.unsqueeze(-1)

    w1 = torch.gather(w_cdf, 1, (idx - 1).clamp(0, S-1))
    t1 = torch.gather(t, 1, (idx - 1).clamp(0, S-1))

    w2 = torch.gather(w_cdf, 1, idx)
    t2 = torch.gather(t, 1, idx)

    # k = ((w2-w1)/(t2-t1))
    # m = w1 - k*t1

    # w = k*tu + m

    # tu = (w-m)/k

    delta_t = (t2-t1)
    delta_t = delta_t.where(delta_t > 0, torch.ones_like(delta_t))

    k = (w2-w1)/delta_t
    m = w1 - k*t1
    tu = (u-m)/k.where(k > 0, torch.ones_like(delta_t))

    return tu.clamp(t1, t2)


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
