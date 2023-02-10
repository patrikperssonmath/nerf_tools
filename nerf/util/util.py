
import torch

def uniform_sample(tn, tf, N):

    dt = (tf - tn)/N

    dt = dt.unsqueeze(-1)

    i = torch.arange(0, N, device=tf.device, dtype=tf.dtype).unsqueeze(0).unsqueeze(-1)

    B,_,_ = dt.shape

    u = torch.rand((B, N, 1), device=tf.device, dtype=tf.dtype)

    t = tn.unsqueeze(-1) + dt*i*u

    return t


