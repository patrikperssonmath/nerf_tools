import argparse


from nerf.nerf_render import NerfRender
from nerf.network import Embedding
from nerf.network import Nerf
from nerf.util import uniform_sample

import torch

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()

    Lp = 10
    Ld = 4
    B = 1024

    render = NerfRender(Embedding(Lp),
                        Embedding(Ld),
                        Nerf(Lp, Ld))

    rays = torch.randn((B, 6))

    #t = torch.rand((B, 64, 1))

    tn = torch.zeros((B, 1))

    tf = torch.rand((B, 1))

    t = uniform_sample(tn, tf, 64)

    color, weight = render.forward(rays, t)

    print(color)

    print(weight)
