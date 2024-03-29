
from distutils.util import strtobool

import torch
from torch import nn

from nerf.nerf.nerf_render import NerfRender
from nerf.util.util import resample, uniform_sample


class Nerf(nn.Module):
    def __init__(self, Lp, Ld, low_res_bins, high_res_bins, homogeneous_projection, **kwargs) -> None:
        super().__init__()

        self.low_res_bins = low_res_bins

        self.high_res_bins = high_res_bins

        self.render = NerfRender(Lp, Ld, homogeneous_projection)

        self.render_low_res = NerfRender(Lp, Ld, homogeneous_projection)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Nerf")

        parser.add_argument("--Lp", type=int, default=10)
        parser.add_argument("--Ld", type=int, default=4)
        parser.add_argument("--low_res_bins", type=int, default=64)
        parser.add_argument("--high_res_bins", type=int, default=128)

        parser.add_argument("--homogeneous_projection",
                            type=strtobool, default=True)

        return parent_parser

    def forward(self, rays, tn, tf, step):

        t = uniform_sample(tn, tf, self.low_res_bins)

        # do one round to find out important sampling regions

        color_low_res, _, w, t = self.render_low_res.forward(rays, t)

        with torch.no_grad():

            # sample according to w
            t_resamp = resample(w, t, self.high_res_bins)

        t_resamp = torch.cat((t, t_resamp), dim=1)

        color_high_res, depth, _, _ = self.render.forward(rays, t_resamp)

        return {"color_high_res": color_high_res, "depth": depth, "color_low_res": color_low_res}
