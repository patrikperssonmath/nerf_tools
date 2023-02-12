
import torch
import pytorch_lightning as pl

from nerf.nerf_render import NerfRender
from nerf.network import Embedding
from nerf.network import Nerf
from nerf.util import uniform_sample, generate_rays
from distutils.util import strtobool
import torchvision


def make_grid(image, output_nbr):
    """ makes a grid for tensorboard output"""

    return torchvision.utils.make_grid(image[0:output_nbr], padding=10, pad_value=1.0)


class LiNerf(pl.LightningModule):
    def __init__(self, Lp, Ld, bins, max_render_batch_power, homogeneous_projection, **kwargs):
        super().__init__()

        self.bins = bins

        self.max_render_batch_power = max_render_batch_power

        self.render = torch.jit.script(NerfRender(Embedding(Lp, homogeneous_projection),
                                                  Embedding(Ld),
                                                  Nerf(Lp, Ld, homogeneous_projection)))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Nerf")

        parser.add_argument("--Lp", type=int, default=10)
        parser.add_argument("--Ld", type=int, default=4)
        parser.add_argument("--bins", type=int, default=64)
        parser.add_argument("--max_render_batch_power", type=int, default=14)

        parser.add_argument("--homogeneous_projection",
                            type=strtobool, default=False)

        return parent_parser

    def validation_step(self, batch, batch_idx):

        #return

        image = batch["image"]
        T = batch["T"]
        intrinsics = batch["intrinsics"]
        tn = batch["tn"]
        tf = batch["tf"]

        B, C, H, W = image.shape

        rays = generate_rays(T, intrinsics, H, W)

        tn = tn.view(B, 1, 1, 1).expand(-1, 1, H, W)
        tf = tf.view(B, 1, 1, 1).expand(-1, 1, H, W)

        rays = rays.view(B, -1, H*W).permute(0, 2, 1).reshape(B*H*W, -1)
        tn = tn.view(B, -1, H*W).permute(0, 2, 1).reshape(B*H*W, -1)
        tf = tf.view(B, -1, H*W).permute(0, 2, 1).reshape(B*H*W, -1)

        t = uniform_sample(tn, tf, self.bins)

        #color, _ = self.render.forward(rays, t)

        color, depth = self.render_frame(rays, t)

        color = color.reshape(B, H*W, -1).permute(0, 2, 1).view(B, -1, H, W)
        depth = depth.reshape(B, H*W, -1).permute(0, 2, 1).view(B, -1, H, W)

        self.logger.experiment.add_image(
            f"img_rendered", make_grid(color, 4), batch_idx)

        depth_max = torch.max(depth.view(B, -1, H*W),
                              dim=-1, keepdim=True)[0].view(-1, 1, 1, 1)
        depth_min = torch.min(depth.view(B, -1, H*W),
                              dim=-1, keepdim=True)[0].view(-1, 1, 1, 1)

        depth = (depth-depth_min)/(depth_max-depth_min)

        self.logger.experiment.add_image(
            f"depth_rendered", make_grid(depth, 4), batch_idx)

        self.logger.experiment.add_image(
            f"img_gt", make_grid(image, 4), batch_idx)

    def render_frame(self, rays, t):

        B, _ = rays.shape

        max_batch = 2**self.max_render_batch_power

        N = B // max_batch

        rem = B % N

        color_list = []

        depth_list = []

        for i in range(N):

            color, depth, _ = self.render.forward(
                rays[i*max_batch:(i+1)*max_batch], t[i*max_batch:(i+1)*max_batch])

            color_list.append(color)
            depth_list.append(depth)

        if rem > 0:

            color, depth, _ = self.render.forward(
                rays[N*max_batch:], t[N*max_batch:])

            color_list.append(color)
            depth_list.append(depth)

        return torch.cat(color_list, dim=0), torch.cat(depth_list, dim=0)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.

        color_gt = batch["rgb"]
        rays = batch["ray"]
        tn = batch["tn"]
        tf = batch["tf"]

        t = uniform_sample(tn, tf, self.bins)

        color, _, _ = self.render.forward(rays, t)

        loss = (color_gt-color).square().mean()

        self.log("loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
