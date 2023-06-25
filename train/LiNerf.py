
import torch
import pytorch_lightning as pl

from nerf.nerf.nerf import Nerf

from nerf.util.util import uniform_sample, generate_rays, resample
from distutils.util import strtobool
import torchvision


def make_grid(image, output_nbr):
    """ makes a grid for tensorboard output"""

    return torchvision.utils.make_grid(image[0:output_nbr], padding=10, pad_value=1.0)


class LiNerf(pl.LightningModule):
    def __init__(self, lr, **kwargs):
        super().__init__()

        self.lr = lr
        self.nerf = Nerf(**kwargs)

        """
        nerf = Nerf(**kwargs).to("cuda:0")
        ray = torch.randn((1024, 6), device="cuda:0", dtype=torch.float32)
        tn = torch.rand((1024, 1), device="cuda:0", dtype=torch.float32)
        tf = tn+torch.rand((1024, 1), device="cuda:0", dtype=torch.float32)

        self.nerf = torch.jit.trace(nerf, (ray, tn, tf))
        """

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LiNerf")

        parser.add_argument("--lr", type=float, default=1e-3)

        Nerf.add_model_specific_args(parent_parser)

        return parent_parser

    def validation_step(self, batch, batch_idx):

        with torch.no_grad():

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

            color, depth = self.render_frame(rays, tn, tf)

            color = color.reshape(B, H*W, -1).permute(0,
                                                      2, 1).view(B, -1, H, W)
            depth = depth.reshape(B, H*W, -1).permute(0,
                                                      2, 1).view(B, -1, H, W)

            self.logger.experiment.add_image(
                f"img_rendered", make_grid(color, 4), self.global_step + batch_idx)

            depth_max = torch.max(depth.view(B, -1, H*W),
                                  dim=-1, keepdim=True)[0].view(-1, 1, 1, 1)
            depth_min = torch.min(depth.view(B, -1, H*W),
                                  dim=-1, keepdim=True)[0].view(-1, 1, 1, 1)

            depth = (depth-depth_min)/(depth_max-depth_min)

            self.logger.experiment.add_image(
                f"depth_rendered", make_grid(depth, 4), self.global_step + batch_idx)

            self.logger.experiment.add_image(
                f"img_gt", make_grid(image, 4), self.global_step + batch_idx)

    def render_frame(self, rays, tn, tf, max_chunk=2048):

        with torch.no_grad():

            color_list = []
            depth_list = []

            rays = torch.split(rays, max_chunk)
            tn = torch.split(tn, max_chunk)
            tf = torch.split(tf, max_chunk)

            for idx, (ray, tn, tf) in enumerate(zip(rays, tn, tf)):

                result = self.nerf(ray, tn, tf)

                color_list.append(result[0])
                depth_list.append(result[1])

                print(f"rendering ray batch {idx} of {len(rays)}", end="\r")

        return torch.cat(color_list, dim=0), torch.cat(depth_list, dim=0)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.

        color_gt = batch["rgb"]

        result = self.nerf(batch["ray"], batch["tn"], batch["tf"])

        loss = (color_gt-result[-1]).abs().mean()

        loss += (color_gt-result[0]).abs().mean()

        loss /= 2.0

        self.log("loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
