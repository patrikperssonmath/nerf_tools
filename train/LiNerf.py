
import pytorch_lightning as pl
import torch
import torchvision

from nerf.util.util import to_rays, to_image


def make_grid(image, output_nbr):
    """ makes a grid for tensorboard output"""

    return torchvision.utils.make_grid(image[0:output_nbr], padding=10, pad_value=1.0)


class LiNerf(pl.LightningModule):
    def __init__(self, model, lr, curv_weight, **kwargs):
        super().__init__()

        self.lr = lr
        self.curv_weight = curv_weight
        self.nerf = model

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LiNerf")

        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--curv_weight", type=float, default=1e-3)

        return parent_parser

    def validation_step(self, batch, batch_idx):

        with torch.no_grad():

            image = batch["image"]
            T = batch["T"]
            intrinsics = batch["intrinsics"]
            tn = batch["tn"]
            tf = batch["tf"]

            B, _, H, W = image.shape

            rays, tn, tf = to_rays(T, intrinsics, tn, tf, B, H, W)

            color, depth = self.render_frame(rays, tn, tf)

            color = to_image(color, B, H, W)
            depth = to_image(depth, B, H, W)

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

    def render_frame(self, rays, tn, tf, max_chunk=1024):

        with torch.no_grad():

            color_list = []
            depth_list = []

            rays = torch.split(rays, max_chunk)
            tn = torch.split(tn, max_chunk)
            tf = torch.split(tf, max_chunk)

            for idx, (ray, tn, tf) in enumerate(zip(rays, tn, tf)):

                result = self.nerf(ray, tn, tf, self.global_step)

                color_list.append(result["color_high_res"])
                depth_list.append(result["depth"])

                print(f"rendering ray batch {idx} of {len(rays)}", end="\r")

        return torch.cat(color_list, dim=0), torch.cat(depth_list, dim=0)

    def training_step(self, batch, batch_idx):

        color_gt = batch["rgb"]

        result = self.nerf(batch["ray"], batch["tn"],
                           batch["tf"], self.global_step)

        loss = (color_gt-result["color_high_res"]).abs().mean()

        if "color_low_res" in result:
            loss += (color_gt-result["color_low_res"]).abs().mean()

            loss /= 2.0

        if "curv" in result:
            loss += self.curv_weight*result["curv"].mean()

        self.log("loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
