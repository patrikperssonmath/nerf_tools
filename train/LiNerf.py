
import torch
import pytorch_lightning as pl

from nerf.nerf_render import NerfRender
from nerf.network import Embedding
from nerf.network import Nerf
from nerf.util import uniform_sample
from distutils.util import strtobool

class LiNerf(pl.LightningModule):
    def __init__(self, Lp, Ld, bins, **kwargs):
        super().__init__()

        self.bins = bins

        self.render = NerfRender(Embedding(Lp),
                                 Embedding(Ld),
                                 Nerf(Lp, Ld))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DeepGA")

        parser.add_argument("--Lp", type=int, default=10)
        parser.add_argument("--Ld", type=int, default=4)
        parser.add_argument("--bins", type=int, default=64)

        #parser.add_argument("--sample", type=strtobool, default=False) 

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.

        color_gt = batch["color_gt"]
        rays = batch["rays"]
        tn = batch["tn"]
        tf = batch["tf"]
        
        t = uniform_sample(tn, tf, self.bins)

        color, _ = self.render.forward(rays, t)

        return (color_gt-color).square().mean()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
