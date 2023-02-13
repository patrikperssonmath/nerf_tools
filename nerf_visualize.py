import argparse

from train.LiNerf import LiNerf
from visualize.nerf_visualize import NerfVisualize

import torch

torch.set_float32_matmul_precision("high")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')

    LiNerf.add_model_specific_args(parser)

    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--threshold", type=int, default=20)
    parser.add_argument("--name", type=str, default="door")

    args = parser.parse_args()

    device = "cuda:0"

    li_nerf = LiNerf.load_from_checkpoint(
        args.model_path, **vars(args)).to(device)

    visualize = NerfVisualize(device, li_nerf.render, [-5, 5, -5, 5, 5, 15],
                              args.samples, args.threshold, args.name)

    visualize.run()
