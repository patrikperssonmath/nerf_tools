import numpy as np
import matplotlib.pyplot as plt
import torch
from nerf_render import NerfRender
import mcubes
import os


class NerfVisualizer:
    def __init__(self, device, nerf_render: NerfRender, limits, nbr_samples, threshold, name="") -> None:
        self.device = device
        self.nerf_render = nerf_render
        self.limits = limits
        self.nbr_samples = nbr_samples
        self.threshold = threshold
        self.name = name

    def run(self):

        with torch.no_grad():

            x_min, x_max, y_min, y_max, z_min, z_max = self.limits

            x = np.linspace(x_min, x_max, self.nbr_samples)
            y = np.linspace(y_min, y_max, self.nbr_samples)
            z = np.linspace(z_min, z_max, self.nbr_samples)

            points = np.stack(np.meshgrid(x, y, z), -1)

            points = torch.tensor(points, device=self.device,
                                  dtype=torch.float32)

            H, W, D, C = points.shape

            sigma, _ = self.nerf_render.evaluate(points.view(H*W*D, 1, C), None)

            sigma = sigma.view(H, W, D)

            sigma = sigma.detach().cpu().numpy()

            print('fraction occupied', np.mean(sigma > self.threshold))

            vertices, triangles = mcubes.marching_cubes(sigma, self.threshold)

            scale = np.array(
                [x_max-x_min, y_max-y_min, z_max-z_min])/(self.nbr_samples-1)
            scale = scale[None, :]

            offset = np.array([x_min, y_min, z_min])
            offset = offset[None, :]

            vertices = scale*vertices + offset

            print('done', vertices.shape, triangles.shape)

            model_folder = "model3d/example"

            if not os.path.exists(model_folder):
                os.makedirs(model_folder)

            mcubes.export_mesh(vertices, triangles, os.path.join(
                model_folder, "{}_{}.dae".format(self.name, self.nbr_samples)), self.name)

            plt.hist(sigma.ravel(), log=True)
            plt.show()
