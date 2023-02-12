import numpy as np

import torch


class NerfVisualize:
    def __init__(self, device, model, limits, nbr_samples) -> None:
        self.device = device
        self.model = model
        self.limits = limits
        self.nbr_samples = nbr_samples

    def run(self):

        x_min, x_max, y_min, y_max, z_min, z_max = self.limits

        x = np.linspace(x_min, x_max, self.nbr_samples+1)
        y = np.linspace(y_min, y_max, self.nbr_samples+1)
        z = np.linspace(z_min, z_max, self.nbr_samples+1)

        points = np.stack((x, y, z))
        directions = np.zeros_like(points)

        points = torch.tensor(points, device=self.device,
                              dtype=torch.float32).permute(1, 0).unsqueeze(0)
        directions = torch.tensor(
            directions, device=self.device, dtype=torch.float32).permute(1, 0).unsqueeze(0)

        
