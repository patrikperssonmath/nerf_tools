

from torch.utils.data import Dataset
from dataset.sfm_solution import SFMSolution
import numpy as np

class RayLoader(Dataset):
    def __init__(self, sfm_solution):
        self.rays = self.extract_rays(sfm_solution)

    def extract_rays(self, sfm_solution:SFMSolution):

        sfm_solution.unit_rescale()

        return sfm_solution.calculate_rays()

    def __len__(self):
        return self.rays.shape[-1]

    def __getitem__(self, idx):

        ray = self.rays[:, idx]

        od, tn, tf, rgb = np.split(ray, [6, 7, 8]) 

        return {"rgb":rgb, "ray":od, "tn":tn, "tf":tf}