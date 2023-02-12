

from torch.utils.data import Dataset
from dataset.sfm_solution import SFMSolution
import numpy as np

class ImageLoader(Dataset):
    def __init__(self, sfm_solution:SFMSolution):
        self.images = sfm_solution.extract_images()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]