from train.LiNerf import LiNerf
from dataset.colmap_solution import ColmapSolution
from dataset.ray_loader import RayLoader
from dataset.image_loader import ImageLoader

from torch.utils.data import DataLoader
import torch


class Trainer:
    def __init__(self, trainer, model_path, dataset_path, batch, num_workers, **kwargs) -> None:
        self.model = LiNerf(**kwargs)

        if model_path:
            self.model_path = model_path
        else:
            self.model_path = None
        
        self.dataset_path = dataset_path
        self.batch = batch
        self.num_workers = num_workers

        self.trainer = trainer

    @staticmethod
    def add_model_specific_args(parent_parser):

        parser = parent_parser.add_argument_group("Trainer")

        parser.add_argument("--model_path", type=str, default="")
        parser.add_argument("--dataset_path", type=str,
                            default="/database/colmap_test")
        parser.add_argument("--batch", type=int, default=64)
        parser.add_argument("--num_workers", type=int, default=16)

        LiNerf.add_model_specific_args(parent_parser)

        return parent_parser

    def load_dataset(self):

        return RayLoader(ColmapSolution(self.dataset_path, 0)), ImageLoader(ColmapSolution(self.dataset_path, 0))

    def run(self):

        train_dataset, test_dataset = self.load_dataset()

        train_loader = DataLoader(train_dataset,
                                  self.batch,
                                  True,
                                  num_workers=self.num_workers)

        test_loader = DataLoader(test_dataset,
                                 1,
                                 False,
                                 num_workers=self.num_workers)

        self.trainer.fit(model=self.model,
                         train_dataloaders=train_loader,
                         val_dataloaders=test_loader,
                         ckpt_path=self.model_path)
