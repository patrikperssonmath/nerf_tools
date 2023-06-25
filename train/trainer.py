
from dataset.colmap_solution import ColmapSolution
from dataset.ray_loader import RayLoader
from dataset.image_loader import ImageLoader

from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, trainer, li_model, model_path, dataset_path, batch, num_workers, width, height, **kwargs) -> None:

        self.model = li_model

        if model_path:
            self.model_path = model_path
        else:
            self.model_path = None

        self.dataset_path = dataset_path
        self.batch = batch
        self.num_workers = num_workers

        self.trainer = trainer
        self.height = height
        self.width = width

    @staticmethod
    def add_model_specific_args(parent_parser):

        parser = parent_parser.add_argument_group("Trainer")

        parser.add_argument("--model_path", type=str, default="")
        parser.add_argument("--dataset_path", type=str, default="")
        parser.add_argument("--batch", type=int, default=64)
        parser.add_argument("--num_workers", type=int, default=16)
        parser.add_argument("--width", type=int, default=320)
        parser.add_argument("--height", type=int, default=320)

        return parent_parser

    def run(self):

        train_dataset = RayLoader(ColmapSolution(
            self.dataset_path, 0, [self.height, self.width]))

        test_dataset = ImageLoader(ColmapSolution(
            self.dataset_path, 0, [self.height, self.width]))

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
