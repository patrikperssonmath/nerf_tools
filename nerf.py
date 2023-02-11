import argparse

from train.trainer import Trainer

import pytorch_lightning as pl

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')

    pl.Trainer.add_argparse_args(parser)

    Trainer.add_model_specific_args(parser)

    args = parser.parse_args()

    pl_trainer = pl.Trainer.from_argparse_args(args)

    trainer = Trainer(trainer=pl_trainer, **vars(args))

    trainer.run()    
