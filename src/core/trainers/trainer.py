"""
File: trainer.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: trainer code
"""
import torch
from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from core.dataloaders.mit_dataloader import MitData
from core.network.custom_nets import FIP


class LitModel(pl.LightningModule):

    """Docstring for LitModel. """

    def __init__(self,
                 data_dir: str,
                 batch_size: int,
                 num_workers: int = 6,
                 lr: float = 1e-1,
                 **kwargs) -> None:
        """
        @lr: learning rate
        """
        super().__init__()

        __import__('pdb').set_trace()
        self.model = FIP()
        self.save_hyperparameters()

    def forward(self, x):
        """forward function for litmodel
        """
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adagrad(self.parameters(),
                                   lr=self.hparams.lr,
                                   weight_decay=5e-8,
                                   eps=1e-6)

    def validation_step(self, batch, batch_idx):
        __import__('pdb').set_trace()
        x, y = batch
        y_hat = self(x)
        loss = y - y_hat
        loss = 0
        return {'loss': loss}

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = y - y_hat
        loss = 0
        return {'loss': loss}

    def setup(self, stage=None):

        train_dir = Path(self.hparams.data_dir).joinpath('train')
        val_dir = Path(self.hparams.data_dir).joinpath('val')

        train_dir_1 = train_dir.joinpath('set-1')
        self.mit_train_1 = MitData(train_dir_1, is_train=True)

        train_dir_2 = train_dir.joinpath('set-2')
        self.mit_train_2 = MitData(train_dir_2, is_train=True)

        self.mit_val = MitData(val_dir, is_train=False)

    def train_dataloader(self):
        if self.current_epoch < 155:
            train_dl = DataLoader(self.mit_train_1,
                                  batch_size=self.hparams.batch_size,
                                  shuffle=True,
                                  num_workers=self.hparams.num_workers)
        else:
            train_dl = DataLoader(self.mit_train_2,
                                  batch_size=self.hparams.batch_size,
                                  shuffle=True,
                                  num_workers=self.hparams.num_workers)

        return train_dl

    def val_dataloader(self):
        val_dl = DataLoader(self.mit_val,
                            batch_size=self.hparams.batch_size,
                            shuffle=False,
                            num_workers=self.hparams.num_workers)

        return val_dl


if __name__ == "__main__":
    data_dir = '/media/nthere/datasets/FastImageProcessing/data/train/set-1/'
    model = LitModel(data_dir=data_dir,
                     batch_size=1)
    trainer = pl.Trainer()
    trainer.fit(model)
