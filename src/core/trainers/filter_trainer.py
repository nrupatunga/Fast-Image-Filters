"""

File: filter_trainer.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: trainer code
"""
from pathlib import Path

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import functional as F
from torch.utils.data import DataLoader

from core.dataloaders.mit_dataloader import MitData
from core.network.custom_nets import FIP


class LitModel(pl.LightningModule):

    """Docstring for LitModel. """

    def __init__(self,
                 data_dir: str,
                 batch_size: int,
                 num_workers: int = 6,
                 lr: float = 1e-4,
                 **kwargs) -> None:
        """
        @lr: learning rate
        """
        super().__init__()

        self.model = FIP()
        self.save_hyperparameters()

    def forward(self, x):
        """forward function for litmodel
        """
        return self.model(x)

    def configure_optimizers(self):
        # eps same as tensorflow adam
        return torch.optim.Adam(self.parameters(),
                                lr=self.hparams.lr, eps=1e-7)

    def setup(self, stage=None):

        train_dir = Path(self.hparams.data_dir).joinpath('train')
        val_dir = Path(self.hparams.data_dir).joinpath('val')

        train_dir_1 = train_dir.joinpath('set-1')
        self.mit_train_1 = MitData(train_dir_1, is_train=True)

        train_dir_2 = train_dir.joinpath('set-2')
        self.mit_train_2 = MitData(train_dir_2, is_train=True)

        self.mit_val = MitData(val_dir, is_train=False)

    def train_dataloader(self):
        if self.current_epoch < 120:
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

    def _vis_images(self, y, idx, prefix='val'):
        y_hat_dbg = y.detach().clone()
        y_hat_dbg = y_hat_dbg[0:10]
        dbg_imgs = y_hat_dbg.cpu().numpy()
        for i in range(dbg_imgs.shape[0]):
            img = dbg_imgs[i]
            img = np.transpose(img, [1, 2, 0])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if idx == 1:
                img = np.clip(img, 0, 1)
            else:
                img = cv2.normalize(img, None, alpha=0, beta=1,
                                    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            dbg_imgs[i] = np.transpose(img, [2, 0, 1])

        dbg_imgs = torch.tensor(dbg_imgs)

        grid = torchvision.utils.make_grid(dbg_imgs)
        if idx == 0:
            self.logger.experiment.add_image(f'{prefix}_gt', grid,
                                             self.global_step)
        elif idx == 1:
            self.logger.experiment.add_image(f'{prefix}_preds', grid,
                                             self.global_step)
        else:
            self.logger.experiment.add_image(f'{prefix}_input', grid,
                                             self.global_step)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y, reduction='sum')

        if batch_idx % 250 == 0:
            self._vis_images(y, 0, prefix='val')
            self._vis_images(y_hat, 1, prefix='val')
            self._vis_images(x, 2, prefix='val')

        tf_logs = {'val_loss': loss}
        return {'loss': loss, 'log': tf_logs}

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y, reduction='sum')

        if batch_idx % 250 == 0:
            self._vis_images(y, 0, prefix='train')
            self._vis_images(y_hat, 1, prefix='train')
            self._vis_images(x, 2, prefix='train')

        tf_logs = {'train_loss': loss}

        return {'loss': loss, 'log': tf_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        return {'val_loss': avg_loss}


if __name__ == "__main__":
    data_dir = '/media/nthere/datasets/FastImageProcessing/data/style/'
    model = LitModel(data_dir=data_dir,
                     batch_size=1)
    ckpt_cb = ModelCheckpoint(filepath='./style-2/', save_top_k=10,
                              save_weights_only=False)
    trainer = pl.Trainer(gpus=[0],
                         max_epochs=180,
                         checkpoint_callback=ckpt_cb,
                         # resume_from_checkpoint='./style/_ckpt_epoch_48.ckpt',
                         reload_dataloaders_every_epoch=True)
    trainer.fit(model)
