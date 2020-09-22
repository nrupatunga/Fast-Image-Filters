"""
File: train.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: training script
"""
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from core.trainers.filter_trainer import LitModel

if __name__ == "__main__":
    data_dir = '/media/nthere/datasets/FastImageProcessing/data/pencil/'
    model = LitModel(data_dir=data_dir,
                     batch_size=1)
    ckpt_cb = ModelCheckpoint(filepath='./ckpt/pencil', save_top_k=10,
                              save_weights_only=False)
    trainer = pl.Trainer(gpus=[0],
                         max_epochs=180,
                         checkpoint_callback=ckpt_cb,
                         resume_from_checkpoint='./ckpt/pencil/_ckpt_epoch_66.ckpt',
                         reload_dataloaders_every_epoch=True)
    trainer.fit(model)
