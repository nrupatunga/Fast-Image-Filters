"""
File: mit_dataloader.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: mit dataset loaders
"""
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from imutils import paths
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from tqdm import tqdm


class MitData(Dataset):

    """Docstring for MitData. """

    def __init__(self,
                 data_dir: Union[str, Path],
                 is_train: bool):
        """
        @data_dir: path to the dataset
        @shuffle: shuffle True/False
        """
        super().__init__()

        self.data_dir = data_dir
        self.is_train = is_train

        input_dir = Path(data_dir).joinpath('input')
        gt_dir = Path(data_dir).joinpath('gt')

        input_imgs = list(paths.list_images(input_dir))
        gt_imgs = list(paths.list_images(gt_dir))

        self.pair_images = list(zip(input_imgs, gt_imgs))

    def __len__(self):
        return len(self.pair_images)

    def __getitem__(self, idx):
        img_path, gt_path = self.pair_images[idx]
        img = cv2.imread(str(img_path))
        gt = cv2.imread(str(gt_path))

        img = np.transpose(img, axes=(2, 0, 1)) / 255.
        gt = np.transpose(gt, axes=(2, 0, 1)) / 255.

        img = torch.from_numpy(img).float()
        gt = torch.from_numpy(gt).float()
        return (img, gt)


if __name__ == "__main__":
    from core.utils.vis_utils import Visualizer

    viz = Visualizer()
    data_dir = '/media/nthere/datasets/FastImageProcessing/data/train/set-1/'

    mit = MitData(data_dir, is_train=True)
    dataloader = DataLoader(mit, batch_size=1, shuffle=True,
                            num_workers=6)

    viz = Visualizer()
    for i, (img, gt) in tqdm(enumerate(dataloader)):
        data = torch.cat((img, gt), 0)
        data = make_grid(data, nrow=2, pad_value=0)
        viz.plot_images_np(data, 'data')
