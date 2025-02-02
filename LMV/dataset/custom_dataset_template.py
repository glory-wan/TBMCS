import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from tqdm import tqdm

from baseDataset import BaseDataset


class CustomDataset(BaseDataset):
    def __init__(self, root_dir, split=None, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images and labels.
            split (str): Subdirectory name, e.g., 'train' or 'val'.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__(root_dir, split, transform)

    def __len__(self):
        """Return the size of the dataset"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Returns an image with corresponding label"""
        image, img_path = super().__getitem__(idx)
        label = None
        return image, label
