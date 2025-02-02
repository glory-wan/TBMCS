import os
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
from functools import lru_cache
import logging
from torch.utils.data import Dataset
from torchvision import transforms
import pickle

import torch
from torchvision import transforms
import matplotlib.pyplot as plt

from LMV.config import Extension
from LMV.dataset.utils import leTransformer
from dataset.generating_labels.yolo2mask import convert_yolo_to_mask

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class BaseDataset(Dataset):
    def __init__(self, root_dir, split=None, transform=None, imgz=640, single_target=True):
        """
        Args:
            root_dir (str): Directory with all the images and labels.
            split (str): Subdirectory name, e.g., 'train' or 'val'.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.split = split
        self.imgz = imgz
        self.single_target = single_target
        self.check_split()

        self.images_dir = os.path.join(root_dir, 'images', self.split)
        self.labels_dir = os.path.join(root_dir, 'labels')

        self.leTransform = transform if transform else leTransformer(imgz)
        self.images = [os.path.join(self.images_dir, f) for f in
                       tqdm(os.listdir(self.images_dir), desc=f'      loading {str(self.split)} dataset')
                       if f.lower().endswith(Extension)]
        self.labels = None

        if not self.images:
            raise ValueError(f"No images found in directory {self.images_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.leTransform(image)

        return image, img_path

    @lru_cache(maxsize=128)
    def load_image(self, img_path):
        return Image.open(img_path).convert("RGB")

    def check_split(self):
        if self.split not in ['train', 'val', 'test']:
            raise ValueError("Unsupported split value. Please use one of ['train', 'val', 'test']")


class ClassificationDataset(BaseDataset):
    def __init__(self, root_dir, split=None, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images and labels.
            split (str): Subdirectory name, e.g., 'train' or 'val'.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            label_file (str): File name for the label file, assumed to be in the root_dir.
        """
        super().__init__(root_dir, split, transform)
        self.label_file = "train_cls.txt" if split == 'train' else 'val_cls.txt'
        self.labels = self.load_labels(os.path.join(self.labels_dir, 'classification', self.label_file))

    @staticmethod
    def load_labels(label_path):
        """
        Loads labels from a given file.

        Args:
            label_path (str): Path to the label file.

        Returns:
            dict: Dictionary mapping image filenames to labels.
        """
        labels = {}
        with open(label_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                image_name = parts[0]
                label = int(parts[1])
                labels[image_name] = label
        return labels

    def __getitem__(self, idx):
        image, img_path = super().__getitem__(idx)
        image_name = os.path.basename(img_path)

        cls_label = self.labels.get(image_name, -1)
        logging.info(f'getting {image_name} and its labels')

        return image, cls_label


class DetectionDataset(BaseDataset):
    def __init__(self, root_dir, split=None, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images and labels.
            split (str): Subdirectory name, e.g., 'train' or 'val'.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__(root_dir, split, transform)
        self.labels = self.load_labels()

    def load_labels(self):
        """
        Loads labels from the label directory.

        Returns:
            dict: Dictionary mapping image filenames to list of labels.
        """
        labels = {}
        for image_path in self.images:
            image_name = os.path.basename(image_path)
            label_path = os.path.join(self.labels_dir, 'detection', self.split,
                                      os.path.splitext(image_name)[0] + '.txt')
            if not os.path.exists(label_path):
                # logging.warning(f"Label file {label_path} not found. Skipping.")
                continue
            labels[image_name] = self.load_single_label(label_path)
        return labels

    @staticmethod
    def load_single_label(label_path):
        """
        Loads a single label file.

        Args:
            label_path (str): Path to the label file.

        Returns:
            list: List of bounding boxes and class labels.
        """
        labels = []
        try:
            with open(label_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    labels.append([class_id, x_center, y_center, width, height])
        except Exception as e:
            logging.warning(f"Error reading {label_path}: {e}")
        return labels

    def __getitem__(self, idx):
        image, img_path = super().__getitem__(idx)
        image_name = os.path.basename(img_path)
        label = self.labels.get(image_name, [])

        if not label:
            logging.warning(f"No label for {image_name}, this image will be skipped")
            return self.__getitem__((idx + 1) % len(self.images))
        logging.info(f'getting {image_name} and its labels')

        return image, label


class SegmentationDataset(BaseDataset):
    def __init__(self, root_dir, split=None, transform=None, imz=512, single_target=True):
        super().__init__(root_dir, split, transform, single_target)
        self.labels = self.get_labels_path()
        self.mask_transform = transforms.Compose([
            transforms.Resize((imz, imz)),
            transforms.ToTensor()
        ])

        self.cache_file = os.path.join(root_dir, f"{self.split}.cache")

        if os.path.exists(self.cache_file):
            print(f"Loading cache from {self.cache_file}")
            with open(self.cache_file, 'rb') as cache:
                cache_data = pickle.load(cache)
                self.images = cache_data['images']
                self.labels = cache_data['labels']
        else:
            print(f"Creating cache file: {self.cache_file}")
            self.images_dir = os.path.join(root_dir, 'images', self.split)
            self.labels_dir = os.path.join(root_dir, 'labels', 'segmentation', self.split)

            self.images = [os.path.join(self.images_dir, f) for f in
                           tqdm(os.listdir(self.images_dir), desc=f'loading {str(self.split)} dataset')
                           if f.lower().endswith('.jpg')]

            self.labels = {}
            for image_path in self.images:
                image_name = os.path.basename(image_path)
                label_path = os.path.join(self.labels_dir, os.path.splitext(image_name)[0] + '.txt')
                if os.path.exists(label_path):
                    self.labels[image_name] = label_path
                else:
                    print(f"Warning: No label found for {image_name}, skipping.")
                    self.images.remove(image_path)

            with open(self.cache_file, 'wb') as cache:
                pickle.dump({'images': self.images, 'labels': self.labels}, cache)

    def get_labels_path(self):
        labels = {}
        for image_path in self.images:
            image_name = os.path.basename(image_path)
            label_path = os.path.join(self.labels_dir, 'segmentation', self.split,
                                      os.path.splitext(image_name)[0] + '.txt')
            if not os.path.exists(label_path):
                # logging.warning(f"Label file {label_path} not found. Skipping.")
                continue
            labels[image_name] = label_path
        return labels

    def __getitem__(self, idx):
        image, img_path = super().__getitem__(idx)
        image_name = os.path.basename(img_path)
        label_path = self.labels.get(image_name, [])

        if not label_path:
            # logging.warning(f"No label for {image_name}, this image will be skipped")
            return self.__getitem__((idx + 1) % len(self.images))
            # return None

        mask_label = self.load_and_convert_mask(img_path, str(label_path))
        mask_label = self.mask_transform(mask_label)

        # logging.info(f'Getting {os.path.basename(img_path)} and its segmentation mask')

        return image, mask_label

    @lru_cache(maxsize=128)
    def load_and_convert_mask(self, img_path, label_path):
        return convert_yolo_to_mask(img_path, str(label_path), single_target=True)


if __name__ == '__main__':
    # classification = ClassificationDataset(
    #     root_dir=r'D:\Code\pycode\dataset_all\tf_version3',
    #     split='train'
    # )
    #
    # img, label = classification.__getitem__(5)
    # print(f"Image shape: {img.shape}, label: {label}")

    # Detection = DetectionDataset(
    #     root_dir=r'D:\Code\pycode\dataset_all\tf_version3',
    #     split='train'
    # )
    # img, label = Detection.__getitem__(2)
    # print(f"Image shape: {img.shape}, label: {label}")

    segmentation = SegmentationDataset(
        root_dir=r'D:\Code\pycode\dataset_all\tf_version3',
        split='train'
    )

    img, label = segmentation.__getitem__(5)
    print(f"Image shape: {img.shape}, label: {label.shape}")
