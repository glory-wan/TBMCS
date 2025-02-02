import os
import logging
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

from dataset.generating_labels.yolo2mask import convert_yolo_to_mask
from LMV.dataset.baseDataset import BaseDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class LmvDataset(BaseDataset):
    def __init__(self, root_dir, split=None, transform=None, imgz=640, single_target=True):
        """
        Args:
            root_dir (str): Directory with all the images and labels.
            split (str): Subdirectory name, e.g., 'train' or 'val'.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            label_file (str): File name for the label file, assumed to be in the root_dir.
        """
        super().__init__(root_dir, split, transform, imgz, single_target)

        # for classification
        self.cls_txt = "train_cls.txt" if split == 'train' else 'val_cls.txt'
        self.cls_labels = self.load_cls_labels(os.path.join(self.labels_dir, 'classification', self.cls_txt))

        # for datection
        self.det_labels = self.load_det_labels()

        # for segmentation
        self.seg_labels = self.get_seg_labels_path()
        self.mask_transform = transforms.Compose([
            transforms.Resize((imgz, imgz)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        image, img_path = super().__getitem__(idx)
        image_name = os.path.basename(img_path)  # e.g. train1.jpg

        # get classification label
        cls_label = self.cls_labels.get(image_name, None)
        # logging.info(f'getting {image_name} and its labels: {cls_label}')

        # get detection label
        det_label = self.det_labels.get(image_name, None)
        # logging.info(f'getting {image_name} and its labels: {det_label}')

        # get segmentation label
        seg_label_path = self.seg_labels.get(image_name, None)
        if seg_label_path is not None:
            seg_label = convert_yolo_to_mask(img_path, str(seg_label_path), single_target=self.single_target)
            seg_label = self.mask_transform(seg_label)
        else:
            seg_label = torch.zeros((1, self.imgz, self.imgz))

        return image, cls_label, det_label, seg_label

    def get_seg_labels_path(self):
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

    def load_det_labels(self):
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
            labels[image_name] = self.load_single_label(str(label_path))
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

    @staticmethod
    def load_cls_labels(label_path):
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


# if __name__ == '__main__':
#     LMVData = LmvDataset(
#         root_dir=r'D:\Code\pycode\dataset_all\tf_version3',
#         split='train'
#     )
#
#     img, cls_label, det_label, mask_label = LMVData.__getitem__(69)
#     print(f"Image shape: {img.shape}, \n"
#           f"cls_label: {cls_label}, \n"
#           f"det_label: {det_label}, \n"
#           f"mask_label: {mask_label.shape}")
