import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch
import random
import numpy as np
import argparse

import os
import yaml

__all__ = (
    "transformer",
    "save_hyperparameters",
    "set_seed",
    "parse_args",
)

tasks = {'classification', 'detection', 'segmentation'}

Extension = (
    ".bmp", ".dng", ".jpeg", ".jpg", ".mpo",
    ".png", ".tif", ".tiff", ".webp", ".pfm"
)


def transformer(imgz):
    transform = transforms.Compose([
        transforms.Resize((imgz, imgz)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform


def save_hyperparameters(gpu, model, loss, optimizer_, learning_rate, save_config_path='train/config.yaml'):
    os.makedirs(os.path.dirname(save_config_path), exist_ok=True)

    hyperparameters = {
        'gpu': gpu,
        'model': model.__class__.__name__,
        'loss': loss,
        'optimizer': optimizer_,
        'learning_rate': learning_rate,
    }

    with open(save_config_path, 'w') as file:
        yaml.dump(hyperparameters, file)

    print(f"Hyperparameters saved to {save_config_path}")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练配置参数')
    parser.add_argument('--use_multi_gpu', action='store_true', help='是否使用多GPU训练')
    parser.add_argument('--log_dir', type=str, default='logs', help='日志保存目录')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='检查点保存目录')

    return parser.parse_args()
