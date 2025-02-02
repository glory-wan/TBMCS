import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import logging

from LMV.dataset.LMVDataset import LmvDataset
from LMV.dataset.utils import lmv_collate_fn

if __name__ == '__main__':
    image_transforms = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])

    root_dir = r'D:\Code\pycode\dataset_all\tf_version3'

    train_dataset = LmvDataset(
        root_dir=root_dir,
        split='train',
        transform=image_transforms,
        imgz=640,
        single_target=True
    )

    # 初始化验证集（如果需要）
    val_dataset = LmvDataset(
        root_dir=root_dir,
        split='val',
        imgz=640,
        single_target=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        collate_fn=lmv_collate_fn
    )

    # 创建验证集的 DataLoader（如果需要）
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        collate_fn=lmv_collate_fn
    )

    for batch_idx, batch in enumerate(train_loader):
        images = batch['images']
        cls_labels = batch['cls_labels']
        det_labels = batch['det_labels']
        seg_labels = batch['seg_labels']

        print(f"Batch {batch_idx + 1}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Classification labels: {cls_labels}")
        print(f"  Number of Detection Labels: {det_labels}")
        print(f"  Segmentation labels shape: {seg_labels.shape}")

        # 将数据移动到 GPU（如果可用）
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # images = images.to(device)
        # cls_labels = cls_labels.to(device)
        # seg_labels = seg_labels.to(device)

        # 在这里添加您的训练代码，例如前向传播、计算损失、反向传播等

        # 为了示例，迭代一个批次后退出
        break
