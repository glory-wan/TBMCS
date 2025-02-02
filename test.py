import torch
import torch.nn as nn
import os
import logging
import argparse
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from LMV.dataset.LMVDataset import LmvDataset
from LMV.dataset.utils import lmv_collate_fn
from LMV.models.module import LMV_cls
from LMV.config import *
from LMV.utils import *


def load_model(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def validate_model(checkpoint_path, data_root, number_class, batch_size, img_size, num_workers, device):
    setup_logging()

    val_dataset = LmvDataset(root_dir=data_root, split='val', transform=transformer(img_size))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, persistent_workers=True,
                            num_workers=int(num_workers / 2), pin_memory=True, collate_fn=lmv_collate_fn)

    lmv = LMV_cls(cls_nc=number_class).to(device)

    lmv = load_model(checkpoint_path, lmv)

    criterion = nn.CrossEntropyLoss()  # 分类任务的交叉熵损失

    lmv.eval()
    cls_all_labels = []
    cls_all_preds = []
    val_running_loss = 0.0  # 累计验证损失
    pbar_val = tqdm(val_loader, desc=f"Validation")

    with torch.no_grad():
        for batch in pbar_val:
            images = batch['images']
            cls_labels = batch['cls_labels']

            images = images.to(device)
            cls_labels = cls_labels.to(device)

            outputs = lmv(images)
            loss = criterion(outputs, cls_labels)

            val_running_loss += loss.item()

            _, cls_predicted = torch.max(outputs, 1)
            cls_all_labels.extend(cls_labels.cpu().numpy())
            cls_all_preds.extend(cls_predicted.cpu().numpy())

            val_accuracy, val_precision, val_recall, val_f1, val_f5 = calculate_metrics_cls(cls_all_labels,
                                                                                            cls_all_preds,
                                                                                            average='macro')
            pbar_val.set_postfix(f1=val_f1, f5=val_f5, accuracy=val_accuracy,
                                 precision=val_precision, recall=val_recall, loss=loss.item())

    average_val_loss = val_running_loss / len(val_loader)
    logging.info(
        f"Validation  : F1: {val_f1:.4f}, F5: {val_f5:.4f}, Accuracy: {val_accuracy:.4f}, "
        f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, Loss: {average_val_loss:.4f},  \n")

    logging.info("Validation complete!")


if __name__ == '__main__':
    checkpoint_path = 'checkpoints/tf_backbone1_medium_epoch_49.pt'
    data_root = r'D:\Code\dataset_all\tf_version3'
    number_class = 4
    batch_size = 8
    img_size = 320
    num_workers = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    validate_model(checkpoint_path, data_root, number_class, batch_size, img_size, num_workers, device)
