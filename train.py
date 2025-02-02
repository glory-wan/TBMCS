import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import csv
import logging
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from LMV.dataset.LMVDataset import LmvDataset
from LMV.dataset.utils import lmv_collate_fn
from LMV.models.module import LMV
from LMV.config import set_seed


def setup_logging(log_dir='logs'):
    """设置日志，将信息同时输出到控制台和文件"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, 'training.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # 输出到控制台
            logging.FileHandler(log_file, mode='a')  # 输出到文件，模式为追加
        ]
    )


def calculate_metrics(y_true, y_pred, average='macro'):
    """
    计算准确度、精度、召回率、F1分数和F5分数
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=1)
    recall = recall_score(y_true, y_pred, average=average, zero_division=1)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=1)

    # F5 score: F5 score 公式为 (1 + 5^2) * (precision * recall) / (5^2 * precision + recall)
    f5 = (1 + 5 ** 2) * (precision * recall) / (5 ** 2 * precision + recall)

    return accuracy, precision, recall, f1, f5


if __name__ == '__main__':
    set_seed()
    setup_logging()  # 设置日志

    number_class = 4
    batch_size = 16
    epochs = 30
    learning_rate = 0.001
    img_size = 320
    num_workers = 8

    data_root = r'D:\Code\pycode\dataset_all\tf_version3'

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }

    train_dataset = LmvDataset(root_dir=data_root, split='train', transform=data_transforms['train'])
    val_dataset = LmvDataset(root_dir=data_root, split='val', transform=data_transforms['val'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, persistent_workers=True,
                              num_workers=num_workers, pin_memory=True, collate_fn=lmv_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, persistent_workers=True,
                            num_workers=num_workers, pin_memory=True, collate_fn=lmv_collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lmv = LMV(nc=number_class).to(device)

    criterion = nn.CrossEntropyLoss()  # 分类任务的交叉熵损失
    optimizer = optim.Adam(lmv.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    scaler = GradScaler()

    for epoch in range(epochs):
        logging.info(f"Epoch {epoch + 1}/{epochs}")
        running_loss = 0.0
        cls_all_labels = []
        cls_all_preds = []
        pbar_train = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}")

        """  ******************         train         ****************************   """
        lmv.train()
        for batch in pbar_train:
            images = batch['images']
            cls_labels = batch['cls_labels']
            # det_labels = batch['det_labels']
            # seg_labels = batch['seg_labels']

            images = images.to(device)
            cls_labels = cls_labels.to(device)

            optimizer.zero_grad()

            # 混合精度训练
            with autocast():
                clssified = lmv(images)
                loss = criterion(clssified, cls_labels)

            # 缩放梯度
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 计算损失和准确率
            running_loss += loss.item()
            _, cls_predicted = torch.max(clssified, 1)

            cls_all_labels.extend(cls_labels.cpu().numpy())
            cls_all_preds.extend(cls_predicted.cpu().numpy())

            accuracy, precision, recall, f1, f5 = calculate_metrics(cls_all_labels, cls_all_preds, average='macro')
            pbar_train.set_postfix(f1=f1, f5=f5, accuracy=accuracy, precision=precision, recall=recall, loss=loss.item())

        epoch_accuracy, epoch_precision, epoch_recall, epoch_f1, epoch_f5 = calculate_metrics(cls_all_labels,
                                                                                                  cls_all_preds,
                                                                                                  average='macro')
        logging.info(
            f"Epoch {epoch + 1}: F1: {epoch_f1:.4f}, F5: {epoch_f5:.4f}, Accuracy: {epoch_accuracy:.4f}, "
            f"Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f} ")
        """  ******************         train over        ****************************   """

        """  ******************         val         ****************************   """
        lmv.eval()
        pbar_val = tqdm(val_loader, desc=f"val Epoch {epoch + 1}/{epochs}")
        with torch.no_grad():
            for batch in pbar_val:
                images = batch['images']
                cls_labels = batch['cls_labels']
                det_labels = batch['det_labels']
                seg_labels = batch['seg_labels']

                images = images.to(device)
                cls_labels = cls_labels.to(device)

                outputs = lmv(images)
                loss = criterion(outputs, cls_labels)

                _, cls_predicted = torch.max(outputs, 1)
                cls_all_labels.extend(cls_labels.cpu().numpy())
                cls_all_preds.extend(cls_predicted.cpu().numpy())

                accuracy, precision, recall, f1, f5 = calculate_metrics(cls_all_labels, cls_all_preds, average='macro')
                pbar_val.set_postfix(f1=f1, f5=f5, accuracy=accuracy, precision=precision, recall=recall, loss=loss.item())

        val_accuracy, val_precision, val_recall, val_f1, val_f5 = calculate_metrics(cls_all_labels,
                                                                                    cls_all_preds,
                                                                                    average='macro')
        logging.info(
            f"Epoch {epoch + 1} val  : F1: {val_f1:.4f}, F5: {val_f5:.4f}, Accuracy: {val_accuracy:.4f}, "
            f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f},  \n")
        """  ******************         val over        ****************************   """

        # 保存模型检查点
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': lmv.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch_loss': running_loss,
            'epoch_accuracy': epoch_accuracy,
            'epoch_precision': epoch_precision,
            'epoch_recall': epoch_recall,
            'epoch_f1': epoch_f1,
            'epoch_f5': epoch_f5,
        }

        torch.save(checkpoint, f"checkpoints/backbone1_deep_shallow.pt")
        scheduler.step()

    logging.info("backbone1 deep_feature : Training complete!")
