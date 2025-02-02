import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import logging

from LMV.dataset.LMVDataset import LmvDataset
from LMV.dataset.utils import lmv_collate_fn
from LMV.models.module import LMV_cls
from LMV.config import *
from LMV.utils import *

if __name__ == '__main__':
    log_dir = 'logs'
    checkpoint_dir = 'checkpoints'
    set_seed()
    setup_logging(log_dir=log_dir)  # 设置日志

    name = 'tf_backbone1_medium'
    number_class = 4
    batch_size = 4
    epochs = 100
    learning_rate = 0.001
    img_size = 320
    num_workers = 8
    use_multi_gpu = True
    patience = 10

    data_root = r'D:\Code\dataset_all\tf_version3'

    train_dataset = LmvDataset(root_dir=data_root, split='train', transform=transformer(img_size))
    val_dataset = LmvDataset(root_dir=data_root, split='val', transform=transformer(img_size))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, persistent_workers=True,
                              num_workers=num_workers, pin_memory=True, collate_fn=lmv_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=int(batch_size / 2), shuffle=False, persistent_workers=True,
                            num_workers=int(num_workers / 2), pin_memory=True, collate_fn=lmv_collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lmv = LMV_cls(cls_nc=number_class).to(device)

    if use_multi_gpu and torch.cuda.device_count() > 1:
        logging.info(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
        lmv = nn.DataParallel(lmv)
    else:
        logging.info(f"使用{device}进行训练")

    criterion = nn.CrossEntropyLoss()  # 分类任务的交叉熵损失
    optimizer = optim.Adam(lmv.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    scaler = GradScaler()

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    best_f1 = 0.0  # 用于保存最佳模型
    best_f5 = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    start_epoch = 0

    for epoch in range(start_epoch, epochs):
        logging.info(f"Epoch {epoch + 1}/{epochs}")
        running_loss = 0.0
        cls_all_labels = []
        cls_all_preds = []
        epoch_accuracy = 0.0
        epoch_precision = 0.0
        epoch_recall = 0.0
        epoch_f1 = 0.0
        epoch_f5 = 0.0
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

            epoch_accuracy, epoch_precision, epoch_recall, epoch_f1, epoch_f5 = calculate_metrics_cls(cls_all_labels,
                                                                                                      cls_all_preds,
                                                                                                      average='macro')
            pbar_train.set_postfix(f1=epoch_f1, f5=epoch_f5, accuracy=epoch_accuracy,
                                   precision=epoch_precision, recall=epoch_recall, Loss=loss.item(), average='macro')
        logging.info(
            f"Epoch {epoch + 1}: F1: {epoch_f1:.4f}, F5: {epoch_f5:.4f}, Accuracy: {epoch_accuracy:.4f}, "
            f"Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f}, Loss: {running_loss}")
        """  ******************         train over        ****************************   """

        """  ******************         val         ****************************   """
        lmv.eval()
        cls_all_labels = []  # 重置验证阶段的标签和预测
        cls_all_preds = []
        val_running_loss = 0.0  # 累计验证损失
        pbar_val = tqdm(val_loader, desc=f"val Epoch {epoch + 1}/{epochs}")
        with torch.no_grad():
            for batch in pbar_val:
                images = batch['images']
                cls_labels = batch['cls_labels']
                # det_labels = batch['det_labels']
                # seg_labels = batch['seg_labels']

                images = images.to(device)
                cls_labels = cls_labels.to(device)

                with autocast():
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
            f"Epoch {epoch + 1} val  : F1: {val_f1:.4f}, F5: {val_f5:.4f}, Accuracy: {val_accuracy:.4f}, "
            f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, Loss: {average_val_loss:.4f},  \n")
        """  ******************         val over        ****************************   """

        # 调整学习率
        scheduler.step()

        if val_f1 + val_f5 > best_f1 + best_f5:
            best_f1 = val_f1
            best_f5 = val_f5
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            checkpoint_path = os.path.join(checkpoint_dir, f"{name}_epoch_{epoch + 1}.pt")

            if use_multi_gpu and torch.cuda.device_count() > 1:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': lmv.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch_loss': running_loss,
                    'epoch_accuracy': epoch_accuracy,
                    'epoch_precision': epoch_precision,
                    'epoch_recall': epoch_recall,
                    'epoch_f1': epoch_f1,
                    'epoch_f5': epoch_f5,
                }, checkpoint_path)
            else:
                torch.save({
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
                }, checkpoint_path)
            logging.info(f"New best model saved to {checkpoint_path} with "
                         f"F1: {best_f1:.4f}, F5: {best_f5:.4f}, best epoch is {epoch + 1}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            from test import validate_model

            checkpoint_path = f'checkpoints/{name}_epoch_{epoch + 1 - patience}.pt'
            num_workers = int(num_workers / 2)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            logging.info(f"Now, loading {checkpoint_path} for validating:")
            validate_model(checkpoint_path, data_root, number_class, batch_size, img_size, num_workers, device)
            logging.info(f"Early stopping at epoch {epoch + 1} due to no improvement in F1 and F5 scores."
                         f"Best epoch is {epoch + 1 - patience}")
            break
