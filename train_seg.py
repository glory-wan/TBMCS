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
from LMV.models.module import LMV_seg
from LMV.Loss.seg_Loss import TverskyLoss
from LMV.config import *
from LMV.utils import *


def validate_model(lmv, val_loader, device, criterion):
    lmv.eval()  # Set model to evaluation mode
    total_metrics = {
        'Precision': 0.0,
        'Recall': 0.0,
        'F1': 0.0,
        'mIoU': 0.0,
        'bIoU': 0.0
    }
    running_loss = 0.0
    num_batches = 0

    with torch.no_grad():  # Disable gradient computation
        for step, batch in enumerate(val_loader):
            images = batch['images'].to(device)
            seg_labels = batch['seg_labels'].to(device)

            seg_predicted = lmv(images)
            loss = criterion(seg_predicted, seg_labels)
            running_loss += loss.item()

            seg_predicted_bin = (torch.sigmoid(seg_predicted) > 0.5).detach().cpu().numpy()
            seg_labels_bin = seg_labels.cpu().numpy()

            metrics = calculate_seg_metrics(seg_predicted_bin, seg_labels_bin)

            for key in total_metrics:
                total_metrics[key] += metrics[key]
            num_batches += 1

            current_avg_metrics = {key: value / (step + 1) for key, value in total_metrics.items()}
            print(f"Validation Step {step + 1}/{len(val_loader)} - Loss: {running_loss / (step + 1):.4f}, "
                  f"P: {current_avg_metrics['Precision']:.4f}, "
                  f"R: {current_avg_metrics['Recall']:.4f}, "
                  f"F1: {current_avg_metrics['F1']:.4f}, "
                  f"mIoU: {current_avg_metrics['mIoU']:.4f}, "
                  f"bIoU: {current_avg_metrics['bIoU']:.4f}")

    avg_loss = running_loss / num_batches
    avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}

    print(f"Validation Results - Loss: {avg_loss:.4f}, "
          f"P: {avg_metrics['Precision']:.4f}, "
          f"R: {avg_metrics['Recall']:.4f}, "
          f"F1: {avg_metrics['F1']:.4f}, "
          f"mIoU: {avg_metrics['mIoU']:.4f}, "
          f"bIoU: {avg_metrics['bIoU']:.4f}")

    return avg_loss, avg_metrics


if __name__ == '__main__':
    log_dir = 'logs'
    checkpoint_dir = 'checkpoints'
    set_seed()
    setup_logging(log_dir=log_dir)  # 设置日志

    name = 'tf_dataset_backbone1_Tversky'
    cls_nc = 4
    seg_nc = 1
    batch_size = 32
    epochs = 100
    learning_rate = 0.001
    img_size = 640
    num_workers = 32
    use_multi_gpu = True
    patience = 30

    data_root = r'D:\Code\pycode\Data_All\tf_version3'

    train_dataset = LmvDataset(root_dir=data_root, split='train', transform=transformer(img_size))
    val_dataset = LmvDataset(root_dir=data_root, split='val', transform=transformer(img_size))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, persistent_workers=True,
                              num_workers=num_workers, pin_memory=True, collate_fn=lmv_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=int(batch_size / 2), shuffle=False, persistent_workers=True,
                            num_workers=int(num_workers / 2), pin_memory=True, collate_fn=lmv_collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lmv = LMV_seg(cls_nc=cls_nc, seg_nc=seg_nc).to(device)

    if use_multi_gpu and torch.cuda.device_count() > 1:
        logging.info(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
        lmv = nn.DataParallel(lmv)
    else:
        logging.info(f"使用{device}进行训练")

    criterion = TverskyLoss()
    optimizer = optim.Adam(lmv.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scaler = GradScaler()

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    best_epoch = 0
    epochs_without_improvement = 0
    start_epoch = 0
    best_mIou = 0.0
    best_bIou = 0.0

    for epoch in range(start_epoch, epochs):
        logging.info(f"Epoch {epoch + 1}/{epochs}")
        running_loss = 0.0
        total_metrics = {
            'Precision': 0.0,
            'Recall': 0.0,
            'F1': 0.0,
            'mIoU': 0.0,
            'bIoU': 0.0
        }
        pbar_train = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}")

        """  ******************         train         ****************************   """
        lmv.train()
        for batch in pbar_train:
            images = batch['images']
            # cls_labels = batch['cls_labels']
            # det_labels = batch['det_labels']
            seg_labels = batch['seg_labels']

            images = images.to(device)
            seg_labels = seg_labels.to(device)

            optimizer.zero_grad()

            # 混合精度训练
            with autocast():
                seg_predicted = lmv(images)
                loss = criterion(seg_predicted, seg_labels)

            # 缩放梯度
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 计算损失和准确率
            running_loss += loss.item()

            seg_predicted_bin = (torch.sigmoid(seg_predicted) > 0.5).detach().cpu().numpy()
            seg_labels_bin = seg_labels.cpu().numpy()
            metrics = calculate_seg_metrics(seg_predicted_bin, seg_labels_bin)

            # for key in total_metrics:
            #     total_metrics[key] += metrics[key]

            pbar_train.set_description(
                f"Loss: {loss.item():.4f}, "
                f"P: {metrics['Precision']:.4f}, "
                f"R: {metrics['Recall']:.4f}, "
                f"F1: {metrics['F1']:.4f}, "
                f"mIoU: {metrics['mIoU']:.4f}, "
                # f"bIoU: {metrics['bIoU']:.4f}"
            )
        """  ******************         train over        ****************************   """

        """  ******************         val         ****************************   """
        avg_loss, avg_metrics = validate_model(lmv, val_loader, device, criterion)
        scheduler.step()

        if avg_metrics['mIoU'] > best_mIou:
            best_mIou = avg_metrics['mIoU']
            # best_bIou = avg_metrics['bIoU']
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
                    'epoch_accuracy': avg_metrics['Precision'],
                    'epoch_precision': avg_metrics['Recall'],
                    'epoch_recall': avg_metrics['F1'],
                    'epoch_f1': avg_metrics['mIoU'],
                    # 'epoch_f5': avg_metrics['bIoU'],
                }, checkpoint_path)
            else:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': lmv.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch_loss': running_loss,
                    'epoch_accuracy': avg_metrics['Precision'],
                    'epoch_precision': avg_metrics['Recall'],
                    'epoch_recall': avg_metrics['F1'],
                    'epoch_f1': avg_metrics['mIoU'],
                    # 'epoch_f5': avg_metrics['bIoU'],
                }, checkpoint_path)
            logging.info(f"New best model saved to {checkpoint_path} with "
                         f"mIou: {avg_metrics['mIoU']:.4f}, bIou: {avg_metrics['bIoU']:.4f}, best epoch is {epoch + 1}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            checkpoint_path = f'checkpoints/{name}_epoch_{epoch + 1 - patience}.pt'
            num_workers = int(num_workers / 2)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            avg_loss, avg_metrics = validate_model(lmv, val_loader, device, criterion)
            logging.info(f"Early stopping at epoch {epoch + 1} due to no improvement in mIou and bIou scores."
                         f"Beast epoch is {epoch + 1 - patience}")
            break
