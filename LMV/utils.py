from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import logging
import os
import torch
import numpy as np

__all__ = (
    "setup_logging",
    "calculate_metrics_cls",
    "load_checkpoint",
    "calculate_seg_metrics",
)


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


def calculate_metrics_cls(y_true, y_pred, average='macro'):
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


def calculate_seg_metrics(seg_predicted, seg_labels):
    """
    计算评估指标，包括 Precision, Recall, F1, mIoU 和 bIoU。
    """
    # Ensure the predictions are binary (0 or 1)
    seg_predicted = (seg_predicted > 0.5).astype(np.uint8)  # 转换为二进制标签
    seg_labels = seg_labels.astype(np.uint8)  # 确保标签也是二进制标签

    # Flatten the arrays to calculate metrics
    seg_predicted = seg_predicted.flatten()
    seg_labels = seg_labels.flatten()

    # Precision, Recall, F1
    precision = precision_score(seg_labels, seg_predicted, zero_division=0)
    recall = recall_score(seg_labels, seg_predicted, zero_division=0)
    f1 = f1_score(seg_labels, seg_predicted, zero_division=0)

    # mIoU (Mean Intersection over Union)
    intersection = np.logical_and(seg_labels, seg_predicted).sum()
    union = np.logical_or(seg_labels, seg_predicted).sum()
    miou = intersection / (union + 1e-6)

    # bIoU (Boundary IoU)
    # boundary_labels = np.logical_xor(seg_labels, np.pad(seg_labels[1:, :-1], ((1, 0), (0, 1)), 'constant'))
    # boundary_pred = np.logical_xor(seg_predicted, np.pad(seg_predicted[1:, :-1], ((1, 0), (0, 1)), 'constant'))
    # boundary_intersection = np.logical_and(boundary_labels, boundary_pred).sum()
    # boundary_union = np.logical_or(boundary_labels, boundary_pred).sum()
    # biou = boundary_intersection / (boundary_union + 1e-6)

    # Return the metrics as a dictionary
    return {
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'mIoU': miou,
        # 'bIoU': biou
    }


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """加载模型和优化器的状态字典"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    best_f1 = checkpoint['epoch_f1']
    best_f5 = checkpoint['epoch_f5']
    return model, optimizer, scheduler, epoch, best_f1, best_f5
