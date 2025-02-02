import torch
import torch.nn as nn


def dice_loss(inputs, targets, num_masks):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)

    return loss.sum() / num_masks


def sigmoid_focal_loss(inputs, targets, num_masks, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        intersection = torch.sum(y_pred * y_true)
        false_positive = torch.sum(y_pred * (1 - y_true))
        false_negative = torch.sum((1 - y_pred) * y_true)

        tversky_index = (intersection + self.smooth) / (
                intersection + self.alpha * false_negative + self.beta * false_positive + self.smooth)
        tversky_loss = 1 - tversky_index

        return tversky_loss

# if __name__ == '__main__':
#     y_pred = torch.randn(2, 1, 4, 4)  # 模拟模型输出
#     y_pred = torch.sigmoid(y_pred)  # 使用 sigmoid 激活函数
#
#     y_true = torch.randint(0, 2, (2, 1, 4, 4)).float()  # 模拟真实标签
#
#     # 计算 Tversky Loss
#     loss_fn = TverskyLoss(alpha=0.7, beta=0.3)
#     loss = loss_fn(y_pred, y_true)
#
#     print("Tversky Loss:", loss.item())
