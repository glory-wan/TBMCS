import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


def lmv_collate_fn(batch):
    images = []
    cls_labels = []
    det_labels = []
    seg_labels = []

    for sample in batch:
        image, cls_label, det_label, seg_label = sample
        images.append(image)
        cls_labels.append(cls_label)
        det_labels.append(det_label)
        seg_labels.append(seg_label)

    images = torch.stack(images, dim=0)
    cls_labels = torch.tensor(cls_labels, dtype=torch.long)
    seg_labels = torch.stack(seg_labels, dim=0)

    return {
        'images': images,
        'cls_labels': cls_labels,
        'det_labels': det_labels,
        'seg_labels': seg_labels
    }


def leTransformer(imgz=640):
    transform = transforms.Compose([
        LetterboxTransform(new_shape=(imgz, imgz)),
        transforms.ToTensor(),
    ])

    return transform


class LetterboxTransform:
    def __init__(self, new_shape=(640, 640), color=(114, 114, 114), auto=True, scale_fill=False, scaleup=True,
                 stride=32):
        self.new_shape = new_shape
        self.color = color
        self.auto = auto
        self.scale_fill = scale_fill
        self.scaleup = scaleup
        self.stride = stride

    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)

        image, _, _, _ = letterbox(image, new_shape=self.new_shape, color=self.color, auto=self.auto,
                                   scale_fill=self.scale_fill, scaleup=self.scaleup, stride=self.stride)

        image = Image.fromarray(image)
        return image


def letterbox(image, new_shape=(640, 640), color=(114, 114, 114), auto=True, scale_fill=False, scaleup=True, stride=32):
    shape = image.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 计算缩放比例
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    # 计算新尺寸
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 填充

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # 根据stride填充

    elif scale_fill:  # 缩放并填充
        dw, dh = 0, 0
        new_unpad = (new_shape[1], new_shape[0])
        r = new_shape[1] / shape[1], new_shape[0] / shape[0]  # 宽高缩放比率

    dw /= 2  # 分成两边填充
    dh /= 2

    if shape[::-1] != new_unpad:  # 调整大小
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(np.ceil(dh)), int(np.floor(dh))
    left, right = int(np.ceil(dw)), int(np.floor(dw))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 添加填充
    return image, r, (dw, dh), new_unpad

# if __name__ == '__main__':
#     transform = transforms.Compose([
#         LetterboxTransform(new_shape=(640, 640)),  # 自定义的 letterbox 变换
#         # transforms.ToTensor(),  # 转换为 Tensor
#         # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
#     ])
#
#     image_path = r"D:\Code\pycode\dataset_all\tf_version3\images\train\train0.jpg"
#     img = Image.open(image_path)
#
#     # 应用组合变换
#     transformed_image = transform(img)
#
#     print(f"original image shape: {img.size}")
#     print(f"transformed image shape: {transformed_image.size}")
#     transformed_image.show
