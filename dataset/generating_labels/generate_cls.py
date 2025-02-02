import json
import os
from natsort import natsorted
from tqdm import tqdm

from LMV.config import Extension

"""
    Suppose you have a classification directory
    Its structure is like this 
        train/
        ├── blur
        ├── broken
        ├── close
        └── stable

    Run this script:
        1. You will get a JSON in folder (classification_dict), 
            containing a label_map dict like:
            label_map = {
                'blur': 0,
                'broken': 1,
                'close': 2,
                'stable': 3
            }

        2. You will get a txt file, containing classification labels, like
            train0.jpg 3
            train1.jpg 1
            train2.jpg 1
                ···
            train9.jpg 3
            train10.jpg 3
"""


def create_label_map(train_dir, dataset_name='unnamed'):
    if not os.path.isdir(train_dir):
        raise ValueError(f"The specified train directory does not exist: {train_dir}")

    # 获取所有子目录名
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]

    if not classes:
        raise ValueError(f"No subdirectories found in the train directory: {train_dir}")

    # 对类别名称进行排序，确保标签分配的一致性
    classes.sort()
    label_map = {cls_name: idx for idx, cls_name in enumerate(classes)}

    # 修正目录名拼写错误，并确保目录存在
    output_dir = 'classification_dict'  # 修正拼写
    os.makedirs(output_dir, exist_ok=True)  # 创建目录（如果不存在）

    output_json = os.path.join(output_dir, f"{dataset_name}.json")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, indent=4, ensure_ascii=False)
    print(f"Label map has been saved to {output_json}")

    return label_map


def save_image_labels_to_txt(dataset_name, train_dir, output_txt):
    image_label_pairs = []
    label_map = create_label_map(train_dir=train_dir, dataset_name=dataset_name)

    for root, _, files in os.walk(train_dir):
        folder_name = os.path.basename(root)
        label = label_map.get(folder_name, -1)
        if label == -1:
            continue  # 跳过未定义标签的文件夹

        for filename in tqdm(files, desc=f'{os.path.basename(root)}'):
            if filename.lower().endswith(Extension):
                image_label_pairs.append((filename, label))

    # 使用natsorted进行自然排序
    image_label_pairs = natsorted(image_label_pairs, key=lambda x: x[0])

    # 保存到txt文件
    with open(output_txt, 'w', encoding='utf-8') as file:
        for filename, label in image_label_pairs:
            file.write(f"{filename} {label}\n")

    print(f"Image labels have been saved to {output_txt}")


if __name__ == '__main__':
    train_directory = r'D:\Code\pycode\Data_All\Database_of_CV\Classification\imagenet\train'  # 请根据实际情况修改路径
    output_txt = r'D:\Code\pycode\Data_All\Database_of_CV\Classification\imagenet\train_cls.txt'  # 输出txt文件路径

    save_image_labels_to_txt(
        dataset_name="ImageNet",
        train_dir=train_directory,
        output_txt=output_txt,
    )
    print(f"Image labels saved to {output_txt}")
