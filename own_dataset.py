import os

import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms, datasets


class ClassificationDataset(Dataset):
    def __init__(self, data_dir, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])):
        # 文件路径
        self.data_dir = data_dir
        # 分类集
        self.classes = os.listdir(self.data_dir)
        # 双重循环构造data list，每个item由(图片路径，label)构成
        self.data = [
            (os.path.join(self.data_dir, path_name, file), label)
            for label, path_name in enumerate(self.classes)
            for file in os.listdir(os.path.join(self.data_dir, path_name))]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (image_path, label) = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, label


# 获取数据集
def get_dataloaders(data_dir):
    # 定义数据路径
    data_dir = 'data/hymenoptera_data'

    # 定义图像的变换和数据增强 transform
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 使用 ImageFolder 类自动构造分类数据集，会将文件夹名作为分类标签
    image_datasets = {x: ClassificationDataset(os.path.join(data_dir, x),
                                                        data_transforms[x])
                      for x in ['train', 'val']}
    # 构造 dataloader
    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=4,
                                                        shuffle=True, num_workers=4, drop_last=True),
                   'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=4, num_workers=4,
                                                      drop_last=True),
                   }
    return image_datasets, dataloaders


# 从tensor 转化为图片并显示
def imshow(inp, title=None):
    # 从 chw -> hwc
    inp = inp.numpy().transpose((1, 2, 0))
    # 还原归一化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    # 显示图像
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(20)


if __name__ == "__main__":
    image_datasets, dataloaders = get_dataloaders(data_dir='data/hymenoptera_data')
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # 获取类的名称集合
    class_names = image_datasets['train'].classes

    # 查看数据集
    inputs, classes = next(iter(dataloaders['train']))
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])
