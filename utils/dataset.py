import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import random


class UnpairedImageDataset(Dataset):
    """
    未配对图像数据集
    用于加载两个域的未配对图像
    """
    def __init__(self, root_A, root_B, transform=None, mode='train'):
        """
        参数:
            root_A: 域A图像的根目录
            root_B: 域B图像的根目录
            transform: 图像转换操作
            mode: 'train' 或 'test'
        """
        self.root_A = root_A
        self.root_B = root_B
        self.transform = transform
        self.mode = mode

        # 获取所有图像文件名
        self.files_A = sorted([os.path.join(root_A, f) for f in os.listdir(root_A)
                               if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        self.files_B = sorted([os.path.join(root_B, f) for f in os.listdir(root_B)
                               if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

        self.len_A = len(self.files_A)
        self.len_B = len(self.files_B)

        if self.len_A == 0 or self.len_B == 0:
            raise ValueError(f"未找到图像文件！请检查路径:\nA域: {root_A}\nB域: {root_B}")

    def __len__(self):
        return max(self.len_A, self.len_B)

    def __getitem__(self, index):
        # 循环读取，避免索引越界
        img_A = Image.open(self.files_A[index % self.len_A]).convert('RGB')
        img_B = Image.open(self.files_B[index % self.len_B]).convert('RGB')

        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return {'A': img_A, 'B': img_B}


def get_transforms(load_size=286, crop_size=256, mode='train'):
    """
    获取图像转换操作
    参数:
        load_size: 加载图像的尺寸 (针对8GB显存优化)
        crop_size: 裁剪后的图像尺寸 (针对8GB显存优化为256)
        mode: 'train' 或 'test'
    """
    transform_list = []

    if mode == 'train':
        # 训练时的数据增强
        transform_list.append(transforms.Resize(load_size, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(crop_size))
        transform_list.append(transforms.RandomHorizontalFlip())
    else:
        # 测试时不做数据增强
        transform_list.append(transforms.Resize(crop_size, Image.BICUBIC))

    transform_list += [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    return transforms.Compose(transform_list)


def create_dataloader(root_A, root_B, batch_size=1, load_size=286, crop_size=256,
                      mode='train', num_workers=2):
    """
    创建数据加载器
    参数:
        root_A: 域A图像的根目录
        root_B: 域B图像的根目录
        batch_size: 批次大小 (针对8GB显存默认为1)
        load_size: 加载图像的尺寸
        crop_size: 裁剪后的图像尺寸
        mode: 'train' 或 'test'
        num_workers: 数据加载的工作线程数
    """
    transform = get_transforms(load_size, crop_size, mode)
    dataset = UnpairedImageDataset(root_A, root_B, transform, mode)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader
