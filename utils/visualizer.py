import os
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class Visualizer:
    """可视化训练过程"""
    def __init__(self, log_dir='./runs'):
        """
        参数:
            log_dir: TensorBoard日志目录
        """
        self.writer = SummaryWriter(log_dir)

    def plot_losses(self, losses, step):
        """记录损失到TensorBoard"""
        for name, value in losses.items():
            self.writer.add_scalar(f'Loss/{name}', value, step)

    def plot_images(self, images_dict, step):
        """记录图像到TensorBoard"""
        for name, images in images_dict.items():
            # 将图像从[-1, 1]转换到[0, 1]
            images = (images + 1) / 2.0
            grid = vutils.make_grid(images, normalize=False)
            self.writer.add_image(name, grid, step)

    def save_images(self, images_dict, save_dir, epoch):
        """保存图像到磁盘"""
        os.makedirs(save_dir, exist_ok=True)
        for name, images in images_dict.items():
            # 将图像从[-1, 1]转换到[0, 1]
            images = (images + 1) / 2.0
            save_path = os.path.join(save_dir, f'{name}_epoch_{epoch}.png')
            vutils.save_image(images, save_path, normalize=False)

    def close(self):
        """关闭writer"""
        self.writer.close()


def denormalize(tensor):
    """
    将归一化的tensor转换回[0, 1]范围
    参数:
        tensor: 形状为[B, C, H, W]的tensor，范围在[-1, 1]
    返回:
        范围在[0, 1]的tensor
    """
    return (tensor + 1) / 2.0


def save_sample_images(real_A, real_B, fake_A, fake_B, rec_A, rec_B, save_path):
    """
    保存样本图像对比图
    参数:
        real_A, real_B: 真实图像
        fake_A, fake_B: 生成的假图像
        rec_A, rec_B: 重建的图像
        save_path: 保存路径
    """
    # 转换到[0, 1]范围
    real_A = denormalize(real_A.cpu())
    real_B = denormalize(real_B.cpu())
    fake_A = denormalize(fake_A.cpu())
    fake_B = denormalize(fake_B.cpu())
    rec_A = denormalize(rec_A.cpu())
    rec_B = denormalize(rec_B.cpu())

    # 创建图像网格
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 第一行：A域
    axes[0, 0].imshow(real_A[0].permute(1, 2, 0))
    axes[0, 0].set_title('Real A')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(fake_B[0].permute(1, 2, 0))
    axes[0, 1].set_title('Fake B (A→B)')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(rec_A[0].permute(1, 2, 0))
    axes[0, 2].set_title('Reconstructed A (A→B→A)')
    axes[0, 2].axis('off')

    # 第二行：B域
    axes[1, 0].imshow(real_B[0].permute(1, 2, 0))
    axes[1, 0].set_title('Real B')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(fake_A[0].permute(1, 2, 0))
    axes[1, 1].set_title('Fake A (B→A)')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(rec_B[0].permute(1, 2, 0))
    axes[1, 2].set_title('Reconstructed B (B→A→B)')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
