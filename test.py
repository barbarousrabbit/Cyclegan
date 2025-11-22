import os
import torch
import argparse
from tqdm import tqdm
from models import CycleGANModel
from utils import create_dataloader
import torchvision.utils as vutils


def test(args):
    """测试CycleGAN"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'A2B'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'B2A'), exist_ok=True)

    # 创建数据加载器
    print(f'加载测试数据集...')
    print(f'域A路径: {args.dataroot_A}')
    print(f'域B路径: {args.dataroot_B}')

    try:
        test_loader = create_dataloader(
            args.dataroot_A,
            args.dataroot_B,
            batch_size=1,  # 测试时batch_size=1
            crop_size=args.crop_size,
            mode='test',
            num_workers=args.num_workers
        )
        print(f'数据集加载成功！共 {len(test_loader)} 个测试样本')
    except ValueError as e:
        print(f'错误: {e}')
        return

    # 创建模型
    print('创建CycleGAN模型...')
    model = CycleGANModel(device=device)

    # 加载预训练模型
    print(f'加载模型: epoch {args.epoch}...')
    try:
        model.load_networks(args.epoch, args.checkpoint_dir)
        print('模型加载成功！')
    except Exception as e:
        print(f'模型加载失败: {e}')
        return

    # 设置为评估模式
    model.netG_A2B.eval()
    model.netG_B2A.eval()

    # 测试循环
    print('\n开始测试...')
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc='测试进度')):
            real_A = batch['A'].to(device)
            real_B = batch['B'].to(device)

            # 生成假图像
            fake_B = model.netG_A2B(real_A)  # A→B
            fake_A = model.netG_B2A(real_B)  # B→A

            # 转换到[0, 1]范围
            fake_B = (fake_B + 1) / 2.0
            fake_A = (fake_A + 1) / 2.0
            real_A = (real_A + 1) / 2.0
            real_B = (real_B + 1) / 2.0

            # 保存图像
            if args.save_mode == 'separate':
                # 分别保存真实图像和生成图像
                vutils.save_image(fake_B, os.path.join(args.output_dir, 'A2B', f'{i:04d}_fake.png'))
                vutils.save_image(fake_A, os.path.join(args.output_dir, 'B2A', f'{i:04d}_fake.png'))

                if args.save_input:
                    vutils.save_image(real_A, os.path.join(args.output_dir, 'A2B', f'{i:04d}_real.png'))
                    vutils.save_image(real_B, os.path.join(args.output_dir, 'B2A', f'{i:04d}_real.png'))

            elif args.save_mode == 'comparison':
                # 保存对比图（真实图像 vs 生成图像）
                comparison_A2B = torch.cat([real_A, fake_B], dim=3)  # 水平拼接
                comparison_B2A = torch.cat([real_B, fake_A], dim=3)

                vutils.save_image(comparison_A2B, os.path.join(args.output_dir, 'A2B', f'{i:04d}_comparison.png'))
                vutils.save_image(comparison_B2A, os.path.join(args.output_dir, 'B2A', f'{i:04d}_comparison.png'))

    print(f'\n测试完成！结果保存在: {args.output_dir}')
    print(f'  - A→B转换结果: {os.path.join(args.output_dir, "A2B")}')
    print(f'  - B→A转换结果: {os.path.join(args.output_dir, "B2A")}')


def test_single_direction(args):
    """单向测试（只测试A→B或B→A）"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 创建模型
    model = CycleGANModel(device=device)

    # 加载预训练模型
    print(f'加载模型: epoch {args.epoch}...')
    model.load_networks(args.epoch, args.checkpoint_dir)

    # 选择生成器
    if args.direction == 'A2B':
        generator = model.netG_A2B
        input_dir = args.input_dir
    else:
        generator = model.netG_B2A
        input_dir = args.input_dir

    generator.eval()

    # 获取输入图像
    from PIL import Image
    from utils import get_transforms

    transform = get_transforms(crop_size=args.crop_size, mode='test')
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    print(f'在 {input_dir} 中找到 {len(image_files)} 张图像')
    print(f'开始 {args.direction} 转换...')

    with torch.no_grad():
        for img_file in tqdm(image_files):
            # 加载并转换图像
            img_path = os.path.join(input_dir, img_file)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)

            # 生成图像
            output = generator(img_tensor)

            # 转换到[0, 1]范围
            output = (output + 1) / 2.0
            img_tensor = (img_tensor + 1) / 2.0

            # 保存
            if args.save_mode == 'separate':
                save_path = os.path.join(args.output_dir, img_file)
                vutils.save_image(output, save_path)
            else:
                comparison = torch.cat([img_tensor, output], dim=3)
                save_path = os.path.join(args.output_dir, img_file)
                vutils.save_image(comparison, save_path)

    print(f'转换完成！结果保存在: {args.output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='测试CycleGAN')

    # 基本参数
    parser.add_argument('--mode', type=str, default='paired', choices=['paired', 'single'],
                        help='测试模式: paired (需要A和B数据) 或 single (只需要一个方向)')
    parser.add_argument('--exp_name', type=str, default='default',
                        help='实验名称 (用于定位模型和输出)')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='模型目录 (默认: experiments/{exp_name}/checkpoints)')
    parser.add_argument('--epoch', type=str, default='final',
                        help='要加载的模型epoch (或 "final")')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录 (默认: experiments/{exp_name}/test_results)')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='图像尺寸')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='数据加载线程数')

    # paired模式参数
    parser.add_argument('--dataroot_A', type=str, default='./data/testA',
                        help='域A测试图像目录')
    parser.add_argument('--dataroot_B', type=str, default='./data/testB',
                        help='域B测试图像目录')

    # single模式参数
    parser.add_argument('--input_dir', type=str, default='./data/test_input',
                        help='输入图像目录 (single模式)')
    parser.add_argument('--direction', type=str, default='A2B', choices=['A2B', 'B2A'],
                        help='转换方向: A2B 或 B2A (single模式)')

    # 保存选项
    parser.add_argument('--save_mode', type=str, default='comparison', choices=['separate', 'comparison'],
                        help='保存模式: separate (分别保存) 或 comparison (对比图)')
    parser.add_argument('--save_input', action='store_true',
                        help='是否保存输入图像 (仅在separate模式下有效)')

    args = parser.parse_args()

    # Set default paths based on experiment name
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f'./experiments/{args.exp_name}/checkpoints'
    if args.output_dir is None:
        args.output_dir = f'./experiments/{args.exp_name}/test_results'

    if args.mode == 'paired':
        test(args)
    else:
        test_single_direction(args)
