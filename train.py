import os
import torch
import argparse
from tqdm import tqdm
from models import CycleGANModel
from utils import create_dataloader, Visualizer, save_sample_images


def train(args):
    """训练CycleGAN"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')

    # 创建目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    # 创建数据加载器
    print(f'加载数据集...')
    print(f'域A路径: {args.dataroot_A}')
    print(f'域B路径: {args.dataroot_B}')

    try:
        train_loader = create_dataloader(
            args.dataroot_A,
            args.dataroot_B,
            batch_size=args.batch_size,
            load_size=args.load_size,
            crop_size=args.crop_size,
            mode='train',
            num_workers=args.num_workers
        )
        print(f'数据集加载成功！共 {len(train_loader)} 个批次')
    except ValueError as e:
        print(f'错误: {e}')
        return

    # 创建模型
    print('创建CycleGAN模型...')
    model = CycleGANModel(
        device=device,
        lr=args.lr,
        beta1=args.beta1,
        lambda_A=args.lambda_A,
        lambda_B=args.lambda_B,
        lambda_identity=args.lambda_identity
    )
    print('模型创建成功！')

    # 加载预训练模型（如果指定）
    if args.load_epoch is not None:
        print(f'加载预训练模型 (Epoch {args.load_epoch})...')
        model.load_networks(args.load_epoch, args.checkpoint_dir)
        args.start_epoch = int(args.load_epoch) if args.load_epoch.isdigit() else 0
        print(f'模型加载成功！将从 Epoch {args.start_epoch + 1} 继续训练')

    # 创建可视化器
    visualizer = Visualizer(log_dir=args.log_dir)

    # 训练循环
    print(f'\n开始训练，共 {args.n_epochs} 个epoch...')
    print(f'优化参数 (针对RTX 4060 8GB):')
    print(f'  - Batch Size: {args.batch_size}')
    print(f'  - 图像尺寸: {args.crop_size}x{args.crop_size}')
    print(f'  - ResNet块数: 6')
    print(f'  - 学习率: {args.lr}')
    print(f'  - Lambda A: {args.lambda_A}, Lambda B: {args.lambda_B}')
    print('-' * 80)

    global_step = args.start_epoch * len(train_loader)
    for epoch in range(args.start_epoch, args.n_epochs):
        model.netG_A2B.train()
        model.netG_B2A.train()

        # 进度条
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.n_epochs}')

        for i, batch in enumerate(pbar):
            real_A = batch['A']
            real_B = batch['B']

            # 设置输入
            model.set_input(real_A, real_B)

            # 优化参数
            model.optimize_parameters()

            # 获取损失
            losses = model.get_current_losses()

            # 更新进度条
            loss_str = ', '.join([f'{k}: {v:.4f}' for k, v in losses.items()])
            pbar.set_postfix_str(loss_str)

            # 记录到TensorBoard
            if global_step % args.print_freq == 0:
                visualizer.plot_losses(losses, global_step)

            # 保存样本图像
            if global_step % args.sample_freq == 0:
                with torch.no_grad():
                    save_sample_images(
                        model.real_A, model.real_B,
                        model.fake_A, model.fake_B,
                        model.rec_A, model.rec_B,
                        os.path.join(args.sample_dir, f'step_{global_step}.png')
                    )

                    # 记录到TensorBoard
                    visualizer.plot_images({
                        'Real_A': model.real_A,
                        'Fake_B': model.fake_B,
                        'Rec_A': model.rec_A,
                        'Real_B': model.real_B,
                        'Fake_A': model.fake_A,
                        'Rec_B': model.rec_B
                    }, global_step)

            global_step += 1

        # 保存模型
        if (epoch + 1) % args.save_epoch_freq == 0:
            print(f'\n保存模型 (Epoch {epoch+1})...')
            model.save_networks(epoch + 1, args.checkpoint_dir)

        # 学习率衰减
        if epoch >= args.n_epochs // 2:
            # 线性衰减学习率
            lr = args.lr * (1.0 - (epoch - args.n_epochs // 2) / (args.n_epochs // 2))
            for param_group in model.optimizer_G.param_groups:
                param_group['lr'] = lr
            for param_group in model.optimizer_D.param_groups:
                param_group['lr'] = lr

    # 保存最终模型
    print('\n保存最终模型...')
    model.save_networks('final', args.checkpoint_dir)
    print('训练完成！')

    visualizer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练CycleGAN')

    # 数据参数
    parser.add_argument('--dataroot_A', type=str, default='./data/trainA',
                        help='域A训练图像目录')
    parser.add_argument('--dataroot_B', type=str, default='./data/trainB',
                        help='域B训练图像目录')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批次大小 (针对8GB显存，建议1)')
    parser.add_argument('--load_size', type=int, default=286,
                        help='加载图像尺寸')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='裁剪图像尺寸 (针对8GB显存，建议256)')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='数据加载线程数')

    # 训练参数
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='起始epoch (用于恢复训练)')
    parser.add_argument('--load_epoch', type=str, default=None,
                        help='加载的模型epoch (例如: 60, final)')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='初始学习率')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Adam优化器的beta1参数')

    # 损失权重
    parser.add_argument('--lambda_A', type=float, default=10.0,
                        help='A域循环一致性损失权重')
    parser.add_argument('--lambda_B', type=float, default=10.0,
                        help='B域循环一致性损失权重')
    parser.add_argument('--lambda_identity', type=float, default=0.5,
                        help='身份映射损失权重')

    # 保存和日志参数
    parser.add_argument('--exp_name', type=str, default='default',
                        help='实验名称 (用于组织输出文件夹)')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='模型保存目录 (默认: experiments/{exp_name}/checkpoints)')
    parser.add_argument('--sample_dir', type=str, default=None,
                        help='样本图像保存目录 (默认: experiments/{exp_name}/samples)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='TensorBoard日志目录 (默认: experiments/{exp_name}/logs)')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='打印/记录频率')
    parser.add_argument('--sample_freq', type=int, default=100,
                        help='保存样本图像频率')
    parser.add_argument('--save_epoch_freq', type=int, default=20,
                        help='保存模型的epoch频率')

    args = parser.parse_args()

    # Set default paths based on experiment name
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f'./experiments/{args.exp_name}/checkpoints'
    if args.sample_dir is None:
        args.sample_dir = f'./experiments/{args.exp_name}/samples'
    if args.log_dir is None:
        args.log_dir = f'./experiments/{args.exp_name}/logs'

    train(args)
