"""
CycleGAN快速开始脚本
检查环境并显示使用指南
"""

import sys
import os
import torch
import subprocess

# 设置UTF-8输出
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def check_environment():
    """检查环境配置"""
    print("=" * 80)
    print("CycleGAN 环境检查")
    print("=" * 80)

    # Python版本
    print(f"\n[1] Python版本: {sys.version}")
    py_version = sys.version_info
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 7):
        print("    ⚠️  警告: 建议使用Python 3.7或更高版本")
    else:
        print("    ✓ Python版本正常")

    # PyTorch
    print(f"\n[2] PyTorch版本: {torch.__version__}")
    print(f"    CUDA可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"    GPU数量: {torch.cuda.device_count()}")
        print(f"    当前GPU: {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"    总显存: {gpu_memory:.2f} GB")

        # 检查可用显存
        torch.cuda.empty_cache()
        free_memory = torch.cuda.mem_get_info()[0] / 1024**3
        used_memory = gpu_memory - free_memory
        print(f"    已用显存: {used_memory:.2f} GB")
        print(f"    可用显存: {free_memory:.2f} GB")

        if free_memory < 2.0:
            print("    ⚠️  警告: 可用显存不足2GB，建议关闭其他占用显存的程序")
        else:
            print("    ✓ 显存充足")
    else:
        print("    ⚠️  警告: CUDA不可用，将使用CPU训练（速度会很慢）")

    # 检查必要的包
    print("\n[3] 检查依赖包:")
    required_packages = [
        'torch', 'torchvision', 'numpy', 'PIL',
        'matplotlib', 'tensorboard', 'tqdm'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            if package == 'PIL':
                __import__('PIL')
            else:
                __import__(package)
            print(f"    ✓ {package}")
        except ImportError:
            print(f"    ✗ {package} (缺失)")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n    ⚠️  缺失的包: {', '.join(missing_packages)}")
        print("    请运行: pip install -r requirements.txt")
    else:
        print("    ✓ 所有依赖包已安装")

    # 检查数据目录
    print("\n[4] 检查数据目录:")
    import os
    data_dirs = ['data/trainA', 'data/trainB', 'data/testA', 'data/testB']
    all_empty = True

    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            files = [f for f in os.listdir(data_dir)
                    if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            if files:
                print(f"    ✓ {data_dir}: {len(files)} 张图像")
                all_empty = False
            else:
                print(f"    ○ {data_dir}: 空目录")
        else:
            print(f"    ✗ {data_dir}: 目录不存在")

    if all_empty:
        print("\n    ⚠️  所有数据目录为空，请先准备数据集")
        print("    参考: data/README.md")

    print("\n" + "=" * 80)


def show_usage_guide():
    """显示使用指南"""
    print("\nCycleGAN 使用指南")
    print("=" * 80)

    print("\n【步骤1】准备数据集")
    print("-" * 80)
    print("将图像放入以下目录:")
    print("  data/trainA/  - 域A的训练图像 (建议500-1000张)")
    print("  data/trainB/  - 域B的训练图像 (建议500-1000张)")
    print("  data/testA/   - 域A的测试图像")
    print("  data/testB/   - 域B的测试图像")
    print("\n详细说明请查看: data/README.md")

    print("\n【步骤2】开始训练")
    print("-" * 80)
    print("基本训练命令:")
    print("  python train.py --dataroot_A ./data/trainA --dataroot_B ./data/trainB")
    print("\n针对RTX 4060优化的训练命令:")
    print("  python train.py \\")
    print("      --dataroot_A ./data/trainA \\")
    print("      --dataroot_B ./data/trainB \\")
    print("      --batch_size 1 \\")
    print("      --crop_size 256 \\")
    print("      --n_epochs 200")

    print("\n【步骤3】监控训练")
    print("-" * 80)
    print("使用TensorBoard监控训练过程:")
    print("  tensorboard --logdir=./runs")
    print("然后在浏览器中打开: http://localhost:6006")

    print("\n【步骤4】测试模型")
    print("-" * 80)
    print("测试命令:")
    print("  python test.py \\")
    print("      --mode paired \\")
    print("      --dataroot_A ./data/testA \\")
    print("      --dataroot_B ./data/testB \\")
    print("      --epoch final")

    print("\n【重要提示】")
    print("-" * 80)
    print("1. 训练时确保显存充足（至少2GB可用）")
    print("2. 如遇到显存不足，减小--crop_size到128")
    print("3. 训练需要较长时间（200 epochs约17-33小时）")
    print("4. 定期检查results/samples/目录查看训练效果")
    print("5. 详细文档请查看: README.md")

    print("\n" + "=" * 80)


def main():
    """主函数"""
    check_environment()
    show_usage_guide()

    print("\n准备就绪！现在可以开始训练CycleGAN了。")
    print("\n如有问题，请查看README.md或检查上述环境配置。")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
