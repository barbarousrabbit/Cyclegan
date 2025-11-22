"""
CycleGAN 统一运行脚本
集成所有常用训练和测试功能
"""

import subprocess
import sys
import os
import argparse

def check_dataset(trainA, trainB):
    """检查数据集是否存在"""
    if not os.path.exists(trainA):
        print(f"Error: {trainA} not found!")
        return False
    if not os.path.exists(trainB):
        print(f"Error: {trainB} not found!")
        return False

    countA = len([f for f in os.listdir(trainA) if f.endswith(('.jpg', '.jpeg', '.png'))])
    countB = len([f for f in os.listdir(trainB) if f.endswith(('.jpg', '.jpeg', '.png'))])

    print(f"   Domain A: {countA} images")
    print(f"   Domain B: {countB} images")
    return True

def run_quick_test():
    """快速测试 (3 epochs)"""
    print("=" * 80)
    print("Quick Test (3 epochs)")
    print("=" * 80)

    trainA = './shared_datasets/horse2zebra/trainB'
    trainB = './shared_datasets/horse2zebra/trainA'

    print("\n[1] Checking dataset...")
    if not check_dataset(trainA, trainB):
        sys.exit(1)

    print("\n[2] Starting quick test...")
    cmd = [
        'py', '-3.11', 'train.py',
        '--exp_name', 'quick_test',
        '--dataroot_A', trainA,
        '--dataroot_B', trainB,
        '--n_epochs', '3',
        '--batch_size', '1',
        '--crop_size', '256',
        '--print_freq', '20',
        '--sample_freq', '100',
        '--save_epoch_freq', '1'
    ]

    result = subprocess.run(cmd)
    return result.returncode == 0

def run_standard_training():
    """标准训练 (batch_size=1, 200 epochs)"""
    print("=" * 80)
    print("Standard Training (Batch Size = 1)")
    print("=" * 80)

    trainA = './shared_datasets/horse2zebra/trainB'
    trainB = './shared_datasets/horse2zebra/trainA'

    print("\n[1] Checking dataset...")
    if not check_dataset(trainA, trainB):
        sys.exit(1)

    print("\n[2] Configuration:")
    print("   - Batch Size: 1")
    print("   - Crop Size: 256x256")
    print("   - Epochs: 200")
    print("   - Expected Time: ~18-20 hours")
    print("   - VRAM Usage: ~4.4 GB")

    cmd = [
        'py', '-3.11', 'train.py',
        '--exp_name', 'horse2zebra_standard',
        '--dataroot_A', trainA,
        '--dataroot_B', trainB,
        '--n_epochs', '200',
        '--batch_size', '1',
        '--crop_size', '256',
        '--lr', '0.0002',
        '--lambda_A', '10.0',
        '--lambda_B', '10.0',
        '--lambda_identity', '0.5',
        '--print_freq', '50',
        '--sample_freq', '200',
        '--save_epoch_freq', '10'
    ]

    print(f"\n[3] Starting training...\n")
    result = subprocess.run(cmd)
    return result.returncode == 0

def run_optimized_training():
    """优化训练 (batch_size=2, 200 epochs)"""
    print("=" * 80)
    print("Optimized Training (Batch Size = 2)")
    print("=" * 80)

    trainA = './shared_datasets/horse2zebra/trainB'
    trainB = './shared_datasets/horse2zebra/trainA'

    print("\n[1] Checking dataset...")
    if not check_dataset(trainA, trainB):
        sys.exit(1)

    print("\n[2] Configuration:")
    print("   - Batch Size: 2 (40-50% faster)")
    print("   - Crop Size: 256x256")
    print("   - Epochs: 200")
    print("   - Expected Time: ~11-12 hours")
    print("   - VRAM Usage: ~5.9 GB")

    cmd = [
        'py', '-3.11', 'train.py',
        '--exp_name', 'horse2zebra_optimized',
        '--dataroot_A', trainA,
        '--dataroot_B', trainB,
        '--n_epochs', '200',
        '--batch_size', '2',
        '--crop_size', '256',
        '--lr', '0.0002',
        '--lambda_A', '10.0',
        '--lambda_B', '10.0',
        '--lambda_identity', '0.5',
        '--print_freq', '50',
        '--sample_freq', '200',
        '--save_epoch_freq', '10'
    ]

    print(f"\n[3] Starting training...\n")
    result = subprocess.run(cmd)
    return result.returncode == 0

def test_optimized_params():
    """测试优化参数 (5 epochs)"""
    print("=" * 80)
    print("Testing Optimized Parameters (Batch Size = 2)")
    print("=" * 80)

    trainA = './shared_datasets/horse2zebra/trainB'
    trainB = './shared_datasets/horse2zebra/trainA'

    print("\n[1] Checking dataset...")
    if not check_dataset(trainA, trainB):
        sys.exit(1)

    print("\n[2] Testing batch_size=2 (5 epochs)...")
    print("   This verifies the optimized parameters work correctly")

    cmd = [
        'py', '-3.11', 'train.py',
        '--exp_name', 'param_test',
        '--dataroot_A', trainA,
        '--dataroot_B', trainB,
        '--n_epochs', '5',
        '--batch_size', '2',
        '--crop_size', '256',
        '--print_freq', '20',
        '--sample_freq', '100',
        '--save_epoch_freq', '5'
    ]

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n" + "=" * 80)
        print("✅ Success! Optimized parameters work correctly!")
        print("=" * 80)
        print("\nYou can now run optimized training with:")
        print("   py -3.11 run.py --mode optimized")
    else:
        print("\n" + "=" * 80)
        print("❌ Test failed! Stick with standard training.")
        print("=" * 80)

    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description='CycleGAN 统一运行工具')
    parser.add_argument('--mode', type=str,
                       choices=['quick', 'standard', 'optimized', 'test'],
                       default='quick',
                       help='运行模式: quick(快速测试), standard(标准训练), optimized(优化训练), test(测试优化参数)')

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("CycleGAN Training Tool")
    print("=" * 80)
    print(f"\nMode: {args.mode}")
    print()

    if args.mode == 'quick':
        success = run_quick_test()
    elif args.mode == 'standard':
        success = run_standard_training()
    elif args.mode == 'optimized':
        success = run_optimized_training()
    elif args.mode == 'test':
        success = test_optimized_params()

    if success:
        print("\n✅ Completed successfully!")
    else:
        print("\n❌ Failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()
