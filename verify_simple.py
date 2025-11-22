"""
Simple verification script for CycleGAN project completeness
"""

import os
import sys
import torch

def check_files():
    """Check if all necessary files exist"""

    print("=" * 60)
    print("CycleGAN Project Verification")
    print("=" * 60)

    missing_files = []

    # Core files
    print("\nChecking Core Files:")
    core_files = [
        "train.py",
        "test.py",
        "requirements.txt",
        "models/cyclegan_model.py",
        "models/networks.py",
        "utils/dataset.py",
        "utils/visualizer.py"
    ]

    for f in core_files:
        if os.path.exists(f):
            print(f"  [OK] {f}")
        else:
            print(f"  [MISSING] {f}")
            missing_files.append(f)

    # Model files
    print("\nChecking Pretrained Models:")
    models = [
        "models/pretrained_weights/netG_A2B_epoch_final.pth",
        "models/pretrained_weights/netG_B2A_epoch_final.pth",
        "models/pretrained_weights/netD_A_epoch_final.pth",
        "models/pretrained_weights/netD_B_epoch_final.pth"
    ]

    for m in models:
        if os.path.exists(m):
            size_mb = os.path.getsize(m) / (1024 * 1024)
            print(f"  [OK] {m} ({size_mb:.1f} MB)")
        else:
            print(f"  [MISSING] {m}")
            missing_files.append(m)

    # Dataset directories
    print("\nChecking Datasets:")
    datasets = {
        "datasets/horse2zebra_balanced/trainA": 1000,
        "datasets/horse2zebra_balanced/trainB": 1000,
        "datasets/horse2zebra_balanced/testA": 120,
        "datasets/horse2zebra_balanced/testB": 140
    }

    for d, expected_count in datasets.items():
        if os.path.exists(d):
            file_count = len([f for f in os.listdir(d) if f.endswith('.jpg')])
            if file_count == expected_count:
                print(f"  [OK] {d} ({file_count} images)")
            else:
                print(f"  [WARNING] {d} ({file_count} images, expected {expected_count})")
        else:
            print(f"  [MISSING] {d}")
            missing_files.append(d)

    # Check PyTorch
    print("\nChecking Environment:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  CUDA: Available ({torch.cuda.get_device_name(0)})")
    else:
        print(f"  CUDA: Not available (CPU mode only)")

    # Summary
    print("\n" + "=" * 60)
    if not missing_files:
        print("PROJECT VERIFICATION: PASSED")
        print("\nThe project is complete and ready to run!")
        print("\nQuick test command:")
        print("  python test.py --dataroot datasets/horse2zebra_balanced")
        print("\nTo use pretrained models:")
        print("  --checkpoints_dir models/pretrained_weights")
        print("  --model_suffix _epoch_final")
    else:
        print("PROJECT VERIFICATION: FAILED")
        print(f"\n{len(missing_files)} files/directories are missing:")
        for f in missing_files[:5]:
            print(f"  - {f}")
        if len(missing_files) > 5:
            print(f"  ... and {len(missing_files)-5} more")

    print("=" * 60)
    return len(missing_files) == 0

if __name__ == "__main__":
    success = check_files()
    sys.exit(0 if success else 1)