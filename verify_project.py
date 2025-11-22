"""
Verify that the CycleGAN project is complete and ready to run
"""

import os
import sys
import torch
import importlib.util

def check_file(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        size_mb = size / (1024 * 1024)
        print(f"✅ {description}: {filepath} ({size_mb:.2f} MB)")
        return True
    else:
        print(f"❌ {description}: {filepath} (MISSING)")
        return False

def check_directory(dirpath, description, expected_files=None):
    """Check if a directory exists and optionally count files"""
    if os.path.exists(dirpath):
        if expected_files is not None:
            file_count = len([f for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f))])
            if file_count == expected_files:
                print(f"✅ {description}: {dirpath} ({file_count} files)")
            else:
                print(f"⚠️  {description}: {dirpath} ({file_count} files, expected {expected_files})")
        else:
            print(f"✅ {description}: {dirpath}")
        return True
    else:
        print(f"❌ {description}: {dirpath} (MISSING)")
        return False

def check_python_import(module_path, module_name):
    """Check if a Python module can be imported"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        print(f"✅ Python module: {module_name}")
        return True
    except Exception as e:
        print(f"❌ Python module: {module_name} - {str(e)}")
        return False

def main():
    print("=" * 60)
    print("CycleGAN Project Verification")
    print("=" * 60)

    all_checks_passed = True

    # 1. Check core Python files
    print("\n📝 Core Python Files:")
    core_files = [
        ("train.py", "Training script"),
        ("test.py", "Testing script"),
        ("requirements.txt", "Dependencies"),
        ("models/cyclegan_model.py", "CycleGAN model"),
        ("models/networks.py", "Network architectures"),
        ("utils/dataset.py", "Dataset loader"),
        ("utils/visualizer.py", "Visualization utilities"),
    ]

    for filepath, desc in core_files:
        if not check_file(filepath, desc):
            all_checks_passed = False

    # 2. Check pretrained models
    print("\n🤖 Pretrained Models:")
    model_files = [
        ("models/pretrained_weights/netG_A2B_epoch_final.pth", "Horse→Zebra Generator"),
        ("models/pretrained_weights/netG_B2A_epoch_final.pth", "Zebra→Horse Generator"),
        ("models/pretrained_weights/netD_A_epoch_final.pth", "Horse Discriminator"),
        ("models/pretrained_weights/netD_B_epoch_final.pth", "Zebra Discriminator"),
    ]

    for filepath, desc in model_files:
        if not check_file(filepath, desc):
            all_checks_passed = False

    # 3. Check datasets
    print("\n📁 Datasets:")
    dataset_dirs = [
        ("datasets/horse2zebra_balanced/trainA", "Training horses", 1000),
        ("datasets/horse2zebra_balanced/trainB", "Training zebras", 1000),
        ("datasets/horse2zebra_balanced/testA", "Test horses", 120),
        ("datasets/horse2zebra_balanced/testB", "Test zebras", 140),
    ]

    for dirpath, desc, expected in dataset_dirs:
        if not check_directory(dirpath, desc, expected):
            all_checks_passed = False

    # 4. Check documentation
    print("\n📚 Documentation:")
    doc_files = [
        ("README.md", "Project README"),
        ("PRETRAINED_MODELS.md", "Model documentation"),
        ("TRAINING_REPORT.md", "Training report"),
        ("training_summary.md", "Training summary"),
    ]

    for filepath, desc in doc_files:
        check_file(filepath, desc)  # Documentation is optional for running

    # 5. Check test results
    print("\n🎨 Test Results:")
    check_directory("test_results_samples", "Test result samples")

    # 6. Check training report
    print("\n📊 Training Report:")
    report_files = [
        ("training_report/training_curves.png", "Loss curves"),
        ("training_report/training_statistics.json", "Training statistics"),
    ]

    for filepath, desc in report_files:
        check_file(filepath, desc)  # Optional for running

    # 7. Check Python environment
    print("\n🐍 Python Environment:")
    print(f"Python version: {sys.version}")

    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   cuDNN version: {torch.backends.cudnn.version()}")
        else:
            print("⚠️  CUDA not available (CPU mode only)")
    except ImportError:
        print("❌ PyTorch not installed")
        all_checks_passed = False

    # 8. Test model loading
    print("\n🔧 Model Loading Test:")
    try:
        # Try to load a generator model
        checkpoint = torch.load('models/pretrained_weights/netG_A2B_epoch_final.pth',
                              map_location='cpu')
        print(f"✅ Successfully loaded generator model")
        print(f"   Model has {sum(p.numel() for p in checkpoint.values())/1e6:.1f}M parameters")
    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")
        all_checks_passed = False

    # 9. Summary
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("✅ PROJECT VERIFICATION PASSED")
        print("The project is complete and ready to run!")
        print("\nTo test the model, run:")
        print("  python test.py --dataroot datasets/horse2zebra_balanced \\")
        print("                 --checkpoints_dir models/pretrained_weights \\")
        print("                 --model_suffix _epoch_final")
        print("\nTo continue training, run:")
        print("  python train.py --dataroot datasets/horse2zebra_balanced \\")
        print("                  --continue_train --epoch_count 301")
    else:
        print("❌ PROJECT VERIFICATION FAILED")
        print("Some required files are missing. Please check the errors above.")

    print("=" * 60)

    return all_checks_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)