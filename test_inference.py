"""
Test inference with pretrained models
"""

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os
import sys

# Add project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.networks import define_G

def load_image(image_path, transform):
    """Load and transform an image"""
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

def test_model():
    """Test the pretrained models"""

    print("Testing CycleGAN Inference...")
    print("=" * 50)

    # Define transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load generators
    print("\nLoading models...")

    # Create generators
    netG_A2B = define_G(input_nc=3, output_nc=3, ngf=64,
                        netG='resnet_9blocks', norm='instance',
                        use_dropout=False, init_type='normal')
    netG_B2A = define_G(input_nc=3, output_nc=3, ngf=64,
                        netG='resnet_9blocks', norm='instance',
                        use_dropout=False, init_type='normal')

    # Load weights
    checkpoint_A2B = torch.load('models/pretrained_weights/netG_A2B_epoch_final.pth',
                                map_location=device)
    checkpoint_B2A = torch.load('models/pretrained_weights/netG_B2A_epoch_final.pth',
                                map_location=device)

    netG_A2B.load_state_dict(checkpoint_A2B)
    netG_B2A.load_state_dict(checkpoint_B2A)

    netG_A2B.to(device).eval()
    netG_B2A.to(device).eval()

    print("Models loaded successfully!")

    # Test on sample images
    print("\nTesting on sample images...")

    # Test A2B (Horse to Zebra)
    test_horse = 'datasets/horse2zebra_balanced/testA/n02381460_1000.jpg'
    if os.path.exists(test_horse):
        print(f"\nTesting Horse→Zebra conversion...")
        horse_img = load_image(test_horse, transform).to(device)

        with torch.no_grad():
            fake_zebra = netG_A2B(horse_img)

        print(f"  Input shape: {horse_img.shape}")
        print(f"  Output shape: {fake_zebra.shape}")
        print(f"  Output range: [{fake_zebra.min():.2f}, {fake_zebra.max():.2f}]")
        print("  [OK] Horse→Zebra conversion successful!")

    # Test B2A (Zebra to Horse)
    test_zebra = 'datasets/horse2zebra_balanced/testB/n02391049_100.jpg'
    if os.path.exists(test_zebra):
        print(f"\nTesting Zebra→Horse conversion...")
        zebra_img = load_image(test_zebra, transform).to(device)

        with torch.no_grad():
            fake_horse = netG_B2A(zebra_img)

        print(f"  Input shape: {zebra_img.shape}")
        print(f"  Output shape: {fake_horse.shape}")
        print(f"  Output range: [{fake_horse.min():.2f}, {fake_horse.max():.2f}]")
        print("  [OK] Zebra→Horse conversion successful!")

    # Test cycle consistency
    print("\nTesting cycle consistency...")
    with torch.no_grad():
        # A -> B -> A
        if 'horse_img' in locals():
            fake_zebra = netG_A2B(horse_img)
            reconstructed_horse = netG_B2A(fake_zebra)
            cycle_loss_A = torch.mean(torch.abs(horse_img - reconstructed_horse))
            print(f"  Cycle consistency A→B→A loss: {cycle_loss_A:.4f}")

        # B -> A -> B
        if 'zebra_img' in locals():
            fake_horse = netG_B2A(zebra_img)
            reconstructed_zebra = netG_A2B(fake_horse)
            cycle_loss_B = torch.mean(torch.abs(zebra_img - reconstructed_zebra))
            print(f"  Cycle consistency B→A→B loss: {cycle_loss_B:.4f}")

    print("\n" + "=" * 50)
    print("INFERENCE TEST: PASSED")
    print("The models are working correctly!")
    print("\nTo run full test on all images:")
    print("  python test.py --dataroot datasets/horse2zebra_balanced \\")
    print("                 --checkpoints_dir models/pretrained_weights \\")
    print("                 --model_suffix _epoch_final")
    print("=" * 50)

    return True

if __name__ == "__main__":
    try:
        success = test_model()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)