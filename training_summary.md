# CycleGAN Training Summary - Horse2Zebra (300 Epochs)

## Training Configuration

### Dataset
- **Training Set**:
  - Horse images (trainA): 1000 images
  - Zebra images (trainB): 1000 images
  - Balanced dataset for improved training stability

- **Test Set**:
  - Horse images (testA): 120 images
  - Zebra images (testB): 140 images

### Model Parameters
- **Architecture**: CycleGAN with ResNet generator
- **Training Epochs**: 300
- **Loss Weights**:
  - λ_A (Cycle consistency A): 10.0
  - λ_B (Cycle consistency B): 10.0
  - λ_identity: 0.5
- **Learning Rate**: 0.0002
- **Batch Size**: 1

### Hardware & Training Time
- **GPU**: NVIDIA GeForce RTX 3060 (12GB VRAM)
- **Training Duration**: Approximately 16-18 hours for 300 epochs
- **Framework**: PyTorch

## File Structure

```
Cyclegan/
├── train.py                     # Main training script
├── test.py                      # Testing/inference script
├── requirements.txt             # Python dependencies
├── models/
│   ├── cyclegan_model.py       # CycleGAN model implementation
│   └── networks.py              # Network architectures
├── utils/
│   ├── dataset.py              # Dataset loading utilities
│   └── visualizer.py           # Visualization tools
├── datasets/
│   └── horse2zebra_balanced/
│       ├── trainA/             # 1000 horse training images
│       ├── trainB/             # 1000 zebra training images
│       ├── testA/              # 120 horse test images
│       └── testB/              # 140 zebra test images
└── experiments/
    └── horse2zebra_balanced/
        ├── checkpoints/         # Saved model weights
        │   ├── netG_A2B_epoch_final.pth (31MB)
        │   ├── netG_B2A_epoch_final.pth (31MB)
        │   ├── netD_A_epoch_final.pth (11MB)
        │   └── netD_B_epoch_final.pth (11MB)
        ├── test_results_300epochs/  # Test results
        │   ├── A2B/            # Horse to Zebra results
        │   └── B2A/            # Zebra to Horse results
        ├── samples/            # Training progress samples
        └── logs/              # TensorBoard logs
```

## Training Results

### Performance Observations
- **A2B (Horse→Zebra)**: Moderate quality conversion with some texture issues
- **B2A (Zebra→Horse)**: Better quality conversion with more consistent results
- **Key Challenges**:
  - Texture preservation in A2B direction
  - Color consistency in complex backgrounds
  - Detail retention in transformed images

### Test Results
- Generated 140 comparison images showing:
  - Original input image
  - Transformed output
  - Reconstructed image (cycle consistency)
- Results available in `test_results_samples/` directory

## Model Files

### Checkpoint Files (Not in Git due to size)
- `netG_A2B_epoch_final.pth`: Generator for Horse→Zebra (31MB)
- `netG_B2A_epoch_final.pth`: Generator for Zebra→Horse (31MB)
- `netD_A_epoch_final.pth`: Discriminator for horses (11MB)
- `netD_B_epoch_final.pth`: Discriminator for zebras (11MB)

Total model size: ~84MB

### Usage
To use the trained models:
```python
python test.py --dataroot datasets/horse2zebra_balanced \
               --checkpoints_dir experiments/horse2zebra_balanced/checkpoints \
               --results_dir results/horse2zebra_balanced
```

## Repository
- **GitHub**: https://github.com/barbarousrabbit/Cyclegan
- **Model Weights**: Available via GitHub Release or cloud storage

## Notes
- This training used a balanced dataset (1000:1000) instead of the original unbalanced dataset
- Training completed successfully after 300 epochs
- TensorBoard logs available for detailed training metrics visualization