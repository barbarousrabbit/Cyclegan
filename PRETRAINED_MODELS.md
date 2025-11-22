# Pretrained Models - CycleGAN Horse2Zebra

## 🎯 模型信息

本仓库包含完整的预训练CycleGAN模型，已在平衡的Horse2Zebra数据集上训练300个epochs。

### 模型文件位置
所有预训练模型位于 `models/pretrained_weights/` 目录：

| 模型文件 | 描述 | 大小 | 用途 |
|---------|------|------|------|
| `netG_A2B_epoch_final.pth` | 马→斑马生成器 | 31MB | 将马的图像转换为斑马 |
| `netG_B2A_epoch_final.pth` | 斑马→马生成器 | 31MB | 将斑马的图像转换为马 |
| `netD_A_epoch_final.pth` | 马判别器 | 11MB | 判断图像是否为真实的马 |
| `netD_B_epoch_final.pth` | 斑马判别器 | 11MB | 判断图像是否为真实的斑马 |

**总大小**: ~84MB

## 🚀 快速使用

### 1. 使用预训练模型进行测试

```python
# 测试单张图片
python test.py --dataroot datasets/horse2zebra_balanced/testA \
               --checkpoints_dir models/pretrained_weights \
               --model_suffix _epoch_final \
               --direction A2B
```

### 2. 批量测试

```python
# 测试整个测试集
python test.py --dataroot datasets/horse2zebra_balanced \
               --checkpoints_dir models/pretrained_weights \
               --model_suffix _epoch_final \
               --num_test 100
```

### 3. 在代码中加载模型

```python
import torch
from models.networks import define_G

# 创建生成器
netG_A2B = define_G(input_nc=3, output_nc=3, ngf=64, netG='resnet_9blocks')

# 加载预训练权重
checkpoint = torch.load('models/pretrained_weights/netG_A2B_epoch_final.pth')
netG_A2B.load_state_dict(checkpoint)
netG_A2B.eval()

# 使用模型
with torch.no_grad():
    fake_B = netG_A2B(real_A)
```

## 📊 训练详情

### 训练配置
- **数据集**: 平衡的Horse2Zebra数据集
  - 训练集: 1000张马 + 1000张斑马
  - 测试集: 120张马 + 140张斑马
- **训练轮数**: 300 epochs
- **批大小**: 1
- **图像尺寸**: 256×256
- **学习率**: 0.0002
- **优化器**: Adam

### 损失函数权重
- **λ_A** (循环一致性A): 10.0
- **λ_B** (循环一致性B): 10.0
- **λ_identity** (身份映射): 0.5

### 硬件环境
- **GPU**: NVIDIA GeForce RTX 3060 (12GB VRAM)
- **训练时间**: ~16-18小时

## 📈 性能评估

### 定性结果
- **A2B (马→斑马)**:
  - 纹理转换效果中等
  - 在复杂背景下有一定的纹理混淆
  - 整体形状保持良好

- **B2A (斑马→马)**:
  - 转换质量较好
  - 颜色还原自然
  - 细节保持完整

### 测试结果示例
查看 `test_results_samples/` 目录中的140张对比图片，展示了模型的实际转换效果。

## 🔄 继续训练

如果想在这些预训练模型的基础上继续训练：

```python
python train.py --dataroot datasets/horse2zebra_balanced \
                --continue_train \
                --epoch_count 301 \
                --n_epochs 400 \
                --checkpoints_dir models/pretrained_weights
```

## ⚠️ 注意事项

1. **模型兼容性**: 这些模型使用PyTorch 2.0+训练，确保您的环境兼容
2. **输入格式**: 模型期望输入为归一化到[-1, 1]范围的RGB图像
3. **图像尺寸**: 最佳性能使用256×256像素，其他尺寸可能需要调整
4. **GPU内存**: 推理时至少需要2GB显存

## 📝 引用

如果您使用这些预训练模型，请引用原始的CycleGAN论文：

```bibtex
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={IEEE International Conference on Computer Vision (ICCV)},
  year={2017}
}
```

## 📄 许可

这些模型仅供学术研究使用。商业使用请联系原作者。