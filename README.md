# CycleGAN Implementation

这是一个针对**华硕天选Air 2024 (RTX 4060 8GB显存)** 优化的CycleGAN实现。

## 硬件配置

- **GPU**: NVIDIA GeForce RTX 4060 Laptop (8GB GDDR6)
- **CPU**: AMD Ryzen 7 R7-8845H / AI 9 HX 370
- **内存**: 16GB/32GB LPDDR5 7500MHz
- **功耗**: 100W TGP

## 针对8GB显存的优化

本实现针对8GB显存进行了以下优化：

### 标准配置 (保守)
1. **Batch Size**: 1
2. **图像尺寸**: 256x256
3. **ResNet块**: 6个 (原版CycleGAN使用9个)
4. **显存占用**: ~4.4 GB
5. **训练时间**: ~18-20小时 (200 epochs)

### 优化配置 (推荐)
1. **Batch Size**: 2 (速度提升40-50%)
2. **图像尺寸**: 256x256
3. **ResNet块**: 6个
4. **显存占用**: ~5.9 GB (安全范围内)
5. **训练时间**: ~11-12小时 (200 epochs)

### 性能对比
| 配置 | Batch Size | 显存占用 | 训练速度 | 200 Epochs耗时 |
|------|-----------|---------|---------|---------------|
| 标准 | 1 | 4.4 GB | 2.7 it/s | 18-20小时 |
| 优化 | 2 | 5.9 GB | 4.0 it/s | 11-12小时 |

## 项目结构

```
Cyclegan/
├── models/                      # 模型定义
│   ├── __init__.py
│   ├── networks.py             # 生成器和判别器网络
│   └── cyclegan_model.py       # CycleGAN模型
├── utils/                       # 工具函数
│   ├── __init__.py
│   ├── dataset.py              # 数据加载
│   └── visualizer.py           # 可视化工具
├── shared_datasets/             # 共享数据集目录
│   └── horse2zebra/            # Horse2Zebra数据集
│       ├── trainA/             # 马的训练图片 (1,067张)
│       ├── trainB/             # 斑马的训练图片 (1,334张)
│       ├── testA/              # 马的测试图片 (120张)
│       └── testB/              # 斑马的测试图片 (140张)
├── experiments/                 # 实验目录 (每个实验独立组织)
│   ├── demo_test/              # Demo训练实验
│   │   ├── checkpoints/        # 保存的模型
│   │   ├── samples/            # 训练样本图片
│   │   └── logs/               # TensorBoard日志
│   ├── horse2zebra_test/       # Horse2Zebra测试实验 (3 epochs)
│   │   ├── dataset_info/       # 数据集信息和引用
│   │   ├── checkpoints/        # 保存的模型
│   │   ├── samples/            # 训练样本图片
│   │   └── logs/               # TensorBoard日志
│   └── horse2zebra_full/       # Horse2Zebra完整训练 (200 epochs)
│       ├── dataset_info/       # 数据集信息和引用
│       ├── checkpoints/        # 保存的模型
│       ├── samples/            # 训练样本图片
│       ├── logs/               # TensorBoard日志
│       └── test_results/       # 测试结果
├── train.py                    # 训练脚本
├── test.py                     # 测试脚本
├── test_horse2zebra.py         # Horse2Zebra快速测试脚本
├── quick_start.py              # 环境检查脚本
├── config.py                   # 配置文件
├── requirements.txt            # 依赖包
└── README.md                   # 本文件
```

**目录说明**：
- `shared_datasets/`: 存放所有数据集，避免重复存储大文件
- `experiments/{exp_name}/`: 每个实验独立组织，包含所有相关文件
  - `dataset_info/`: 记录该实验使用的数据集信息和路径
  - `checkpoints/`: 训练过程中保存的模型
  - `samples/`: 训练过程中生成的样本图片
  - `logs/`: TensorBoard训练日志
  - `test_results/`: 测试阶段的输出结果

## 安装

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据集

Horse2Zebra数据集已包含在 `shared_datasets/` 目录中。如需添加其他数据集，将数据组织到 `shared_datasets/` 目录：

```
shared_datasets/
└── your_dataset/
    ├── trainA/     # 域A的训练图像 (建议500-1000张)
    ├── trainB/     # 域B的训练图像 (建议500-1000张)
    ├── testA/      # 域A的测试图像
    └── testB/      # 域B的测试图像
```

**数据集要求**：
- 图像格式: PNG, JPG, JPEG, BMP
- 图像数量: 每个域至少500-1000张（更多更好）
- 图像质量: 高质量、清晰的图像
- 图像内容: 确保两个域有相似的内容分布

**常用数据集**：
- horse2zebra: 马↔斑马 (已包含)
- apple2orange: 苹果↔橙子
- summer2winter: 夏天↔冬天
- monet2photo: 莫奈画作↔照片

## 使用方法

### 快速开始

使用统一运行脚本 `run.py`:

```bash
# 1. 快速测试 (3 epochs, 推荐首次运行)
py -3.11 run.py --mode quick

# 2. 测试优化参数 (5 epochs, 验证 batch_size=2)
py -3.11 run.py --mode test

# 3. 标准训练 (batch_size=1, 18-20小时)
py -3.11 run.py --mode standard

# 4. 优化训练 (batch_size=2, 11-12小时, 推荐)
py -3.11 run.py --mode optimized
```

### 训练模式对比

| 模式 | Batch Size | 训练时间 | 显存占用 | 推荐场景 |
|------|-----------|---------|---------|---------|
| quick | 1 | 5-10分钟 | 4.4 GB | 首次测试 |
| test | 2 | 10-15分钟 | 5.9 GB | 验证优化参数 |
| standard | 1 | 18-20小时 | 4.4 GB | 保守训练 |
| **optimized** | **2** | **11-12小时** | **5.9 GB** | **推荐** ⭐ |

结果将自动保存到 `experiments/{exp_name}/` 目录，包含：
- `checkpoints/`: 训练的模型文件
- `samples/`: 训练过程中的样本图片
- `logs/`: TensorBoard日志
- `dataset_info/`: 数据集信息

自定义参数训练：

```bash
py -3.11 train.py \
    --exp_name my_experiment \
    --dataroot_A ./shared_datasets/your_dataset/trainA \
    --dataroot_B ./shared_datasets/your_dataset/trainB \
    --batch_size 1 \
    --crop_size 256 \
    --n_epochs 200 \
    --lr 0.0002 \
    --lambda_A 10.0 \
    --lambda_B 10.0
```

**重要参数说明**：

- `--batch_size`: 批次大小（标准配置1，优化配置2）
- `--crop_size`: 图像尺寸（建议256，最大512）
- `--n_epochs`: 训练轮数（建议200-300）
- `--lr`: 学习率（默认0.0002）
- `--lambda_A` / `--lambda_B`: 循环一致性损失权重（默认10.0）
- `--lambda_identity`: 身份映射损失权重（默认0.5）

### 监控训练

使用TensorBoard监控训练过程：

```bash
# 监控特定实验
tensorboard --logdir=./experiments/horse2zebra_full/logs

# 或监控所有实验
tensorboard --logdir=./experiments
```

然后在浏览器中打开 `http://localhost:6006`

### 测试

**1. 双向测试（需要A和B两个域的测试数据）**：

```bash
py -3.11 test.py \
    --exp_name horse2zebra_full \
    --mode paired \
    --dataroot_A ./shared_datasets/horse2zebra/testA \
    --dataroot_B ./shared_datasets/horse2zebra/testB \
    --epoch final
```

结果将保存到 `experiments/horse2zebra_full/test_results/`

**2. 单向测试（只测试一个方向，例如A→B）**：

```bash
py -3.11 test.py \
    --exp_name horse2zebra_full \
    --mode single \
    --input_dir ./shared_datasets/horse2zebra/testA \
    --direction A2B \
    --epoch final
```

**测试参数说明**：

- `--exp_name`: 实验名称（用于定位模型）
- `--mode`: 测试模式（`paired` 或 `single`）
- `--epoch`: 要加载的模型（例如 `100` 或 `final`）
- `--save_mode`: 保存模式
  - `separate`: 分别保存原图和生成图
  - `comparison`: 保存对比图（原图 | 生成图）
- `--direction`: 转换方向（`A2B` 或 `B2A`，仅single模式）

## 性能优化建议

### 如果遇到显存不足 (OOM)

1. 减小图像尺寸：`--crop_size 128`
2. 确保批次大小为1：`--batch_size 1`
3. 关闭其他占用显存的程序
4. 减少ResNet块数量（需修改代码）

### 如果想加快训练

1. 增加数据加载线程：`--num_workers 4`
2. 减少训练轮数：`--n_epochs 100`
3. 使用较小的图像尺寸：`--crop_size 128`

### 如果想提升生成质量

1. 增加训练轮数：`--n_epochs 300`
2. 使用更大的数据集（>1000张/域）
3. 调整损失权重：
   - 如果循环一致性不够好，增加 `--lambda_A` 和 `--lambda_B`
   - 如果有颜色偏移问题，增加 `--lambda_identity`
4. 使用更大的图像尺寸（如果显存允许）：`--crop_size 512`

## 训练时间估计

基于RTX 4060 8GB的训练时间估计：

- **图像尺寸256x256, Batch Size 1**:
  - 每个epoch约需5-10分钟（取决于数据集大小）
  - 200 epochs约需17-33小时

- **图像尺寸128x128, Batch Size 1**:
  - 每个epoch约需2-4分钟
  - 200 epochs约需7-13小时

## 显存使用

典型显存占用：

- **256x256, Batch 1**: ~5-6GB
- **512x512, Batch 1**: ~7-8GB（接近上限）
- **128x128, Batch 1**: ~3-4GB

**注意**: 实际显存占用会根据当前系统状态波动。建议训练时关闭其他占用显存的程序（浏览器、游戏等）。

## CycleGAN原理简介

CycleGAN是一种用于图像到图像转换的深度学习模型，特点是**不需要配对的训练数据**。

### 核心组件

1. **两个生成器**：
   - G_A2B: 将域A的图像转换为域B
   - G_B2A: 将域B的图像转换为域A

2. **两个判别器**：
   - D_A: 判断图像是否来自域A
   - D_B: 判断图像是否来自域B

### 损失函数

1. **对抗损失 (Adversarial Loss)**: 使生成的图像看起来真实
2. **循环一致性损失 (Cycle Consistency Loss)**: 确保 A→B→A 能恢复原图
3. **身份映射损失 (Identity Loss)**: 保持颜色一致性（可选）

## 常见问题

### Q1: 训练过程中出现 "CUDA out of memory" 错误？

**A**: 减小 `--crop_size` 到 128，确保 `--batch_size` 为 1，并关闭其他占用显存的程序。

### Q2: 生成的图像质量不好？

**A**:
- 确保数据集质量高且数量足够（>1000张/域）
- 增加训练轮数到300
- 调整损失权重参数
- 检查两个域的图像内容是否相似

### Q3: 训练速度很慢？

**A**:
- 增加 `--num_workers`
- 使用较小的图像尺寸
- 确保使用GPU训练（检查 `torch.cuda.is_available()`）

### Q4: 生成的图像有颜色偏移？

**A**: 增加身份映射损失权重：`--lambda_identity 1.0`

### Q5: 如何恢复训练？

**A**: 使用 `load_networks()` 方法加载之前保存的模型，然后继续训练（需要修改训练脚本）。

## 参考资料

- **论文**: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
- **官方实现**: [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

## License

MIT License

## 致谢

本实现基于CycleGAN论文，并针对华硕天选Air 2024笔记本进行了优化。
