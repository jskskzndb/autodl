# WGN增强ResNet50-UNet使用指南

## 概述

本指南介绍如何使用新集成的WGN（Wavelet-Gated Recursive Convolution）增强功能来改进ResNet50-UNet模型的性能。

## 什么是WGN增强？

WGN增强是一种创新的深度学习技术，结合了：

1. **小波变换（DWT）**: 将特征分解为低频（结构信息）和高频（细节信息）子带
2. **递归门控卷积（GnConv）**: 提供高阶空间交互能力
3. **双域处理**: 分别处理不同频域的特征，实现更精细的特征学习

## 架构设计（方案三：混合架构）

我们采用的是混合架构方案，在ResNet50编码器的基础上添加WGN增强：

```
ResNet50编码器 + WGN增强模块
├── conv1 + bn1 + relu + maxpool
├── layer1 (256通道) → WGN增强1 (order_low=3, order_high=2)
├── layer2 (512通道) → WGN增强2 (order_low=4, order_high=3)  
├── layer3 (1024通道) → WGN增强3 (order_low=5, order_high=4)
└── 标准UNet解码器
```

## 使用方法

### 1. 基本使用

启用WGN增强的ResNet50-UNet：

```bash
python train.py --use-resnet-encoder --use-wgn-enhancement
```

### 2. 自定义WGN配置

#### 使用基础order参数：

```bash
python train.py --use-resnet-encoder --use-wgn-enhancement --wgn-base-order 4
```

这将使用以下配置：
- layer1: (4, 3)
- layer2: (5, 4)  
- layer3: (6, 5)

#### 使用完全自定义配置：

```bash
python train.py --use-resnet-encoder --use-wgn-enhancement --wgn-orders "2,1,3,2,4,3"
```

格式：`"layer1_low,layer1_high,layer2_low,layer2_high,layer3_low,layer3_high"`

### 3. 完整训练命令示例

```bash
# 基础WGN增强训练
python train.py \
    --epochs 100 \
    --batch-size 16 \
    --learning-rate 1e-4 \
    --use-resnet-encoder \
    --use-wgn-enhancement \
    --use-cafm \
    --amp

# 自定义WGN配置训练
python train.py \
    --epochs 100 \
    --batch-size 16 \
    --learning-rate 1e-4 \
    --use-resnet-encoder \
    --use-wgn-enhancement \
    --wgn-orders "3,2,4,3,5,4" \
    --use-cafm \
    --amp
```

## 参数说明

### 新增命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use-wgn-enhancement` | flag | False | 启用WGN增强模块 |
| `--wgn-base-order` | int | 3 | WGN块的基础order值 |
| `--wgn-orders` | str | None | 自定义WGN orders配置 |

### WGN Order参数含义

- **order_low**: 低频子带的递归深度，控制全局结构信息的处理复杂度
- **order_high**: 高频子带的递归深度，控制局部细节信息的处理复杂度
- **更大的order**: 更强的特征交互能力，但参数量和计算量也更大

## 性能特点

### 优势

1. **多尺度特征学习**: 小波分解提供天然的多尺度信息
2. **频域感知**: 分别优化结构和细节信息的处理
3. **高阶交互**: GnConv提供比传统卷积更强的特征交互
4. **参数效率**: 可通过调整order控制模型复杂度

### 适用场景

- **医学图像分割**: 需要精确边缘和细节保持
- **高分辨率图像**: 小波变换对大图像更有效
- **多尺度目标**: 同时包含大小不同的分割目标

### 参数量对比

根据不同的order配置，WGN增强会增加一定的参数量：

- 默认配置 (3,2,4,3,5,4): 约增加15-25%参数量
- 轻量配置 (2,1,3,2,4,3): 约增加10-15%参数量
- 重量配置 (4,3,5,4,6,5): 约增加25-35%参数量

## 依赖要求

确保安装以下依赖：

```bash
pip install pytorch-wavelets>=1.3.0 timm>=0.6.0
```

或使用项目的requirements.txt：

```bash
pip install -r requirements.txt
```

## 实验建议

### 1. 对比实验

建议进行以下对比实验：

```bash
# 基线：标准ResNet50-UNet
python train.py --use-resnet-encoder

# 对比1：ResNet50-UNet + CAFM
python train.py --use-resnet-encoder --use-cafm

# 对比2：ResNet50-UNet + WGN增强
python train.py --use-resnet-encoder --use-wgn-enhancement

# 对比3：ResNet50-UNet + CAFM + WGN增强（完整版）
python train.py --use-resnet-encoder --use-cafm --use-wgn-enhancement
```

### 2. 超参数调优

WGN增强模型可能需要调整以下超参数：

- **学习率**: 可能需要稍微降低（如从1e-4到8e-5）
- **批次大小**: 由于参数量增加，可能需要减小批次大小
- **正则化**: 可能需要增加权重衰减

### 3. 消融研究

测试不同order组合的效果：

```bash
# 轻量级配置
python train.py --use-resnet-encoder --use-wgn-enhancement --wgn-orders "2,1,2,1,3,2"

# 中等配置（默认）
python train.py --use-resnet-encoder --use-wgn-enhancement --wgn-orders "3,2,4,3,5,4"

# 重量级配置
python train.py --use-resnet-encoder --use-wgn-enhancement --wgn-orders "4,3,5,4,6,5"
```

## 故障排除

### 常见问题

1. **内存不足**: 降低批次大小或使用较小的order值
2. **训练速度慢**: WGN增强会增加计算量，这是正常的
3. **收敛困难**: 尝试降低学习率或增加warmup

### 调试技巧

1. 使用`--amp`启用混合精度训练节省显存
2. 监控训练日志中的参数量信息
3. 对比不同配置的训练曲线

## 总结

WGN增强为ResNet50-UNet提供了强大的特征学习能力，特别适合需要精细分割的任务。通过合理配置order参数，可以在性能和效率之间找到最佳平衡点。

建议从默认配置开始，根据具体任务和硬件条件进行调优。
