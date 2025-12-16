"""
boundary_stream.py
[创新模块] 显式边界流 (Explicit Boundary Stream)
功能：从浅层特征中提取高频边缘信息，用于：
1. 计算边缘损失 (Auxiliary Loss)
2. 为 D-UBM 提供几何先验 (Geometric Prior)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryStream(nn.Module):
    def __init__(self, in_channels, internal_channels=64):
        super().__init__()
        
        # 1. 降维 (1x1 Conv)
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True)
        )
        
        # 2. 边缘特征提取 (Gated Shape Layer 的简化版 - 强调高频)
        # 使用残差结构保护梯度
        self.res_conv = nn.Sequential(
            nn.Conv2d(internal_channels, internal_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(internal_channels, internal_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(internal_channels)
        )
        
        # 3. 边缘预测头 (输出 1 通道 logits)
        self.edge_out = nn.Conv2d(internal_channels, 1, kernel_size=1)

    def forward(self, x):
        # x: Encoder 浅层特征 (e.g., 1/4 分辨率)
        feat = self.conv_in(x)
        residual = self.res_conv(feat)
        feat = F.relu(feat + residual)
        
        # 输出 logits (训练时用于计算 BCEWithLogitsLoss，推理时 sigmoid 后用于先验)
        edge_logits = self.edge_out(feat)
        
        return edge_logits