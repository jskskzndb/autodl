"""
unet/unet_cnext_standard.py
[Ablation Baseline] ConvNeXt V2 + Standard UNet Decoder
用途: 终极消融实验 Baseline
结构: 
  - Encoder: ConvNeXt V2 Base (Pretrained)
  - Decoder: Standard DoubleConv (Conv-BN-ReLU x2)
  - Skip: Direct Concatenation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class DoubleConv(nn.Module):
    """
    标准的 UNet 解码单元: (Conv3x3 -> BN -> ReLU) * 2
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Standard_Up(nn.Module):
    """
    标准的上采样模块
    Upsample -> Concat -> DoubleConv
    """
    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        # 使用双线性插值上采样
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 计算拼接后的通道数
        # 输入经过上采样后通道不变，拼接上 skip_channels
        concat_channels = in_channels + skip_channels
        
        # 通过双卷积将通道数融合并降维
        self.conv = DoubleConv(concat_channels, out_channels, mid_channels=in_channels // 2)

    def forward(self, x1, x2):
        # x1: 深层特征 (Decoder Input)
        # x2: 浅层特征 (Skip Connection)
        x1 = self.up(x1)
        
        # 处理可能的尺寸不匹配 (Padding)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffX != 0 or diffY != 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        
        # 拼接
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet_CNext_Standard(nn.Module):
    def __init__(self, n_classes=1, cnext_type='convnextv2_base', **kwargs):
        super().__init__()
        
        # 🔥 [修复1] 必须初始化 n_classes，train.py 需要读取它
        self.n_classes = n_classes
        
        print(f"🧪 [Ablation Baseline] ConvNeXt + Standard UNet Decoder")
        print(f"   - Encoder: {cnext_type}")
        print(f"   - Decoder: Standard DoubleConv")
        
        # --- 1. Encoder: ConvNeXt V2 ---
        # 🔥 [修复2] 改名为 'spatial_encoder' 以匹配 train.py 的差分学习率逻辑
        self.spatial_encoder = timm.create_model(
            cnext_type, 
            pretrained=True, 
            features_only=True, 
            out_indices=(0, 1, 2, 3)
        )
        
        # 获取通道数 (Base: [128, 256, 512, 1024])
        dims = self.spatial_encoder.feature_info.channels()
        c1, c2, c3, c4 = dims

        # --- 2. Decoder: Standard UNet Style ---
        # Up 1: Input=1024(s4), Skip=512(s3) -> Out=512
        self.up1 = Standard_Up(c4, c3, skip_channels=c3)
        
        # Up 2: Input=512(d1), Skip=256(s2) -> Out=256
        self.up2 = Standard_Up(c3, c2, skip_channels=c2)
        
        # Up 3: Input=256(d2), Skip=128(s1) -> Out=128
        self.up3 = Standard_Up(c2, c1, skip_channels=c1)
        
        # --- 3. Final Output ---
        # ConvNeXt Stem 缩放了 4 倍，所以最后需要上采样 4 倍
        self.final_up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.outc = nn.Conv2d(c1, n_classes, kernel_size=1)

    def forward(self, x):
        # === Encoder ===
        # 🔥 [修复2] 这里调用也要改名
        s1, s2, s3, s4 = self.spatial_encoder(x)

        # === Decoder ===
        d1 = self.up1(s4, s3)
        d2 = self.up2(d1, s2)
        d3 = self.up3(d2, s1)
        
        # === Output ===
        out = self.final_up(d3)
        logits = self.outc(out)
        
        return logits