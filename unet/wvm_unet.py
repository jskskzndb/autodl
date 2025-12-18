"""
wvm_unet.py
----------------------------------------------------------------
Architecture: WVM-UNet (Wavelet-Visual-Mamba UNet)
Encoder: ConvNeXt V2
Decoder: Wavelet-Visual-Mamba (WVM) Upsampler
----------------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from pathlib import Path
import sys

# 添加路径以导入 Mamba 模块
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from decoder.hybrid_decoder import VisualStateSpaceBlock
except ImportError:
    print("❌ Error: Could not import VisualStateSpaceBlock from decoder.hybrid_decoder")
    VisualStateSpaceBlock = None


# ================================================================
# 1. Haar 小波变换
# ================================================================
class HaarWaveletTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def dwt(self, x):
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        ll = (x00 + x01 + x10 + x11) / 2
        lh = (x00 + x01 - x10 - x11) / 2
        hl = (x00 - x01 + x10 - x11) / 2
        hh = (x00 - x01 - x10 + x11) / 2
        return ll, lh, hl, hh

    def idwt(self, ll, lh, hl, hh):
        x00 = (ll + lh + hl + hh) / 2
        x01 = (ll + lh - hl - hh) / 2
        x10 = (ll - lh + hl - hh) / 2
        x11 = (ll - lh - hl + hh) / 2
        b, c, h, w = ll.shape
        out = torch.zeros(b, c, h * 2, w * 2, device=ll.device, dtype=ll.dtype)
        out[:, :, 0::2, 0::2] = x00
        out[:, :, 0::2, 1::2] = x01
        out[:, :, 1::2, 0::2] = x10
        out[:, :, 1::2, 1::2] = x11
        return out


# ================================================================
# 2. WVM 上采样器 (核心模块)
# ================================================================
class WVM_Upsampler(nn.Module):
    def __init__(self, deep_channels, skip_channels, out_channels):
        super().__init__()
        
        if VisualStateSpaceBlock is None:
            raise ImportError("Mamba module not found.")

        self.dwt_idwt = HaarWaveletTransform()
        self.mid_channels = out_channels
        
        # 投影层
        self.deep_proj = nn.Conv2d(deep_channels, self.mid_channels, 1)
        self.skip_proj = nn.Conv2d(skip_channels, self.mid_channels, 1)
        
        # 融合层 (输入 4 个分量)
        self.fusion_conv = nn.Conv2d(self.mid_channels * 4, self.mid_channels * 4, 1)
        
        # Mamba 频域筛选
        self.mamba_selector = VisualStateSpaceBlock(dim=self.mid_channels * 4)
        
        # 输出平滑
        self.out_conv = nn.Sequential(
            nn.Conv2d(self.mid_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_deep, x_skip):
        # 1. 准备 (Preparation)
        feat_ll = self.deep_proj(x_deep) 
        _, skip_lh, skip_hl, skip_hh = self.dwt_idwt.dwt(x_skip)
        
        feat_lh = self.skip_proj(skip_lh)
        feat_hl = self.skip_proj(skip_hl)
        feat_hh = self.skip_proj(skip_hh)
        
        # 2. 筛选 (Selection)
        combined = torch.cat([feat_ll, feat_lh, feat_hl, feat_hh], dim=1)
        combined = self.fusion_conv(combined)
        combined_refined = self.mamba_selector(combined)
        ref_ll, ref_lh, ref_hl, ref_hh = torch.chunk(combined_refined, 4, dim=1)
        
        # 3. 重建 (Reconstruction)
        out = self.dwt_idwt.idwt(ref_ll, ref_lh, ref_hl, ref_hh)
        return self.out_conv(out)


# ================================================================
# 3. 主模型: WVM-UNet
# ================================================================
class WVM_UNet(nn.Module):
    # **kwargs 用于接收并忽略不需要的参数 (如 use_dsis 等)
    def __init__(self, n_channels=3, n_classes=1, cnext_type='convnextv2_tiny', **kwargs):
        super().__init__()
        
        print(f"🚀 [WVM-UNet] Initializing Model...")
        if kwargs:
            print(f"   - Ignored legacy args: {list(kwargs.keys())}")
        
        self.n_classes = n_classes
        
        # --- A. Encoder: ConvNeXt V2 ---
        self.enc_model = timm.create_model(
            cnext_type, pretrained=True, features_only=True, 
            out_indices=[0, 1, 2, 3], in_chans=n_channels
        )
        c1, c2, c3, c4 = self.enc_model.feature_info.channels()
        
        # --- B. Decoder: WVM Stages ---
        # Up 1: 1/32 -> 1/16
        self.up1 = WVM_Upsampler(deep_channels=c4, skip_channels=c3, out_channels=c3)
        # Up 2: 1/16 -> 1/8
        self.up2 = WVM_Upsampler(deep_channels=c3, skip_channels=c2, out_channels=c2)
        # Up 3: 1/8 -> 1/4
        self.up3 = WVM_Upsampler(deep_channels=c2, skip_channels=c1, out_channels=c1)
        
        # --- C. Final Head ---
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(c1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        features = self.enc_model(x)
        s1, s2, s3, x4 = features[0], features[1], features[2], features[3]

        d1 = self.up1(x_deep=x4, x_skip=s3)
        d2 = self.up2(x_deep=d1, x_skip=s2)
        d3 = self.up3(x_deep=d2, x_skip=s1)
        
        d4 = self.final_up(d3)
        return self.outc(d4)