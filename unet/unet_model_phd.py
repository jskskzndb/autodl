"""
cnext_fme_unet.py
----------------------------------------------------------------
Architecture: ConvNeXt V2 + PHD Decoder + FME (Frequency-Mamba Enhancement)
Description: 
    这是你的最终涨点模型。
    - Encoder: ConvNeXt V2 (Base/Tiny)
    - Decoder: PHD (Parallel Hybrid Decoder)
    - Skip: FME (频域 Mamba 增强，可消融)
----------------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from pathlib import Path
import sys

# 添加路径以导入 Decoder 模块
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入核心 PHD 模块
try:
    from decoder.hybrid_decoder import PHD_DecoderBlock, VisualStateSpaceBlock
except ImportError:
    print("❌ Error: Could not import modules from decoder.hybrid_decoder")
    PHD_DecoderBlock = None
    VisualStateSpaceBlock = None

# ================================================================
# 1. 工具类: Haar 小波变换
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
        out = torch.zeros(ll.size(0), ll.size(1), ll.size(2)*2, ll.size(3)*2, device=ll.device)
        out[:, :, 0::2, 0::2] = x00
        out[:, :, 0::2, 1::2] = x01
        out[:, :, 1::2, 0::2] = x10
        out[:, :, 1::2, 1::2] = x11
        return out

# ================================================================
# 2. 核心涨点模块: FME (Frequency-Mamba Enhancement)
# ================================================================
class FME_Block(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.dwt_idwt = HaarWaveletTransform()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_dim = channels
        
        # Mamba 序列建模 (2x2 Patch = 4 频段)
        self.mamba = VisualStateSpaceBlock(dim=channels)
        
        # 权重生成
        self.weight_gen = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, H, W]
        ll, lh, hl, hh = self.dwt_idwt.dwt(x)
        
        # 生成 Token 并堆叠
        t_ll = self.avg_pool(ll).flatten(1)
        t_lh = self.avg_pool(lh).flatten(1)
        t_hl = self.avg_pool(hl).flatten(1)
        t_hh = self.avg_pool(hh).flatten(1)
        
        # [B, C, 4] -> [B, C, 2, 2] (伪装成图片)
        tokens = torch.stack([t_ll, t_lh, t_hl, t_hh], dim=2) 
        tokens = tokens.view(x.size(0), self.channel_dim, 2, 2)
        
        # Mamba 交互
        feats_out = self.mamba(tokens) 
        feats_out = feats_out.view(x.size(0), self.channel_dim, 4)
        
        # 生成权重
        w_ll = self.weight_gen(feats_out[:, :, 0]).view(x.size(0), -1, 1, 1)
        w_lh = self.weight_gen(feats_out[:, :, 1]).view(x.size(0), -1, 1, 1)
        w_hl = self.weight_gen(feats_out[:, :, 2]).view(x.size(0), -1, 1, 1)
        w_hh = self.weight_gen(feats_out[:, :, 3]).view(x.size(0), -1, 1, 1)
        
        out = self.dwt_idwt.idwt(ll*w_ll, lh*w_lh, hl*w_hl, hh*w_hh)
        return out + x

# ================================================================
# 3. 适配器: Up_PHD (集成 FME 开关)
# ================================================================
class Up_PHD(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, skip_channels=0, use_dcn=False, use_fme=False):
        super().__init__()
        
        # 🔥 FME 开关
        if use_fme and skip_channels > 0:
            self.fme = FME_Block(skip_channels)
        else:
            self.fme = None

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            conv_in_channels = in_channels + skip_channels
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            conv_in_channels = (in_channels // 2) + skip_channels

        # 核心解码
        self.conv = PHD_DecoderBlock(
            in_channels=conv_in_channels, 
            out_channels=out_channels, 
            use_dcn=use_dcn, 
            use_dubm=False 
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Padding
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffX != 0 or diffY != 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        # 🔥 FME 增强
        if self.fme is not None:
            x2 = self.fme(x2)
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# ================================================================
# 4. 主模型: ConvNeXt + FME + PHD
# ================================================================
class CNext_FME_UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, cnext_type='convnextv2_tiny', 
                 use_dcn_in_phd=False, use_fme=False, **kwargs):
        super().__init__()
        
        print(f"🚀 [CNext-FME-UNet] Initializing...")
        print(f"   - Encoder: {cnext_type}")
        print(f"   - FME Module: {'✅ Enabled' if use_fme else '❌ Disabled'}")
        
        self.n_classes = n_classes
        
        # --- Encoder: ConvNeXt V2 ---
        self.enc_model = timm.create_model(
            cnext_type, 
            pretrained=True, 
            features_only=True, 
            out_indices=[0, 1, 2, 3], 
            in_chans=n_channels
        )
        c1, c2, c3, c4 = self.enc_model.feature_info.channels()
        
        # --- Decoder ---
        self.up1 = Up_PHD(c4, c3, skip_channels=c3, use_dcn=use_dcn_in_phd, use_fme=use_fme)
        self.up2 = Up_PHD(c3, c2, skip_channels=c2, use_dcn=use_dcn_in_phd, use_fme=use_fme)
        self.up3 = Up_PHD(c2, c1, skip_channels=c1, use_dcn=use_dcn_in_phd, use_fme=use_fme)
        
        # --- Head ---
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

        d1 = self.up1(x4, s3)
        d2 = self.up2(d1, s2)
        d3 = self.up3(d2, s1)
        
        d4 = self.final_up(d3)
        return self.outc(d4)