"""
phd_unet.py
----------------------------------------------------------------
Architecture: Pure PHD-UNet
Encoder: ConvNeXt V2
Decoder: PHD (Parallel Hybrid Decoder with Mamba & DCN)
Description: 
    A standalone, clean implementation focusing solely on the 
    ConvNeXt V2 + PHD combination.
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
    from decoder.hybrid_decoder import PHD_DecoderBlock
except ImportError:
    print("❌ Error: Could not import PHD_DecoderBlock from decoder.hybrid_decoder")
    PHD_DecoderBlock = None

# ================================================================
# 1. 适配器：Up_PHD
#    负责上采样 + Padding对齐 + 拼接 + 调用 PHD Block
# ================================================================
class Up_PHD(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, skip_channels=0, use_dcn=False):
        super().__init__()
        
        # 1. 上采样部分
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            conv_in_channels = in_channels + skip_channels
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            conv_in_channels = (in_channels // 2) + skip_channels

        # 2. 核心解码部分 (PHD: Mamba + DCN)
        # 注意：这里我们默认关闭了 dubm，只保留核心的 dcn 和 mamba
        self.conv = PHD_DecoderBlock(
            in_channels=conv_in_channels, 
            out_channels=out_channels, 
            use_dcn=use_dcn, 
            use_dubm=False # 纯净版不使用 DUBM
        )

    def forward(self, x1, x2):
        # x1: Deep features (Low Res)
        # x2: Skip connection (High Res)
        x1 = self.up(x1)
        
        # 处理尺寸不匹配 (Padding)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffX != 0 or diffY != 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        # 拼接
        x = torch.cat([x2, x1], dim=1)
        
        # PHD 解码
        # 纯净版不需要 edge_prior
        return self.conv(x)


# ================================================================
# 2. 主模型: PHD_UNet
# ================================================================
class PHD_UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, cnext_type='convnextv2_tiny', 
                 use_dcn_in_phd=False, **kwargs):
        """
        Args:
            n_channels: 输入通道数
            n_classes: 输出类别数
            cnext_type: ConvNeXt 型号
            use_dcn_in_phd: 是否在 PHD 中开启 DCN (对应 --use-dcn)
            **kwargs: 接收并忽略 train.py 传入的其他多余参数 (如 use_dsis, use_wgn 等)
        """
        super().__init__()
        
        print(f"🚀 [PHD-UNet] Initializing Clean Model...")
        print(f"   - Encoder: {cnext_type}")
        print(f"   - Decoder: PHD (Mamba + {'DCNv3' if use_dcn_in_phd else 'StripConv'})")
        
        # 打印被忽略的参数 (用于调试确认)
        if kwargs:
            ignored_keys = list(kwargs.keys())
            # print(f"   - Ignored legacy args: {ignored_keys}")

        self.n_classes = n_classes
        
        # --- A. Encoder: ConvNeXt V2 ---
        self.enc_model = timm.create_model(
            cnext_type, 
            pretrained=True, 
            features_only=True, 
            out_indices=[0, 1, 2, 3], 
            in_chans=n_channels
        )
        # 获取各层通道数 (Tiny: [96, 192, 384, 768])
        c1, c2, c3, c4 = self.enc_model.feature_info.channels()
        
        # --- B. Decoder: PHD Stages ---
        # Up 1: 1/32 (c4) + 1/16 (c3) -> c3
        self.up1 = Up_PHD(c4, c3, skip_channels=c3, use_dcn=use_dcn_in_phd)
        
        # Up 2: 1/16 (c3) + 1/8 (c2) -> c2
        self.up2 = Up_PHD(c3, c2, skip_channels=c2, use_dcn=use_dcn_in_phd)
        
        # Up 3: 1/8 (c2) + 1/4 (c1) -> c1
        self.up3 = Up_PHD(c2, c1, skip_channels=c1, use_dcn=use_dcn_in_phd)
        
        # --- C. Final Head ---
        # ConvNeXt Stem 下采样了 4 倍，所以最后需要上采样 4 倍恢复到原图尺寸
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(c1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # 1. Encoder
        features = self.enc_model(x)
        s1, s2, s3, x4 = features[0], features[1], features[2], features[3]

        # 2. Decoder (PHD)
        d1 = self.up1(x4, s3)
        d2 = self.up2(d1, s2)
        d3 = self.up3(d2, s1)
        
        # 3. Head
        d4 = self.final_up(d3)
        logits = self.outc(d4)
        
        return logits