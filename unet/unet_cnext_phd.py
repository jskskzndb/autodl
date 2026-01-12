"""
unet_cnext_phd.py
[Ablation Model] Pure ConvNeXt V2 + PHD Decoder
用途: 验证 PHD 解码器 (Mamba + DCN + SK-Fusion) 在标准 U-Net 架构下的有效性。
特点: 
  - 移除 Dual-Stream, Frequency Branch, DSIS, UNet3+ 等冗余模块。
  - 仅保留核心的 ConvNeXt 编码器和 PHD 解码器。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import sys

# ================================================================
# 1. Mamba 核心组件 (从 mamba_helper.py 提取)
# ================================================================
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    print("⚠️ Warning: mamba-ssm not found. PHD Decoder will fail if Mamba is required.")
    HAS_MAMBA = False

class MambaLayer2D(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if not HAS_MAMBA:
            raise ImportError("Mamba module not found. Please install mamba-ssm.")
            
        self.mamba = Mamba(
            d_model=dim,      # 输入通道数
            d_state=d_state,  # 状态维度
            d_conv=d_conv,    # 局部卷积宽度
            expand=expand     # 扩张系数
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        with torch.cuda.amp.autocast(enabled=False): # 强制 FP32 防止溢出
            x = x.float()
            x_seq = x.flatten(2).transpose(1, 2) # [B, L, C]
            x_seq = self.norm(x_seq)
            x_seq = self.mamba(x_seq) 
            x_out = x_seq.transpose(1, 2).view(B, C, H, W)
        return x_out

# ================================================================
# 2. DCNv3 核心组件 (尝试导入)
# ================================================================
try:
    import sys
    # 假设你的 DCNv3 代码在 ops_dcnv3 目录下，根据实际情况调整
    sys.path.append("./ops_dcnv3") 
    from modules.dcnv3 import DCNv3
    HAS_DCN = True
except ImportError:
    HAS_DCN = False
    # print("⚠️ Warning: DCNv3 not found. StripConvBlock will fallback to standard Conv.")

# ================================================================
# 3. PHD 解码器组件 (从 hybrid_decoder.py 提取)
# ================================================================

# --- A. 局部形状分支 (DCN / Conv) ---
class StripConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, use_dcn=True):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True)
        )
        self.use_dcn = use_dcn and HAS_DCN
        
        if self.use_dcn:
            # 使用 DCNv3 进行长条形卷积模拟
            dcn_group = 4 
            self.strip_h = DCNv3(channels=out_channels, kernel_size=(1, kernel_size), stride=1, 
                                 pad=(0, padding), group=dcn_group, offset_scale=1.0)
            self.strip_v = DCNv3(channels=out_channels, kernel_size=(kernel_size, 1), stride=1, 
                                 pad=(padding, 0), group=dcn_group, offset_scale=1.0)
            self.norm_h = nn.BatchNorm2d(out_channels)
            self.norm_v = nn.BatchNorm2d(out_channels)
            self.act = nn.ReLU(inplace=True)
        else:
            # Fallback: 使用普通的分组卷积
            self.strip_h = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (1, kernel_size), padding=(0, padding), 
                          groups=out_channels, bias=False), 
                nn.BatchNorm2d(out_channels), 
                nn.ReLU(inplace=True)
            )
            self.strip_v = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (kernel_size, 1), padding=(padding, 0), 
                          groups=out_channels, bias=False), 
                nn.BatchNorm2d(out_channels), 
                nn.ReLU(inplace=True)
            )
        self.fusion_conv = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.proj(x)
        if self.use_dcn:
            h = self.act(self.norm_h(self.strip_h(x)))
            v = self.act(self.norm_v(self.strip_v(x)))
        else:
            h = self.strip_h(x)
            v = self.strip_v(x)
        return self.fusion_conv(h + v)

# --- B. 全局上下文分支 (Omni-Mamba) ---
class OmniMambaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        if in_channels != out_channels:
            self.align = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.align = nn.Identity()

        self.core_op = MambaLayer2D(out_channels)
        
    def forward(self, x):
        x = self.align(x)
        residual = x 
        # 四向扫描模拟
        x1 = self.core_op(x) # 正向
        x2 = torch.flip(self.core_op(torch.flip(x, dims=[2, 3])), dims=[2, 3]) # 反向
        x3 = self.core_op(x.transpose(2, 3)).transpose(2, 3) # 垂直正向
        x4 = torch.transpose(torch.flip(self.core_op(torch.flip(x.transpose(2, 3), dims=[2, 3])), dims=[2, 3]), 2, 3) # 垂直反向
        
        mamba_out = (x1 + x2 + x3 + x4) / 4.0
        return mamba_out + residual

# --- C. 融合模块 (SK-Fusion) ---
class SKFusion(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        mid_channels = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False), 
            nn.ReLU(inplace=True), 
            nn.Linear(mid_channels, 2 * channels, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_local, x_global):
        B, C, H, W = x_local.shape
        U = x_local + x_global 
        s = self.avg_pool(U).view(B, C)
        z = self.fc(s).view(B, 2, C)
        weights = self.softmax(z)
        w_local = weights[:, 0].view(B, C, 1, 1)
        w_global = weights[:, 1].view(B, C, 1, 1)
        return w_local * x_local + w_global * x_global

# --- D. PHD Decoder Block (总装) ---
class PHD_DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_dcn=True):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 分支 1: 局部细节 (DCN/StripConv)
        self.local_branch = StripConvBlock(out_channels, out_channels, use_dcn=use_dcn)
        # 分支 2: 全局上下文 (Mamba)
        self.global_branch = OmniMambaBlock(out_channels, out_channels)
        # 融合: SK-Fusion
        self.fusion = SKFusion(out_channels)

    def forward(self, x):
        x = self.relu(self.bn(self.reduce(x)))
        feat_local = self.local_branch(x)
        feat_global = self.global_branch(x)
        return self.fusion(feat_local, feat_global)

# ================================================================
# 4. 上采样适配器
# ================================================================
class Up_PHD(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_dcn=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        conv_in_channels = in_channels + skip_channels
        self.conv = PHD_DecoderBlock(in_channels=conv_in_channels, out_channels=out_channels, use_dcn=use_dcn)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if x2 is not None:
            # Padding 对齐
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            if diffX != 0 or diffY != 0:
                x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        return self.conv(x)

# ================================================================
# 5. 主模型: UNet_CNext_PHD
# ================================================================
class UNet_CNext_PHD(nn.Module):
    def __init__(self, n_classes=1, cnext_type='convnextv2_base', use_dcn=True, **kwargs):
        super().__init__()
        
        print(f"🧪 [Ablation Model] Initializing...")
        print(f"   - Encoder: {cnext_type}")
        print(f"   - Decoder: PHD (Mamba + DCNv3 + SK-Fusion)")
        print(f"   - Dual Stream: ❌ Disabled")
        
        # 保存 n_classes 供 train.py 使用
        self.n_classes = n_classes

        # --- 1. Encoder: ConvNeXt V2 ---
        self.spatial_encoder = timm.create_model(
            cnext_type, 
            pretrained=True, 
            features_only=True, 
            out_indices=(0, 1, 2, 3),
            drop_path_rate=0.0  # 🔥 [关键] 强制关闭 DropPath，让模型火力全开！
        )
        dims = self.spatial_encoder.feature_info.channels()
        c1, c2, c3, c4 = dims # Base: [128, 256, 512, 1024]

        # --- 2. Decoder: PHD Blocks ---
        # Up 1: Input=c4, Skip=c3 -> Out=c3
        self.up1 = Up_PHD(c4, c3, skip_channels=c3, use_dcn=use_dcn)
        
        # Up 2: Input=c3, Skip=c2 -> Out=c2
        self.up2 = Up_PHD(c3, c2, skip_channels=c2, use_dcn=use_dcn)
        
        # Up 3: Input=c2, Skip=c1 -> Out=c1
        self.up3 = Up_PHD(c2, c1, skip_channels=c1, use_dcn=use_dcn)
        
        # --- 3. Output Head ---
        self.final_up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.outc = nn.Conv2d(c1, n_classes, kernel_size=1)

    def forward(self, x):
        # === Encoder ===
        s1, s2, s3, s4 = self.spatial_encoder(x)

        # === Decoder ===
        d1 = self.up1(s4, s3)
        d2 = self.up2(d1, s2)
        d3 = self.up3(d2, s1)
        
        # === Output ===
        out = self.final_up(d3)
        logits = self.outc(out)
        
        # 兼容 train.py 的 Tuple 检查 (虽然这里只返回一个)
        return logits