"""
wgn_conv_block_cbam.py
WGN Variant: Replacing GnConv with CBAM (Convolutional Block Attention Module).

Structure:
    Input -> [3x3 Conv Feature Extraction] -> [Channel Attention] -> [Spatial Attention] -> Output

This maintains the "Gating" capability (calculating weights) while using standard operators.
"""

import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse


# =====================================================================================
# 基础组件 1: 软阈值去噪模块 (保留 V3 的优良特性)
# =====================================================================================
class LearnableSoftThresholding(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(0.02), requires_grad=True)

    def forward(self, x):
        tau = torch.abs(self.threshold)
        return torch.sign(x) * torch.relu(torch.abs(x) - tau)


# =====================================================================================
# 基础组件 2: CBAM 模块 (替代 GnConv 的核心)
# =====================================================================================

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # 如果通道数太少，ratio设小一点防止压缩到0
        hidden_planes = max(in_planes // ratio, 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享 MLP
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden_planes, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Avg分支
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # Max分支
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # 叠加 + Sigmoid 生成通道权重
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 压缩通道为2 (Max + Avg)
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道维度的 AvgPool 和 MaxPool
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        # 卷积生成空间权重
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class CBAMConv(nn.Module):
    """
    替代 GnConv 的组合模块：
    1. 3x3 Conv: 提取局部特征 (类似于 GnConv 里的 DWConv)
    2. CBAM: 计算注意力权重并进行门控 (Gating)
    """

    def __init__(self, dim, order=None):  # order 参数是为了兼容接口，这里不用
        super().__init__()

        # 1. 特征提取 (Feature Extraction)
        # 必须先做卷积，有了特征才能算 Attention
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        # 2. 门控机制 (Gating / Attention)
        self.ca = ChannelAttention(dim)
        self.sa = SpatialAttention()

    def forward(self, x):
        # 先提取特征
        feat = self.conv(x)

        # 应用通道注意力 (Channel Gating)
        # Weight * Feature
        feat = self.ca(feat) * feat

        # 应用空间注意力 (Spatial Gating)
        # Weight * Feature
        feat = self.sa(feat) * feat

        return feat


# =====================================================================================
# 主模块: Wg_nConv_Block (集成 CBAM 版)
# =====================================================================================
class Wg_nConv_Block(nn.Module):
    def __init__(self, channels, order_low=4, order_high=3):
        super().__init__()

        self.dwt = DWTForward(J=1, wave='haar', mode='zero')
        self.idwt = DWTInverse(wave='haar', mode='zero')

        # --- 1. 低频路径 ---
        # 使用 CBAMConv 替代 GnConv
        self.gnconv_low_freq = CBAMConv(dim=channels)

        # 引导生成器 (保持 V3 的 LL-Guided 逻辑)
        self.guidance_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

        # --- 2. 高频路径 (三路独立) ---

        # A. 水平边缘 (LH)
        self.branch_lh = nn.Sequential(
            LearnableSoftThresholding(channels),
            nn.Dropout(p=0.2),
            CBAMConv(dim=channels)  # 🔥 替换为 CBAM
        )

        # B. 垂直边缘 (HL)
        self.branch_hl = nn.Sequential(
            LearnableSoftThresholding(channels),
            nn.Dropout(p=0.2),
            CBAMConv(dim=channels)  # 🔥 替换为 CBAM
        )

        # C. 对角边缘 (HH)
        self.branch_hh = nn.Sequential(
            LearnableSoftThresholding(channels),
            nn.Dropout(p=0.2),
            CBAMConv(dim=channels)  # 🔥 替换为 CBAM
        )

        self.high_freq_proj_out = nn.Conv2d(channels * 3, channels * 3, 1)

    def forward(self, x):
        identity = x
        b, c, h, w = x.shape

        # 1. 小波分解
        ll, high_freq_list = self.dwt(x)
        high_freq = high_freq_list[0].view(b, c * 3, h // 2, w // 2)

        # 2. 低频处理
        ll_enhanced = self.gnconv_low_freq(ll)

        # 生成引导 Mask
        guidance_mask = self.guidance_conv(ll_enhanced)

        # 3. 高频处理
        lh, hl, hh = torch.chunk(high_freq, 3, dim=1)

        lh_out = self.branch_lh(lh)
        hl_out = self.branch_hl(hl)
        hh_out = self.branch_hh(hh)

        # 应用低频引导 (LL-Guided)
        # 注意：这里是双重 Gating！
        # 1. 内部有 CBAM 做 Self-Gating
        # 2. 外部有 LL 做 Cross-Gating
        lh_out = lh_out * guidance_mask
        hl_out = hl_out * guidance_mask
        hh_out = hh_out * guidance_mask

        high_feat_combined = torch.cat([lh_out, hl_out, hh_out], dim=1)
        high_feat_final = self.high_freq_proj_out(high_feat_combined)

        # 4. 小波重构
        high_freq_out_list = [high_feat_final.view(b, c, 3, h // 2, w // 2)]
        y = self.idwt((ll_enhanced, high_freq_out_list))

        return identity + y, high_feat_final


if __name__ == '__main__':
    print("Testing WGN with CBAM...")
    x = torch.randn(2, 64, 32, 32)
    block = Wg_nConv_Block(64)
    out, high = block(x)
    print(f"Output shape: {out.shape}")
    print("✅ CBAM Integration Successful!")