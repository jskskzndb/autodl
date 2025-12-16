"""
wgn_conv_block.py - V3.1 (Spatially Adaptive Thresholding Version)

Upgrades from V3:
1.  [Pixel-wise Denoising]: Replaced global LearnableSoftThresholding with
    SpatiallyAdaptiveThresholding. Now the threshold is dynamic per pixel.
2.  [Direction-Aware]: Retains independent LH/HL/HH processing from V3.
"""

import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse


# =====================================================================================
# 基础组件 1: 空间自适应软阈值去噪 (🔥 核心修改点)
# =====================================================================================
class SpatiallyAdaptiveThresholding(nn.Module):
    """
    像素级自适应软阈值。
    不再使用单一的阈值参数，而是通过一个小型的卷积网络，
    根据当前的特征图内容，为每一个像素点预测一个专属的噪声阈值。
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        # 一个轻量级的预测网络
        # input -> 降维 -> ReLU -> 3x3卷积看邻域 -> Sigmoid -> 阈值图
        self.threshold_net = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()  # 输出 0~1 之间的系数
        )
        # 一个可学习的基准缩放因子 (控制整体去噪力度)
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # 1. 生成像素级阈值图 [B, C, H, W]
        # 边缘处阈值会自动变小(保留)，平坦处阈值变大(抑制)
        thresh_map = self.threshold_net(x) * self.scale

        # 2. 执行软阈值公式: sign(x) * max(|x| - thresh, 0)
        return torch.sign(x) * torch.relu(torch.abs(x) - thresh_map)


# =====================================================================================
# 基础组件 2: GnConv (保持原样)
# =====================================================================================
def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)


class GnConv(nn.Module):
    def __init__(self, dim, order=5, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)
        self.dwconv = get_dwconv(sum(self.dims), 7, True)
        self.proj_out = nn.Conv2d(dim, dim, 1)
        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )
        self.scale = s

    def forward(self, x):
        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)
        dw_abc = self.dwconv(abc) * self.scale
        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]
        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]
        x = self.proj_out(x)
        return x


# =====================================================================================
# 主模块: Wg_nConv_Block V3.1
# =====================================================================================
class Wg_nConv_Block(nn.Module):
    def __init__(self, channels, order_low=4, order_high=3):
        super().__init__()

        # 小波工具
        self.dwt = DWTForward(J=1, wave='haar', mode='zero')
        self.idwt = DWTInverse(wave='haar', mode='zero')

        # --- 1. 低频路径 (主导) ---
        self.gnconv_low_freq = GnConv(dim=channels, order=order_low)

        # 引导掩码生成器
        self.guidance_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

        # --- 2. 高频路径 (三路独立 + 自适应去噪) ---

        # A. 水平边缘 (LH)
        self.branch_lh = nn.Sequential(
            # 🔥 修改点：使用自适应阈值
            SpatiallyAdaptiveThresholding(channels),
            nn.Dropout(p=0.2),
            GnConv(dim=channels, order=order_high)
        )

        # B. 垂直边缘 (HL)
        self.branch_hl = nn.Sequential(
            # 🔥 修改点：使用自适应阈值
            SpatiallyAdaptiveThresholding(channels),
            nn.Dropout(p=0.2),
            GnConv(dim=channels, order=order_high)
        )

        # C. 对角边缘 (HH)
        self.branch_hh = nn.Sequential(
            # 🔥 修改点：使用自适应阈值
            SpatiallyAdaptiveThresholding(channels),
            nn.Dropout(p=0.2),
            GnConv(dim=channels, order=order_high)
        )

        # 最后的融合投影 (3C -> 3C)
        self.high_freq_proj_out = nn.Conv2d(channels * 3, channels * 3, 1)

    def forward(self, x):
        identity = x
        b, c, h, w = x.shape

        # 1. 分解
        ll, high_freq_list = self.dwt(x)
        high_freq = high_freq_list[0].view(b, c * 3, h // 2, w // 2)

        # 2. 低频处理
        ll_enhanced = self.gnconv_low_freq(ll)
        guidance_mask = self.guidance_conv(ll_enhanced)

        # 3. 高频处理 (拆分 -> 独立处理 -> 合并)
        lh, hl, hh = torch.chunk(high_freq, 3, dim=1)

        # 独立处理 (内部包含自适应去噪)
        lh_out = self.branch_lh(lh)
        hl_out = self.branch_hl(hl)
        hh_out = self.branch_hh(hh)

        # 引导交互
        lh_out = lh_out * guidance_mask
        hl_out = hl_out * guidance_mask
        hh_out = hh_out * guidance_mask

        # 拼接回去 (3C 通道)
        high_feat_combined = torch.cat([lh_out, hl_out, hh_out], dim=1)

        # 最终融合
        high_feat_final = self.high_freq_proj_out(high_feat_combined)

        # 4. 重构
        high_freq_out_list = [high_feat_final.view(b, c, 3, h // 2, w // 2)]
        y = self.idwt((ll_enhanced, high_freq_out_list))

        # 5. 返回双结果
        return identity + y, high_feat_final


# 测试代码
if __name__ == '__main__':
    print("Testing WGN V3.1 (Adaptive Threshold)...")
    x = torch.randn(2, 64, 32, 32)
    block = Wg_nConv_Block(64)

    out, high_feat = block(x)

    print(f"Input: {x.shape}")
    print(f"Fused Output: {out.shape}")
    print(f"High-Freq Feature: {high_feat.shape}")

    assert out.shape == x.shape
    assert high_feat.shape == (2, 64 * 3, 16, 16)

    print("✅ V3.1 Upgrade Successful!")