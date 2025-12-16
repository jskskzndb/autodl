"""
wgn_conv_block.py - V2 Enhanced Version

Upgrades:
1. Learnable Soft Thresholding: For noise suppression in high-frequency domain.
2. Cross-Frequency Guidance: Low-frequency features guide high-frequency enhancement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse

# =====================================================================================
# 基础组件 1: 软阈值去噪模块 (新增)
# =====================================================================================
class LearnableSoftThresholding(nn.Module):
    """
    可学习的软阈值层。
    公式: y = sign(x) * max(|x| - tau, 0)
    作用: 自动过滤掉幅值较小的值（通常是噪声），保留显著特征。
    """
    def __init__(self, channels):
        super().__init__()
        # 每个通道独立学习一个阈值，初始化为 0.02 (经验值，稍微有点去噪能力)
        self.threshold = nn.Parameter(torch.tensor(0.02), requires_grad=True)
        self.channels = channels
        # 也可以为每个通道单独设置阈值: self.threshold = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        # 保证阈值非负
        tau = torch.abs(self.threshold)
        # 软阈值操作
        return torch.sign(x) * torch.relu(torch.abs(x) - tau)

# =====================================================================================
# 基础组件 2: GnConv (保持原样，这是HorNet的核心)
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
# 主模块: Wg_nConv_Block V2 (升级版)
# =====================================================================================
class Wg_nConv_Block(nn.Module):
    """
    V2 升级版: 包含软阈值去噪和跨频率引导
    """
    def __init__(self, channels, order_low=4, order_high=3):
        super().__init__()
        
        # 1. 小波变换工具
        self.dwt = DWTForward(J=1, wave='haar', mode='zero')
        self.idwt = DWTInverse(wave='haar', mode='zero')
        
        # 2. 低频处理路径 (Master Path)
        self.gnconv_low_freq = GnConv(dim=channels, order=order_low)
        
        # [🔥 新增 1] 引导生成器
        # 将低频特征转化为注意力掩码 (Channel Attention / Spatial Attention)
        self.guidance_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid() # 输出 0~1 的权重
        )

        # 3. 高频处理路径 (Slave Path)
        self.high_freq_proj_in = nn.Conv2d(channels * 3, channels, 1)
        
        # [🔥 新增 2] 软阈值去噪模块 (加在高频投影之后，增强之前)
        self.soft_threshold = LearnableSoftThresholding(channels)
        
        self.gnconv_high_freq = GnConv(dim=channels, order=order_high)
        self.high_freq_proj_out = nn.Conv2d(channels, channels * 3, 1)
        
    def forward(self, x):
        identity = x
        b, c, h, w = x.shape

        # --- 1. 小波分解 ---
        ll, high_freq_list = self.dwt(x)
        # high_freq 原本是 list，取出来 reshape 成 [B, 3C, H/2, W/2]
        high_freq = high_freq_list[0].view(b, c * 3, h // 2, w // 2)

        # --- 2. 低频处理 (主导) ---
        # 低频代表结构信息，用高阶卷积提取全局特征
        ll_enhanced = self.gnconv_low_freq(ll)
        
        # [🔥 新增逻辑] 生成引导 Mask
        # 含义：低频告诉高频，哪里是物体(权重趋近1)，哪里是背景(权重趋近0)
        guidance_mask = self.guidance_conv(ll_enhanced)

        # --- 3. 高频处理 (从属) ---
        # a. 投影: 3C -> C
        high_feat = self.high_freq_proj_in(high_freq)
        
        # [🔥 新增逻辑] 软阈值去噪
        # 自动去除微小的高频噪声
        high_feat = self.soft_threshold(high_feat)
        
        # b. 增强: GnConv
        high_feat = self.gnconv_high_freq(high_feat)
        
        # [🔥 新增逻辑] 跨频率引导交互
        # 用低频生成的 Mask 过滤高频特征
        high_feat = high_feat * guidance_mask
        
        # c. 还原: C -> 3C
        high_freq_reconstructed = self.high_freq_proj_out(high_feat)
        
        # --- 4. 小波重构 ---
        high_freq_out_list = [high_freq_reconstructed.view(b, c, 3, h // 2, w // 2)]
        y = self.idwt((ll_enhanced, high_freq_out_list))
        
        # --- 5. 残差连接 ---
        #return identity + y
        return identity + y, high_feat
# 测试代码
# 测试代码
if __name__ == '__main__':
    print("Testing V2 Wg^nConv Block...")
    x = torch.randn(2, 64, 32, 32)
    block = Wg_nConv_Block(64)

    # --- 修改这里：接收两个返回值 ---
    out, high_freq_feat = block(x)

    print(f"Input: {x.shape}")
    print(f"Fused Output: {out.shape}")
    print(f"High-Freq Feature: {high_freq_feat.shape}")  # 打印高频特征的尺寸

    assert x.shape == out.shape, "Fused output shape mismatch!"
    # 验证高频特征的尺寸应该是输入的一半 (因为是小波分解后的)
    expected_high_shape = (x.shape[0], x.shape[1], x.shape[2] // 2, x.shape[3] // 2)
    assert high_freq_feat.shape == expected_high_shape, f"High-freq shape mismatch! Got {high_freq_feat.shape}, expected {expected_high_shape}"

    print("✅ V2 Upgrade Successful! (Returns tuple correctly)")