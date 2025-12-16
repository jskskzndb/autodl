"""
wgn_conv_block.py - V3 (Direction-Aware Version)

Upgrades from V2:
1.  [Direction-Aware]: High-frequency components (LH, HL, HH) are processed INDEPENDENTLY.
    (No more 3C -> C compression that mixes directions).
2.  [Spatially Adaptive]: Uses Learnable Soft Thresholding (retained from V2).
3.  [Guidance]: Low-frequency guides all three high-frequency branches (retained from V2).
"""

import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse

# =====================================================================================
# 基础组件 1: 软阈值去噪模块
# =====================================================================================
class LearnableSoftThresholding(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(0.02), requires_grad=True)

    def forward(self, x):
        tau = torch.abs(self.threshold)
        return torch.sign(x) * torch.relu(torch.abs(x) - tau)

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
# 主模块: Wg_nConv_Block V3 (方向感知版)
# =====================================================================================
class Wg_nConv_Block(nn.Module):
    def __init__(self, channels, order_low=4, order_high=3):
        super().__init__()
        
        # 小波工具
        self.dwt = DWTForward(J=1, wave='haar', mode='zero')
        self.idwt = DWTInverse(wave='haar', mode='zero')
        
        # --- 1. 低频路径 (主导) ---
        self.gnconv_low_freq = GnConv(dim=channels, order=order_low)
        
        # 引导掩码生成器 (1x1 Conv + Sigmoid)
        self.guidance_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

        # --- 2. 高频路径 (V3 核心升级: 三路独立) ---
        # 以前 V2 是把 3C 压缩成 C (混合了方向)。
        # 现在 V3 是建立 3 个独立的 GnConv，分别处理 LH, HL, HH。
        
        # A. 水平边缘 (LH) 处理流
        self.branch_lh = nn.Sequential(
            LearnableSoftThresholding(channels), # 先去噪
            nn.Dropout(p=0.2),  # 🔥 新增这一行
            GnConv(dim=channels, order=order_high) # 再增强
        )
        
        # B. 垂直边缘 (HL) 处理流
        self.branch_hl = nn.Sequential(
            LearnableSoftThresholding(channels),
            nn.Dropout(p=0.2),  # 🔥 新增这一行
            GnConv(dim=channels, order=order_high)
        )
        
        # C. 对角边缘 (HH) 处理流
        self.branch_hh = nn.Sequential(
            LearnableSoftThresholding(channels),
            nn.Dropout(p=0.2),  # 🔥 新增这一行
            GnConv(dim=channels, order=order_high)
        )
        
        # 最后的融合投影 (3C -> 3C)
        self.high_freq_proj_out = nn.Conv2d(channels * 3, channels * 3, 1)
        
    def forward(self, x):
        identity = x
        b, c, h, w = x.shape

        # 1. 小波分解
        ll, high_freq_list = self.dwt(x)
        high_freq = high_freq_list[0].view(b, c * 3, h // 2, w // 2)

        # 2. 低频处理
        ll_enhanced = self.gnconv_low_freq(ll)
        
        # 生成引导 Mask (Batch, C, H/2, W/2)
        guidance_mask = self.guidance_conv(ll_enhanced)

        # 3. 高频处理 (V3: 拆分 -> 独立处理 -> 合并)
        # 将 3C 拆分为 LH, HL, HH (每份是 C)
        lh, hl, hh = torch.chunk(high_freq, 3, dim=1)
        
        # 分别进入各自的“单间”进行处理
        lh_out = self.branch_lh(lh)
        hl_out = self.branch_hl(hl)
        hh_out = self.branch_hh(hh)
        
        # 都在这里应用低频引导 (Mask 广播给三个分支)
        lh_out = lh_out * guidance_mask
        hl_out = hl_out * guidance_mask
        hh_out = hh_out * guidance_mask
        
        # 拼接回去 (Batch, 3C, H/2, W/2)
        high_feat_combined = torch.cat([lh_out, hl_out, hh_out], dim=1)
        
        # 最终融合一下特征
        high_feat_final = self.high_freq_proj_out(high_feat_combined)
        
        # 4. 小波重构
        high_freq_out_list = [high_feat_final.view(b, c, 3, h // 2, w // 2)]
        y = self.idwt((ll_enhanced, high_freq_out_list))
        
        # 5. 返回双结果 (适配你的双流解码器)
        # 注意：这里把“处理好的高频特征”传给 Edge Decoder
        return identity + y, high_feat_final

# 测试代码
if __name__ == '__main__':
    print("Testing V3 (Direction-Aware) WGN Block...")
    x = torch.randn(2, 64, 32, 32)
    block = Wg_nConv_Block(64)
    
    out, high_feat = block(x)
    
    print(f"Input: {x.shape}")
    print(f"Fused Output: {out.shape}")
    print(f"High-Freq Feature: {high_feat.shape}")
    
    assert out.shape == x.shape
    # 高频特征应该是输入分辨率的一半，通道数的3倍
    assert high_feat.shape == (2, 64*3, 16, 16)
    
    print("✅ V3 Upgrade Successful! LH/HL/HH are processed independently.")