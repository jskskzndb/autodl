"""
unet/unet_s_dmfnet.py
[S-DMFNet Pro] Enhanced Dual-Stream Mutual-Guided Frequency-Aware Network
版本特性:
1. 集成 Bi-FGF (双向互导) 模块
2. 移除 MFAM 瓶颈层，采用轻量化融合
3. Edge Head 使用语义清洗后的频率特征
4. 完全复刻 Up_PHD 接口
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import sys
from pathlib import Path

# ================================================================
# 0. 动态导入依赖 (保持原有目录结构的兼容性)
# ================================================================

# 尝试导入 PHD 解码器核心块
try: 
    from decoder.hybrid_decoder import PHD_DecoderBlock
except ImportError: 
    try: 
        from unet.hybrid_decoder import PHD_DecoderBlock
    except ImportError: 
        PHD_DecoderBlock = None
        print("Warning: PHD_DecoderBlock import failed.")

# 尝试导入 Mamba (用于右路)
try: 
    from decoder.mamba_helper import MambaLayer2D
except ImportError: 
    try: 
        from unet.mamba_helper import MambaLayer2D
    except ImportError: 
        MambaLayer2D = None
        print("Warning: MambaLayer2D import failed, using Identity.")

# ================================================================
# 1. 基础工具类 (Haar 小波) - 保持不变
# ================================================================

class HaarWaveletTransform(nn.Module):
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        self.register_buffer('filters', torch.stack([ll, lh, hl, hh]).unsqueeze(1))

    def dwt(self, x):
        B, C, H, W = x.shape
        # 偶数填充处理
        if H % 2 != 0 or W % 2 != 0:
            x = F.pad(x, (0, W % 2, 0, H % 2), mode='reflect')
        filters = self.filters.repeat(C, 1, 1, 1)
        output = F.conv2d(x, filters, stride=2, groups=C)
        output = output.view(B, C, 4, output.shape[2], output.shape[3])
        return output[:, :, 0], output[:, :, 1], output[:, :, 2], output[:, :, 3]

# ================================================================
# 2. 右路核心模块 (WaveletMamba) - 保持不变
# ================================================================

class WaveletMambaBlock(nn.Module):
    """ 右路：小波-Mamba 编码器块 """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dwt = HaarWaveletTransform()
        
        # LL (低频): Mamba 捕捉全局结构
        self.low_process = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            MambaLayer2D(dim=out_channels) if MambaLayer2D else nn.Identity()
        )
        
        # High (高频): Conv 捕捉局部边缘
        self.high_process = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 融合
        self.fusion = nn.Conv2d(out_channels * 2, out_channels, 1)

    def forward(self, x):
        ll, lh, hl, hh = self.dwt.dwt(x)
        ll_feat = self.low_process(ll)
        high_cat = torch.cat([lh, hl, hh], dim=1)
        high_feat = self.high_process(high_cat)
        out = self.fusion(torch.cat([ll_feat, high_feat], dim=1))
        return out

# ================================================================
# 3. [🔥核心升级] Bi-FGF 双向互导模块
#    学术对标: RSBuilding (2024)
# ================================================================

class Bi_FGF_Module(nn.Module):
    """ 
    Bi-Directional Frequency-Guided Fusion 
    双向互导频率融合模块
    """
    def __init__(self, s_channels, f_channels):
        super().__init__()
        
        # --- Path 1: Freq -> Spatial (频率清洗语义) ---
        # 利用边缘信息 (Freq) 生成 Attention，去除 Spatial 中的平坦背景噪声
        self.freq_gate = nn.Sequential(
            nn.Conv2d(f_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        # 频率特征注入对齐
        self.freq_align = nn.Conv2d(f_channels, s_channels, kernel_size=1)

        # --- Path 2: Spatial -> Freq (语义抑制频率) ---
        # 利用语义置信度 (Spatial) 生成 Attention，去除 Freq 中的虚假纹理(如波纹)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(s_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        # 语义特征注入对齐
        self.spatial_align = nn.Conv2d(s_channels, f_channels, kernel_size=1)

    def forward(self, x_s, x_f):
        # 如果尺寸不匹配(通常不会发生，但为了鲁棒性)
        if x_f.shape[2:] != x_s.shape[2:]:
            x_f = F.interpolate(x_f, size=x_s.shape[2:], mode='bilinear', align_corners=False)

        # 1. 正向引导 (Freq -> Spatial)
        # 逻辑: 语义特征 * 边缘权重 + 频率细节补充
        att_map_f2s = self.freq_gate(x_f)
        s_out = (x_s * att_map_f2s) + self.freq_align(x_f)

        # 2. 反向引导 (Spatial -> Freq)
        # 逻辑: 频率特征 * 语义权重 + 语义上下文补充
        att_map_s2f = self.spatial_gate(x_s)
        f_out = (x_f * att_map_s2f) + self.spatial_align(x_s)

        return s_out, f_out
# ================================================================
# 🔥 [新增] SK-Fusion: 涨点神器
# ================================================================
class SK_Fusion(nn.Module):
    """
    Selective Kernel Fusion (SK-Fusion)
    作用: 动态学习 Semantic流 和 Frequency流 的融合权重
    输入: 两个维度相同的特征图 x_s, x_f
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.d = max(channels // reduction, 32)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, self.d),
            nn.BatchNorm1d(self.d),
            nn.ReLU(inplace=True)
        )
        self.fc_selection = nn.Linear(self.d, channels * 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_s, x_f):
        B, C, H, W = x_s.shape
        # 1. 初始叠加
        U = x_s + x_f
        # 2. 全局描述符
        s = self.avg_pool(U).view(B, C)
        # 3. 压缩激励
        z = self.fc(s)
        # 4. 生成竞争权重
        weights = self.fc_selection(z).view(B, 2, C)
        weights = self.softmax(weights)
        # 5. 加权融合
        w_s = weights[:, 0, :].view(B, C, 1, 1)
        w_f = weights[:, 1, :].view(B, C, 1, 1)
        return (x_s * w_s) + (x_f * w_f)
# ================================================================
# 4. [适配器] Up_PHD - 完全复刻原代码
# ================================================================

class Up_PHD(nn.Module):
    """
    负责：上采样 -> Padding对齐 -> 拼接 -> 调用 PHD_DecoderBlock
    """
    def __init__(self, in_channels, out_channels, bilinear=True, skip_channels=0, 
                 use_dcn=False, use_dubm=False, use_strg=False):
        super().__init__()
        
        # 1. 定义上采样
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            conv_in_channels = in_channels + skip_channels
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            conv_in_channels = (in_channels // 2) + skip_channels

        # 2. 核心：调用 PHD_DecoderBlock
        self.conv = PHD_DecoderBlock(in_channels=conv_in_channels, out_channels=out_channels, 
                                     use_dcn=use_dcn, use_dubm=use_dubm)

    def forward(self, x1, x2=None, edge_prior=None):
        # x1: 深层特征 (需要上采样)
        # x2: 浅层 Skip 特征 (需要拼接)
        
        x1 = self.up(x1)
        
        if x2 is not None:
            # Padding 对齐
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            if diffX != 0 or diffY != 0:
                x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            
            # 拼接
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        
        # 调用 PHD Block
        return self.conv(x, edge_prior=edge_prior)

# ================================================================
# 5. S_DMFNet 主模型 (Bi-FGF 版)
# ================================================================

class S_DMFNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, 
                 encoder_name='cnextv2', cnext_type='convnextv2_base', # 推荐 Base
                 decoder_name='phd', use_dcn=True,
                 # 接收兼容参数
                 use_dsis=False, use_dual_stream=False, use_wavelet_denoise=False, 
                 use_wgn_enhancement=False, use_cafm=False, use_edge_loss=False, 
                 use_dubm=False, use_strg=False, **kwargs):
        super(S_DMFNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        print(f"🚀 [S-DMFNet Pro] 初始化... Encoder: {cnext_type}, Decoder: {decoder_name}")
        print(f"   ✨ Features: Bi-FGF (Enabled), MFAM (Removed), EdgeHead (Enhanced)")

        # --- 1. 左路: Spatial Encoder (ConvNeXt V2 Base) ---
        backbone_name = cnext_type if cnext_type else 'convnextv2_base'
        self.spatial_encoder = timm.create_model(backbone_name, pretrained=True, features_only=True, out_indices=(0, 1, 2, 3))
        s_dims = self.spatial_encoder.feature_info.channels() 
        self.dims = s_dims

        # --- 2. 右路: Frequency Encoder (通道数 1/4) ---
        f_dims = [c // 4 for c in s_dims]
        self.freq_stem = nn.Sequential(
            nn.Conv2d(3, f_dims[0], 4, stride=4, padding=0),
            nn.BatchNorm2d(f_dims[0]),
            nn.ReLU(inplace=True)
        )
        self.freq_layers = nn.ModuleList()
        for i in range(3):
            self.freq_layers.append(WaveletMambaBlock(f_dims[i], f_dims[i+1]))
        self.freq_stage4 = WaveletMambaBlock(f_dims[3], f_dims[3])

        # --- 3. [升级] 交互: Bi-FGF Modules ---
        # 替换了原来的 FGF_Module
        self.bi_fgf_modules = nn.ModuleList([Bi_FGF_Module(s_dims[i], f_dims[i]) for i in range(4)])

        # --- 4. 🔥 [新增] Fusion: SK-Fusion ---
        # 为每一层(包括瓶颈层)准备一个 SK 融合模块
        self.sk_fusions = nn.ModuleList([
            SK_Fusion(s_dims[0]), # Layer 1
            SK_Fusion(s_dims[1]), # Layer 2
            SK_Fusion(s_dims[2]), # Layer 3
            SK_Fusion(s_dims[3])  # Layer 4 (Neck)
        ])
        
        # 保留对齐层 (为了将 f 对齐到 s，供 SK-Fusion 使用)
        # 注意：其实 Bi-FGF 里已经有对齐层了，我们可以复用 Bi-FGF 里的参数，
        # 但为了逻辑清晰，SK-Fusion 之前我们调用 Bi-FGF 里的 freq_align 即可，不需要额外定义。

        # --- 5. 解码器: 使用 Up_PHD 包装器 ---
        c1, c2, c3, c4 = s_dims
        
        # Up 1: x4_fused + s3
        self.up1 = Up_PHD(c4, c3, bilinear, skip_channels=c3, use_dcn=use_dcn, use_dubm=use_dubm)
        # Up 2: d1 + s2
        self.up2 = Up_PHD(c3, c2, bilinear, skip_channels=c2, use_dcn=use_dcn, use_dubm=use_dubm)
        # Up 3: d2 + s1
        self.up3 = Up_PHD(c2, c1, bilinear, skip_channels=c1, use_dcn=use_dcn, use_dubm=use_dubm)
        
        self.final_up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.outc = nn.Conv2d(c1, n_classes, kernel_size=1)

        # --- 6. [增强] 边缘预测头 ---
        # 输入维度依然是 f_dims[0]，但传入的内容将是被语义清洗过的特征
        self.edge_head = nn.Sequential(
            nn.Conv2d(f_dims[0], 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        # === Encoder ===
        s_feats = list(self.spatial_encoder(x)) # [s1, s2, s3, s4]
        
        f_feats = []
        f_curr = self.freq_stem(x)
        f_feats.append(f_curr) # f1
        for layer in self.freq_layers:
            f_curr = layer(f_curr)
            f_feats.append(f_curr) # f2, f3, f4
        f_feats[-1] = self.freq_stage4(f_feats[-1])

        # === Interaction (Bi-FGF) ===
        s_clean = []    # 用于跳跃连接
        f_enhanced = [] # 用于边缘监督和深层融合
        
        for i in range(4):
            # 🔥 Bi-FGF 双向互洗
            s_new, f_new = self.bi_fgf_modules[i](s_feats[i], f_feats[i])
            s_clean.append(s_new)
            f_enhanced.append(f_new)
            
        s1, s2, s3, x4 = s_clean
        f1_enh, f2_enh, f3_enh, f4_enh = f_enhanced

        # === Neck (SK-Fusion) ===
        # 1. 复用 Bi-FGF 中的对齐层，把 f4 变成 s4 的通道数
        f4_aligned = self.bi_fgf_modules[3].freq_align(f4_enh)
        # 2. SK-Fusion: 智能融合语义和频率
        x4_fused = self.sk_fusions[3](x4, f4_aligned)

        # === Decoder (带 SK-Fusion 跳跃连接) ===
        
        # Layer 3 Skip
        f3_aligned = self.bi_fgf_modules[2].freq_align(f3_enh)
        skip3 = self.sk_fusions[2](s3, f3_aligned) # 🔥 SK 融合
        d1 = self.up1(x4_fused, skip3)
        
        # Layer 2 Skip
        f2_aligned = self.bi_fgf_modules[1].freq_align(f2_enh)
        skip2 = self.sk_fusions[1](s2, f2_aligned) # 🔥 SK 融合
        d2 = self.up2(d1, skip2)
        
        # Layer 1 Skip
        f1_aligned = self.bi_fgf_modules[0].freq_align(f1_enh)
        skip1 = self.sk_fusions[0](s1, f1_aligned) # 🔥 SK 融合
        d3 = self.up3(d2, skip1)
        
        d4 = self.final_up(d3)
        logits = self.outc(d4)
        
        # === Auxiliary Output ===
        if self.training:
            # 🔥 关键改进: 使用 f_enhanced[0] 而不是 f_feats[0]
            # 这里送入 Edge Head 的特征已经被 s1 (语义流) 清洗过，
            # 抑制了水波纹/斑马线等伪边缘，Loss 计算更准。
            edge_logits_small = self.edge_head(f_enhanced[0])
            edge_logits = F.interpolate(edge_logits_small, size=logits.shape[2:], mode='bilinear', align_corners=True)
            
            return logits, edge_logits
            
        return logits