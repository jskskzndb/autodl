"""
unet_s_dmfnet.py
[S-DMFNet] Simplified Dual-Stream Mutual-Guided Frequency-Aware Network     S-DMFNet V1 (Baseline) - 单向引导 + MFAM + 统一学习率
完全复刻 unet_model_unified.py 的解码器写法 (Up_PHD)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import sys
from pathlib import Path

# 尝试导入 PHD 解码器核心块 (保持和你原有代码一致的导入逻辑)
try: from decoder.hybrid_decoder import PHD_DecoderBlock
except ImportError: 
    try: from unet.hybrid_decoder import PHD_DecoderBlock
    except ImportError: PHD_DecoderBlock = None

# 尝试导入 Mamba (用于右路)
try: from decoder.mamba_helper import MambaLayer2D
except ImportError: 
    try: from unet.mamba_helper import MambaLayer2D
    except ImportError: MambaLayer2D = None

# ================================================================
# 1. 基础工具类 (Haar 小波)
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
        if H % 2 != 0 or W % 2 != 0:
            x = F.pad(x, (0, W % 2, 0, H % 2), mode='reflect')
        filters = self.filters.repeat(C, 1, 1, 1)
        output = F.conv2d(x, filters, stride=2, groups=C)
        output = output.view(B, C, 4, output.shape[2], output.shape[3])
        return output[:, :, 0], output[:, :, 1], output[:, :, 2], output[:, :, 3]

# ================================================================
# 2. 右路核心模块 (WaveletMamba) & 交互模块 (FGF) & 瓶颈 (MFAM)
# ================================================================

class WaveletMambaBlock(nn.Module):
    """ 右路：小波-Mamba 编码器块 """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dwt = HaarWaveletTransform()
        
        # LL: Mamba 捕捉结构
        self.low_process = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            MambaLayer2D(dim=out_channels) if MambaLayer2D else nn.Identity()
        )
        
        # High: Conv 捕捉边缘
        self.high_process = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.Conv2d(out_channels * 2, out_channels, 1)

    def forward(self, x):
        ll, lh, hl, hh = self.dwt.dwt(x)
        ll_feat = self.low_process(ll)
        high_cat = torch.cat([lh, hl, hh], dim=1)
        high_feat = self.high_process(high_cat)
        out = self.fusion(torch.cat([ll_feat, high_feat], dim=1))
        return out

class FGF_Module(nn.Module):
    """ 频率引导融合模块 """
    def __init__(self, spatial_dim, freq_dim):
        super().__init__()
        self.freq_to_att = nn.Sequential(
            nn.Conv2d(freq_dim, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.align = nn.Conv2d(freq_dim, spatial_dim, 1)

    def forward(self, x_spatial, x_freq):
        if x_freq.shape[2:] != x_spatial.shape[2:]:
            x_freq = F.interpolate(x_freq, size=x_spatial.shape[2:], mode='bilinear', align_corners=False)
        att_map = self.freq_to_att(x_freq)
        x_guided = x_spatial * att_map
        return x_guided + self.align(x_freq)

class MFAM(nn.Module):
    """ 混合频率注意力 (Neck) - 复刻 FDENet """
    def __init__(self, in_channels):
        super().__init__()
        reduction = 4
        mid_channels = max(16, in_channels // reduction)

        self.phi_h = nn.Parameter(torch.ones(1, in_channels, 1, 1), requires_grad=True)
        self.phi_l = nn.Parameter(torch.ones(1, in_channels, 1, 1), requires_grad=True)

        self.proj_h_hor = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1))
        self.proj_h_ver = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0))
        self.proj_l_hor = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1))
        self.proj_l_ver = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0))
        self.gamma_h = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.gamma_l = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_h = nn.Sequential(nn.Linear(in_channels, mid_channels), nn.ReLU(), nn.Linear(mid_channels, in_channels), nn.Sigmoid())
        self.fc_l = nn.Sequential(nn.Linear(in_channels, mid_channels), nn.ReLU(), nn.Linear(mid_channels, in_channels), nn.Sigmoid())
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.fusion_conv = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, f_spatial, f_freq):
        if f_freq.shape[2:] != f_spatial.shape[2:]:
            f_freq = F.interpolate(f_freq, size=f_spatial.shape[2:], mode='bilinear', align_corners=False)
        
        B, C, H, W = f_spatial.shape
        f_star = self.phi_h * f_freq + self.phi_l * f_spatial
        
        d_h = self.proj_h_hor(f_freq) + self.proj_h_ver(f_freq)
        d_l = self.proj_l_hor(f_spatial) + self.proj_l_ver(f_spatial)
        f_dir = self.gamma_h * d_h + self.gamma_l * d_l
        
        u_h = self.gap(f_freq).view(B, C)
        u_l = self.gap(f_spatial).view(B, C)
        w_h_c = self.fc_h(self.alpha * u_h + self.beta * u_l).view(B, C, 1, 1)
        w_l_c = self.fc_l(self.alpha * u_l + self.beta * u_h).view(B, C, 1, 1)
        w_c = 0.5 * (w_h_c + w_l_c)
        
        return f_spatial + (self.fusion_conv(f_star) + f_dir) * w_c

# ================================================================
# 3. [关键] 复刻 unet_model_unified.py 的 Up_PHD
# ================================================================

class Up_PHD(nn.Module):
    """
    完全复刻你 unet_model_unified.py 中的 Up_PHD 类
    负责：上采样 -> Padding对齐 -> 拼接 -> 调用 PHD_DecoderBlock
    """
    def __init__(self, in_channels, out_channels, bilinear=True, skip_channels=0, 
                 use_dcn=False, use_dubm=False, use_strg=False):
        super().__init__()
        
        # 1. 定义上采样
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # 计算拼接后的总通道数
            conv_in_channels = in_channels + skip_channels
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            conv_in_channels = (in_channels // 2) + skip_channels

        # 2. 核心：调用 PHD_DecoderBlock
        # 这里不需要手动处理 cat，因为 cat 在 forward 里做完后，通道数就是 conv_in_channels
        # PHD_DecoderBlock 会处理这个维度的输入
        self.conv = PHD_DecoderBlock(in_channels=conv_in_channels, out_channels=out_channels, 
                                     use_dcn=use_dcn, use_dubm=use_dubm)

    def forward(self, x1, x2=None, edge_prior=None):
        # x1: 深层特征 (需要上采样)
        # x2: 浅层 Skip 特征 (需要拼接)
        
        x1 = self.up(x1)
        
        if x2 is not None:
            # Padding 对齐 (防止奇数尺寸不匹配)
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            if diffX != 0 or diffY != 0:
                x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            
            # 拼接
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        
        # 调用 PHD Block (支持传 edge_prior)
        return self.conv(x, edge_prior=edge_prior)

# ================================================================
# 4. S_DMFNet 主模型
# ================================================================

class S_DMFNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, 
                 encoder_name='cnextv2', cnext_type='convnextv2_base', # 强制 Base
                 decoder_name='phd', use_dcn=True,
                 # 接收兼容参数
                 use_mfam=Ture, use_dsis=False, use_dual_stream=False, use_wavelet_denoise=False, 
                 use_wgn_enhancement=False, use_cafm=False, use_edge_loss=False, 
                 use_dubm=False, use_strg=False, **kwargs):
        super(S_DMFNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_mfam = use_mfam  # <--- 🔥 [新增] 2. 保存参数状态
        print(f"🚀 [S-DMFNet] 初始化... Encoder: {cnext_type} (Base), Decoder: {decoder_name}")

        # --- 1. 左路: Spatial Encoder (ConvNeXt V2 Base) ---
        backbone_name = 'convnextv2_base' 
        self.spatial_encoder = timm.create_model(backbone_name, pretrained=True, features_only=True, out_indices=(0, 1, 2, 3))
        s_dims = [128, 256, 512, 1024] 
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

        # --- 3. 交互: FGF Modules ---
        self.fgf_modules = nn.ModuleList([FGF_Module(s_dims[i], f_dims[i]) for i in range(4)])

        # --- 4. 瓶颈: MFAM ---
        self.neck_freq_align = nn.Conv2d(f_dims[-1], s_dims[-1], 1)
        self.neck_mfam = MFAM(in_channels=s_dims[-1])
        if self.use_mfam:  # <--- 🔥 [新增] 3. 加判断
            self.neck_freq_align = nn.Conv2d(f_dims[-1], s_dims[-1], 1)
            self.neck_mfam = MFAM(in_channels=s_dims[-1])
            print("   ✅ MFAM (Neck) Enabled")
        else:
            print("   🚫 MFAM (Neck) Disabled for Ablation")
        # --- 5. 解码器: 使用 Up_PHD 包装器 (完全复刻原代码风格) ---
        c1, c2, c3, c4 = s_dims
        
        # Up 1: x4(1024) + s3(512) -> 输出 c3(512)
        # 这里的 Up_PHD 会自动处理 x4 的上采样和与 s3 的 concat
        self.up1 = Up_PHD(c4, c3, bilinear, skip_channels=c3, use_dcn=use_dcn)
        
        # Up 2: d1(512) + s2(256) -> 输出 c2(256)
        self.up2 = Up_PHD(c3, c2, bilinear, skip_channels=c2, use_dcn=use_dcn)
        
        # Up 3: d2(256) + s1(128) -> 输出 c1(128)
        self.up3 = Up_PHD(c2, c1, bilinear, skip_channels=c1, use_dcn=use_dcn)
        
        self.final_up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.outc = nn.Conv2d(c1, n_classes, kernel_size=1)

        # --- [新增] 边缘预测头 (Edge Head) ---
        # 利用右路第一层(f1)特征预测边缘，用于辅助 Loss
        self.edge_head = nn.Sequential(
            nn.Conv2d(f_dims[0], 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        # === Encoder ===
        s_feats = list(self.spatial_encoder(x))
        
        f_feats = []
        f_curr = self.freq_stem(x)
        f_feats.append(f_curr)
        for layer in self.freq_layers:
            f_curr = layer(f_curr)
            f_feats.append(f_curr)
        f_feats[-1] = self.freq_stage4(f_feats[-1])

        # === Interaction (FGF) ===
        s_clean = []
        for i in range(4):
            s_out = self.fgf_modules[i](s_feats[i], f_feats[i])
            s_clean.append(s_out)
        s1, s2, s3, x4 = s_clean

        # === Neck (MFAM) ===
        if self.use_mfam:  # <--- 🔥 [新增] 4. 加判断
            f4_aligned = self.neck_freq_align(f_feats[3])
            x4_enhanced = self.neck_mfam(x4, f4_aligned)
        else:
            # 如果不使用 MFAM，直接跳过，把 x4 原封不动传给后面
            x4_enhanced = x4

        # === Decoder (使用 Up_PHD 接口) ===
        # Up_PHD.forward(x1, x2) -> x1是深层(x4), x2是浅层Skip(s3)
        d1 = self.up1(x4_enhanced, s3)
        d2 = self.up2(d1, s2)
        d3 = self.up3(d2, s1)
        
        d4 = self.final_up(d3)
        logits = self.outc(d4)
        
        # === 返回逻辑 ===
        if self.training:
            # 计算边缘辅助输出
            # 将 f1 (1/4分辨率) 预测为边缘，再上采样回原图
            edge_logits_small = self.edge_head(f_feats[0])
            edge_logits = F.interpolate(edge_logits_small, size=logits.shape[2:], mode='bilinear', align_corners=True)
            
            # 返回双结果：主分割 + 边缘
            return logits, edge_logits
            
        return logits