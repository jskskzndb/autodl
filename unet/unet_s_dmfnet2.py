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
    

class InverseHaarWaveletTransform(nn.Module):
    """
    小波逆变换 (IDWT)
    将 LL, LH, HL, HH 四个分量还原回空间图像尺寸 (2x)
    """
    def __init__(self):
        super().__init__()
        # 定义逆变换的卷积核 (基于 Haar 小波定义)
        # 这里的权重是为了配合 Forward 的 0.5 系数进行还原
        ll = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        lh = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
        hl = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]])
        hh = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
        
        # 使用 Transposed Conv 来实现上采样+求和
        self.register_buffer('filters', torch.stack([ll, lh, hl, hh]).unsqueeze(1) / 2.0)

    def idwt(self, ll, lh, hl, hh):
        # 输入: [B, C, H, W] * 4
        # 输出: [B, C, 2H, 2W]
        B, C, H, W = ll.shape
        # 将 4 个分量拼接为 [B, 4C, H, W]
        x = torch.cat([ll, lh, hl, hh], dim=1)
        # 使用 Group ConvTranspose2d 进行独立通道的逆变换
        # groups=C 保证每个通道独立还原
        out = F.conv_transpose2d(
            x, 
            self.filters.repeat(C, 1, 1, 1), 
            stride=2, 
            groups=C
        )
        return out
# ================================================================
# 2. 右路核心模块 (WaveletMamba) - 保持不变
# ================================================================

class WaveletMambaBlock(nn.Module):
    """ 
    [Modified] High-Frequency Aware Mamba
    动机：利用 Mamba 的长序列能力，修复高频分量中不连续的建筑物边缘
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dwt = HaarWaveletTransform()
        self.idwt = InverseHaarWaveletTransform() # 🔥 新增逆变换
        # 1. Low Freq (LL): 使用普通卷积捕捉粗略结构
        self.low_process = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 2. High Freq (LH+HL+HH): 使用 Mamba 进行全局边缘连通性建模
        # 输入维度是 in_channels * 3
        self.high_process = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, 1), # 降维对齐
            nn.BatchNorm2d(out_channels),
            # 🔥 Mamba 放这里！处理高频边缘
            MambaLayer2D(dim=out_channels) if MambaLayer2D else nn.Identity() 
        )
        # 🔥 [新增] 高频恢复层：把 Mamba 融合后的 1路高频 拆回 3路 (LH, HL, HH)
        self.high_restore = nn.Conv2d(out_channels, out_channels * 3, 1)
        # 🔥 [新增] 交互对齐层：IDWT 后尺寸变大 2倍，需要下采样回原尺寸以便交互
        # 为什么？因为 s_feats[i] 的尺寸和当前的 out_next 是一样的。
        # IDWT 变大后如果不缩回来，就没法和 ConvNeXt 分支对应层交互了。
        self.inter_downsample = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 3. 融合
        self.fusion = nn.Conv2d(out_channels * 2, out_channels, 1)

    def forward(self, x):
        # 1. DWT 分解
        ll, lh, hl, hh = self.dwt.dwt(x)
        
        # 2. 处理
        ll_feat = self.low_process(ll)
        high_cat = torch.cat([lh, hl, hh], dim=1)
        high_feat = self.high_process(high_cat) # [B, C, H/2, W/2]
        
        # === 路 1: 传给下一层 (保持现状，拼接融合) ===
        out_next = self.fusion(torch.cat([ll_feat, high_feat], dim=1))
        
        # === 路 2: 去交互 (IDWT 还原空间域) ===
        # A. 把 Mamba 处理完的高频特征，尝试拆解回 3 个分量
        high_restored = self.high_restore(high_feat)
        lh_rec, hl_rec, hh_rec = torch.chunk(high_restored, 3, dim=1)
        
        # B. 执行 IDWT (尺寸变大 2倍: H/2 -> H)
        # 这里利用了 IDWT 的归纳偏置，把特征变回“类图像”结构
        out_spatial_large = self.idwt.idwt(ll_feat, lh_rec, hl_rec, hh_rec)
        
        # C. 再次下采样 (H -> H/2) 以匹配 ConvNeXt 对应层的尺寸
        out_inter = self.inter_downsample(out_spatial_large)
        
        # 返回两个：一个去下一层，一个去交互
        return out_next, out_inter
# ================================================================
# 3. [🔥核心升级] Bi-FGF 双向互导模块
#    学术对标: RSBuilding (2024)
# ================================================================

class Cross_GL_FGF(nn.Module):
    """
    [SOTA级交互] Cross Global-Local Frequency-Guided Fusion
    论文图示：X-Structure (Serial)
    逻辑：Global Channel Gating (Denoise) -> Local Spatial Gating (Align) -> Injection (Fusion)
    """
    def __init__(self, s_channels, f_channels, reduction=16):
        super().__init__()
        
        # 安全计算隐藏层维度
        s_mid = max(s_channels // reduction, 4)
        f_mid = max(f_channels // reduction, 4)

        # --- Stage 1: Global Channel Interaction (宏观去噪) ---
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # S -> F (语义指导频率：去噪)
        self.mlp_s2f = nn.Sequential(
            nn.Linear(s_channels, f_mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(f_mid, f_channels, bias=False),
            nn.Sigmoid()
        )
        # F -> S (频率指导语义：关注细节)
        self.mlp_f2s = nn.Sequential(
            nn.Linear(f_channels, s_mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(s_mid, s_channels, bias=False),
            nn.Sigmoid()
        )

        # --- Stage 2: Local Spatial Interaction (微观精修) ---
        self.spatial_conv_s2f = nn.Sequential(
            nn.Conv2d(s_channels, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.spatial_conv_f2s = nn.Sequential(
            nn.Conv2d(f_channels, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        # --- Stage 3: Feature Injection (特征融合) ---
        self.s_align = nn.Conv2d(s_channels, f_channels, 1)
        self.f_align = nn.Conv2d(f_channels, s_channels, 1)
        
        # Zero-Init: 保证训练初期互不干扰
        nn.init.constant_(self.s_align.weight, 0)
        nn.init.constant_(self.s_align.bias, 0)
        nn.init.constant_(self.f_align.weight, 0)
        nn.init.constant_(self.f_align.bias, 0)
        
        # 最终融合卷积 (Concatenate -> Conv)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(s_channels * 2, s_channels, 1),
            nn.BatchNorm2d(s_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_s, x_f):
        B, Cs, H, W = x_s.shape
        _, Cf, _, _ = x_f.shape
        
        # 尺寸对齐
        if x_f.shape[2:] != x_s.shape[2:]:
            x_f = F.interpolate(x_f, size=(H, W), mode='bilinear', align_corners=False)

        # 1. 全局去噪 (Channel Gating)
        s_vec = self.gap(x_s).view(B, Cs)
        f_vec = self.gap(x_f).view(B, Cf)
        w_s2f = self.mlp_s2f(s_vec).view(B, Cf, 1, 1) 
        w_f2s = self.mlp_f2s(f_vec).view(B, Cs, 1, 1) 
        f_clean = x_f * w_s2f
        s_clean = x_s * w_f2s
        
        # 2. 局部精修 (Spatial Attention)
        m_s2f = self.spatial_conv_s2f(s_clean) 
        m_f2s = self.spatial_conv_f2s(f_clean)
        f_refined = f_clean * m_s2f + f_clean
        s_refined = s_clean * m_f2s + s_clean

        # 3. 交叉融合 (Fusion for Skip)
        # 将 F 对齐并注入
        f_injected = self.f_align(f_refined)
        # 拼接 + 卷积融合 (生成跳跃连接特征)
        out = self.fusion_conv(torch.cat([s_refined, f_injected], dim=1))
        
        # 返回: (跳跃连接特征, 增强后的频率特征)
        return out, f_refined
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
# 5. S_DMFNet 主模型 (Refined)
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
        
        print(f"🚀 [S-DMFNet Pro] Rebuttal Version | Encoder: {cnext_type}")
        print(f"   ✨ Features: Cross-GL-FGF (SOTA Interaction), High-Freq Mamba, No SK-Fusion")

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

        # --- 3. [升级] 交互: Cross_GL_FGF Modules ---
        self.bi_fgf_modules = nn.ModuleList([Cross_GL_FGF(s_dims[i], f_dims[i]) for i in range(4)])

        
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
        s_feats = list(self.spatial_encoder(x))
        
        f_feats = []
        # Stem 层处理 (保持不变)
        f_curr = self.freq_stem(x) 
        f_feats.append(f_curr) # f1 (对应 s1)
        
        # === 核心修改部分 ===
        # 循环处理频率层
        for layer in self.freq_layers:
            # 🔥 接收两个输出：f_next (去下一层), f_inter (去交互)
            f_next, f_inter = layer(f_curr)
            
            f_feats.append(f_inter) # 存入去交互的特征
            f_curr = f_next         # 更新当前流，继续往下走
            
        # 处理最后一层 (Stage 4 通常没有下一层了，直接当做交互特征)
        # 注意：你需要检查 freq_stage4 是否也需要改成上面的结构，
        # 或者简单处理。通常最后一层可以不需要分流，因为后面没有下一层了。
        # 这里假设 freq_stage4 还是原来的结构，或者你也把它改成新的 Block。
        # 如果 freq_stage4 是 WaveletMambaBlock，它会返回两个值。
        f_last_next, f_last_inter = self.freq_stage4(f_curr)
        f_feats[-1] = f_last_inter # 更新最后一个特征

        # === Interaction (Cross-GL-FGF) ===
        skips = []      # 用于 Skip Connection
        f_enhanced = [] # 用于 Edge Head
        
        for i in range(4):
            # fusion_out: 融合后的特征 (Skip)
            # f_out: 增强后的频率特征 (Deep Supervision)
            fusion_out, f_out = self.bi_fgf_modules[i](s_feats[i], f_feats[i])
            skips.append(fusion_out)
            f_enhanced.append(f_out)

        s1_fused, s2_fused, s3_fused, s4_fused = skips

        # === Decoder ===
        d1 = self.up1(s4_fused, s3_fused)
        d2 = self.up2(d1, s2_fused)
        d3 = self.up3(d2, s1_fused)
        
        d4 = self.final_up(d3)
        logits = self.outc(d4)
        
        # === Auxiliary Output ===
        if self.training:
            # 🔥 [关键修正] 输入使用 f_enhanced[0] (清洗后的频率特征)
            # 理由：利用语义流抑制了背景纹理噪声，使边缘监督更精准
            edge_logits_small = self.edge_head(f_enhanced[0])
            edge_logits = F.interpolate(edge_logits_small, size=logits.shape[2:], mode='bilinear', align_corners=True)
            return logits, edge_logits
            
        return logits