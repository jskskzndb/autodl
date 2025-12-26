"""
unet_s_dmfnet.py
[S-DMFNet] Simplified Dual-Stream Mutual-Guided Frequency-Aware Network
架构说明：
1. 左路主干：ConvNeXt V2 Base (语义流)
2. 右路主干：Wavelet-Mamba Encoder (频率/边界流)
3. 交互模块：FGF (Frequency-Guided Fusion) - 每一层交互
4. 瓶颈融合：MFAM (Mixed-Frequency Attention Mechanism)
5. 解码器：PHD Decoder (Single Stream) - 仅负责主体预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import sys
from pathlib import Path

# 尝试导入 PHD 解码器块 (适配你的目录结构)
try:
    from decoder.hybrid_decoder import PHD_DecoderBlock
except ImportError:
    # 兼容性处理：如果直接在根目录运行
    try:
        from unet.hybrid_decoder import PHD_DecoderBlock
    except ImportError:
        print("❌ 警告: 未找到 PHD_DecoderBlock，请检查 decoder/hybrid_decoder.py 是否存在。")
        PHD_DecoderBlock = None

# 尝试导入 Mamba 辅助类 (用于右路频率流)
try:
    from decoder.mamba_helper import MambaLayer2D
except ImportError:
    try:
        from unet.mamba_helper import MambaLayer2D
    except ImportError:
        print("❌ 警告: 未找到 MambaLayer2D，请检查 mamba_helper.py。")
        MambaLayer2D = None

# ================================================================
# 1. 基础工具类 (小波变换 & 频率处理)
# ================================================================

class HaarWaveletTransform(nn.Module):
    """ 离散小波变换 (DWT) 和 逆变换 (IWT) """
    def __init__(self):
        super().__init__()
        # Haar 滤波器
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        
        self.register_buffer('filters', torch.stack([ll, lh, hl, hh]).unsqueeze(1))

    def dwt(self, x):
        B, C, H, W = x.shape
        # Pad if needed
        if H % 2 != 0 or W % 2 != 0:
            x = F.pad(x, (0, W % 2, 0, H % 2), mode='reflect')
        
        # Group convolution for channel-wise DWT
        filters = self.filters.repeat(C, 1, 1, 1)
        output = F.conv2d(x, filters, stride=2, groups=C)
        
        # Split into subbands
        B, C4, H2, W2 = output.shape
        # output structure: [C*4, H/2, W/2] -> 0::4 is LL, 1::4 is LH...
        # Reshape to easily extract: [B, C, 4, H/2, W/2]
        output = output.view(B, C, 4, H2, W2)
        
        ll, lh, hl, hh = output[:, :, 0], output[:, :, 1], output[:, :, 2], output[:, :, 3]
        return ll, lh, hl, hh

    def idwt(self, ll, lh, hl, hh):
        # 简化的 IWT (使用转置卷积或插值+加权，这里为了效率使用 Upsample 近似或直接反向逻辑)
        # 为保证梯度传播和精确重构，这里使用 Upsample + Conv 的可学习逆变换方式替代标准 IDWT，
        # 或者为了严格复现，我们使用反向 Haar 卷积。
        # 这里为了代码稳定性，采用特征拼接后上采样融合，模拟 IWT 效果。
        return torch.cat([ll, lh, hl, hh], dim=1) 

# ================================================================
# 2. 核心模块: FGF & MFAM & WaveletMambaBlock
# ================================================================

class WaveletMambaBlock(nn.Module):
    """
    [右路] 小波-Mamba 编码器块
    结构: DWT -> LL(Mamba) + High(Conv) -> Fusion
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dwt = HaarWaveletTransform()
        
        # 1. 低频处理 (LL): 使用 Mamba 捕捉全局结构
        # 输入通道是 in_channels (DWT 后 spatial 减半，通道不变)
        self.low_process = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            MambaLayer2D(dim=out_channels) if MambaLayer2D else nn.Identity()
        )
        
        # 2. 高频处理 (LH, HL, HH): 使用卷积捕捉边缘
        # 输入通道是 in_channels * 3
        self.high_process = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3. 融合层 (输出给下一层 或 FGF)
        self.fusion = nn.Conv2d(out_channels * 2, out_channels, 1)

    def forward(self, x):
        # 1. DWT 分解
        ll, lh, hl, hh = self.dwt.dwt(x)
        
        # 2. 双流处理
        ll_feat = self.low_process(ll)
        high_cat = torch.cat([lh, hl, hh], dim=1)
        high_feat = self.high_process(high_cat)
        
        # 3. 融合 (模拟 IWT 的信息整合)
        out = self.fusion(torch.cat([ll_feat, high_feat], dim=1))
        
        # 对齐尺寸 (如果 DWT 导致分辨率减半，这里 out 已经是 H/2, W/2)
        return out

class FGF_Module(nn.Module):
    """
    [交互] 频率引导融合模块 (Frequency-Guided Fusion)
    逻辑: 右路(Freq) 生成 Attention Map -> 指导 左路(Spatial)
    """
    def __init__(self, spatial_dim, freq_dim):
        super().__init__()
        # 将右路特征映射为 1 通道注意力图
        self.freq_to_att = nn.Sequential(
            nn.Conv2d(freq_dim, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        # 简单的通道对齐 (可选，用于残差增强)
        self.align = nn.Conv2d(freq_dim, spatial_dim, 1)

    def forward(self, x_spatial, x_freq):
        # 1. 对齐尺寸 (如果右路是下采样后的，需要上采样对齐左路)
        # 左路如果是 [H, W]，右路同层输入前是 [H, W] (但经过 Block 后可能变了)
        # 这里假设 x_freq 已经被调整到与 x_spatial 同尺寸，或者需要插值
        if x_freq.shape[2:] != x_spatial.shape[2:]:
            x_freq = F.interpolate(x_freq, size=x_spatial.shape[2:], mode='bilinear', align_corners=False)

        # 2. 生成频率注意力图 (0~1)
        att_map = self.freq_to_att(x_freq)
        
        # 3. 指导: 空间特征 * 注意力图 (抑制噪声)
        x_guided = x_spatial * att_map
        
        # 4. 残差补充: 将频率特征加回去增强边缘
        x_out = x_guided + self.align(x_freq)
        
        return x_out

class MFAM(nn.Module):
    """
    [瓶颈] 混合频率注意力机制 (Mixed-Frequency Attention Mechanism)
    """
    def __init__(self, in_channels):
        super().__init__()
        reduction = 4
        mid_channels = max(16, in_channels // reduction)

        # 自适应频率平衡参数
        self.phi_h = nn.Parameter(torch.ones(1, in_channels, 1, 1), requires_grad=True)
        self.phi_l = nn.Parameter(torch.ones(1, in_channels, 1, 1), requires_grad=True)

        # 方向信息提取 (模拟水平/垂直卷积)
        self.proj_h_hor = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1))
        self.proj_h_ver = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0))
        self.proj_l_hor = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1))
        self.proj_l_ver = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0))
        self.gamma_h = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.gamma_l = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        # 通道相关性建模
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_h = nn.Sequential(nn.Linear(in_channels, mid_channels), nn.ReLU(), nn.Linear(mid_channels, in_channels), nn.Sigmoid())
        self.fc_l = nn.Sequential(nn.Linear(in_channels, mid_channels), nn.ReLU(), nn.Linear(mid_channels, in_channels), nn.Sigmoid())
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        self.fusion_conv = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, f_spatial, f_freq):
        # 尺寸对齐
        if f_freq.shape[2:] != f_spatial.shape[2:]:
            f_freq = F.interpolate(f_freq, size=f_spatial.shape[2:], mode='bilinear', align_corners=False)

        B, C, H, W = f_spatial.shape
        
        # 1. 平衡
        f_star = self.phi_h * f_freq + self.phi_l * f_spatial

        # 2. 方向提取
        d_h = self.proj_h_hor(f_freq) + self.proj_h_ver(f_freq)
        d_l = self.proj_l_hor(f_spatial) + self.proj_l_ver(f_spatial)
        f_dir = self.gamma_h * d_h + self.gamma_l * d_l

        # 3. 通道交互
        u_h = self.gap(f_freq).view(B, C)
        u_l = self.gap(f_spatial).view(B, C)
        
        # 简化的互相关计算 (为了节省显存，不直接算 BxCxC 的矩阵，而是用线性层模拟交互)
        # 这里严格复现论文逻辑需要 BxCxC，但对于 Base 模型可能 OOM，采用近似复现：
        w_h_c = self.fc_h(self.alpha * u_h + self.beta * u_l).view(B, C, 1, 1)
        w_l_c = self.fc_l(self.alpha * u_l + self.beta * u_h).view(B, C, 1, 1)
        w_c = 0.5 * (w_h_c + w_l_c)

        # 4. 重构
        f_fused = (self.fusion_conv(f_star) + f_dir) * w_c
        return f_spatial + f_fused

# ================================================================
# 3. S_DMFNet 主模型
# ================================================================

class S_DMFNet(nn.Module):
    """
    S-DMFNet 模型主类
    完全适配 train.py 的调用接口
    """
    def __init__(self, n_channels, n_classes, bilinear=False, 
                 encoder_name='cnextv2', cnext_type='convnextv2_base', # 强制 Base
                 decoder_name='phd', use_dcn=True,
                 # 接收所有 train.py 可能传入的参数，防止报错 (但不一定都使用)
                 use_dsis=False, use_dual_stream=False, use_wavelet_denoise=False, 
                 use_wgn_enhancement=False, use_cafm=False, use_edge_loss=False, 
                 use_dubm=False, use_strg=False, **kwargs):
        super(S_DMFNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        print(f"🚀 [S-DMFNet] 初始化... Encoder: {cnext_type} (Base), Decoder: {decoder_name}")

        # --- 1. 左路: Spatial Encoder (ConvNeXt V2 Base) ---
        # 强制使用 convnextv2_base，忽略传入的 cnext_type 如果它不是 base
        backbone_name = 'convnextv2_base' 
        self.spatial_encoder = timm.create_model(
            backbone_name, 
            pretrained=True, 
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )
        
        # Base 版本的通道数: [128, 256, 512, 1024]
        # s1(4x), s2(8x), s3(16x), s4(32x)
        s_dims = [128, 256, 512, 1024] 
        self.dims = s_dims

        # --- 2. 右路: Frequency Encoder (Lightweight Wavelet Stream) ---
        # 为了显存平衡，右路通道数设为左路的 1/4
        f_dims = [c // 4 for c in s_dims] # [32, 64, 128, 256]
        
        # Stem: 快速下采样到 4x (对齐 s1)
        self.freq_stem = nn.Sequential(
            nn.Conv2d(3, f_dims[0], 4, stride=4, padding=0),
            nn.BatchNorm2d(f_dims[0]),
            nn.ReLU(inplace=True)
        )
        
        self.freq_layers = nn.ModuleList()
        # Stage 1->2, 2->3, 3->4
        for i in range(3):
            self.freq_layers.append(WaveletMambaBlock(f_dims[i], f_dims[i+1]))
            
        # 最后一个 Stage 4 的处理
        self.freq_stage4 = WaveletMambaBlock(f_dims[3], f_dims[3])

        # --- 3. 交互: FGF Modules (每一层) ---
        self.fgf_modules = nn.ModuleList([
            FGF_Module(s_dims[i], f_dims[i]) for i in range(4)
        ])

        # --- 4. 瓶颈: MFAM (Deep Fusion) ---
        # 先对齐右路通道到左路
        self.neck_freq_align = nn.Conv2d(f_dims[-1], s_dims[-1], 1)
        self.neck_mfam = MFAM(in_channels=s_dims[-1])

        # --- 5. 解码器: PHD Decoder (仅单流) ---
        # 重新映射通道变量，适配 copy 来的代码习惯
        c1, c2, c3, c4 = s_dims
        
        # 定义上采样模块 (PHD Blocks)
        # PHD_DecoderBlock(in_ch, skip_ch, out_ch)
        # Up 1: x4 (1024) + s3 (512) -> 512
        self.up1 = PHD_DecoderBlock(c4, c3, c3, use_dcn=use_dcn)
        # Up 2: d1 (512) + s2 (256) -> 256
        self.up2 = PHD_DecoderBlock(c3, c2, c2, use_dcn=use_dcn)
        # Up 3: d2 (256) + s1 (128) -> 128
        self.up3 = PHD_DecoderBlock(c2, c1, c1, use_dcn=use_dcn)
        
        # Final Up: d3 (128) -> Original Res
        self.final_up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.outc = nn.Conv2d(c1, n_classes, kernel_size=1)

    def forward(self, x):
        # === Encoder Pass ===
        
        # 1. 左路 ConvNeXt
        # s_feats = [s1, s2, s3, s4]
        s_feats = list(self.spatial_encoder(x))
        
        # 2. 右路 Wavelet
        f_feats = []
        f_curr = self.freq_stem(x) # [B, 32, H/4, W/4] -> Align with s1
        f_feats.append(f_curr)
        
        # 逐级处理: f1->f2, f2->f3, f3->f4
        for layer in self.freq_layers:
            f_curr = layer(f_curr) # DWT Downsample inside
            f_feats.append(f_curr)
        
        # 处理最后一层 f4 (保持分辨率)
        f_feats[-1] = self.freq_stage4(f_feats[-1])

        # === Interaction (FGF) ===
        # 每一层进行融合清洗
        s_clean = []
        for i in range(4):
            # s_feats[i] 和 f_feats[i] 分辨率应当一致
            s_out = self.fgf_modules[i](s_feats[i], f_feats[i])
            s_clean.append(s_out)
            
        s1, s2, s3, x4 = s_clean

        # === Neck (MFAM) ===
        # 深度融合 x4 和 f4
        f4_aligned = self.neck_freq_align(f_feats[3])
        x4_enhanced = self.neck_mfam(x4, f4_aligned)

        # === Decoder Pass (PHD) ===
        # 使用清洗后的 skip features (s1, s2, s3) 和 增强后的深层特征 (x4_enhanced)
        
        # d1: [B, 512, H/16, W/16]
        d1 = self.up1(x4_enhanced, s3)
        
        # d2: [B, 256, H/8, W/8]
        d2 = self.up2(d1, s2)
        
        # d3: [B, 128, H/4, W/4]
        d3 = self.up3(d2, s1)
        
        # Final: [B, n_classes, H, W]
        d4 = self.final_up(d3)
        logits = self.outc(d4)
        
        return logits