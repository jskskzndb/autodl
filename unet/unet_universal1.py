"""
unet/unet_universal.py
[Universal Model] 全能型 UNet
功能: 一个类实现 4 种架构组合，通过参数控制。
核心组件:
  1. Encoder: ConvNeXt V2 (Spatial) + [Optional] WaveletMamba (Frequency)
  2. Interaction: [Optional] Bi-FGF
  3. Decoder: [Selectable] PHD_Pro (Inverted Bottleneck+FFN) OR Standard (DoubleConv)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
# === 新增：DCNv3 依赖检查 ===
try:
    # 假设你的 DCN 模块在 ops_dcnv3 文件夹下
    from ops_dcnv3.modules.dcnv3 import DCNv3
    HAS_DCN = True
    print("✅ [Universal Model] DCNv3 module loaded.")
except ImportError:
    HAS_DCN = False
    print("⚠️ [Universal Model] DCNv3 not found! Fallback to standard Conv.")
# ================================================================
# 0. 基础依赖与环境检查
# ================================================================
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    print("⚠️ Warning: mamba-ssm not found.")
    HAS_MAMBA = False

class MambaLayer2D(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if not HAS_MAMBA: self.mamba = nn.Identity()
        else:
            self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()
            x_seq = x.flatten(2).transpose(1, 2)
            x_seq = self.norm(x_seq)
            if HAS_MAMBA: x_seq = self.mamba(x_seq)
            x_out = x_seq.transpose(1, 2).view(B, C, H, W)
        return x_out

# ================================================================
# 1. 小波变换工具 (用于频率流)
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
        if H % 2 != 0 or W % 2 != 0: x = F.pad(x, (0, W % 2, 0, H % 2), mode='reflect')
        filters = self.filters.repeat(C, 1, 1, 1)
        output = F.conv2d(x, filters, stride=2, groups=C)
        output = output.view(B, C, 4, output.shape[2], output.shape[3])
        return output[:, :, 0], output[:, :, 1], output[:, :, 2], output[:, :, 3]

class InverseHaarWaveletTransform(nn.Module):
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        lh = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
        hl = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]])
        hh = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
        self.register_buffer('filters', torch.stack([ll, lh, hl, hh]).unsqueeze(1) / 2.0)

    def idwt(self, ll, lh, hl, hh):
        B, C, H, W = ll.shape
        x = torch.cat([ll, lh, hl, hh], dim=1)
        return F.conv_transpose2d(x, self.filters.repeat(C, 1, 1, 1), stride=2, groups=C)

# ================================================================
# 2. 核心模块 (PHD Pro, Standard, Bi-FGF)
# ================================================================

# --- A. 频率流模块 ---
class WaveletMambaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dwt = HaarWaveletTransform()
        self.idwt = InverseHaarWaveletTransform()
        self.low_process = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels), MambaLayer2D(dim=out_channels))
        self.high_process = nn.Sequential(nn.Conv2d(in_channels * 3, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.high_restore = nn.Conv2d(out_channels, out_channels * 3, 1)
        self.inter_downsample = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.fusion_next = nn.Sequential(nn.Conv2d(out_channels * 2, out_channels, 1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        ll, lh, hl, hh = self.dwt.dwt(x)
        ll_feat = self.low_process(ll)
        high_feat = self.high_process(torch.cat([lh, hl, hh], dim=1))
        out_next = self.fusion_next(torch.cat([ll_feat, high_feat], dim=1))
        high_restored = self.high_restore(high_feat)
        lh_rec, hl_rec, hh_rec = torch.chunk(high_restored, 3, dim=1)
        out_spatial = self.idwt.idwt(ll_feat, lh_rec, hl_rec, hh_rec)
        return out_next, self.inter_downsample(out_spatial)

# --- B. 交互模块 (Bi-FGF) ---
class Cross_GL_FGF(nn.Module):
    def __init__(self, s_channels, f_channels, reduction=16):
        super().__init__()
        s_mid, f_mid = max(s_channels // reduction, 4), max(f_channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp_s2f = nn.Sequential(nn.Linear(s_channels, f_mid, bias=False), nn.ReLU(), nn.Linear(f_mid, f_channels, bias=False), nn.Sigmoid())
        self.mlp_f2s = nn.Sequential(nn.Linear(f_channels, s_mid, bias=False), nn.ReLU(), nn.Linear(s_mid, s_channels, bias=False), nn.Sigmoid())
        self.spatial_s2f = nn.Sequential(nn.Conv2d(s_channels, 1, 7, padding=3, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())
        self.spatial_f2s = nn.Sequential(nn.Conv2d(f_channels, 1, 7, padding=3, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())
        self.f_align = nn.Conv2d(f_channels, s_channels, 1)
        self.fusion_conv = nn.Sequential(nn.Conv2d(s_channels * 2, s_channels, 1), nn.BatchNorm2d(s_channels), nn.ReLU(inplace=True))

    def forward(self, x_s, x_f):
        B, Cs, H, W = x_s.shape
        _, Cf, _, _ = x_f.shape
        if x_f.shape[2:] != x_s.shape[2:]: x_f = F.interpolate(x_f, size=(H, W), mode='bilinear')
        s_vec, f_vec = self.gap(x_s).view(B, Cs), self.gap(x_f).view(B, Cf)
        f_clean = x_f * self.mlp_s2f(s_vec).view(B, Cf, 1, 1)
        s_clean = x_s * self.mlp_f2s(f_vec).view(B, Cs, 1, 1)
        f_refined = f_clean * self.spatial_s2f(s_clean) + f_clean
        s_refined = s_clean * self.spatial_f2s(f_clean) + s_clean
        out = self.fusion_conv(torch.cat([s_refined, self.f_align(f_refined)], dim=1))
        return out, f_refined
class StripConvBlock(nn.Module):
    """
    [Original PHD Style] 可变形条形卷积模块
    核心逻辑: 使用 DCNv3 的长方形核 (1xK, Kx1) 来模拟条形卷积，
    既保留了条形卷积对道路/边缘的敏感性，又利用 DCN 适应不规则弯曲。
    """
    def __init__(self, in_channels, out_channels, kernel_size=7, use_dcn=True):
        super().__init__()
        padding = (kernel_size - 1) // 2
        
        # 1. 投影层
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True)
        )
        
        # 检查是否开启 DCN
        self.use_dcn = use_dcn and HAS_DCN
        
        if self.use_dcn:
            # === 你的经典设计：用 DCN 实现 Strip Conv ===
            dcn_group = 4 
            # 水平 DCN Strip (1 x K)
            self.strip_h = DCNv3(channels=out_channels, kernel_size=(1, kernel_size), stride=1, 
                                 pad=(0, padding), group=dcn_group, offset_scale=1.0)
            # 垂直 DCN Strip (K x 1)
            self.strip_v = DCNv3(channels=out_channels, kernel_size=(kernel_size, 1), stride=1, 
                                 pad=(padding, 0), group=dcn_group, offset_scale=1.0)
            
            # DCN 后的归一化
            self.norm_h = nn.BatchNorm2d(out_channels)
            self.norm_v = nn.BatchNorm2d(out_channels)
            self.act = nn.ReLU(inplace=True)
        else:
            # === 普通 Strip Conv (回退方案) ===
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
            # DCN 路径
            h = self.act(self.norm_h(self.strip_h(x)))
            v = self.act(self.norm_v(self.strip_v(x)))
        else:
            # 普通路径
            h = self.strip_h(x)
            v = self.strip_v(x)
            
        # 融合：H + V
        return self.fusion_conv(h + v)
# --- C. 解码器组件 1: 标准 UNet Block ---
class StandardDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)
# ================================================================
# [New Architecture] ProtoFormer Decoder Components
# ================================================================

class PrototypeInteractionBlock(nn.Module):
    """
    [ProtoFormer Core] 原型交互单元 (完全体)
    功能: 利用可学习的原型(Prototypes)对特征图进行语义重构。
    包含防御机制:
      1. 动态位置编码 (Learnable PE) -> 防止空间信息丢失
      2. 局部细节卷积 (Local Conv) -> 补充高频纹理
    """
    def __init__(self, channels, num_prototypes=16):
        super().__init__()
        self.channels = channels
        
        # 1. 定义原型 (Learnable Prompts) [1, N, C]
        # 这些向量在训练中会学成"建筑"、"道路"、"背景"等概念
        self.prototypes = nn.Parameter(torch.randn(1, num_prototypes, channels))
        
        # 2. 动态位置编码库 (Learnable PE)
        # 初始化一个 64x64 的位置编码，使用时自动插值
        self.pos_embed = nn.Parameter(torch.randn(1, channels, 64, 64) * 0.02)
        
        # 3. 投影层 (用于 Attention)
        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        
        self.out_proj = nn.Conv2d(channels, channels, 1)
        
        # GroupNorm 比 BN 在这种 Attention 结构下更稳
        self.norm = nn.GroupNorm(8, channels)
        
        # 4. 局部细节补充 (救命稻草)
        # 专门用来提取 Attention 忽略掉的细碎边缘
        self.local_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        
        # --- A. 注入位置信息 (防御空间丢失) ---
        # 动态插值位置编码到当前尺寸
        pos = F.interpolate(self.pos_embed, size=(H, W), mode='bilinear', align_corners=False)
        x_with_pos = x + pos 
        
        # --- B. 准备 Query (图像) ---
        # [B, C, H, W] -> [B, HW, C]
        q = self.q_proj(x_with_pos).flatten(2).transpose(1, 2)
        
        # --- C. 准备 Key/Value (原型) ---
        # [1, N, C] -> [B, N, C]
        protos = self.prototypes.repeat(B, 1, 1)
        k = self.k_proj(protos)
        v = self.v_proj(protos)
        
        # --- D. 语义交互 (Cross Attention) ---
        # 计算像素与原型的相似度: [B, HW, C] @ [B, C, N] -> [B, HW, N]
        scale = C ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1) # 归一化: 每个像素必须选一个最像的原型
        
        # --- E. 重构特征 ---
        # 用原型的信息重组图像: [B, HW, N] @ [B, N, C] -> [B, HW, C]
        out = attn @ v
        out = out.transpose(1, 2).view(B, C, H, W)
        
        # --- F. 融合与输出 ---
        out = self.out_proj(out)
        
        # 加上局部卷积，补充丢失的纹理
        out = out + self.local_conv(out)
        
        return self.norm(out + residual)


class PrototypeQueryHead(nn.Module):
    """
    [Optional] 原型查询输出头
    替代最后的 1x1 Conv，用作语义精炼。
    """
    def __init__(self, in_channels, num_prototypes=32, num_classes=1):
        super().__init__()
        self.num_prototypes = num_prototypes
        # 独立的头原型
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, in_channels))
        # 简单的融合层
        self.merge_conv = nn.Conv2d(num_prototypes, num_classes, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        pixel_features = x.view(B, C, -1) 
        # 准备原型 [1, C, N]
        queries = self.prototypes.unsqueeze(0).transpose(1, 2).repeat(B, 1, 1)
        # 相似度匹配 [B, HW, N]
        sim_map = torch.bmm(pixel_features.transpose(1, 2), queries)
        # 还原空间 [B, N, H, W]
        mask_predictions = sim_map.permute(0, 2, 1).view(B, self.num_prototypes, H, W)
        # 输出
        return self.merge_conv(mask_predictions)


class PHD_DecoderBlock_Pro(nn.Module):
    """
    [New Decoder Wrapper] ProtoFormer 解码块
    完全替代旧版 PHD，对外接口保持不变。
    """
    def __init__(self, in_channels, out_channels, expand_ratio=None): 
        super().__init__()
        
        # 1. 空间对齐 (3x3 Conv)
        self.align = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 2. 原型交互核心 (每层独立拥有 16 个原型)
        self.proto_block = PrototypeInteractionBlock(out_channels, num_prototypes=8)
        
    def forward(self, x):
        # 对齐
        x = self.align(x)
        # 交互重构
        x = self.proto_block(x)
        return x

# --- E. 通用上采样包装器 ---
class Up_Universal(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, decoder_type='phd'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        conv_in = in_channels + skip_channels
        
        if decoder_type == 'phd':
            # 使用 PHD Pro
            self.conv = PHD_DecoderBlock_Pro(conv_in, out_channels)
        else:
            # 使用标准 DoubleConv (中间降维)
            self.conv = StandardDoubleConv(conv_in, out_channels)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if x2 is not None:
            diffY, diffX = x2.size()[2] - x1.size()[2], x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        return self.conv(x)

# ================================================================
# 3. 主模型: UniversalUNet
# ================================================================
class UniversalUNet(nn.Module):
    def __init__(self, 
                 n_classes=1, 
                 cnext_type='convnextv2_tiny', 
                 pretrained=True,
                 decoder_type='phd',       # 'phd' or 'standard'
                 use_dual_stream=True,     # True or False
                 **kwargs):
        super().__init__()
        self.n_classes = n_classes
        self.use_dual_stream = use_dual_stream
        self.decoder_type = decoder_type
        
        print(f"🤖 [Universal Model] Initialized with:")
        print(f"   - Encoder: {cnext_type} (Pretrained={pretrained})")
        print(f"   - Dual Stream: {'✅ ON' if use_dual_stream else '❌ OFF'}")
        print(f"   - Decoder: {'🏆 PHD Pro' if decoder_type=='phd' else '🔹 Standard UNet'}")

        # 1. Spatial Encoder
        self.spatial_encoder = timm.create_model(cnext_type, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3), drop_path_rate=0.0)
        s_dims = self.spatial_encoder.feature_info.channels()
        c1, c2, c3, c4 = s_dims

        # 2. Frequency Encoder & Bi-FGF (Only if Dual Stream)
        if self.use_dual_stream:
            f_dims = [c // 4 for c in s_dims]
            self.freq_stem = nn.Sequential(nn.Conv2d(3, f_dims[0], 4, stride=4, padding=0), nn.BatchNorm2d(f_dims[0]), nn.ReLU(True))
            self.freq_layers = nn.ModuleList([WaveletMambaBlock(f_dims[i], f_dims[i+1]) for i in range(3)])
            self.bi_fgf_modules = nn.ModuleList([Cross_GL_FGF(s_dims[i], f_dims[i]) for i in range(4)])
            
            # 辅助边缘头 (只在双流时有意义)
            self.edge_head = nn.Sequential(nn.Conv2d(f_dims[0], 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.Conv2d(64, 1, 1))

        # 3. Decoder
        self.up1 = Up_Universal(c4, c3, skip_channels=c3, decoder_type=decoder_type)
        self.up2 = Up_Universal(c3, c2, skip_channels=c2, decoder_type=decoder_type)
        self.up3 = Up_Universal(c2, c1, skip_channels=c1, decoder_type=decoder_type)
        
        self.final_up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.outc = nn.Conv2d(c1, n_classes, kernel_size=1)

    def forward(self, x):
        # 1. Encoder Pass
        s_feats = list(self.spatial_encoder(x))
        
        # 2. Dual Stream & Interaction Pass
        edge_logits = None
        if self.use_dual_stream:
            f_curr = self.freq_stem(x)
            f_feats = [f_curr]
            for layer in self.freq_layers:
                f_next, f_inter = layer(f_curr)
                f_feats.append(f_inter)
                f_curr = f_next
            
            # Interaction
            s_fused_list = []
            f_enhanced_list = []
            for i in range(4):
                s_out, f_out = self.bi_fgf_modules[i](s_feats[i], f_feats[i])
                s_fused_list.append(s_out)
                f_enhanced_list.append(f_out)
            
            # Update s_feats with fused features
            s_feats = s_fused_list
            
            # Edge Head
            if self.training:
                edge_small = self.edge_head(f_enhanced_list[0])
                edge_logits = F.interpolate(edge_small, size=x.shape[2:], mode='bilinear', align_corners=True)

        # 3. Decoder Pass (s_feats 可能是原始的，也可能是融合过的)
        s1, s2, s3, s4 = s_feats
        d1 = self.up1(s4, s3)
        d2 = self.up2(d1, s2)
        d3 = self.up3(d2, s1)
        
        logits = self.outc(self.final_up(d3))
        
        if self.training and self.use_dual_stream and edge_logits is not None:
            return logits, edge_logits
        return logits