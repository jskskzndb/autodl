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

# --- C. 解码器组件 1: 标准 UNet Block ---
class StandardDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

# --- D. 解码器组件 2: PHD Pro (改进版) ---
class PHD_DecoderBlock_Pro(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=4):
        super().__init__()
        hidden_dim = int(out_channels * expand_ratio)
        self.align = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        
        # Inverted Bottleneck
        self.expand = nn.Sequential(nn.Conv2d(out_channels, hidden_dim, 1, bias=False), nn.BatchNorm2d(hidden_dim), nn.GELU())
        
        # Dual Stream (Local + Global)
        self.local = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, (1,7), padding=(0,3), groups=hidden_dim, bias=False),
            nn.Conv2d(hidden_dim, hidden_dim, (7,1), padding=(3,0), groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True)
        )
        self.global_branch = MambaLayer2D(hidden_dim)
        
        # Fusion (Simplified SK)
        self.fusion_conv = nn.Conv2d(hidden_dim*2, hidden_dim, 1)
        
        self.proj = nn.Sequential(nn.Conv2d(hidden_dim, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))
        
        # FFN
        ffn_dim = out_channels * 4
        self.ffn = nn.Sequential(
            nn.Conv2d(out_channels, ffn_dim, 1), nn.BatchNorm2d(ffn_dim), nn.GELU(),
            nn.Conv2d(ffn_dim, out_channels, 1), nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.align(x)
        shortcut = x
        x_exp = self.expand(x)
        x_loc = self.local(x_exp)
        x_glo = self.global_branch(x_exp)
        x_fused = self.fusion_conv(torch.cat([x_loc, x_glo], dim=1))
        x_out = self.proj(x_fused)
        x = shortcut + x_out
        x = x + self.ffn(x)
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