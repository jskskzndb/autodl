"""
unet/unet_universal_v9_dynamic_final.py
[Universal Model] 全能型 UNet (Final Dynamic Version)
架构特点:
  1. Spatial Encoder: 支持 ConvNeXt V2 / Swin (语义提取)
  2. Frequency Encoder: Omni-SFDA Block (集成了方向感知、循环稀疏编码与软阈值去噪)
  3. Interaction: SK-Fusion (基于SK-Net思想的动态选择性融合)
  4. Decoder: Dynamic PHD Decoder (最终优化版：移除Tanh，引入Alpha自适应系数，增加Pre-LayerNorm)
  5. Auxiliary Supervision: 复刻 DeepSwinLite 的 SE-AuxHead
  6. Return Logic: 统一返回 List 接口，支持深度监督训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# ================================================================
# 0. 基础工具类 (小波变换 & 核心组件)
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

class GlobalAttention(nn.Module):
    """ 全局注意力 (FP32 Safe & Logits Clamp) """
    def __init__(self, dim, num_heads=4, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        if x.dim() == 4:
            B, C, H, W = x.shape
            x_in = x.flatten(2).transpose(1, 2)
            is_spatial = True
        else:
            B, N, C = x.shape
            x_in = x
            is_spatial = False

        x_norm = self.norm(x_in)
        with torch.cuda.amp.autocast(enabled=False):
            x_32 = x_norm.float()
            qkv = self.qkv(x_32).reshape(B, -1, 3, self.num_heads, x_32.shape[-1] // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn_logits = (q @ k.transpose(-2, -1)) * self.scale
            attn_logits = torch.clamp(attn_logits, min=-30, max=30) 
            attn = attn_logits.softmax(dim=-1)
            x_out = (attn @ v) 
        
        x_out = x_out.to(x_in.dtype) 
        x_out = x_out.transpose(1, 2).reshape(B, -1, x_norm.shape[-1])
        x_out = self.proj(x_out)
        x_out = x_in + x_out

        if is_spatial:
            x_out = x_out.transpose(1, 2).reshape(B, C, H, W)
        return x_out

# ================================================================
# 1. 辅助监督模块: DeepSwinLite AuxHead
# ================================================================

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(self.avg_pool(x))

class DeepSwinAuxHead(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super().__init__()
        mid_channels = in_channels // 2
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True)
        )
        self.se = SEBlock(mid_channels, reduction=8)
        self.dropout = nn.Dropout2d(p=0.3)
        self.out_conv = nn.Conv2d(mid_channels, num_classes, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.se(x)
        x = self.dropout(x)
        return self.out_conv(x)

# ================================================================
# 2. 频率流模块: Omni-SFDA Components
# ================================================================

class FrequencyChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        return x * y

class LearnableSoftThresholding(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor([0.02] * channels).view(1, channels, 1, 1))
    def forward(self, x):
        thresh = torch.abs(self.threshold)
        return torch.sign(x) * F.relu(torch.abs(x) - thresh)

class DirectionalEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_lh = nn.Sequential(nn.Conv2d(in_channels, out_channels, (3, 1), padding=(1, 0), bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True))
        self.conv_hl = nn.Sequential(nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1), bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True))
        self.conv_hh = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True))
        self.fusion = nn.Conv2d(out_channels * 3, out_channels, 1, bias=False)
    def forward(self, lh, hl, hh):
        return self.fusion(torch.cat([self.conv_lh(lh), self.conv_hl(hl), self.conv_hh(hh)], dim=1))

class RecurrentSparseBlock(nn.Module):
    def __init__(self, channels, iterations=2):
        super().__init__()
        self.iterations = iterations
        self.encoder = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.decoder = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.loop_norm = nn.BatchNorm2d(channels)
        self.threshold = LearnableSoftThresholding(channels)

    def forward(self, x):
        z = self.threshold(self.bn(self.encoder(x)))
        for _ in range(self.iterations):
            x_hat = self.decoder(z)
            error = x - x_hat
            z = self.loop_norm(z + 0.1 * self.encoder(error))
            z = torch.clamp(z, min=-20.0, max=20.0)
            z = self.threshold(z)
        return z

class SFDABlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4):
        super().__init__()
        self.dwt = HaarWaveletTransform()
        self.freq_select = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, 1),
            nn.BatchNorm2d(out_channels), nn.ReLU(True),
            FrequencyChannelAttention(out_channels)
        )
        self.lo_proj = nn.Conv2d(in_channels, out_channels, 1)
        self.lo_process = nn.Sequential(
            nn.AvgPool2d(2), GlobalAttention(out_channels, num_heads=num_heads), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.hi_proj = nn.Conv2d(in_channels, out_channels, 1)
        self.directional_encoder = DirectionalEncoder(out_channels, out_channels)
        self.recurrent_denoiser = RecurrentSparseBlock(out_channels)
        self.fusion = nn.Sequential(nn.Conv2d(out_channels * 3, out_channels, 1), nn.BatchNorm2d(out_channels), nn.ReLU(True))
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.AvgPool2d(2), nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = nn.AvgPool2d(2)

    def forward(self, x):
        res = self.shortcut(x)
        x = torch.clamp(x, -50, 50)
        ll, lh, hl, hh = self.dwt.dwt(x)
        
        feat_sel = self.freq_select(torch.cat([ll, lh, hl, hh], dim=1))
        feat_lo = self.lo_process(self.lo_proj(ll))
        
        feat_hi_dir = self.directional_encoder(self.hi_proj(lh), self.hi_proj(hl), self.hi_proj(hh))
        feat_hi_final = self.recurrent_denoiser(feat_hi_dir)
        
        out = self.fusion(torch.cat([feat_sel, feat_lo, feat_hi_final], dim=1)) + res
        return out, out

# ================================================================
# 3. 交互模块 (SK-Fusion)
# ================================================================

class SK_Fusion_Block(nn.Module):
    def __init__(self, s_channels, f_channels, reduction=16):
        super().__init__()
        self.f_align = nn.Sequential(nn.Conv2d(f_channels, s_channels, 1, bias=False), nn.BatchNorm2d(s_channels), nn.ReLU(True))
        mid_dim = max(s_channels // reduction, 32)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Conv2d(s_channels, mid_dim, 1, bias=False), nn.ReLU(True), nn.Conv2d(mid_dim, s_channels * 2, 1, bias=False))
        self.softmax = nn.Softmax(dim=1)
        self.out_conv = nn.Sequential(nn.Conv2d(s_channels, s_channels, 3, padding=1, bias=False), nn.BatchNorm2d(s_channels), nn.ReLU(True))

    def forward(self, x_s, x_f):
        if x_f.shape[2:] != x_s.shape[2:]:
            x_f = F.interpolate(x_f, size=x_s.shape[2:], mode='bilinear', align_corners=True)
        x_f_aligned = self.f_align(x_f)
        u = x_s + x_f_aligned
        z = self.mlp(self.avg_pool(u)).view(x_s.size(0), 2, x_s.size(1), 1, 1)
        weights = self.softmax(z)
        out = self.out_conv(weights[:, 0] * x_s + weights[:, 1] * x_f_aligned)
        return out, x_f_aligned

# ================================================================
# 4. 解码器组件: Dynamic PHD Decoder (方式A: 加性偏移 - 最终修复版)
# ================================================================

class DynamicPrototypeModulator(nn.Module):
    """ 
    🔥 [方式A: 加性偏移 - 最终版] 
    1. 移除 Tanh: 解除幅度限制，救活浅层
    2. 引入 Alpha: 自适应缩放，平衡深层过热与浅层过冷
    """
    def __init__(self, channels, num_prototypes):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.channels = channels
        self.seed_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, num_prototypes * channels, 1, bias=False),
            # 🔥 移除了 nn.Tanh() 以释放动态范围
        )
        
        # 🔥 可学习的缩放系数，初始化为 0.1 以实现软启动
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, static_prototypes):
        B, C, H, W = x.shape
        # 提取视觉种子偏移量
        offset = self.seed_extractor(x).view(B, self.num_prototypes, C)
        # 执行加性偏移调制: Static + Alpha * Dynamic
        return static_prototypes + self.alpha * offset

class PrototypeInteractionBlock(nn.Module):
    """ 
    🔥 升级版: 动态原型交互块 (集成 Pre-LayerNorm) 
    """
    def __init__(self, channels, num_prototypes=32): # 🔥 默认原型增加到 32
        super().__init__()
        self.channels = channels
        self.num_prototypes = num_prototypes
        
        # 1. 静态原型 (基准锚点)
        self.prototypes = nn.Parameter(torch.randn(1, num_prototypes, channels))
        
        # 2. 动态调制器 (方式A)
        self.modulator = DynamicPrototypeModulator(channels, num_prototypes)
        
        # 🔥 [关键修复] 前置 LayerNorm，解决浅层特征值过小导致的归一化冲突
        self.pre_norm = nn.LayerNorm(channels)
        
        self.pos_embed = nn.Parameter(torch.randn(1, channels, 64, 64) * 0.02)
        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        self.norm = nn.BatchNorm2d(channels)
        self.local_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False), 
            nn.BatchNorm2d(channels), nn.GELU()
        )
        self.gamma = nn.Parameter(torch.ones(channels) * 1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        pos = F.interpolate(self.pos_embed, size=(H, W), mode='bilinear', align_corners=False)
        x = x + pos 
        
        # --- 核心修改区 ---
        # 1. 归一化输入特征，防止数值过大或过小导致动态分支失效
        x_flat = x.flatten(2).transpose(1, 2) # [B, N, C]
        x_norm = self.pre_norm(x_flat)        # LayerNorm
        x_in = x_norm.transpose(1, 2).view(B, C, H, W) # 还原回 [B, C, H, W]
        
        # 2. 生成动态原型 (传入归一化后的特征)
        dynamic_protos = self.modulator(x_in, self.prototypes)
        # ----------------
        
        q = self.q_proj(x).flatten(2).transpose(1, 2)
        k = self.k_proj(dynamic_protos)
        v = self.v_proj(dynamic_protos)
        
        with torch.cuda.amp.autocast(enabled=False):
            q_32, k_32, v_32 = q.float(), k.float(), v.float()
            scale = C ** -0.5
            attn_logits = (q_32 @ k_32.transpose(-2, -1)) * scale
            attn_logits = torch.clamp(attn_logits, min=-30, max=30)
            attn = attn_logits.softmax(dim=-1)
            out = (attn @ v_32).to(x.dtype)
            
        out = out.transpose(1, 2).view(B, C, H, W)
        out = self.out_proj(out)
        out = out + self.local_conv(out)
        return self.norm(residual + out * self.gamma.view(1, -1, 1, 1))

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1), nn.GELU(), nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, 1), nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)

class PHD_DecoderBlock_Pro(nn.Module):
    def __init__(self, in_channels, out_channels, depth=2): 
        super().__init__()
        self.align = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True))
        self.layers = nn.ModuleList([])
        self.gamma_ffn = nn.Parameter(torch.ones(depth, out_channels) * 1e-5)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # 🔥 num_prototypes 自动继承默认值 32
                PrototypeInteractionBlock(out_channels, num_prototypes=32),
                FeedForward(out_channels, out_channels * 4)
            ]))
    def forward(self, x):
        x = self.align(x)
        for i, (proto_block, ffn) in enumerate(self.layers):
            x = proto_block(x)
            x = x + self.gamma_ffn[i].view(1, -1, 1, 1) * ffn(x)
        return x

class Up_Universal(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, decoder_type='phd'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        conv_in = in_channels + skip_channels
        if decoder_type == 'phd':
            self.conv = PHD_DecoderBlock_Pro(conv_in, out_channels, depth=2)
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(conv_in, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True)
            )
    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if x2 is not None:
            if x2.size()[2:] != x1.size()[2:]:
                x1 = F.pad(x1, [0, x2.size(3)-x1.size(3), 0, x2.size(2)-x1.size(2)])
            x = torch.cat([x2, x1], dim=1)
        else: x = x1
        return self.conv(x)

# ================================================================
# 5. 主模型: UniversalUNet
# ================================================================

class UniversalUNet(nn.Module):
    def __init__(self, n_classes=1, cnext_type='convnextv2_tiny', pretrained=True, decoder_type='phd', 
                 use_dual_stream=False, use_deep_supervision=False, **kwargs):
        super().__init__()
        self.n_classes = n_classes
        self.use_dual_stream = use_dual_stream
        self.decoder_type = decoder_type
        self.use_deep_supervision = use_deep_supervision
        self.encoder_name = cnext_type.lower()
        
        print(f"🤖 [Dynamic Universal Model Final] Initialized:")
        print(f"   - Encoder: {cnext_type} | Dual Stream: {use_dual_stream}")
        print(f"   - Decoder: {decoder_type.upper()} (Alpha-Dynamic + PreNorm)")
        
        # 1. Spatial Encoder
        self.spatial_encoder = timm.create_model(cnext_type, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3))
        s_dims = self.spatial_encoder.feature_info.channels()
        
        # 2. Frequency Encoder (Omni-SFDA)
        if self.use_dual_stream:
            f_dims = [c // 2 for c in s_dims]
            self.freq_stem = nn.Sequential(nn.Conv2d(3, f_dims[0], 4, stride=4), nn.BatchNorm2d(f_dims[0]), nn.ReLU(True))
            self.freq_layers = nn.ModuleList([SFDABlock(f_dims[i], f_dims[i+1]) for i in range(3)])
            self.bi_fgf_modules = nn.ModuleList([SK_Fusion_Block(s_dims[i], f_dims[i]) for i in range(4)])
            self.edge_head = nn.Sequential(nn.Conv2d(f_dims[0], 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.Conv2d(64, 1, 1))

        # 3. Decoder
        self.up1 = Up_Universal(s_dims[3], s_dims[2], skip_channels=s_dims[2], decoder_type=decoder_type)
        self.up2 = Up_Universal(s_dims[2], s_dims[1], skip_channels=s_dims[1], decoder_type=decoder_type)
        self.up3 = Up_Universal(s_dims[1], s_dims[0], skip_channels=s_dims[0], decoder_type=decoder_type)
        self.final_up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.outc = nn.Conv2d(s_dims[0], n_classes, kernel_size=1)
        
        # 4. Deep Supervision (DeepSwinLite Style)
        if self.use_deep_supervision:
            self.head_up2 = DeepSwinAuxHead(s_dims[1], n_classes)
            self.head_up3 = DeepSwinAuxHead(s_dims[0], n_classes)

    def forward(self, x):
        s_feats = list(self.spatial_encoder(x))
        
        # Swin 特征维度翻转安全修复
        if 'swin' in self.encoder_name:
            if s_feats[0].shape[-1] == self.spatial_encoder.feature_info.channels()[0]:
                s_feats = [f.permute(0, 3, 1, 2).contiguous() for f in s_feats]
        
        edge_logits = None
        if self.use_dual_stream:
            f_curr = self.freq_stem(x)
            f_feats = [f_curr]
            for layer in self.freq_layers:
                f_next, f_inter = layer(f_curr)
                f_feats.append(f_inter)
                f_curr = f_next
            
            s_fused = []
            for i in range(4):
                out, _ = self.bi_fgf_modules[i](s_feats[i], f_feats[i])
                s_fused.append(out)
            s_feats = s_fused
            
            if self.training:
                edge_logits = F.interpolate(self.edge_head(f_feats[0]), size=x.shape[2:], mode='bilinear', align_corners=True)

        # Decoding
        d1 = self.up1(s_feats[3], s_feats[2])
        d2 = self.up2(d1, s_feats[1])
        d3 = self.up3(d2, s_feats[0])
        logits = self.outc(self.final_up(d3))
        
        if self.training:
            outputs = [logits]
            if self.use_deep_supervision:
                outputs.extend([self.head_up2(d2), self.head_up3(d3)])
            if self.use_dual_stream and edge_logits is not None:
                outputs.append(edge_logits)
            return outputs
            
        return logits

def UNet(n_channels=3, n_classes=1, **kwargs):
    """ 兼容接口 """
    return UniversalUNet(n_classes=n_classes, **kwargs)