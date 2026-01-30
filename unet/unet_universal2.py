"""
unet/unet_universal1.py
[Universal Model] 全能型 UNet (Final Version)
架构特点:
  1. Spatial Encoder: ConvNeXt V2 (语义提取)
  2. Frequency Encoder: SFDA Block (频谱-频率解耦注意力, Hi-Lo Attention)
     - 包含 FP32 精度保护 (防 NaN)
     - 包含 LayerNorm + Residual (防梯度消失)
  3. Interaction: Bi-FGF (双向门控融合)
  4. Decoder: ProtoFormer (原型交互解码器, FP32 保护)
  5. Deep Supervision: 支持多尺度辅助监督
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# ================================================================
# 0. 基础工具类 (小波变换 & Attention组件)
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
        # Padding 防止奇数尺寸报错
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

# --- 辅助模块: 全局注意力 (FP32 Safe + Residual + Norm) ---
class GlobalAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
        # 🔥 [关键优化] LayerNorm，保证深层训练稳定
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x input: [B, C, H, W] or [B, N, C]
        if x.dim() == 4:
            B, C, H, W = x.shape
            x_in = x.flatten(2).transpose(1, 2) # [B, N, C]
            is_spatial = True
        else:
            B, N, C = x.shape
            x_in = x
            is_spatial = False

        # Pre-Norm
        x_norm = self.norm(x_in)

        # 🔥🔥🔥 [FP32 安全区] 防止 Attention 溢出导致 NaN 🔥🔥🔥
        with torch.cuda.amp.autocast(enabled=False):
            x_32 = x_norm.float()
            qkv = self.qkv(x_32).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            x_out = (attn @ v)
        
        x_out = x_out.to(x.dtype) # 转回 FP16/FP32

         # 🔥🔥🔥 [核心修复] 合并多头维度 🔥🔥🔥
        # [B, heads, N, dim_head] -> [B, N, heads, dim_head] -> [B, N, C]
        x_out = x_out.transpose(1, 2).reshape(B, -1, C)
        x_out = self.proj(x_out)
        
        # 🔥 [关键优化] 加上残差连接 Input + Output
        x_out = x_in + x_out

        if is_spatial:
            x_out = x_out.transpose(1, 2).reshape(B, C, H, W)
            
        return x_out

# --- 辅助模块: 窗口局部注意力 (FP32 Safe + Residual) ---
class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=7):
        super().__init__()
        self.window_size = window_size
        # 复用 GlobalAttention (内部已有 Norm 和 Residual)
        self.attn = GlobalAttention(dim, num_heads) 

    def forward(self, x):
        B, C, H, W = x.shape
        # Pad 如果尺寸不能被 window_size 整除
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        x_padded = F.pad(x, (0, pad_w, 0, pad_h))
        
        _, _, Hp, Wp = x_padded.shape
        
        # Window Partition
        # [B, C, Hp, Wp] -> [B*NumWin, C, WinSize, WinSize]
        x_windows = F.unfold(x_padded, kernel_size=self.window_size, stride=self.window_size)
        x_windows = x_windows.transpose(1, 2).contiguous().view(B, -1, C, self.window_size, self.window_size)
        x_windows = x_windows.permute(0, 1, 3, 4, 2).contiguous().view(-1, C, self.window_size, self.window_size)
        
        # Attention (内部有 Residual)
        # 这里的 Residual 是针对 window 内部特征的
        attn_windows = self.attn(x_windows)
        
        # Window Reverse
        attn_windows = attn_windows.view(B, -1, C, self.window_size, self.window_size).permute(0, 2, 3, 4, 1)
        attn_windows = attn_windows.contiguous().view(B, C * self.window_size * self.window_size, -1)
        x_out = F.fold(attn_windows, output_size=(Hp, Wp), kernel_size=self.window_size, stride=self.window_size)
        
        # Crop Padding
        return x_out[:, :, :H, :W]
# ================================================================
# 新增模块: ASPP (Atrous Spatial Pyramid Pooling)
# 作用: 增加感受野，显著增加有效参数量 (针对策略2)
# ================================================================
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        modules = []
        
        # 1. 分支1: 1x1 卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        # 2. 分支2-4: 不同扩张率的 3x3 空洞卷积
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)))

        self.aspp_blocks = nn.ModuleList(modules)
        
        # 3. 分支5: 全局平均池化 (Image Pooling)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

        # 4. 融合投影层
        # 输入通道数 = 5 个分支 * out_channels
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(atrous_rates) + 2), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1) # 防止过拟合
        )

    def forward(self, x):
        res = []
        # 计算卷积分支
        for block in self.aspp_blocks:
            res.append(block(x))
        
        # 计算池化分支并上采样
        res.append(F.interpolate(self.global_avg_pool(x), size=x.shape[2:], mode='bilinear', align_corners=True))
        
        # 拼接
        res = torch.cat(res, dim=1)
        
        # 融合输出
        return self.project(res)
# ================================================================
# 1. 核心模块: SFDA Block (替代 WaveletMambaBlock)
# ================================================================

class SFDABlock(nn.Module):
    """
    [New Core] Spectral-Frequency Decoupled Attention Block
    频率流核心：低频全局 + 高频局部 + 门控融合 + 残差修正
    """
    def __init__(self, in_channels, out_channels, num_heads=4):
        super().__init__()
        self.dwt = HaarWaveletTransform()
        
        # 1. 低频路径 (Lo-Path): 处理 LL
        self.lo_proj = nn.Conv2d(in_channels, out_channels, 1)
        self.lo_process = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2), # 下采样
            GlobalAttention(out_channels, num_heads=num_heads), # 内部有Res+Norm
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        # 2. 高频路径 (Hi-Path): 处理 LH, HL, HH
        self.hi_proj = nn.Conv2d(in_channels * 3, out_channels, 1)
        self.hi_process = WindowAttention(out_channels, num_heads=num_heads, window_size=7)
        
        # 3. 优化后的门控融合
        self.gate = nn.Sequential(
            nn.Conv2d(out_channels * 2, 1, 1),
            nn.Sigmoid()
        )
        
        # 4. 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 5. 🔥 [核心修复 1] 残差路径必须下采样！
        # 因为 DWT 会让主路尺寸减半，所以残差路也要减半才能相加
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2), # 空间下采样
            nn.Conv2d(in_channels, out_channels, 1), # 通道对齐
            nn.BatchNorm2d(out_channels)
        )

        # 6. 🔥 [核心修复 2] 移除 self.downsample
        # SFDA Block 本身通过 DWT 已经完成了下采样 (Stride 2)，
        # 不需要再在末尾加 downsample，否则一个 Block 降采样 4 倍会导致和 ConvNeXt 对不上。

    def forward(self, x):
        # 0. 准备残差 (现在 residual 也是 H/2, W/2 了)
        residual = self.shortcut(x)

        # 1. DWT 分解 (H/2, W/2)
        ll, lh, hl, hh = self.dwt.dwt(x)
        
        # 2. Lo-Path
        x_lo = self.lo_proj(ll)
        out_lo = self.lo_process(x_lo)
        
        # 3. Hi-Path
        x_hi = torch.cat([lh, hl, hh], dim=1)
        x_hi = self.hi_proj(x_hi)
        out_hi = self.hi_process(x_hi)
        
        # 4. Gated Fusion
        gate_input = torch.cat([out_lo, out_hi], dim=1)
        gate_map = self.gate(gate_input)
        
        out_fused = self.fusion(torch.cat([out_lo, out_hi * gate_map], dim=1))
        
        # 5. 残差相加 (现在尺寸匹配了！)
        out_fused = out_fused + residual
        
        # 🔥 [核心修复 3] 直接返回 out_fused
        # out_fused 已经是下一层需要的尺寸 (Stride 2)
        # next_layer_input = out_fused
        # interaction_feat = out_fused
        return out_fused, out_fused

# ================================================================
# 2. 交互模块 (Bi-FGF)
# ================================================================
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

# ================================================================
# 3. 解码器组件: ProtoFormer & Standard
# ================================================================
class StandardDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class PrototypeInteractionBlock(nn.Module):
    """
    [ProtoFormer Core] 原型交互单元 (FP32 Safe)
    """
    def __init__(self, channels, num_prototypes=16):
        super().__init__()
        self.channels = channels
        self.prototypes = nn.Parameter(torch.randn(1, num_prototypes, channels))
        self.pos_embed = nn.Parameter(torch.randn(1, channels, 64, 64) * 0.02)
        
        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(8, channels)
        self.local_conv = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False), nn.BatchNorm2d(channels), nn.GELU())

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        pos = F.interpolate(self.pos_embed, size=(H, W), mode='bilinear', align_corners=False)
        x = x + pos 
        q = self.q_proj(x).flatten(2).transpose(1, 2)
        protos = self.prototypes.repeat(B, 1, 1)
        k = self.k_proj(protos)
        v = self.v_proj(protos)
        
        # 🔥🔥🔥 [FP32 安全区] 防止 Decoder NaN 🔥🔥🔥
        with torch.cuda.amp.autocast(enabled=False):
            q_32, k_32, v_32 = q.float(), k.float(), v.float()
            scale = C ** -0.5
            attn = (q_32 @ k_32.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            out = attn @ v_32
            
        out = out.to(x.dtype)
        out = out.transpose(1, 2).view(B, C, H, W)
        out = self.out_proj(out)
        out = out + self.local_conv(out)
        return self.norm(out + residual)

class PHD_DecoderBlock_Pro(nn.Module):
    def __init__(self, in_channels, out_channels): 
        super().__init__()
        self.align = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        # 每个解码层独立的原型
        self.proto_block = PrototypeInteractionBlock(out_channels, num_prototypes=16)
        
    def forward(self, x):
        return self.proto_block(self.align(x))

class Up_Universal(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, decoder_type='phd'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        conv_in = in_channels + skip_channels
        if decoder_type == 'phd':
            self.conv = PHD_DecoderBlock_Pro(conv_in, out_channels)
        else:
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
# 4. 主模型: UniversalUNet (最终组装)
# ================================================================
class UniversalUNet(nn.Module):
    def __init__(self, 
                 n_classes=1, 
                 cnext_type='convnextv2_tiny', 
                 pretrained=True,
                 decoder_type='phd',       
                 use_dual_stream=True,     
                 use_deep_supervision=False,
                 **kwargs):
        super().__init__()
        self.n_classes = n_classes
        self.use_dual_stream = use_dual_stream
        self.decoder_type = decoder_type
        self.use_deep_supervision = use_deep_supervision
        
        print(f"🤖 [Universal Model] Initialized with:")
        print(f"   - Encoder: {cnext_type} (Pretrained={pretrained})")
        print(f"   - Dual Stream (SFDA + HiLo): {'✅ ON' if use_dual_stream else '❌ OFF'}")
        print(f"   - Decoder: {decoder_type}")
        print(f"   - Deep Supervision: {'✅ ON' if use_deep_supervision else '❌ OFF'}")

        # 1. Spatial Encoder
        self.spatial_encoder = timm.create_model(cnext_type, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3), drop_path_rate=0.0)
        s_dims = self.spatial_encoder.feature_info.channels() # [96, 192, 384, 768] for tiny
        
        # 2. Frequency Encoder (SFDA Stream)
        if self.use_dual_stream:
            f_dims = [c // 4 for c in s_dims]
            self.freq_stem = nn.Sequential(nn.Conv2d(3, f_dims[0], 4, stride=4, padding=0), nn.BatchNorm2d(f_dims[0]), nn.ReLU(True))
            
            # 🔥 使用修复后的 SFDABlock (带 Shortcut 和 Gate优化)
            self.freq_layers = nn.ModuleList([
                SFDABlock(in_channels=f_dims[i], out_channels=f_dims[i+1]) 
                for i in range(3)
            ])
            
            self.bi_fgf_modules = nn.ModuleList([Cross_GL_FGF(s_dims[i], f_dims[i]) for i in range(4)])
            self.edge_head = nn.Sequential(nn.Conv2d(f_dims[0], 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.Conv2d(64, 1, 1))
# 🔥🔥🔥 [新增修改] 策略2: 定义语义桥梁 ASPP 🔥🔥🔥
        # 放在 Encoder 最深层 (s_dims[3]=768) 之后
        # 输入输出保持一致，只为了提取特征和增加参数
        self.bridge = ASPP(in_channels=s_dims[3], out_channels=s_dims[3])
        # 3. Decoder
        self.up1 = Up_Universal(s_dims[3], s_dims[2], skip_channels=s_dims[2], decoder_type=decoder_type)
        self.up2 = Up_Universal(s_dims[2], s_dims[1], skip_channels=s_dims[1], decoder_type=decoder_type)
        self.up3 = Up_Universal(s_dims[1], s_dims[0], skip_channels=s_dims[0], decoder_type=decoder_type)
        
        self.final_up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.outc = nn.Conv2d(s_dims[0], n_classes, kernel_size=1)
        
        # 4. Deep Supervision Heads
        if self.use_deep_supervision:
            # Scale 1/8
            self.head_up2 = nn.Sequential(
                nn.Conv2d(s_dims[1], 32, 3, padding=1), 
                nn.BatchNorm2d(32), nn.ReLU(), 
                nn.Conv2d(32, n_classes, 1)
            )
            # Scale 1/4
            self.head_up3 = nn.Sequential(
                nn.Conv2d(s_dims[0], 32, 3, padding=1), 
                nn.BatchNorm2d(32), nn.ReLU(), 
                nn.Conv2d(32, n_classes, 1)
            )

    def forward(self, x):
        # 1. Encoder Pass
        s_feats = list(self.spatial_encoder(x))
        
        # 2. Dual Stream Pass (SFDA)
        edge_logits = None
        if self.use_dual_stream:
            f_curr = self.freq_stem(x)
            f_feats = [f_curr]
            for layer in self.freq_layers:
                f_next, f_inter = layer(f_curr) # next是下一层输入，inter是当前层用于交互的特征
                f_feats.append(f_inter)
                f_curr = f_next
            
            # Interaction
            s_fused_list = []
            f_enhanced_list = []
            for i in range(4):
                s_out, f_out = self.bi_fgf_modules[i](s_feats[i], f_feats[i])
                s_fused_list.append(s_out)
                f_enhanced_list.append(f_out)
            s_feats = s_fused_list
            
            if self.training:
                edge_small = self.edge_head(f_enhanced_list[0])
                edge_logits = F.interpolate(edge_small, size=x.shape[2:], mode='bilinear', align_corners=True)
# 🔥🔥🔥 [新增修改] 策略2: 在进入解码器之前，先过桥 🔥🔥🔥
        # s_feats[3] 是最深层语义特征 (H/32)
        # 通过 ASPP 增强其全局感受野
        s_feats[3] = self.bridge(s_feats[3])
        # 3. Decoder Pass
        s1, s2, s3, s4 = s_feats
        
        d1 = self.up1(s4, s3)
        d2 = self.up2(d1, s2)
        d3 = self.up3(d2, s1)
        
        logits = self.outc(self.final_up(d3))
        
        # 4. Deep Supervision Return Logic
        if self.training and self.use_deep_supervision:
            aux2 = self.head_up2(d2)
            aux3 = self.head_up3(d3)
            outputs = [logits, aux2, aux3]
            if self.use_dual_stream and edge_logits is not None:
                outputs.append(edge_logits)
            return outputs
        
        # Legacy Return Logic
        if self.training and self.use_dual_stream and edge_logits is not None:
            return logits, edge_logits
            
        return logits