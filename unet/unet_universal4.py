"""
unet/unet_universal3.py
[Universal Model] 全能型 UNet (Heavy Decoder Version + Body-Edge Decoupling)
架构特点:
  1. Spatial Encoder: ConvNeXt V2 (语义提取)
  2. Frequency Encoder: SFDA Block (Hi-Lo Attention, FP32 Protected)
  3. Interaction: Bi-FGF (双向门控融合)
  4. Decoder: Heavy ProtoFormer (3级级联交互，参数量增强)
  5. Deep Supervision: 支持多尺度辅助监督
  6. Decoupling: 体-缘解耦双解码器 (Body Stream + Edge Stream -> Fusion)
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

# --- 辅助模块: 全局注意力 (修复维度 Bug + FP32 Safe) ---
class GlobalAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
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

        # 🔥🔥🔥 [FP32 安全区] 🔥🔥🔥
        with torch.cuda.amp.autocast(enabled=False):
            x_32 = x_norm.float()
            # qkv: [B, N, 3*C] -> [B, N, 3, heads, dim_head] -> [3, B, heads, N, dim_head]
            qkv = self.qkv(x_32).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            # 🔥 [修改后] 限制 Logits 的最大值，防止 Softmax 过于极化
            attn_logits = (q @ k.transpose(-2, -1)) * self.scale
            
            # 这里的 30 是经验值，e^30 约为 1e13，足够大但不会溢出 FP32
            # 这一步能极大缓解梯度爆炸
            attn_logits = torch.clamp(attn_logits, min=-30, max=30) 
            
            attn = attn_logits.softmax(dim=-1)
            x_out = (attn @ v) # [B, heads, N, dim_head]
        
        x_out = x_out.to(x.dtype) # 转回 FP16/FP32
        
        # 🔥🔥🔥 [修复] 合并多头维度 🔥🔥🔥
        # [B, heads, N, dim_head] -> [B, N, heads, dim_head] -> [B, N, C]
        x_out = x_out.transpose(1, 2).reshape(B, -1, C)
        
        x_out = self.proj(x_out)
        
        # 残差连接
        x_out = x_in + x_out

        if is_spatial:
            x_out = x_out.transpose(1, 2).reshape(B, C, H, W)
            
        return x_out

# --- 辅助模块: 窗口局部注意力 ---
class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=7):
        super().__init__()
        self.window_size = window_size
        self.attn = GlobalAttention(dim, num_heads) 

    def forward(self, x):
        B, C, H, W = x.shape
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        x_padded = F.pad(x, (0, pad_w, 0, pad_h))
        
        _, _, Hp, Wp = x_padded.shape
        
        # Window Partition
        x_windows = F.unfold(x_padded, kernel_size=self.window_size, stride=self.window_size)
        x_windows = x_windows.transpose(1, 2).contiguous().view(B, -1, C, self.window_size, self.window_size)
        x_windows = x_windows.permute(0, 1, 3, 4, 2).contiguous().view(-1, C, self.window_size, self.window_size)
        
        # Attention
        attn_windows = self.attn(x_windows)
        
        # Window Reverse
        attn_windows = attn_windows.view(B, -1, C, self.window_size, self.window_size).permute(0, 2, 3, 4, 1)
        attn_windows = attn_windows.contiguous().view(B, C * self.window_size * self.window_size, -1)
        x_out = F.fold(attn_windows, output_size=(Hp, Wp), kernel_size=self.window_size, stride=self.window_size)
        
        return x_out[:, :, :H, :W]

# ================================================================
# 1. 核心模块: SFDA Block (频率流)
# ================================================================

class SFDABlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4):
        super().__init__()
        self.dwt = HaarWaveletTransform()
        
        # 1. 低频路径
        self.lo_proj = nn.Conv2d(in_channels, out_channels, 1)
        self.lo_process = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2), 
            GlobalAttention(out_channels, num_heads=num_heads), 
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        # 2. 高频路径
        self.hi_proj = nn.Conv2d(in_channels * 3, out_channels, 1)
        self.hi_process = WindowAttention(out_channels, num_heads=num_heads, window_size=7)
        
        # 3. 门控融合
        self.gate = nn.Sequential(nn.Conv2d(out_channels * 2, 1, 1), nn.Sigmoid())
        
        # 4. 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        
        # 5. Block 级残差 Shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2), 
                nn.Conv2d(in_channels, out_channels, 1), 
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        ll, lh, hl, hh = self.dwt.dwt(x)
        
        x_lo = self.lo_proj(ll)
        out_lo = self.lo_process(x_lo)
        
        x_hi = torch.cat([lh, hl, hh], dim=1)
        x_hi = self.hi_proj(x_hi)
        out_hi = self.hi_process(x_hi)
        
        gate_input = torch.cat([out_lo, out_hi], dim=1)
        gate_map = self.gate(gate_input)
        
        out_fused = self.fusion(torch.cat([out_lo, out_hi * gate_map], dim=1))
        out_fused = out_fused + residual
        
        return out_fused, out_fused # 这里的 out_fused 已经是下采样后的尺寸

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
# 3. 解码器组件: Heavy ProtoFormer (级联版)
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
    """[ProtoFormer Core] 原型交互单元 (FP32 Safe)"""
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
        # 🔥 [新增] LayerScale 参数 (初始为 1e-5，非常小，保证稳定启动)
        self.gamma = nn.Parameter(torch.ones(channels) * 1e-5)
    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        pos = F.interpolate(self.pos_embed, size=(H, W), mode='bilinear', align_corners=False)
        x = x + pos 
        q = self.q_proj(x).flatten(2).transpose(1, 2)
        protos = self.prototypes.repeat(B, 1, 1)
        k = self.k_proj(protos)
        v = self.v_proj(protos)
        
        # 🔥 FP32 Safe Attention
        with torch.cuda.amp.autocast(enabled=False):
            q_32, k_32, v_32 = q.float(), k.float(), v.float()
            scale = C ** -0.5
            # 🔥 新增: 限制数值范围
            attn_logits = (q_32 @ k_32.transpose(-2, -1)) * scale
            attn_logits = torch.clamp(attn_logits, min=-30, max=30)
            
            attn = attn_logits.softmax(dim=-1)
            # --- 👆 修改结束 👆 ---
            out = attn @ v_32
            
        out = out.to(x.dtype)
        out = out.transpose(1, 2).view(B, C, H, W)
        out = self.out_proj(out)
        out = out + self.local_conv(out)
        # 🔥 [修正] 正确的 LayerScale 位置：只缩放增量部分
        # Output = Norm(Input + gamma * Delta)
        return self.norm(residual + out * self.gamma.view(1, -1, 1, 1))

class FeedForward(nn.Module):
    """简单 FFN"""
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)

class PHD_DecoderBlock_Pro(nn.Module):
    """
    [Heavy Version] 重型化解码块
    depth=3: 串联 3 个 ProtoBlock，参数量增加约 5-10M
    """
    def __init__(self, in_channels, out_channels, depth=2): 
        super().__init__()
        self.align = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        
        # 🔥 级联堆叠
        self.layers = nn.ModuleList([])
        # 🔥 FFN 的 Gamma 还是要留着，保平安
        self.gamma_ffn = nn.Parameter(torch.ones(depth, out_channels) * 1e-5)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PrototypeInteractionBlock(out_channels, num_prototypes=16),
                FeedForward(out_channels, out_channels * 4)
            ]))

    def forward(self, x):
        x = self.align(x)
        # 迭代精修
        for i, (proto_block, ffn) in enumerate(self.layers):
            x = proto_block(x)
            # FFN 加上 gamma 保护
            gamma = self.gamma_ffn[i].view(1, -1, 1, 1)
            x = x + gamma * ffn(x)# FFN 残差
        return x

class Up_Universal(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, decoder_type='phd'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        conv_in = in_channels + skip_channels
        if decoder_type == 'phd':
            # 🔥 开启重型模式: depth=3
            self.conv = PHD_DecoderBlock_Pro(conv_in, out_channels, depth=2)
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
# 4. 主模型: UniversalUNet (最终组装 + Decoupling)
# ================================================================
class UniversalUNet(nn.Module):
    def __init__(self, 
                 n_classes=1, 
                 cnext_type='convnextv2_tiny', 
                 pretrained=True,
                 decoder_type='phd',       
                 use_dual_stream=True,     
                 use_deep_supervision=False,
                 use_decouple=False, # 🔥 [新参数] 默认关闭
                 **kwargs):
        super().__init__()
        self.n_classes = n_classes
        self.use_dual_stream = use_dual_stream
        self.decoder_type = decoder_type
        self.use_deep_supervision = use_deep_supervision
        self.use_decouple = use_decouple
        
        print(f"🤖 [Universal Model] Initialized with:")
        print(f"   - Encoder: {cnext_type}")
        print(f"   - Dual Stream (SFDA): {'✅ ON' if use_dual_stream else '❌ OFF'}")
        print(f"   - Decoder: Heavy ProtoFormer (Depth=3)")
        print(f"   - Body-Edge Decoupling: {'✅ ON' if use_decouple else '❌ OFF'}")
        
        # 1. Spatial Encoder
        self.spatial_encoder = timm.create_model(cnext_type, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3), drop_path_rate=0.0)
        s_dims = self.spatial_encoder.feature_info.channels() # [96, 192, 384, 768]
        
        # 2. Frequency Encoder
        if self.use_dual_stream:
            # 🔥 加宽频率流: c // 2
            f_dims = [c // 2 for c in s_dims]
            self.freq_stem = nn.Sequential(nn.Conv2d(3, f_dims[0], 4, stride=4, padding=0), nn.BatchNorm2d(f_dims[0]), nn.ReLU(True))
            
            self.freq_layers = nn.ModuleList([
                SFDABlock(in_channels=f_dims[i], out_channels=f_dims[i+1]) 
                for i in range(3)
            ])
            
            self.bi_fgf_modules = nn.ModuleList([Cross_GL_FGF(s_dims[i], f_dims[i]) for i in range(4)])
            self.edge_head = nn.Sequential(nn.Conv2d(f_dims[0], 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.Conv2d(64, 1, 1))

        # 3. Decoder
        self.up1 = Up_Universal(s_dims[3], s_dims[2], skip_channels=s_dims[2], decoder_type=decoder_type)
        self.up2 = Up_Universal(s_dims[2], s_dims[1], skip_channels=s_dims[1], decoder_type=decoder_type)
        self.up3 = Up_Universal(s_dims[1], s_dims[0], skip_channels=s_dims[0], decoder_type=decoder_type)
        
        self.final_up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        
        # 🔥🔥🔥 [修改区域: 解耦头] 🔥🔥🔥
        final_dim = s_dims[0]
        if self.use_decouple:
            # A. Body Head (预测实体)
            self.body_head = nn.Sequential(
                nn.Conv2d(final_dim, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, n_classes, 1)
            )
            # B. Edge Head (预测边缘)
            self.edge_head_decouple = nn.Sequential(
                nn.Conv2d(final_dim, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, n_classes, 1)
            )
            # C. Fusion Head (融合)
            self.fusion_head = nn.Sequential(
                nn.Conv2d(n_classes * 2, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, n_classes, 1)
            )
        else:
            # 传统单输出
            self.outc = nn.Conv2d(s_dims[0], n_classes, kernel_size=1)
        
        # 4. Deep Supervision
        if self.use_deep_supervision:
            self.head_up2 = nn.Sequential(nn.Conv2d(s_dims[1], 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, n_classes, 1))
            self.head_up3 = nn.Sequential(nn.Conv2d(s_dims[0], 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, n_classes, 1))

    def forward(self, x):
        s_feats = list(self.spatial_encoder(x))
        
        # Dual Stream
        edge_logits = None
        if self.use_dual_stream:
            f_curr = self.freq_stem(x)
            f_feats = [f_curr]
            for layer in self.freq_layers:
                f_next, f_inter = layer(f_curr)
                f_feats.append(f_inter)
                f_curr = f_next
            
            s_fused_list = []
            for i in range(4):
                s_out, _ = self.bi_fgf_modules[i](s_feats[i], f_feats[i])
                s_fused_list.append(s_out)
            s_feats = s_fused_list
            
            if self.training:
                edge_small = self.edge_head(f_feats[0])
                edge_logits = F.interpolate(edge_small, size=x.shape[2:], mode='bilinear', align_corners=True)

        # Decoder
        d1 = self.up1(s_feats[3], s_feats[2])
        d2 = self.up2(d1, s_feats[1])
        d3 = self.up3(d2, s_feats[0])
        
        # 解码器最终特征
        dec_feat = self.final_up(d3)
        
        # 🔥🔥🔥 [修改区域: 解耦前向传播] 🔥🔥🔥
        if self.use_decouple:
            # 1. 独立预测 Body 和 Edge
            body_out = self.body_head(dec_feat)
            edge_out = self.edge_head_decouple(dec_feat)
            
            # 2. 拼接并融合得到最终结果
            cat_feat = torch.cat([body_out, edge_out], dim=1)
            final_out = self.fusion_head(cat_feat)
        else:
            final_out = self.outc(dec_feat)
            
        # 返回逻辑
        if self.training:
            outputs = [final_out]
            
            # Deep Supervision
            if self.use_deep_supervision:
                aux2 = self.head_up2(d2)
                aux3 = self.head_up3(d3)
                outputs.extend([aux2, aux3])
            
            # Decouple Outputs (为了计算 Loss)
            if self.use_decouple:
                outputs.extend([body_out, edge_out])

            # Dual Stream Legacy Edge
            if self.use_dual_stream and edge_logits is not None:
                outputs.append(edge_logits)
            
            # 如果是单输出模式且无其他 head，直接返回 tensor
            return outputs if len(outputs) > 1 else final_out
            
        return final_out