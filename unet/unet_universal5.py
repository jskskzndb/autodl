"""
unet/unet_universal1.py
[Universal Model] 全能型 UNet (Final Version)
架构特点:
  1. Spatial Encoder: ConvNeXt V2 (语义提取)
  2. Frequency Encoder: Omni-SFDA Block (集成了方向感知、循环稀疏编码与软阈值去噪，修复了Shortcut维度Bug)
  3. Interaction: SK-Fusion (基于SK-Net思想的动态选择性融合，自动抗噪)
  4. Decoder: Heavy ProtoFormer (3级级联交互，参数量增强) 或 Standard (稳定型 GroupNorm)
  5. Return Logic: 统一返回 List 接口，防止训练循环解包错误
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


# ================================================================
# [新增] SCSE 模块 (Spatial & Channel Squeeze Excitation)
# 作用: 稳健的跳跃连接增强，带残差结构，防止掉点
# ================================================================
class SCSEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # 1. 空间分支 (Spatial cSE) - 关注"在哪里" (边缘)
        self.spatial_se = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        
        # 2. 通道分支 (Channel sSE) - 关注"是什么" (语义)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_se = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, H, W]
        
        # --- 分支 1: 空间注意力 ---
        s_attn = self.spatial_se(x)
        x_spatial = x * s_attn
        
        # --- 分支 2: 通道注意力 ---
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        c_attn = self.channel_se(y).view(b, c, 1, 1)
        x_channel = x * c_attn
        
        # --- 融合 ---
        attention_out = x_spatial + x_channel
        
        # 🔥🔥🔥 关键：残差连接 (Identity Mapping) 🔥🔥🔥
        # 保证最差情况也是原始特征，绝对稳健
        return x + attention_out
# --- 辅助模块: 全局注意力 (FP32 Safe) ---
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
            qkv = self.qkv(x_32).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            # 限制 Logits 最大值，防止 Softmax 爆炸
            attn_logits = (q @ k.transpose(-2, -1)) * self.scale
            attn_logits = torch.clamp(attn_logits, min=-30, max=30) 
            
            attn = attn_logits.softmax(dim=-1)
            x_out = (attn @ v) 
        
        x_out = x_out.to(x.dtype) 
        x_out = x_out.transpose(1, 2).reshape(B, -1, C)
        x_out = self.proj(x_out)
        
        # 残差连接
        x_out = x_in + x_out

        if is_spatial:
            x_out = x_out.transpose(1, 2).reshape(B, C, H, W)
            
        return x_out

# ================================================================
# 1. 核心模块: SFDA Block (频率流 - 全能型)
# ================================================================

# [组件] 频率通道注意力
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
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# [组件] 可学习软阈值
class LearnableSoftThresholding(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor([0.02] * channels).view(1, channels, 1, 1))

    def forward(self, x):
        thresh = torch.abs(self.threshold)
        return torch.sign(x) * F.relu(torch.abs(x) - thresh)

# [组件] 方向感知编码器
class DirectionalEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # LH (水平低/垂直高) -> 垂直纹理强 -> 用 3x1 卷积提取
        self.conv_lh = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # HL (水平高/垂直低) -> 水平纹理强 -> 用 1x3 卷积提取
        self.conv_hl = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # HH (对角线) -> 无特定方向 -> 用普通 3x3
        self.conv_hh = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.Conv2d(out_channels * 3, out_channels, 1, bias=False)

    def forward(self, lh, hl, hh):
        f_lh = self.conv_lh(lh)
        f_hl = self.conv_hl(hl)
        f_hh = self.conv_hh(hh)
        return self.fusion(torch.cat([f_lh, f_hl, f_hh], dim=1))

# [组件] 循环稀疏编码块
# [修改后的 RecurrentSparseBlock] 增加 GroupNorm 和 Clamp 以防止 NaN
class RecurrentSparseBlock(nn.Module):
    def __init__(self, channels, iterations=2):
        super().__init__()
        self.iterations = iterations
        
        # 使用 Kaiming 初始化防止初始值过大
        self.encoder = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.decoder = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        
        # 🔥 修改点 1: 将 BatchNorm 改为 GroupNorm (对小 BatchSize 和 频率特征 更稳)
        self.bn = nn.GroupNorm(8, channels) 
        
        # 🔥 修改点 2: 循环内部的归一化，防止迭代发散
        self.loop_norm = nn.GroupNorm(8, channels)
        
        self.threshold = LearnableSoftThresholding(channels)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 初始编码
        z = self.threshold(self.bn(self.encoder(x)))
        
        for _ in range(self.iterations):
            x_hat = self.decoder(z)
            error = x - x_hat
            delta_z = self.encoder(error)
            
            # 🔥 修改点 3: 限制更新步长，防止一步跨太大
            delta_z = 0.1 * delta_z
            
            # 🔥 修改点 4: 残差连接后立即做归一化
            z = z + delta_z
            z = self.loop_norm(z) 
            
            # 🔥 修改点 5: 物理截断 (Hard Clamp)，防止数值溢出 FP16 范围
            # 20.0 对于特征图来说已经非常大了，足够保留信息但不会溢出
            z = torch.clamp(z, min=-20.0, max=20.0)
            
            z = self.threshold(z)
            
        return z

# [修改后的 SFDABlock]
class SFDABlock(nn.Module):
    """
    🔥 [Omni-SFDA 稳定版]
    集成了：FCA, Directional Conv, Recurrent Sparse Coding
    修复了 Shortcut 维度 Bug，并增强了数值稳定性。
    """
    def __init__(self, in_channels, out_channels, num_heads=4):
        super().__init__()
        self.dwt = HaarWaveletTransform()
        
        # 1. 频率选择
        self.freq_select = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, 1),
            nn.GroupNorm(8, out_channels), # 改为 GroupNorm
            nn.ReLU(inplace=True),
            FrequencyChannelAttention(out_channels) 
        )

        # 2. 低频路径
        self.lo_proj = nn.Conv2d(in_channels, out_channels, 1)
        self.lo_process = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2), 
            GlobalAttention(out_channels, num_heads=num_heads), 
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        # 3. 高频路径
        self.hi_proj_layer = nn.Conv2d(in_channels, out_channels, 1) 
        self.directional_encoder = DirectionalEncoder(out_channels, out_channels)
        # 使用上面定义的新版 RecurrentSparseBlock
        self.recurrent_denoiser = RecurrentSparseBlock(out_channels, iterations=2)
        
        # 4. 融合
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 1),
            nn.GroupNorm(8, out_channels), # 🔥 改为 GroupNorm，防止融合时均值漂移
            nn.ReLU(inplace=True)
        )
        
        # 5. Shortcut (保持之前的修复逻辑)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2), 
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels) # Shortcut 可以保留 BN
            )
        else:
            self.shortcut = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        residual = self.shortcut(x)
        
        # 为了稳定性，可以加上 eps 防止除零 (虽然 DWT 一般没事)
        x = torch.clamp(x, min=-50, max=50) 
        
        ll, lh, hl, hh = self.dwt.dwt(x)
        
        # Selected View
        all_freq = torch.cat([ll, lh, hl, hh], dim=1)
        feat_selected = self.freq_select(all_freq)
        
        # Low View
        x_lo = self.lo_proj(ll)
        feat_lo = self.lo_process(x_lo)
        
        # High View
        x_lh = self.hi_proj_layer(lh)
        x_hl = self.hi_proj_layer(hl)
        x_hh = self.hi_proj_layer(hh)
        
        feat_hi_dir = self.directional_encoder(x_lh, x_hl, x_hh)
        feat_hi_final = self.recurrent_denoiser(feat_hi_dir)
        
        # Fusion
        # 🔥 安全检查：如果有分支是 NaN，替换为 0 (极端保命措施，可选)
        if torch.isnan(feat_hi_final).any():
             feat_hi_final = torch.zeros_like(feat_hi_final)

        out_fused = self.fusion(torch.cat([feat_selected, feat_lo, feat_hi_final], dim=1))
        out_fused = out_fused + residual
        
        return out_fused, out_fused

# ================================================================
# 2. 交互模块 (SK-Fusion)
# ================================================================
class SK_Fusion_Block(nn.Module):
    """
    🔥 [SK-Fusion] 动态选择融合模块
    """
    def __init__(self, s_channels, f_channels, reduction=16):
        super().__init__()
        # 1. 频率流对齐
        self.f_align = nn.Sequential(
            nn.Conv2d(f_channels, s_channels, 1, bias=False),
            nn.BatchNorm2d(s_channels),
            nn.ReLU(inplace=True)
        )
        
        dim = s_channels
        mid_dim = max(dim // reduction, 32)
        
        # 2. 全局信息描述符
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 3. 权重生成器
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mid_dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, dim * 2, 1, bias=False)
        )
        
        self.softmax = nn.Softmax(dim=1)
        
        # 4. 最终整合
        self.out_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_s, x_f):
        if x_f.shape[2:] != x_s.shape[2:]:
            x_f = F.interpolate(x_f, size=x_s.shape[2:], mode='bilinear', align_corners=True)
        
        x_f_aligned = self.f_align(x_f)
        u = x_s + x_f_aligned
        s = self.avg_pool(u)
        z = self.mlp(s)
        
        b, c, _, _ = x_s.size()
        z = z.view(b, 2, c, 1, 1)
        weights = self.softmax(z)
        
        out = weights[:, 0] * x_s + weights[:, 1] * x_f_aligned
        out = self.out_conv(out)
        
        return out, x_f_aligned

# ================================================================
# 3. 解码器组件: Heavy ProtoFormer
# ================================================================

# 🔥 [关键修改] 标准解码器双卷积：将 BatchNorm 改为 GroupNorm (稳定版)
class StandardDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),  # ✅ BN -> GN
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),  # ✅ BN -> GN
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class PrototypeInteractionBlock(nn.Module):
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
        
        with torch.cuda.amp.autocast(enabled=False):
            q_32, k_32, v_32 = q.float(), k.float(), v.float()
            scale = C ** -0.5
            attn_logits = (q_32 @ k_32.transpose(-2, -1)) * scale
            attn_logits = torch.clamp(attn_logits, min=-30, max=30)
            attn = attn_logits.softmax(dim=-1)
            out = attn @ v_32
            
        out = out.to(x.dtype)
        out = out.transpose(1, 2).view(B, C, H, W)
        out = self.out_proj(out)
        out = out + self.local_conv(out)
        return self.norm(residual + out * self.gamma.view(1, -1, 1, 1))

class FeedForward(nn.Module):
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
    def __init__(self, in_channels, out_channels, depth=3): 
        super().__init__()
        self.align = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        self.layers = nn.ModuleList([])
        self.gamma_ffn = nn.Parameter(torch.ones(depth, out_channels) * 1e-5)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PrototypeInteractionBlock(out_channels, num_prototypes=16),
                FeedForward(out_channels, out_channels * 4)
            ]))

    def forward(self, x):
        x = self.align(x)
        for i, (proto_block, ffn) in enumerate(self.layers):
            x = proto_block(x)
            gamma = self.gamma_ffn[i].view(1, -1, 1, 1)
            x = x + gamma * ffn(x)
        return x

class Up_Universal(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, decoder_type='phd', use_sparse_skip=False): 
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 🔥🔥🔥 [修改这里] 实例化 SCSE 模块 🔥🔥🔥
        self.skip_refiner = None
        if use_sparse_skip and skip_channels > 0:
            print(f"   ✨ Enable SCSE Refiner (Robust) for channel {skip_channels}")
            self.skip_refiner = SCSEModule(skip_channels) # <--- 改成这个类名！

        conv_in = in_channels + skip_channels
        
        if decoder_type == 'phd':
            self.conv = PHD_DecoderBlock_Pro(conv_in, out_channels, depth=2)
        else:
            self.conv = StandardDoubleConv(conv_in, out_channels)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if x2 is not None:
            # 这里不用改，只要 self.skip_refiner 不为空就会自动调用
            if self.skip_refiner is not None:
                x2 = self.skip_refiner(x2)
            
            diffY, diffX = x2.size()[2] - x1.size()[2], x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        return self.conv(x)

# ================================================================
# 4. 主模型: UniversalUNet
# ================================================================
class UniversalUNet(nn.Module):
    def __init__(self, 
                 n_classes=1, 
                 cnext_type='convnextv2_tiny', 
                 pretrained=True,
                 decoder_type='phd',       
                 use_dual_stream=True,     
                 use_deep_supervision=False,
                 use_sparse_skip=False,  # 🔥🔥🔥 [新增] 开关参数
                 **kwargs):
        super().__init__()
        self.n_classes = n_classes
        self.use_dual_stream = use_dual_stream
        self.decoder_type = decoder_type
        self.use_deep_supervision = use_deep_supervision
        
        print(f"🤖 [Universal Model] Initialized with:")
        print(f"   - Encoder: {cnext_type}")
        print(f"   - Dual Stream (SFDA): {'✅ ON (Omni-Optimized)' if use_dual_stream else '❌ OFF'}")
        print(f"   - Interaction: SK-Fusion")
        print(f"   - Decoder: {decoder_type.upper()}")
        print(f"   - Sparse Skip Refiner: {'✅ ON' if use_sparse_skip else '❌ OFF'}") # 打印一下状态
        # 1. Spatial Encoder
        self.spatial_encoder = timm.create_model(cnext_type, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3), drop_path_rate=0.0)
        s_dims = self.spatial_encoder.feature_info.channels() 
        
        # 2. Frequency Encoder
        if self.use_dual_stream:
            f_dims = [c // 2 for c in s_dims]
            self.freq_stem = nn.Sequential(nn.Conv2d(3, f_dims[0], 4, stride=4, padding=0), nn.BatchNorm2d(f_dims[0]), nn.ReLU(True))
            
            self.freq_layers = nn.ModuleList([
                SFDABlock(in_channels=f_dims[i], out_channels=f_dims[i+1]) 
                for i in range(3)
            ])
            
            self.bi_fgf_modules = nn.ModuleList([SK_Fusion_Block(s_dims[i], f_dims[i]) for i in range(4)])
            self.edge_head = nn.Sequential(nn.Conv2d(f_dims[0], 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.Conv2d(64, 1, 1))

        # 3. Decoder
        self.up1 = Up_Universal(s_dims[3], s_dims[2], skip_channels=s_dims[2], decoder_type=decoder_type, use_sparse_skip=use_sparse_skip)
        self.up2 = Up_Universal(s_dims[2], s_dims[1], skip_channels=s_dims[1], decoder_type=decoder_type, use_sparse_skip=use_sparse_skip)
        self.up3 = Up_Universal(s_dims[1], s_dims[0], skip_channels=s_dims[0], decoder_type=decoder_type, use_sparse_skip=use_sparse_skip)
        
        self.final_up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
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
        
        logits = self.outc(self.final_up(d3))
        
        # 🔥🔥🔥 [统一返回 List] 🔥🔥🔥
        if self.training:
            outputs = [logits]
            
            if self.use_deep_supervision:
                aux2 = self.head_up2(d2)
                aux3 = self.head_up3(d3)
                outputs.extend([aux2, aux3])
            
            if self.use_dual_stream and edge_logits is not None:
                outputs.append(edge_logits)
                
            return outputs # 始终返回 List，train02.py 舒服了

        return logits