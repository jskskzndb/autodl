import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

# ==========================================
# 1. 核心组件: Bilateral Hybrid Attention
# ==========================================
class BilateralHybridAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=4):
        super().__init__()
        self.num_heads = num_heads
        
        # 降维策略: Attention 内部维度为 C/4
        self.internal_dim = dim // 4
        self.head_dim = self.internal_dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.sr_ratio = sr_ratio

        # Path 1: Query (Conv 4x4)
        self.sr_conv = nn.Conv2d(dim, self.internal_dim, kernel_size=sr_ratio, stride=sr_ratio)
        self.norm_q = nn.LayerNorm(self.internal_dim)
        self.q_proj = nn.Linear(self.internal_dim, self.internal_dim, bias=qkv_bias)

        # Path 2: SAP (MaxPool)
        self.max_pool = nn.MaxPool2d(kernel_size=sr_ratio, stride=sr_ratio)
        self.k1_proj = nn.Linear(dim, self.internal_dim, bias=qkv_bias)
        self.v1_proj = nn.Linear(dim, self.internal_dim, bias=qkv_bias)

        # Path 3: OAP (AvgPool)
        self.avg_pool = nn.AvgPool2d(kernel_size=sr_ratio, stride=sr_ratio)
        self.k2_proj = nn.Linear(dim, self.internal_dim, bias=qkv_bias)
        self.v2_proj = nn.Linear(dim, self.internal_dim, bias=qkv_bias)

        # Relative Position Bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * 7 - 1) * (2 * 7 - 1), num_heads)) 
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.internal_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.upsample = nn.Upsample(scale_factor=sr_ratio, mode='bilinear', align_corners=True)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. Query
        q_feat = self.sr_conv(x)
        H_sr, W_sr = q_feat.shape[2], q_feat.shape[3]
        N = H_sr * W_sr
        q_feat = q_feat.flatten(2).transpose(1, 2)
        q = self.q_proj(self.norm_q(q_feat))

        # 2. SAP (K1, V1)
        x_max = self.max_pool(x).flatten(2).transpose(1, 2)
        k1 = self.k1_proj(x_max)
        v1 = self.v1_proj(x_max)

        # 3. OAP (K2, V2)
        x_avg = self.avg_pool(x).flatten(2).transpose(1, 2)
        k2 = self.k2_proj(x_avg)
        v2 = self.v2_proj(x_avg)

        # 4. Attention
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k1 = k1.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v1 = v1.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k2 = k2.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v2 = v2.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn1 = (q @ k1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        z_sap = (attn1 @ v1)

        attn2 = (q @ k2.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)
        z_oap = (attn2 @ v2)

        # 5. Fusion & Restore
        z = z_sap + z_oap
        z = z.transpose(1, 2).reshape(B, N, self.internal_dim)
        z = self.proj(self.proj_drop(z))
        z = z.transpose(1, 2).reshape(B, C, H_sr, W_sr)
        z = self.upsample(z)
        
        return z

# ==========================================
# 2. 核心模块: BHAFormer Block
# ==========================================
class BHAFormerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.input_conv = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.GELU())
        
        self.attn = BilateralHybridAttention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.output_conv = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.Dropout(drop))

        self.norm2 = nn.BatchNorm2d(dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, int(dim * mlp_ratio), 1), nn.GELU(), nn.Dropout(drop),
            nn.Conv2d(int(dim * mlp_ratio), dim, 1), nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.drop_path(self.output_conv(self.attn(self.input_conv(self.norm1(x)))))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x

# ==========================================
# 3. 适配器: Up_BHA (用于 U-Net)
# ==========================================
class Up_BHA(nn.Module):
    """ 
    U-Net Decoder Block using BHAFormer 
    结构: Upsample -> Concat -> 1x1 Conv(Fusion) -> BHAFormerBlock
    """
    def __init__(self, in_channels, out_channels, bilinear=True, skip_channels=0):
        super().__init__()
        
        # 1. 上采样
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            concat_channels = in_channels + skip_channels
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            concat_channels = (in_channels // 2) + skip_channels

        # 2. 融合通道并降维
        self.fusion_conv = nn.Conv2d(concat_channels, out_channels, kernel_size=1)
        
        # 3. 核心处理: BHAFormer
        # 注意: out_channels 必须能被 4 整除 (因为内部有 dim//4 操作)
        # 如果 out_channels 太小(如64)，heads设为4或2
        n_heads = 8 if out_channels >= 128 else 4
        self.bha_block = BHAFormerBlock(dim=out_channels, num_heads=n_heads)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        
        # 拼接 Skip Connection
        if x2 is not None:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
            
        # 融合与特征提取
        x = self.fusion_conv(x)
        x = self.bha_block(x)
        return x