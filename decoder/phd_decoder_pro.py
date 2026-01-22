"""
decoder/phd_decoder_pro.py
[PHD Decoder Pro] 增强版混合解码器
特性:
  1. Inverted Bottleneck (倒残差): 先升维(4x)再处理，大幅增加参数量和特征容量。
  2. FFN (Feed-Forward Network): 引入两层感知机，增强非线性变换能力。
  3. Residual Connections: 内部多重残差，防止梯度消失，支持深层堆叠。
  4. 包含所有依赖组件 (Mamba, StripConv, SK-Fusion)，无需额外 import。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ================================================================
# 0. 基础依赖组件 (Mamba & DCN)
# ================================================================

# --- A. Mamba 环境检查 ---
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    print("⚠️ Warning: mamba-ssm not found. PHD Decoder Pro will fail if Mamba is required.")
    HAS_MAMBA = False

class MambaLayer2D(nn.Module):
    """ Mamba 的 2D 适配封装 """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if not HAS_MAMBA:
            raise ImportError("Mamba module not found. Please install mamba-ssm.")
            
        self.mamba = Mamba(
            d_model=dim,      # 输入通道数
            d_state=d_state,  # 状态维度
            d_conv=d_conv,    # 局部卷积宽度
            expand=expand     # 扩张系数
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        # 强制 FP32 防止 Mamba 溢出
        with torch.cuda.amp.autocast(enabled=False): 
            x = x.float()
            x_seq = x.flatten(2).transpose(1, 2) # [B, L, C]
            x_seq = self.norm(x_seq)
            x_seq = self.mamba(x_seq) 
            x_out = x_seq.transpose(1, 2).view(B, C, H, W)
        return x_out

# --- B. DCNv3 环境检查 ---
try:
    # 假设您的 DCNv3 路径如下，请根据实际情况调整
    # from ops_dcnv3.modules.dcnv3 import DCNv3
    # 暂时用占位符，如果没有 DCN 会自动回退到 StripConv
    HAS_DCN = False 
except ImportError:
    HAS_DCN = False

# ================================================================
# 1. 核心子模块
# ================================================================

# --- 1.1 Strip Conv Block (局部细节流) ---
class StripConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, use_dcn=True):
        super().__init__()
        padding = (kernel_size - 1) // 2
        
        # 投影层
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True)
        )
        
        # 判断是否使用 DCN (需要库支持)
        self.use_dcn = use_dcn and HAS_DCN
        
        if self.use_dcn:
            # DCN 实现 (略，需外部库支持)
            pass 
        else:
            # Fallback: 使用长条形卷积 (Strip Convolution)
            # 水平条卷积 (1 x K)
            self.strip_h = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (1, kernel_size), padding=(0, padding), 
                          groups=out_channels, bias=False), 
                nn.BatchNorm2d(out_channels), 
                nn.ReLU(inplace=True)
            )
            # 垂直条卷积 (K x 1)
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
            # 占位逻辑
            return x 
        else:
            h = self.strip_h(x)
            v = self.strip_v(x)
            return self.fusion_conv(h + v)

# --- 1.2 Omni-Mamba Block (全局语义流) ---
class OmniMambaBlock(nn.Module):
    """ 四向扫描 Mamba """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        if in_channels != out_channels:
            self.align = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.align = nn.Identity()

        self.core_op = MambaLayer2D(out_channels)
        
    def forward(self, x):
        x = self.align(x)
        residual = x 
        
        # 四向扫描: 正向、反向、垂直正向、垂直反向
        x1 = self.core_op(x)
        x2 = torch.flip(self.core_op(torch.flip(x, dims=[2, 3])), dims=[2, 3])
        x3 = self.core_op(x.transpose(2, 3)).transpose(2, 3)
        x4 = torch.transpose(torch.flip(self.core_op(torch.flip(x.transpose(2, 3), dims=[2, 3])), dims=[2, 3]), 2, 3)
        
        mamba_out = (x1 + x2 + x3 + x4) / 4.0
        return mamba_out + residual

# --- 1.3 SK-Fusion (自适应融合) ---
class SKFusion(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        mid_channels = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False), 
            nn.ReLU(inplace=True), 
            nn.Linear(mid_channels, 2 * channels, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_local, x_global):
        B, C, H, W = x_local.shape
        U = x_local + x_global 
        s = self.avg_pool(U).view(B, C)
        z = self.fc(s).view(B, 2, C)
        weights = self.softmax(z)
        w_local = weights[:, 0].view(B, C, 1, 1)
        w_global = weights[:, 1].view(B, C, 1, 1)
        return w_local * x_local + w_global * x_global

# ================================================================
# 2. 🔥 [PRO 版本] 增强版解码模块
# ================================================================

class PHD_DecoderBlock_Pro(nn.Module):
    """
    [Pro Version] 增重版 PHD 解码块
    策略: 
    1. 引入 Expansion Factor (默认4倍)，先升维再处理。
    2. 增加 FFN 模块，增强非线性特征变换。
    3. 适合小数据集下的暴力涨点。
    """
    def __init__(self, in_channels, out_channels, expand_ratio=4, use_dcn=True):
        super().__init__()
        
        # 1. 计算中间高维空间的维度
        hidden_dim = int(out_channels * expand_ratio)
        
        # 2. 通道对齐 (如果输入输出不一致，先对齐到 out_channels)
        self.align = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # === Stage A: 混合建模 (Inverted Bottleneck) ===
        # 3. 升维投影 (1x1 Conv) -> 变宽
        self.expand_conv = nn.Sequential(
            nn.Conv2d(out_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU() # 使用 GELU 激活
        )
        
        # 4. 双流处理 (在高维空间操作)
        # 局部流: Strip Conv
        self.local_branch = StripConvBlock(hidden_dim, hidden_dim, use_dcn=use_dcn)
        # 全局流: Omni-Mamba
        self.global_branch = OmniMambaBlock(hidden_dim, hidden_dim)
        
        # 5. 融合 (SK-Fusion)
        self.fusion = SKFusion(hidden_dim)
        
        # 6. 降维投影 (1x1 Conv) -> 变回原宽度
        self.proj_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # === Stage B: FFN 增强 (Feed-Forward Network) ===
        # 7. 类似于 Transformer 的 MLP 块
        ffn_dim = out_channels * 4
        self.ffn = nn.Sequential(
            nn.Conv2d(out_channels, ffn_dim, 1, bias=False),
            nn.BatchNorm2d(ffn_dim),
            nn.GELU(),
            nn.Dropout(0.1), # 防止过拟合
            nn.Conv2d(ffn_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # 缩放因子 (可选)
        self.gamma = nn.Parameter(torch.zeros(1)) 

    def forward(self, x):
        # 1. 对齐通道
        x = self.align(x)
        shortcut = x
        
        # --- Block 1: Inverted Bottleneck ---
        # 升维
        x_exp = self.expand_conv(x)
        
        # 双流处理
        x_local = self.local_branch(x_exp)
        x_global = self.global_branch(x_exp)
        
        # 融合
        x_fused = self.fusion(x_local, x_global)
        
        # 降维
        x_out = self.proj_conv(x_fused)
        
        # 残差连接 1
        x = shortcut + x_out
        
        # --- Block 2: FFN ---
        # 残差连接 2
        x = x + self.ffn(x)
        
        return x