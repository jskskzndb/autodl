"""
hybrid_decoder.py
支持参数穿透 (edge_prior) 的解码器
"""
import torch
import torch.nn as nn
from decoder.mamba_helper import MambaLayer2D

try: from unet.dubm_module import DUBM_Block
except ImportError: DUBM_Block = None

try:
    import sys
    sys.path.append("./ops_dcnv3")
    from modules.dcnv3 import DCNv3
    HAS_DCN = True
except ImportError: HAS_DCN = False

# StripConvBlock 保持不变，但增加参数接收
class StripConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, use_dcn=False):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.proj = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.use_dcn = use_dcn and HAS_DCN
        if self.use_dcn:
            dcn_group = 4 
            self.strip_h = DCNv3(channels=out_channels, kernel_size=(1, kernel_size), stride=1, pad=(0, padding), group=dcn_group, offset_scale=1.0)
            self.strip_v = DCNv3(channels=out_channels, kernel_size=(kernel_size, 1), stride=1, pad=(padding, 0), group=dcn_group, offset_scale=1.0)
            self.norm_h = nn.BatchNorm2d(out_channels); self.norm_v = nn.BatchNorm2d(out_channels); self.act = nn.ReLU(inplace=True)
        else:
            self.strip_h = nn.Sequential(nn.Conv2d(out_channels, out_channels, (1, kernel_size), padding=(0, padding), groups=out_channels, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
            self.strip_v = nn.Sequential(nn.Conv2d(out_channels, out_channels, (kernel_size, 1), padding=(padding, 0), groups=out_channels, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.fusion_conv = nn.Conv2d(out_channels, out_channels, 1)

    # 🔥 接收 edge_prior 但忽略它，保证接口统一
    def forward(self, x, edge_prior=None):
        x = self.proj(x)
        if self.use_dcn:
            h = self.act(self.norm_h(self.strip_h(x)))
            v = self.act(self.norm_v(self.strip_v(x)))
        else:
            h = self.strip_h(x); v = self.strip_v(x)
        return self.fusion_conv(h + v)
class OmniMambaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 1. 通道对齐层 (Channel Alignment)
        # 如果输入输出通道不同，必须先对齐，否则无法做残差相加。
        # 如果通道相同，标准的 Block 通常不在这里再加卷积，直接进残差。
        if in_channels != out_channels:
            self.align = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.align = nn.Identity() # 通道一致时，直接透传，不做多余计算

        # 2. 核心 Mamba 层
        # MambaLayer2D 内部包含了: Norm -> Linear(升维) -> Conv1d -> SSM -> Linear(降维)
        self.core_op = MambaLayer2D(out_channels)
        
    def forward(self, x):
        # === 步骤 1: 维度对齐 ===
        x = self.align(x)
        
        # === 步骤 2: 存储残差 (Shortcut) ===
        # 这是“标准版”的灵魂：保留原始信息
        residual = x 

        # === 步骤 3: Mamba 核心处理 ===
        # 注意：这里不需要再做 self.norm(x)，因为 core_op 内部第一步就是 LayerNorm (Pre-Norm 结构)
        
        # 3.1 正向
        x1 = self.core_op(x)
        
        # 3.2 水平翻转 (模仿从右到左扫描)
        x2 = torch.flip(self.core_op(torch.flip(x, dims=[2, 3])), dims=[2, 3])
        
        # 3.3 垂直翻转 (模仿从下到上扫描)
        # 先转置(H,W互换) -> 此时的 flip 相当于原图的垂直操作
        x3 = self.core_op(x.transpose(2, 3)).transpose(2, 3)
        
        # 3.4 反向垂直 (模仿从上到下扫描)
        x4 = torch.transpose(torch.flip(self.core_op(torch.flip(x.transpose(2, 3), dims=[2, 3])), dims=[2, 3]), 2, 3)
        
        # 融合四向结果
        # 标准 VMamba 可能会用 Linear 做融合，但求平均 (Mean) 是最稳健且不增加参数的标准做法
        mamba_out = (x1 + x2 + x3 + x4) / 4.0
        
        # === 步骤 4: 残差连接 ===
        # Output = Input + Mamba(Norm(Input))
        return mamba_out + residual

# SKFusion 保持不变
class SKFusion(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        mid_channels = max(channels // reduction, 4)
        self.fc = nn.Sequential(nn.Linear(channels, mid_channels, bias=False), nn.ReLU(inplace=True), nn.Linear(mid_channels, 2 * channels, bias=False))
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x_local, x_global):
        B, C, H, W = x_local.shape
        U = x_local + x_global 
        s = self.avg_pool(U).view(B, C)
        z = self.fc(s).view(B, 2, C)
        weights = self.softmax(z)
        return weights[:, 0].view(B, C, 1, 1) * x_local + weights[:, 1].view(B, C, 1, 1) * x_global

# PHD_DecoderBlock 修改接口
class PHD_DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_dcn=False, use_dubm=False):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.use_dubm = use_dubm and (DUBM_Block is not None)
        
        if self.use_dubm: self.local_branch = DUBM_Block(out_channels)
        elif use_dcn: self.local_branch = StripConvBlock(out_channels, out_channels, use_dcn=True)
        else: self.local_branch = StripConvBlock(out_channels, out_channels, use_dcn=False)
        
        self.global_branch = OmniMambaBlock(out_channels, out_channels)
        self.fusion = SKFusion(out_channels)

    # 🔥 关键：接收并传递 edge_prior
    def forward(self, x, edge_prior=None):
        x = self.relu(self.bn(self.reduce(x)))
        
        if self.use_dubm:
            # 只有 D-UBM 真正使用这个参数
            feat_local, _ = self.local_branch(x, edge_prior=edge_prior)
        else:
            feat_local = self.local_branch(x, edge_prior=edge_prior) # 其他模式会忽略
            
        feat_global = self.global_branch(x)
        return self.fusion(feat_local, feat_global)
        # ================================================================
# 5. [新增] VisualStateSpaceBlock 
# (这是为了适配 wvm_unet.py 的调用接口)
# ================================================================
class VisualStateSpaceBlock(nn.Module):
    """
    WVM 模型需要的接口包装器。
    它接收 'dim' 参数，并在内部调用 OmniMambaBlock。
    """
    def __init__(self, dim):
        super().__init__()
        # WVM 传入的是 dim (例如 256)，OmniMambaBlock 接收 in/out channels
        self.block = OmniMambaBlock(in_channels=dim, out_channels=dim)

    def forward(self, x):
        return self.block(x)