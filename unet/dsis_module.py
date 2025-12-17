import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """ Squeeze-and-Excitation Block """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DSIS_Module(nn.Module):
    """
    Dual-Stream Interactive Skip Module (DSIS)
    输入:
        s1: Encoder Stage 1 特征 (H/4)
        s2: Encoder Stage 2 特征 (H/8)
    输出:
        out1: 增强后的 s1 (H/4, c_base) -> 送入 Decoder Up3
        out2: 增强后的 s2 (H/8, c_base) -> 送入 Decoder Up2
    """
    def __init__(self, c1_in, c2_in, c_base=64):
        super().__init__()
        
        # 1. 独立增强分支 (Unification & Attention)
        # S1 (High Res, Low Semantics)
        self.s1_proj = nn.Sequential(
            nn.Conv2d(c1_in, c_base, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_base),
            nn.ReLU(inplace=True),
            SEBlock(c_base)
        )
        
        # S2 (Med Res, Med Semantics)
        self.s2_proj = nn.Sequential(
            nn.Conv2d(c2_in, c_base, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_base),
            nn.ReLU(inplace=True),
            SEBlock(c_base)
        )

        # 2. 交互融合 (Interaction)
        # S2 -> S1 (上采样提供语义)
        self.up_fusion = nn.Sequential(
            nn.Conv2d(c_base, c_base, kernel_size=1),
            nn.BatchNorm2d(c_base),
            nn.Sigmoid() # 门控机制
        )
        
        # S1 -> S2 (下采样提供边缘)
        self.down_fusion = nn.Sequential(
            nn.Conv2d(c_base, c_base, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c_base),
            nn.ReLU(inplace=True)
        )
        
        # 3. 输出平滑
        self.s1_out = nn.Conv2d(c_base, c_base, 3, padding=1)
        self.s2_out = nn.Conv2d(c_base, c_base, 3, padding=1)

    def forward(self, s1, s2):
        # 1. 投影与注意力
        x1 = self.s1_proj(s1)
        x2 = self.s2_proj(s2)
        
        # 2. 交互
        # S2 支援 S1
        x2_up = F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=True)
        x1_gate = self.up_fusion(x2_up)
        x1_fused = x1 + (x1 * x1_gate) # 语义门控增强
        
        # S1 支援 S2
        x1_down = self.down_fusion(x1)
        x2_fused = x2 + x1_down # 边缘残差增强
        
        # 3. 输出
        out1 = self.s1_out(x1_fused)
        out2 = self.s2_out(x2_fused)
        
        return out1, out2