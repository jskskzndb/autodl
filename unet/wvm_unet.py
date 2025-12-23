"""
wvm_unet.py
----------------------------------------------------------------
Architecture: WVM-UNet (Wavelet-Visual-Mamba UNet)
Encoder: ConvNeXt V2
Decoder: Wavelet-Visual-Mamba (WVM) Upsampler
----------------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from pathlib import Path
import sys

# 添加路径以导入 Mamba 模块
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    # 引入你现有的 StripConvBlock
    from decoder.hybrid_decoder import VisualStateSpaceBlock, StripConvBlock
except ImportError:
    print("❌ Error: Could not import VisualStateSpaceBlock from decoder.hybrid_decoder")
    VisualStateSpaceBlock = None


class SqueezeBodyEdge(nn.Module):
    def __init__(self, inplane):
        super(SqueezeBodyEdge, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2, padding=1),
            nn.BatchNorm2d(inplane),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2, padding=1),
            nn.BatchNorm2d(inplane),
            nn.ReLU(inplace=True)
        )
        self.flow_make = nn.Conv2d(inplane * 2, 2, kernel_size=3, padding=1, bias=True)
        
        # 3. 【核心步骤】强制零初始化
        # 让初始的光流场全为 0，图像不发生任何扭曲
        nn.init.constant_(self.flow_make.weight, 0)
        nn.init.constant_(self.flow_make.bias, 0)

    def forward(self, x):
        size = x.size()[2:]
        seg_down = self.down(x)
        seg_down = F.interpolate(seg_down, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([x, seg_down], dim=1))
        seg_flow_warp = self.flow_warp(x, flow, size)
        seg_edge = x - seg_flow_warp
        return seg_flow_warp, seg_edge

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h_grid = torch.linspace(-1.0, 1.0, out_h, device=input.device).view(-1, 1).repeat(1, out_w)
        w_grid = torch.linspace(-1.0, 1.0, out_w, device=input.device).repeat(out_h, 1)
        grid = torch.cat((w_grid.unsqueeze(2), h_grid.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1)
        grid = grid + flow.permute(0, 2, 3, 1) / norm
        return F.grid_sample(input, grid, align_corners=True)

# 定义一个生成标准检测头的辅助函数 (Conv3x3 -> Conv1x1)
def make_head(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(in_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_ch, out_ch, kernel_size=1)
    )

class SE_Projection(nn.Module):
    """
    带通道注意力机制的投影层 (SE-Weighted Projection)
    作用：在压缩通道之前，先判断哪些通道是重要的（比如这是房子），哪些是不重要的（比如这是背景），
          给重要的通道加权，不重要的抑制，然后再进行 1x1 卷积压缩。
    """
    def __init__(self, in_channels, out_channels, reduction=16):
        super().__init__()
        # 1. SE 模块 (计算通道权重)
        # 确保 reduction 不会让通道数变成 0，最小为 4
        mid = max(in_channels // reduction, 4)
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),             # Squeeze: 全局池化
            nn.Conv2d(in_channels, mid, 1),      # Excitation 1: 降维
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, 1),      # Excitation 2: 升维
            nn.Sigmoid()                         # 输出 0~1 的权重
        )
        
        # 2. 投影层 (原始的 1x1 卷积)
        self.proj = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        # x: [B, C, H, W]
        w = self.se(x)     # 计算权重
        x = x * w          # 重新加权 (Reweight)
        return self.proj(x) # 投影压缩
# ================================================================
# 1. Haar 小波变换
# ================================================================
class HaarWaveletTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def dwt(self, x):
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        ll = (x00 + x01 + x10 + x11) / 2
        lh = (x00 + x01 - x10 - x11) / 2
        hl = (x00 - x01 + x10 - x11) / 2
        hh = (x00 - x01 - x10 + x11) / 2
        return ll, lh, hl, hh

    def idwt(self, ll, lh, hl, hh):
        x00 = (ll + lh + hl + hh) / 2
        x01 = (ll + lh - hl - hh) / 2
        x10 = (ll - lh + hl - hh) / 2
        x11 = (ll - lh - hl + hh) / 2
        b, c, h, w = ll.shape
        out = torch.zeros(b, c, h * 2, w * 2, device=ll.device, dtype=ll.dtype)
        out[:, :, 0::2, 0::2] = x00
        out[:, :, 0::2, 1::2] = x01
        out[:, :, 1::2, 0::2] = x10
        out[:, :, 1::2, 1::2] = x11
        return out


# ================================================================
# 2. WVM 上采样器 (核心模块)
# ================================================================
class WVM_Upsampler(nn.Module):
    def __init__(self, deep_channels, skip_channels, out_channels, use_dcn=False):
        super().__init__()
        
        if VisualStateSpaceBlock is None:
            raise ImportError("Mamba module not found.")

        self.dwt_idwt = HaarWaveletTransform()
        self.mid_channels = out_channels
        
        # 投影层
        
        # 🔥普通卷积
        self.deep_proj = nn.Conv2d(deep_channels, self.mid_channels, 1)
        
        # ✅ 新代码:
        #self.deep_proj = SE_Projection(deep_channels, self.mid_channels)
        
        # 2. 跳跃连接保持普通卷积 (保留原始细节)
        self.skip_proj = nn.Conv2d(skip_channels, self.mid_channels, 1)
        
        # 融合层 (输入 4 个分量)
        #self.fusion_conv = nn.Conv2d(self.mid_channels * 4, self.mid_channels * 4, 1)
        
        # 🔥🔥🔥 新增：空间对齐模块 (Strip Convolution) 🔥🔥🔥
        # 放在融合之后，Mamba 之前
        # 输入通道是 mid_channels * 4 (因为拼接了4个分量)
        #self.align_module = StripConvBlock(
        #    in_channels=self.mid_channels * 4, 
        #    out_channels=self.mid_channels * 4, 
        #    kernel_size=7, # 你想要的大核
        #    use_dcn=use_dcn  # 设为 True 就是用 DCN，设为 False 就是用 Strip Conv
        #)

        # Mamba 频域筛选
        #self.mamba_selector = VisualStateSpaceBlock(dim=self.mid_channels * 4)
        
        # [修改] 移除 fusion_conv，改为分流处理
        
        # [分支 A] 低频语义 (LL) -> Mamba
        # 只处理 1 个分量，参数量大幅降低
        self.mamba_ll = VisualStateSpaceBlock(dim=self.mid_channels)

        # [分支 B] 高频边缘 (LH, HL, HH) -> Strip DCN
        # 处理 3 个分量，使用 1x7 和 7x1 并行卷积捕捉几何边缘
        # 强制开启 use_dcn=True 以处理倾斜/不规则边缘
        self.edge_align = StripConvBlock(
            in_channels=self.mid_channels * 3,
            out_channels=self.mid_channels * 3,
            kernel_size=7,    
            use_dcn=True      
        )


        # 输出平滑
        self.out_conv = nn.Sequential(
            nn.Conv2d(self.mid_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_deep, x_skip):
        # 1. 准备 (Preparation)
        feat_ll = self.deep_proj(x_deep) 
        _, skip_lh, skip_hl, skip_hh = self.dwt_idwt.dwt(x_skip)
        
        feat_lh = self.skip_proj(skip_lh)
        feat_hl = self.skip_proj(skip_hl)
        feat_hh = self.skip_proj(skip_hh)
        
        # 2. 筛选 (Selection)
        #combined = torch.cat([feat_ll, feat_lh, feat_hl, feat_hh], dim=1)
        #combined = self.fusion_conv(combined)

        # 🔥🔥🔥 新增：先对齐，再筛选 🔥🔥🔥
        # Strip Conv 会利用 7x7 的视野，把语义和纹理在空间上对准
        # combined = self.align_module(combined)

        #combined_refined = self.mamba_selector(combined)
        #ref_ll, ref_lh, ref_hl, ref_hh = torch.chunk(combined_refined, 4, dim=1)
        
# ==================== [核心修改 Start] ====================
        # 2. 分流处理 (Divide and Conquer)
        
        # Path A: 低频走 Mamba (学习全局语义)
        ref_ll = self.mamba_ll(feat_ll)
        
        # Path B: 高频走 Strip DCN (学习几何边缘)
        # 拼接三个高频分量
        high_freq_stack = torch.cat([feat_lh, feat_hl, feat_hh], dim=1)
        
        # Strip DCN 处理
        ref_high = self.edge_align(high_freq_stack)
        
        # 拆分回三个分量
        ref_lh, ref_hl, ref_hh = torch.chunk(ref_high, 3, dim=1)
        # ==================== [核心修改 End] ======================

        # 3. 重建 (Reconstruction)
        out = self.dwt_idwt.idwt(ref_ll, ref_lh, ref_hl, ref_hh)
        return self.out_conv(out)


# ================================================================
# 3. 主模型: WVM-UNet
# ================================================================
class WVM_UNet(nn.Module):
    # **kwargs 用于接收并忽略不需要的参数 (如 use_dsis 等)
    def __init__(self, n_channels=3, n_classes=1, cnext_type='convnextv2_base', use_decouple=False, **kwargs):
        super().__init__()
        self.use_decouple = use_decouple  # 保存开关状态
        use_dcn = kwargs.get('use_dcn', False)
        print(f"🚀 [WVM-UNet] Initializing Model...")
        print(f"   - Alignment Mode: {'Deformable Conv (DCN)' if use_dcn else 'Strip Conv'}")
        print(f"   - MDBES Decoupling: {'ENABLED ✅' if use_decouple else 'DISABLED ❌'}") # 打印状态
        
        self.n_classes = n_classes

        self.encoder_name = 'cnextv2'
        
        # --- A. Encoder: ConvNeXt V2 ---
        self.enc_model = timm.create_model(
            cnext_type, pretrained=True, features_only=True, 
            out_indices=[0, 1, 2, 3], in_chans=n_channels
        )
        c1, c2, c3, c4 = self.enc_model.feature_info.channels()
        
        # --- B. Decoder: WVM Stages ---
        # Up 1: 1/32 -> 1/16
        self.up1 = WVM_Upsampler(deep_channels=c4, skip_channels=c3, out_channels=c3, use_dcn=use_dcn)
        # Up 2: 1/16 -> 1/8
        self.up2 = WVM_Upsampler(deep_channels=c3, skip_channels=c2, out_channels=c2, use_dcn=use_dcn)
        # Up 3: 1/8 -> 1/4
        self.up3 = WVM_Upsampler(deep_channels=c2, skip_channels=c1, out_channels=c1, use_dcn=use_dcn)
        
        # --- C. Final Head ---
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(c1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # ================= [修改点 2: 根据开关初始化解耦模块] =================
        if self.use_decouple:
            # 1. 初始化解耦器 (输入 64 通道)
            self.decoupler = SqueezeBodyEdge(64)
            
            # 2. 初始化两个辅助头 (Version B: 带3x3缓冲)
            # Body 和 Edge 都是二分类任务，所以输出通道为 1
            self.head_body = make_head(64, 1)
            self.head_edge = make_head(64, 1)
            
            # 3. 最终分割头
            # 输入依然是 64，因为 Body+Edge=FusedFeature，通道数不变
            self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        else:
            # 原始逻辑：直接接分类头
            self.outc = nn.Conv2d(64, n_classes, kernel_size=1)


    def forward(self, x):
        features = self.enc_model(x)
        s1, s2, s3, x4 = features[0], features[1], features[2], features[3]

        d1 = self.up1(x_deep=x4, x_skip=s3)
        d2 = self.up2(x_deep=d1, x_skip=s2)
        d3 = self.up3(x_deep=d2, x_skip=s1)
        
        d4 = self.final_up(d3)
        if self.use_decouple:
            # 1. 显式解耦
            feat_body, feat_edge = self.decoupler(d4)
            
            # 2. 重耦 (Re-couple) -> 融合一致性与锐利度
            feat_fuse = feat_body + feat_edge
            
            # 3. 主分割预测
            out_seg = self.outc(feat_fuse)
            
            # 4. 训练模式返回三元组 (用于计算那个三合一 Loss)
            if self.training:
                out_body = self.head_body(feat_body)
                out_edge = self.head_edge(feat_edge)
                return out_seg, out_body, out_edge
            else:
                # 验证/推理模式只返回最终分割
                return out_seg
        else:
            # 原始 Baseline 逻辑 (无解耦)
            return self.outc(d4)