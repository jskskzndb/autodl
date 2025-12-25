"""
wvm_unet.py
----------------------------------------------------------------
Architecture: WVM-UNet (Wavelet-Visual-Mamba UNet)
Encoder: ConvNeXt V2
Decoder: Wavelet-Visual-Mamba (WVM) Upsampler - [Enhanced Fusion Mode]
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
    # 引入你现有的 StripConvBlock 和 VisualStateSpaceBlock
    from decoder.hybrid_decoder import VisualStateSpaceBlock, StripConvBlock
except ImportError:
    print("❌ Error: Could not import modules from decoder.hybrid_decoder")
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

def make_head(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(in_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_ch, out_ch, kernel_size=1)
    )

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
# 2. WVM 上采样器 (增强版：找回丢失的 skip_ll)
# ================================================================
class WVM_Upsampler(nn.Module):
    def __init__(self, deep_channels, skip_channels, out_channels, use_dcn=False):
        super().__init__()
        
        if VisualStateSpaceBlock is None:
            raise ImportError("Mamba module not found.")

        self.dwt_idwt = HaarWaveletTransform()
        self.mid_channels = out_channels
        
        # 1. 投影层
        self.deep_proj = nn.Conv2d(deep_channels, self.mid_channels, 1)
        self.skip_proj = nn.Conv2d(skip_channels, self.mid_channels, 1) # 用于高频
        
        # 🔥 [新增] 专门用于 skip_ll 的投影
        # 我们需要把它压缩到 mid_channels 以便和 deep特征 融合
        self.skip_ll_proj = nn.Conv2d(skip_channels, self.mid_channels, 1) 
        
        # ==================== [核心修改 Start] ====================
        
        # 【分支 A】 低频融合与建模 (LL Fusion & Mamba)
        # 以前：Mamba( Deep ) -> 缺少细节
        # 现在：Mamba( Deep + Skip_LL ) -> 语义与结构兼备
        self.mamba_ll = VisualStateSpaceBlock(dim=self.mid_channels)
        
        # 【分支 B】 高频边缘 (LH, HL, HH) -> Strip DCN
        # 保持不变，这是你成功的关键
        self.edge_align = StripConvBlock(
            in_channels=self.mid_channels * 3,
            out_channels=self.mid_channels * 3,
            kernel_size=7,    
            use_dcn=use_dcn   
        )
        # ==================== [核心修改 End] ======================
        
        # 输出重建
        self.out_conv = nn.Sequential(
            nn.Conv2d(self.mid_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_deep, x_skip):
        # 1. 投影深层语义 (Semantic)
        feat_ll_deep = self.deep_proj(x_deep) 
        
        # 2. 分解跳跃连接 (Structure & Edge)
        # 🔥 [关键修改] 不再丢弃第一个返回值，而是捕获 skip_ll
        skip_ll, skip_lh, skip_hl, skip_hh = self.dwt_idwt.dwt(x_skip)
        
        # 3. 投影跳跃分量
        feat_ll_skip = self.skip_ll_proj(skip_ll) # 投影 skip_ll
        
        feat_lh = self.skip_proj(skip_lh)
        feat_hl = self.skip_proj(skip_hl)
        feat_hh = self.skip_proj(skip_hh)
        
        # ==================== [核心修改 Start] ====================
        # 2. 分流与融合 (Split, Fuse & Conquer)
        
        # Path A: 低频处理 (LL)
        # 🔥 [关键修改] 融合 Deep语义 和 Skip结构
        # 简单的相加融合 (ResNet风格)，既保留了语义，又找回了丢失的空间灰度信息
        feat_ll_combined = feat_ll_deep + feat_ll_skip
        
        # 送入 Mamba 进行上下文建模
        ref_ll = self.mamba_ll(feat_ll_combined)
        
        # Path B: 高频处理 (High Freq)
        # 拼接三个高频分量
        high_freq_stack = torch.cat([feat_lh, feat_hl, feat_hh], dim=1)
        
        # Strip DCN 处理
        ref_high = self.edge_align(high_freq_stack)
        
        # 拆分回三个分量
        ref_lh, ref_hl, ref_hh = torch.chunk(ref_high, 3, dim=1)
        # ==================== [核心修改 End] ======================
        
        # 3. 小波反变换重建
        out = self.dwt_idwt.idwt(ref_ll, ref_lh, ref_hl, ref_hh)
        
        return self.out_conv(out)


# ================================================================
# 3. 主模型: WVM-UNet
# ================================================================
class WVM_UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, cnext_type='convnextv2_base', use_decouple=False, **kwargs):
        super().__init__()
        self.use_decouple = use_decouple 
        
        # 兼容性参数接收
        use_dcn = kwargs.get('use_dcn', False) or kwargs.get('use_dcn_in_phd', False)
        
        print(f"🚀 [WVM-UNet] Initializing Model...")
        print(f"   - Alignment Mode: {'Deformable Conv (DCN)' if use_dcn else 'Strip Conv (Standard)'}")
        print(f"   - MDBES Decoupling: {'ENABLED ✅' if use_decouple else 'DISABLED ❌'}") 
        
        self.n_classes = n_classes
        self.encoder_name = 'cnextv2'
        
        self.enc_model = timm.create_model(
            cnext_type, pretrained=True, features_only=True, 
            out_indices=[0, 1, 2, 3], in_chans=n_channels
        )
        c1, c2, c3, c4 = self.enc_model.feature_info.channels()
        
        self.up1 = WVM_Upsampler(deep_channels=c4, skip_channels=c3, out_channels=c3, use_dcn=use_dcn)
        self.up2 = WVM_Upsampler(deep_channels=c3, skip_channels=c2, out_channels=c2, use_dcn=use_dcn)
        self.up3 = WVM_Upsampler(deep_channels=c2, skip_channels=c1, out_channels=c1, use_dcn=use_dcn)
        
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(c1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        if self.use_decouple:
            self.decoupler = SqueezeBodyEdge(64)
            self.head_body = make_head(64, 1)
            self.head_edge = make_head(64, 1)
            self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        else:
            self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        features = self.enc_model(x)
        s1, s2, s3, x4 = features[0], features[1], features[2], features[3]

        d1 = self.up1(x_deep=x4, x_skip=s3)
        d2 = self.up2(x_deep=d1, x_skip=s2)
        d3 = self.up3(x_deep=d2, x_skip=s1)
        
        d4 = self.final_up(d3)
        if self.use_decouple:
            feat_body, feat_edge = self.decoupler(d4)
            feat_fuse = feat_body + feat_edge
            out_seg = self.outc(feat_fuse)
            if self.training:
                out_body = self.head_body(feat_body)
                out_edge = self.head_edge(feat_edge)
                return out_seg, out_body, out_edge
            else:
                return out_seg
        else:
            return self.outc(d4)