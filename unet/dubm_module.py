"""
dubm_module.py (Fixed Arguments)
修复日志：
1. [参数修正] DCNv3Function.apply 补全了漏掉的 dilation_w 参数。
2. 包含之前的 edge_prior、GroupNorm 和 Contiguous 修复。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 尝试导入 DCNv3 底层算子函数
try:
    import sys
    sys.path.append("./ops_dcnv3")
    from functions.dcnv3_func import DCNv3Function
    HAS_DCN = True
except ImportError:
    HAS_DCN = False
    print("⚠️ D-UBM Error: DCNv3 functions not found! Please compile ops_dcnv3 first.")

class UncertaintyDCN(nn.Module):
    """
    [魔改版 DCN] 支持不确定性调制的 DCNv3
    """
    def __init__(self, channels, kernel_size=3, group=4, offset_scale=1.0):
        super().__init__()
        if not HAS_DCN:
            raise ImportError("Cannot initialize UncertaintyDCN: DCNv3 not compiled.")

        self.channels = channels
        self.kernel_size = kernel_size
        self.group = group
        self.offset_scale = offset_scale
        
        # 1. 上下文提取网络 (GroupNorm 修复版)
        self.dw_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=(kernel_size-1)//2, groups=channels),
            nn.GroupNorm(1, channels), 
            nn.GELU()
        )

        # 2. 线性投影层
        self.offset_mask_linear = nn.Linear(
            channels, 
            group * kernel_size * kernel_size * 3
        )
        
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.offset_mask_linear.weight, 0.)
        nn.init.constant_(self.offset_mask_linear.bias, 0.)

    def forward(self, x, uncertainty_map):
        """
        x: [N, C, H, W]
        uncertainty_map: [N, 1, H, W]
        """
        N, C, H, W = x.shape
        
        # 准备数据格式
        x_in = x.permute(0, 2, 3, 1).contiguous() 
        
        # --- A. 预测 Offset 和 Mask ---
        feat_ctx = self.dw_conv(x).permute(0, 2, 3, 1).contiguous()
        offset_mask = self.offset_mask_linear(feat_ctx)
        
        offset_dim = self.group * self.kernel_size * self.kernel_size * 2
        
        # 拆分并强制连续化
        offset = offset_mask[..., :offset_dim].contiguous()
        mask = offset_mask[..., offset_dim:].contiguous()
        
        # Mask 归一化
        mask = mask.reshape(N, H, W, self.group, -1)
        mask = F.softmax(mask, dim=-1)
        
        # --- B. 不确定性调制 ---
        u_map = uncertainty_map.permute(0, 2, 3, 1).unsqueeze(-1)
        mask = mask * u_map 
        
        # Reshape 并强制连续化
        mask = mask.reshape(N, H, W, -1).contiguous()

        # --- C. 调用底层算子 ---
        # 🔥🔥🔥 [参数修正点] 🔥🔥🔥
        x_out = DCNv3Function.apply(
            x_in, 
            offset, 
            mask,
            self.kernel_size, self.kernel_size,
            1, 1, # stride_h, stride_w
            (self.kernel_size-1)//2, (self.kernel_size-1)//2, # pad_h, pad_w
            1, 1, # 🔥 dilation_h, dilation_w (之前漏了一个1)
            self.group, 
            self.channels // self.group, 
            self.offset_scale,
            256, # im2col_step
            False # remove_center
        )
        
        return x_out.permute(0, 3, 1, 2).contiguous()


class DUBM_Block(nn.Module):
    """
    [D-UBM 完整模块]
    支持双流互补：接收外部边缘先验
    """
    def __init__(self, in_channels):
        super().__init__()
        self.seg_head = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.dcn_refine = UncertaintyDCN(in_channels, kernel_size=3, group=4)
        self.fusion = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    # 支持 edge_prior
    def forward(self, x, edge_prior=None):
        # 1. 内源性不确定性
        p = torch.sigmoid(self.seg_head(x))
        u_self = 1 - torch.abs(2 * p - 1)
        
        # 2. 互补融合逻辑
        if edge_prior is not None:
            edge_prob = torch.sigmoid(edge_prior)
            if edge_prob.shape[2:] != u_self.shape[2:]:
                edge_prob = F.interpolate(edge_prob, size=u_self.shape[2:], mode='bilinear', align_corners=True)
            u_final = torch.max(u_self, edge_prob)
        else:
            u_final = u_self
        
        # 3. DCN 精修
        feat_refined = self.dcn_refine(x, u_final)
        out = x + self.fusion(feat_refined)
        
        return out, u_final