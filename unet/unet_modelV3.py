"""
unet_model_encoder_wgn.py

"""

from .unet_parts import *
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from advanced_cafm_module import Advanced_CAFM


class EdgeDecoder(nn.Module):
    """
    边缘解码器：接收 WGN 的高频特征进行边缘重建
    """

    def __init__(self):
        super().__init__()
        # Layer 3 High Freq (1/16 size)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3072, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        # Layer 2 High Freq (1/8 size)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(512 + 1536, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # Layer 1 High Freq (1/4 size)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = nn.Sequential(
            nn.Conv2d(256 + 768, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Final Output (1/4 -> 1/1)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x3_h, x2_h, x1_h):
        x = self.conv1(self.up1(x3_h))
        x = torch.cat([x, x2_h], dim=1)
        x = self.conv2(self.up2(x))
        x = torch.cat([x, x1_h], dim=1)
        x = self.conv3(self.up3(x))
        # 此时是 1/4 尺寸，上采样 4 倍回原图
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        return self.final_conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, use_advanced_cafm=False,
                 use_resnet_encoder=False, use_wgn_enhancement=False, use_edge_loss=False, wgn_orders=None):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_advanced_cafm = use_advanced_cafm
        self.use_resnet_encoder = use_resnet_encoder
        self.use_wgn_enhancement = use_wgn_enhancement
        self.use_edge_loss = use_edge_loss  # 边缘损失开关
        self.checkpointing = False

        if use_resnet_encoder:
            # ========== ResNet50 Encoder (Standard Stride=2) ==========
            from torchvision.models import resnet50, ResNet50_Weights

            if wgn_orders is None:
                wgn_orders = {'layer1': (3, 2), 'layer2': (4, 3), 'layer3': (5, 4)}

            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

            # 提取各层
            self.conv1 = resnet.conv1  # 1/2
            self.bn1 = resnet.bn1
            self.relu = resnet.relu
            self.maxpool = resnet.maxpool  # 1/4

            self.layer1 = resnet.layer1  # 1/4, 256ch
            self.layer2 = resnet.layer2  # 1/8, 512ch
            self.layer3 = resnet.layer3  # 1/16, 1024ch
            self.layer4 = resnet.layer4  # 1/32, 2048ch

            # ========== WGN Enhancement (In-Encoder) ==========
            if use_wgn_enhancement:
                # 直接从 wgn 包导入 (默认指向 __init__.py 里定义的那个)
                from wgn import Wg_nConv_Block
                # 这里命名为 enhance 而不是 skip，暗示它在主干工作
                self.wgn_enhance1 = Wg_nConv_Block(256, *wgn_orders['layer1'])
                self.wgn_enhance2 = Wg_nConv_Block(512, *wgn_orders['layer2'])
                self.wgn_enhance3 = Wg_nConv_Block(1024, *wgn_orders['layer3'])

                if use_edge_loss:
                    self.edge_decoder = EdgeDecoder()

            # ========== Bottleneck ==========
            if use_advanced_cafm:
                self.cafm = Advanced_CAFM(n_feat=2048, n_head=8)
            else:
                self.bottleneck_conv = DoubleConv(2048, 2048)

            # ========== Decoder (5 Stages) ==========
            # 结构与 V2 保持一致，保证公平对比
            self.up1 = Up(2048, 1024, bilinear, skip_channels=1024)
            self.up2 = Up(1024, 512, bilinear, skip_channels=512)
            self.up3 = Up(512, 256, bilinear, skip_channels=256)
            self.up4 = Up(256, 128, bilinear, skip_channels=64)  # 拼接 Stem

            self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.final_conv_block = DoubleConv(128, 64)
            self.outc = OutConv(64, n_classes)

        else:
            
            # ========== 原始 U-Net 逻辑 (保持不变以兼容旧代码) ==========
            self.inc = (DoubleConv(n_channels, 64))
            self.down1 = (Down(64, 128))
            self.down2 = (Down(128, 256))
            self.down3 = (Down(256, 512))
            factor = 2 if bilinear else 1
            self.down4 = (Down(512, 1024 // factor))

            if self.use_advanced_cafm:
                bottleneck_channels = 1024 // factor
                self.advanced_cafm_bottleneck = Advanced_CAFM(n_feat=bottleneck_channels)

            self.up1 = (Up(1024, 512 // factor, bilinear))
            self.up2 = (Up(512, 256 // factor, bilinear))
            self.up3 = (Up(256, 128 // factor, bilinear))
            self.up4 = (Up(128, 64, bilinear))
            self.outc = (OutConv(64, n_classes))
            pass

    def forward(self, x):
        if self.use_resnet_encoder:
            # === 1. Encoder (带有串联 WGN) ===

            # Stem
            x0 = self.relu(self.bn1(self.conv1(x)))  # 1/2
            x_stem = self.maxpool(x0)  # 1/4

            # Layer 1
            x1 = self.layer1(x_stem)  # 1/4
            x1_high = None
            if self.use_wgn_enhancement:
                # 🔥 关键修改：WGN 处理后的结果 x1 直接替换原 x1
                # 这意味着下一层 Layer2 接收的是 WGN 处理过的特征
                x1, x1_high = self.wgn_enhance1(x1)

            # Layer 2
            x2 = self.layer2(x1)  # 1/8 (输入是 WGN 后的 x1)
            x2_high = None
            if self.use_wgn_enhancement:
                x2, x2_high = self.wgn_enhance2(x2)

            # Layer 3
            x3 = self.layer3(x2)  # 1/16 (输入是 WGN 后的 x2)
            x3_high = None
            if self.use_wgn_enhancement:
                x3, x3_high = self.wgn_enhance3(x3)

            # Layer 4
            x4 = self.layer4(x3)  # 1/32 (输入是 WGN 后的 x3)

            # 准备跳跃连接 (此时 x1, x2, x3 已经是被 WGN 增强过的了)
            s3, s2, s1, s0 = x3, x2, x1, x0

            # === 2. Bottleneck ===
            if self.use_advanced_cafm:
                x_bot = self.cafm(x4)
            else:
                x_bot = self.bottleneck_conv(x4)  # 假设 DoubleConv 已适配

            # === 3. Decoder ===
            d1 = self.up1(x_bot, s3)
            d2 = self.up2(d1, s2)
            d3 = self.up3(d2, s1)
            d4 = self.up4(d3, s0)

            d5 = self.final_up(d4)
            d5 = self.final_conv_block(d5)
            logits = self.outc(d5)

            # 辅助边缘任务
            if self.training and self.use_wgn_enhancement and self.use_edge_loss:
                logits_edge = self.edge_decoder(x3_high, x2_high, x1_high)
                return logits, logits_edge
            else:
                return logits

        else:
            if self.checkpointing:
                x1 = torch.utils.checkpoint.checkpoint(self.inc, x, use_reentrant=False)
                x2 = torch.utils.checkpoint.checkpoint(self.down1, x1, use_reentrant=False)
                x3 = torch.utils.checkpoint.checkpoint(self.down2, x2, use_reentrant=False)
                x4 = torch.utils.checkpoint.checkpoint(self.down3, x3, use_reentrant=False)
                x5 = torch.utils.checkpoint.checkpoint(self.down4, x4, use_reentrant=False)

                if self.use_advanced_cafm:
                    x5 = torch.utils.checkpoint.checkpoint(self.advanced_cafm_bottleneck, x5, use_reentrant=False)

                x = torch.utils.checkpoint.checkpoint(self.up1, x5, x4, use_reentrant=False)
                x = torch.utils.checkpoint.checkpoint(self.up2, x, x3, use_reentrant=False)
                x = torch.utils.checkpoint.checkpoint(self.up3, x, x2, use_reentrant=False)
                x = torch.utils.checkpoint.checkpoint(self.up4, x, x1, use_reentrant=False)
                logits = torch.utils.checkpoint.checkpoint(self.outc, x, use_reentrant=False)
            else:
                x1 = self.inc(x)
                x2 = self.down1(x1)
                x3 = self.down2(x2)
                x4 = self.down3(x3)
                x5 = self.down4(x4)

                if self.use_advanced_cafm:
                    x5 = self.advanced_cafm_bottleneck(x5)

                x = self.up1(x5, x4)
                x = self.up2(x, x3)
                x = self.up3(x, x2)
                x = self.up4(x, x1)
                logits = self.outc(x)
            return logits
            return self.outc(x)

    def use_checkpointing(self):
        self.checkpointing = True

    def disable_checkpointing(self):
        self.checkpointing = False