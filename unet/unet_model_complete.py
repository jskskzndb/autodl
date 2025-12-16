"""
unet_model_complete.py
最终完整版：修复了 Standard U-Net 分支缺失 WGN 和边缘监督的问题。
支持全面的消融实验。
"""

from .unet_parts import *
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# 添加项目根目录到路径，以便导入 advanced_cafm_module
sys.path.insert(0, str(Path(__file__).parent.parent))
from advanced_cafm_module import Advanced_CAFM

# 假设你已经建立了 wgn 文件夹并配置了 __init__.py
# 如果没有，请修改为 from wgn_conv_block import Wg_nConv_Block
from wgn import Wg_nConv_Block


class EdgeDecoder(nn.Module):
    """
    通用边缘解码器：动态适配输入通道数
    接收 WGN 提取的三层高频特征 (High, Mid, Low scales)
    """

    def __init__(self, high_ch, mid_ch, low_ch):
        super().__init__()

        # Note: WGN 的高频输出通道通常是输入的 3 倍 (LH, HL, HH)

        # Stage 1: High Scale Processing (e.g., 1/16 size)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(high_ch * 3, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True)
        )

        # Stage 2: Mid Scale Processing (e.g., 1/8 size)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 输入 = 上层上采样结果(mid_ch) + 本层WGN高频(mid_ch * 3)
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_ch + mid_ch * 3, low_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(low_ch),
            nn.ReLU(inplace=True)
        )

        # Stage 3: Low Scale Processing (e.g., 1/4 size)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 输入 = 上层上采样结果(low_ch) + 本层WGN高频(low_ch * 3)
        self.conv3 = nn.Sequential(
            nn.Conv2d(low_ch + low_ch * 3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Final Output (1/2 -> 1/1 size)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x_high, x_mid, x_low):
        # x_high: deepest feature (e.g., 1/16)
        x = self.conv1(self.up1(x_high))

        # x_mid: middle feature (e.g., 1/8)
        x = torch.cat([x, x_mid], dim=1)
        x = self.conv2(self.up2(x))

        # x_low: shallow feature (e.g., 1/4)
        x = torch.cat([x, x_low], dim=1)
        x = self.conv3(self.up3(x))

        # Final Upsample to Original Size (assuming current is 1/2 size)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
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
        self.use_edge_loss = use_edge_loss
        self.checkpointing = False

        # 默认 WGN 阶数配置
        if wgn_orders is None:
            wgn_orders = {'layer1': (3, 2), 'layer2': (4, 3), 'layer3': (5, 4)}

        # =================================================================
        # 分支 1: ResNet50 Encoder
        # =================================================================
        if use_resnet_encoder:
            from torchvision.models import resnet50, ResNet50_Weights
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

            # Encoder Layers (Standard Stride)
            self.conv1 = resnet.conv1  # 64ch, 1/2
            self.bn1 = resnet.bn1
            self.relu = resnet.relu
            self.maxpool = resnet.maxpool  # 64ch, 1/4

            self.layer1 = resnet.layer1  # 256ch, 1/4
            self.layer2 = resnet.layer2  # 512ch, 1/8
            self.layer3 = resnet.layer3  # 1024ch, 1/16
            self.layer4 = resnet.layer4  # 2048ch, 1/32

            # ResNet WGN Setup
            if use_wgn_enhancement:
                self.wgn_skip1 = Wg_nConv_Block(256, *wgn_orders['layer1'])
                self.wgn_skip2 = Wg_nConv_Block(512, *wgn_orders['layer2'])
                self.wgn_skip3 = Wg_nConv_Block(1024, *wgn_orders['layer3'])

                if use_edge_loss:
                    # ResNet channels: High=1024, Mid=512, Low=256
                    self.edge_decoder = EdgeDecoder(high_ch=1024, mid_ch=512, low_ch=256)

            # Bottleneck
            if use_advanced_cafm:
                self.cafm = Advanced_CAFM(n_feat=2048, n_head=8)
            else:
                self.bottleneck_conv = DoubleConv(2048, 2048)

            # Decoder (5 Stages for 1/32 downsampling)
            self.up1 = Up(2048, 1024, bilinear, skip_channels=1024)
            self.up2 = Up(1024, 512, bilinear, skip_channels=512)
            self.up3 = Up(512, 256, bilinear, skip_channels=256)
            self.up4 = Up(256, 128, bilinear, skip_channels=64)  # Concat Stem(64)

            self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.final_conv_block = DoubleConv(128, 64)
            self.outc = OutConv(64, n_classes)

        # =================================================================
        # 分支 2: Standard U-Net Encoder
        # =================================================================
        else:
            self.inc = (DoubleConv(n_channels, 64))
            self.down1 = (Down(64, 128))  # 1/2
            self.down2 = (Down(128, 256))  # 1/4
            self.down3 = (Down(256, 512))  # 1/8
            factor = 2 if bilinear else 1
            self.down4 = (Down(512, 1024 // factor))  # 1/16

            # Standard WGN Setup (修复了之前版本缺失的问题)
            if use_wgn_enhancement:
                # 对应层级: Down3(512), Down2(256), Down1(128)
                self.wgn_skip3 = Wg_nConv_Block(512, *wgn_orders['layer3'])  # High
                self.wgn_skip2 = Wg_nConv_Block(256, *wgn_orders['layer2'])  # Mid
                self.wgn_skip1 = Wg_nConv_Block(128, *wgn_orders['layer1'])  # Low

                if use_edge_loss:
                    # Standard channels: High=512, Mid=256, Low=128
                    self.edge_decoder = EdgeDecoder(high_ch=512, mid_ch=256, low_ch=128)

            # Bottleneck
            if use_advanced_cafm:
                self.advanced_cafm_bottleneck = Advanced_CAFM(n_feat=1024 // factor)

            # Decoder
            self.up1 = (Up(1024, 512 // factor, bilinear))
            self.up2 = (Up(512, 256 // factor, bilinear))
            self.up3 = (Up(256, 128 // factor, bilinear))
            self.up4 = (Up(128, 64, bilinear))
            self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        # =================================================================
        # Forward: ResNet Encoder
        # =================================================================
        if self.use_resnet_encoder:
            # 1. Encoder
            x0 = self.relu(self.bn1(self.conv1(x)))  # Stem, 1/2
            x_stem = self.maxpool(x0)  # 1/4

            x1 = self.layer1(x_stem)  # 1/4, 256ch
            x2 = self.layer2(x1)  # 1/8, 512ch
            x3 = self.layer3(x2)  # 1/16, 1024ch
            x4 = self.layer4(x3)  # 1/32, 2048ch

            # 2. WGN Skip Enhancement
            x1_h, x2_h, x3_h = None, None, None
            if self.use_wgn_enhancement:
                skip3, x3_h = self.wgn_skip3(x3)
                skip2, x2_h = self.wgn_skip2(x2)
                skip1, x1_h = self.wgn_skip1(x1)
            else:
                skip3, skip2, skip1 = x3, x2, x1
            skip0 = x0

            # 3. Bottleneck
            if self.use_advanced_cafm:
                x_bot = self.cafm(x4)
            else:
                x_bot = self.bottleneck_conv(x4)

            # 4. Decoder
            d1 = self.up1(x_bot, skip3)
            d2 = self.up2(d1, skip2)
            d3 = self.up3(d2, skip1)
            d4 = self.up4(d3, skip0)

            d5 = self.final_up(d4)
            d5 = self.final_conv_block(d5)
            logits = self.outc(d5)

            # 5. Edge Output
            if self.training and self.use_wgn_enhancement and self.use_edge_loss:
                logits_edge = self.edge_decoder(x3_h, x2_h, x1_h)
                return logits, logits_edge
            else:
                return logits

        # =================================================================
        # Forward: Standard Encoder
        # =================================================================
        else:
            # 1. Encoder
            x1 = self.inc(x)  # 64ch
            x2 = self.down1(x1)  # 128ch, 1/2
            x3 = self.down2(x2)  # 256ch, 1/4
            x4 = self.down3(x3)  # 512ch, 1/8
            x5 = self.down4(x4)  # 1024ch, 1/16

            # 2. WGN Skip Enhancement (修复了之前版本缺失逻辑)
            x2_h, x3_h, x4_h = None, None, None
            if self.use_wgn_enhancement:
                # 注意：Standard UNet 里的 skip 是 x4, x3, x2
                skip3, x4_h = self.wgn_skip3(x4)  # High (Down3)
                skip2, x3_h = self.wgn_skip2(x3)  # Mid (Down2)
                skip1, x2_h = self.wgn_skip1(x2)  # Low (Down1)
            else:
                skip3, skip2, skip1 = x4, x3, x2

            # 3. Bottleneck
            if self.use_advanced_cafm:
                x5 = self.advanced_cafm_bottleneck(x5)

            # 4. Decoder
            x = self.up1(x5, skip3)
            x = self.up2(x, skip2)
            x = self.up3(x, skip1)
            x = self.up4(x, x1)  # 最浅层通常不加 WGN
            logits = self.outc(x)

            # 5. Edge Output
            if self.training and self.use_wgn_enhancement and self.use_edge_loss:
                # 传入顺序：High(x4_h), Mid(x3_h), Low(x2_h)
                logits_edge = self.edge_decoder(x4_h, x3_h, x2_h)
                return logits, logits_edge
            else:
                return logits

    def use_checkpointing(self):
        self.checkpointing = True

    def disable_checkpointing(self):
        self.checkpointing = False