"""
unet_model.py
集成 PHD (Parallel Hybrid Decoder) + ResNet Encoder + WGN (Encoder-Series)
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

# 1. 导入 WGN (WGN 增强模块)
from wgn import Wg_nConv_Block

# 🔥 2. [关键修复] 导入你的新解码器 PHD
from decoder.hybrid_decoder import PHD_DecoderBlock


# 🔥 3. [关键修复] 定义 Up_PHD 适配器
# 用来替代原来的 'Up' 类，把 PHD_DecoderBlock 包装成能进行上采样和拼接的模块
class Up_PHD(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, skip_channels=0):
        super().__init__()
        
        # A. 上采样 (Upsample)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # 双线性插值不改变通道数
            conv_in_channels = in_channels + skip_channels
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # 转置卷积会让通道减半
            conv_in_channels = (in_channels // 2) + skip_channels

        # B. 特征融合与解码 (使用 PHD 替代普通 DoubleConv)
        self.conv = PHD_DecoderBlock(in_channels=conv_in_channels, out_channels=out_channels)

    def forward(self, x1, x2):
        # x1: 深层特征 (需要上采样)
        # x2: 跳跃连接特征 (High Res)
        x1 = self.up(x1)
        
        # 处理尺寸不匹配 (Padding)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # 拼接
        x = torch.cat([x2, x1], dim=1)
        
        # 混合解码
        return self.conv(x)


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
        # Final Output
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x3_h, x2_h, x1_h):
        x = self.conv1(self.up1(x3_h))
        x = torch.cat([x, x2_h], dim=1)
        x = self.conv2(self.up2(x))
        x = torch.cat([x, x1_h], dim=1)
        x = self.conv3(self.up3(x))
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
        self.use_edge_loss = use_edge_loss
        self.checkpointing = False

        if use_resnet_encoder:
            # ========== ResNet50 Encoder ==========
            from torchvision.models import resnet50, ResNet50_Weights

            if wgn_orders is None:
                wgn_orders = {'layer1': (3, 2), 'layer2': (4, 3), 'layer3': (5, 4)}

            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.conv1 = resnet.conv1
            self.bn1 = resnet.bn1
            self.relu = resnet.relu
            self.maxpool = resnet.maxpool
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4

            # ========== WGN Enhancement (串联在 Encoder 中) ==========
            if use_wgn_enhancement:
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

            # ========== Decoder ==========
            # 🔥 4. [关键修复] 使用 Up_PHD 替代 Up
            # 只有这里改了，才能用到你的 hybrid_decoder.py
            self.up1 = Up_PHD(2048, 1024, bilinear, skip_channels=1024)
            self.up2 = Up_PHD(1024, 512, bilinear, skip_channels=512)
            self.up3 = Up_PHD(512, 256, bilinear, skip_channels=256)
            self.up4 = Up_PHD(256, 128, bilinear, skip_channels=64)

            self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.final_conv_block = DoubleConv(128, 64)
            self.outc = OutConv(64, n_classes)

        else:
            # ========== Standard Encoder ==========
            self.inc = (DoubleConv(n_channels, 64))
            self.down1 = (Down(64, 128))
            self.down2 = (Down(128, 256))
            self.down3 = (Down(256, 512))
            factor = 2 if bilinear else 1
            self.down4 = (Down(512, 1024 // factor))

            if self.use_advanced_cafm:
                bottleneck_channels = 1024 // factor
                self.advanced_cafm_bottleneck = Advanced_CAFM(n_feat=bottleneck_channels)

            # 这里可以保留标准解码器，作为 Baseline 对比
            self.up1 = (Up(1024, 512 // factor, bilinear))
            self.up2 = (Up(512, 256 // factor, bilinear))
            self.up3 = (Up(256, 128 // factor, bilinear))
            self.up4 = (Up(128, 64, bilinear))
            self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        if self.use_resnet_encoder:
            # ... (这部分逻辑是你原本的逻辑，保持不变) ...
            x0 = self.relu(self.bn1(self.conv1(x)))
            x_stem = self.maxpool(x0)

            x1 = self.layer1(x_stem)
            x1_high = None
            if self.use_wgn_enhancement:
                x1, x1_high = self.wgn_enhance1(x1)

            x2 = self.layer2(x1)
            x2_high = None
            if self.use_wgn_enhancement:
                x2, x2_high = self.wgn_enhance2(x2)

            x3 = self.layer3(x2)
            x3_high = None
            if self.use_wgn_enhancement:
                x3, x3_high = self.wgn_enhance3(x3)

            x4 = self.layer4(x3)

            s3, s2, s1, s0 = x3, x2, x1, x0

            if self.use_advanced_cafm:
                x_bot = self.cafm(x4)
            else:
                x_bot = self.bottleneck_conv(x4)

            d1 = self.up1(x_bot, s3)
            d2 = self.up2(d1, s2)
            d3 = self.up3(d2, s1)
            d4 = self.up4(d3, s0)

            d5 = self.final_up(d4)
            d5 = self.final_conv_block(d5)
            logits = self.outc(d5)

            if self.training and self.use_wgn_enhancement and self.use_edge_loss:
                logits_edge = self.edge_decoder(x3_high, x2_high, x1_high)
                return logits, logits_edge
            else:
                return logits

        else:
            # Standard UNet Forward (保持不变)
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

    def use_checkpointing(self):
        self.checkpointing = True

    def disable_checkpointing(self):
        self.checkpointing = False