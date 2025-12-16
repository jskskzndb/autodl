""" Full assembly of the parts to form the complete network """

from .unet_parts import *
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# 添加项目根目录到路径，以便导入 advanced_cafm_module
sys.path.insert(0, str(Path(__file__).parent.parent))
from advanced_cafm_module import Advanced_CAFM
from wgn import Wg_nConv_Block

class EdgeDecoder(nn.Module):
    """
    边缘解码器分支：适配 WGN V3 的 3倍通道输出
    从跳跃连接中提取的高频特征进行边缘重建
    """

    def __init__(self):
        super().__init__()

        # --- Layer 3 High Freq (来自 ResNet Layer 3, 1/16 size) ---
        # WGN V3 输出: 1024 * 3 = 3072 通道
        # 16x16 -> 32x32
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3072, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # --- Layer 2 High Freq (来自 ResNet Layer 2, 1/8 size) ---
        # WGN V3 输出: 512 * 3 = 1536 通道
        # 32x32 + 32x32 -> 64x64
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(512 + 1536, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # --- Layer 1 High Freq (来自 ResNet Layer 1, 1/4 size) ---
        # WGN V3 输出: 256 * 3 = 768 通道
        # 64x64 + 64x64 -> 128x128
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = nn.Sequential(
            nn.Conv2d(256 + 768, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 最终输出层 (从 1/4 尺寸直接上采样到原图)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x3_h, x2_h, x1_h):
        # x3_h: [B, 3072, H/16, W/16]
        x = self.conv1(self.up1(x3_h))  # -> [512, H/8, W/8]

        # x2_h: [B, 1536, H/8, W/8]
        x = torch.cat([x, x2_h], dim=1)
        x = self.conv2(self.up2(x))  # -> [256, H/4, W/4]

        # x1_h: [B, 768, H/4, W/4]
        x = torch.cat([x, x1_h], dim=1)
        x = self.conv3(self.up3(x))  # -> [64, H/2, W/2]

        # 最终上采样回原图 (H/2 -> H)
        # 注意：这里我们目前是在 1/4 尺度，需要 x4 上采样
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        return self.final_conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, use_advanced_cafm=False, use_resnet_encoder=False,
                 use_wgn_enhancement=False, use_edge_loss=False, wgn_orders=None):
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
            # ========== ResNet50 Encoder (Standard - No Stride Hacks) ==========
            from torchvision.models import resnet50, ResNet50_Weights

            # 默认 WGN 参数
            if wgn_orders is None:
                wgn_orders = {
                    'layer1': (3, 2),
                    'layer2': (4, 3),
                    'layer3': (5, 4)
                }

            # 加载预训练权重
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

            # 提取各层 (保持原样 Stride)
            # Input: 512x512
            self.conv1 = resnet.conv1  # Out: 64ch, Stride=2 -> 256x256 (1/2)
            self.bn1 = resnet.bn1
            self.relu = resnet.relu
            self.maxpool = resnet.maxpool  # Out: 64ch, Stride=2 -> 128x128 (1/4)

            self.layer1 = resnet.layer1  # Out: 256ch, Stride=1 -> 128x128 (1/4)
            self.layer2 = resnet.layer2  # Out: 512ch, Stride=2 -> 64x64   (1/8)
            self.layer3 = resnet.layer3  # Out: 1024ch, Stride=2 -> 32x32  (1/16)
            self.layer4 = resnet.layer4  # Out: 2048ch, Stride=2 -> 16x16  (1/32) <-- 启用 Layer 4

            # ========== WGN Enhancement on Skip Connections (侧路增强) ==========
            if use_wgn_enhancement:
                # 直接从 wgn 包导入 (默认指向 __init__.py 里定义的那个)
                from wgn import Wg_nConv_Block
                # 分别对 Layer 1, 2, 3 的跳跃连接进行增强
                # 放在跳跃连接上，不影响主干预训练权重的特征提取
                self.wgn_skip1 = Wg_nConv_Block(256, *wgn_orders['layer1'])
                self.wgn_skip2 = Wg_nConv_Block(512, *wgn_orders['layer2'])
                self.wgn_skip3 = Wg_nConv_Block(1024, *wgn_orders['layer3'])

                # 边缘解码器
                if use_edge_loss:
                    self.edge_decoder = EdgeDecoder()

            # ========== Bottleneck ==========
            # 输入 Layer 4 (2048ch)
            if use_advanced_cafm:
                # CAFM 需要处理 2048 通道
                self.cafm = Advanced_CAFM(n_feat=2048, n_head=8)
            else:
                self.bottleneck_conv = DoubleConv(2048, 2048)

            # ========== Decoder (5 Stages) ==========
            # 因为下采样到了 1/32，我们需要 5 次上采样

            # Stage 1: 1/32 -> 1/16 (Concat Layer3 Skip: 1024ch)
            # In: 2048(Bottleneck) + 1024(Skip)
            self.up1 = Up(in_channels=2048, out_channels=1024, bilinear=bilinear, skip_channels=1024)

            # Stage 2: 1/16 -> 1/8 (Concat Layer2 Skip: 512ch)
            # In: 1024(Prev) + 512(Skip)
            self.up2 = Up(in_channels=1024, out_channels=512, bilinear=bilinear, skip_channels=512)

            # Stage 3: 1/8 -> 1/4 (Concat Layer1 Skip: 256ch)
            # In: 512(Prev) + 256(Skip)
            self.up3 = Up(in_channels=512, out_channels=256, bilinear=bilinear, skip_channels=256)

            # Stage 4: 1/4 -> 1/2 (Concat Stem Skip: 64ch)
            # In: 256(Prev) + 64(Skip from conv1)
            # 注意：Stem层特征对细节恢复很重要
            self.up4 = Up(in_channels=256, out_channels=64, bilinear=bilinear, skip_channels=64)

            # Stage 5: 1/2 -> 1/1 (Final Recovery)
            # 不需要拼接了，直接上采样 + 卷积
            self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.final_conv_block = DoubleConv(64, 64)

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

    def forward(self, x):
        if self.use_resnet_encoder:
            # === 1. Encoder (Standard ResNet50) ===
            # x: [B, 3, 512, 512]

            # Stem
            x0 = self.relu(self.bn1(self.conv1(x)))  # [64, 256, 256] (1/2) -> Skip 0 (Stem)
            x_stem = self.maxpool(x0)  # [64, 128, 128] (1/4)

            # Layers (原汁原味的前向传播，不做任何修改)
            x1 = self.layer1(x_stem)  # [256, 128, 128] (1/4) -> Skip 1
            x2 = self.layer2(x1)  # [512, 64, 64]   (1/8) -> Skip 2
            x3 = self.layer3(x2)  # [1024, 32, 32]  (1/16)-> Skip 3
            x4 = self.layer4(x3)  # [2048, 16, 16]  (1/32)-> Bottleneck Input

            # === 2. Skip Connection Enhancement (WGN 在这里工作!) ===
            # 将 WGN 挂载在跳跃连接上，"清洗"特征，而不干扰主干
            x1_high, x2_high, x3_high = None, None, None

            if self.use_wgn_enhancement:
                # 增强 Layer 3 特征 (给 Up1 用)
                skip3, x3_high = self.wgn_skip3(x3)
                # 增强 Layer 2 特征 (给 Up2 用)
                skip2, x2_high = self.wgn_skip2(x2)
                # 增强 Layer 1 特征 (给 Up3 用)
                skip1, x1_high = self.wgn_skip1(x1)
            else:
                skip3, skip2, skip1 = x3, x2, x1

            # Stem 特征通常保留原样
            skip0 = x0

            # === 3. Bottleneck ===
            if self.use_advanced_cafm:
                x_bot = self.cafm(x4)  # [2048, 16, 16]
            else:
                x_bot = self.bottleneck_conv(x4)  # [2048, 16, 16]

            # === 4. Decoder (5 Stages) ===
            # Stage 1: 1/32 -> 1/16 (与 Skip3 融合)
            d1 = self.up1(x_bot, skip3)

            # Stage 2: 1/16 -> 1/8 (与 Skip2 融合)
            d2 = self.up2(d1, skip2)

            # Stage 3: 1/8 -> 1/4 (与 Skip1 融合)
            d3 = self.up3(d2, skip1)

            # Stage 4: 1/4 -> 1/2 (与 Skip0/Stem 融合)
            d4 = self.up4(d3, skip0)

            # Stage 5: 1/2 -> 1/1 (Final Recovery)
            d5 = self.final_up(d4)
            d5 = self.final_conv_block(d5)

            # === 5. Head ===
            logits = self.outc(d5)

            # 辅助边缘任务 (仅训练时)
            if self.training and self.use_wgn_enhancement and self.use_edge_loss:
                # 将 WGN 提取的高频特征传给边缘解码器
                logits_edge = self.edge_decoder(x3_high, x2_high, x1_high)
                return logits, logits_edge
            else:
                return logits

        else:
            # 原始 Forward 逻辑 (兼容旧代码)
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

    def use_checkpointing(self):
        self.checkpointing = True

    def disable_checkpointing(self):
        self.checkpointing = False