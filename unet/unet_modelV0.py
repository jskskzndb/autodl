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


class EdgeDecoder(nn.Module):
    """
    边缘解码器分支：适配 WGN V3 的 3倍通道输出
    级联上采样，最终输出原图尺寸边缘图
    """

    def __init__(self):
        super().__init__()

        # --- Layer 3 (High Freq) ---
        # WGN V3 输出: 1024 * 3 = 3072 通道
        # 输入 16x16 -> 上采样 -> 32x32
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3072, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # --- Layer 2 (High Freq) ---
        # WGN V3 输出: 512 * 3 = 1536 通道
        # 输入 32x32 + 32x32 -> 拼接 -> 64x64
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(512 + 1536, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # --- Layer 1 (High Freq) ---
        # WGN V3 输出: 256 * 3 = 768 通道
        # 输入 64x64 + 64x64 -> 拼接 -> 128x128
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = nn.Sequential(
            nn.Conv2d(256 + 768, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 最终输出层
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x5_h, x4_h, x3_h):
        # x5_h: [B, 3072, 16, 16]
        x = self.conv1(self.up1(x5_h))  # -> [512, 32, 32]

        # x4_h: [B, 1536, 32, 32]
        x = torch.cat([x, x4_h], dim=1)  # -> [2048, 32, 32]
        x = self.conv2(self.up2(x))  # -> [256, 64, 64]

        # x3_h: [B, 768, 64, 64]
        x = torch.cat([x, x3_h], dim=1)  # -> [1024, 64, 64]
        x = self.conv3(self.up3(x))  # -> [64, 128, 128]

        # 最终上采样回原图 (128 -> 256)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)  # -> [64, 256, 256]
        return self.final_conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, use_advanced_cafm=False, use_resnet_encoder=False, use_wgn_enhancement=False, wgn_orders=None):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_advanced_cafm = use_advanced_cafm
        self.use_resnet_encoder = use_resnet_encoder
        self.use_wgn_enhancement = use_wgn_enhancement
        self.checkpointing = False  # 默认不启用 gradient checkpointing
        
        if use_resnet_encoder:
            # ========== ResNet50编码器 ==========
            from torchvision.models import resnet50, ResNet50_Weights
            
            # 设置默认的WGN order配置
            if wgn_orders is None:
                wgn_orders = {
                    'layer1': (3, 2),  # 256通道，较小的order
                    'layer2': (4, 3),  # 512通道，中等order  
                    'layer3': (5, 4)   # 1024通道，较大的order
                }
            
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            
            # 提取ResNet50各层
            self.conv1 = resnet.conv1      # Out: 64ch, Stride=2 (Size 128x128)
            self.bn1 = resnet.bn1
            self.relu = resnet.relu
            self.maxpool = resnet.maxpool   # Out: 64ch, Stride=2 (Size 64x64)
            
            self.layer1 = resnet.layer1     # Out: 256ch, Stride=1 (Size 64x64)
            self.layer2 = resnet.layer2     # Out: 512ch, 原Stride=2
            self.layer3 = resnet.layer3     # Out: 1024ch, 原Stride=2
            
            # 🔥【关键修改 1】: 强制修改 Layer2 和 Layer3 的 Stride 为 1
            # 这样它们就不会在内部进行下采样了
            self.layer2[0].conv2.stride = (1, 1)
            self.layer2[0].downsample[0].stride = (1, 1)
            
            self.layer3[0].conv2.stride = (1, 1)
            self.layer3[0].downsample[0].stride = (1, 1)

            # 🔥【关键修改 2】: 定义显式的下采样层
            # 顺序: WGN增强 -> 跳跃连接引出 -> 下采样 -> 下一层
            self.explicit_down_to_layer2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.explicit_down_to_layer3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.explicit_down_to_bottleneck = nn.MaxPool2d(kernel_size=2, stride=2)
            
            # ========== WGN增强模块（可选）==========
            if use_wgn_enhancement:
                # 直接从 wgn 包导入 (默认指向 __init__.py 里定义的那个)
                from wgn import Wg_nConv_Block
                self.wgn_enhance1 = Wg_nConv_Block(256, *wgn_orders['layer1'])  # layer1后增强
                self.wgn_enhance2 = Wg_nConv_Block(512, *wgn_orders['layer2'])  # layer2后增强  
                self.wgn_enhance3 = Wg_nConv_Block(1024, *wgn_orders['layer3']) # layer3后增强
                # 初始化边缘解码器
                self.edge_decoder = EdgeDecoder()

            # ========== 瓶颈层处理模块 (CAFM / Conv) ==========
            # 输入来自 Layer3(1024) -> Down -> Bottleneck(1024)
            if use_advanced_cafm:
                self.cafm = Advanced_CAFM(n_feat=1024, n_head=8)
            self.bottleneck_conv = DoubleConv(1024, 1024)
            
            # ========== 解码器（对称结构）==========
            # 按照图示逻辑配置通道数
            
            # Up1: 接收 Bottleneck(1024), 拼接 WGN3/Layer3(1024) -> 输出 512
            self.up1 = Up(
                in_channels=1024,
                out_channels=512,
                bilinear=bilinear,
                skip_channels=1024
            )
            
            # Up2: 接收 Up1(512), 拼接 WGN2/Layer2(512) -> 输出 256
            self.up2 = Up(
                in_channels=512,
                out_channels=256,
                bilinear=bilinear,
                skip_channels=512
            )
            
            # Up3: 接收 Up2(256), 拼接 WGN1/Layer1(256) -> 输出 64
            # 注意: 下一步要拼 Conv1(64)，所以这里输出 64
            self.up3 = Up(
                in_channels=256,
                out_channels=64,
                bilinear=bilinear,
                skip_channels=256
            )
            
            # Up4: 接收 Up3(64), 拼接 Conv1(64) -> 输出 64
            self.up4 = Up(
                in_channels=64,
                out_channels=64,
                bilinear=bilinear,
                skip_channels=64
            )
            
            # ========== 输出层 ==========
            self.outc = OutConv(64, n_classes)
            
        else:
            # ========== 原始U-Net编码器 (保留原始逻辑) ==========
            # 编码器（下采样）部分
            self.inc = (DoubleConv(n_channels, 64))
            self.down1 = (Down(64, 128))
            self.down2 = (Down(128, 256))
            self.down3 = (Down(256, 512))
            factor = 2 if bilinear else 1
            self.down4 = (Down(512, 1024 // factor))
            
            # 条件性地创建 Advanced_CAFM 模块
            if self.use_advanced_cafm:
                bottleneck_channels = 1024 // factor
                self.advanced_cafm_bottleneck = Advanced_CAFM(n_feat=bottleneck_channels)
            
            # 解码器（上采样）部分
            self.up1 = (Up(1024, 512 // factor, bilinear))
            self.up2 = (Up(512, 256 // factor, bilinear))
            self.up3 = (Up(256, 128 // factor, bilinear))
            self.up4 = (Up(128, 64, bilinear))
            self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        if self.use_resnet_encoder:
            # ========== ResNet50编码器前向传播 (重构版) ==========
            # 假设输入: [B, 3, 256, 256]
            
            # 1. Stem (Conv1)
            x1 = self.relu(self.bn1(self.conv1(x))) # [B, 64, 128, 128]
            x1_skip = x1 # 保存用于 Up4
            
            x2 = self.maxpool(x1) # [B, 64, 64, 64] -> 下采样
            
            # 2. Layer 1
            x3 = self.layer1(x2) # [B, 256, 64, 64]
            x3_high = None
            if self.use_wgn_enhancement:
                x3, x3_high = self.wgn_enhance1(x3) # WGN增强
            
            x3_skip = x3 # 保存用于 Up3 (未下采样)
            x3_down = self.explicit_down_to_layer2(x3) # [B, 256, 32, 32] -> 下采样
            
            # 3. Layer 2 (Stride已改1)
            x4 = self.layer2(x3_down) # [B, 512, 32, 32]
            x4_high = None
            if self.use_wgn_enhancement:
                x4, x4_high = self.wgn_enhance2(x4)
                
            x4_skip = x4 # 保存用于 Up2 (未下采样)
            x4_down = self.explicit_down_to_layer3(x4) # [B, 512, 16, 16] -> 下采样
            
            # 4. Layer 3 (Stride已改1)
            x5 = self.layer3(x4_down) # [B, 1024, 16, 16]
            x5_high = None
            if self.use_wgn_enhancement:
                x5, x5_high = self.wgn_enhance3(x5)
            
            x5_skip = x5 # 保存用于 Up1 (未下采样)
            x5_down = self.explicit_down_to_bottleneck(x5) # [B, 1024, 8, 8] -> 下采样
            
            # ========== 瓶颈层处理 ==========
            if self.use_advanced_cafm:
                x_bot = self.cafm(x5_down) # [B, 1024, 8, 8]
            else:
                x_bot = self.bottleneck_conv(x5_down)
            
            # ========== 解码器 (严格对应跳跃连接) ==========
            # Up1: 8->16, Concat x5_skip (Layer3 WGN out)
            x = self.up1(x_bot, x5_skip) # -> [B, 512, 16, 16]
            
            # Up2: 16->32, Concat x4_skip (Layer2 WGN out)
            x = self.up2(x, x4_skip) # -> [B, 256, 32, 32]
            
            # Up3: 32->64, Concat x3_skip (Layer1 WGN out)
            x = self.up3(x, x3_skip) # -> [B, 64, 64, 64]
            
            # Up4: 64->128, Concat x1_skip (Conv1 out)
            x = self.up4(x, x1_skip) # -> [B, 64, 128, 128]
            
            # ========== 输出 ==========
            logits = self.outc(x) # [B, n_classes, 128, 128]
            
            # 最后上采样回原图 (128 -> 256)
            logits = F.interpolate(logits, scale_factor=2, mode='bilinear', align_corners=True)
            
            # 训练时返回边缘解码器结果
            if self.use_wgn_enhancement and self.training:
                logits_edge = self.edge_decoder(x5_high, x4_high, x3_high)
                return logits, logits_edge
            else:
                return logits
        
        else:
            # ========== 原始U-Net前向传播 ==========
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
        """启用梯度检查点"""
        self.checkpointing = True
        
    def disable_checkpointing(self):
        """禁用梯度检查点"""
        self.checkpointing = False