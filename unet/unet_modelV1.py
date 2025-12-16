""" Full assembly of the parts to form the complete network """

from .unet_parts import *
# 从同包下的 unet_parts.py 导入所有构件（DoubleConv / Down / Up / OutConv）
import sys
from pathlib import Path
# 添加项目根目录到路径，以便导入 advanced_cafm_module
sys.path.insert(0, str(Path(__file__).parent.parent))
from advanced_cafm_module import Advanced_CAFM


# 在 unet_model.py 的 class UNet(nn.Module): 之前插入以下类

class EdgeDecoder(nn.Module):
    """
    边缘解码器分支：适配 WGN V3 的 3倍通道输出
    """

    def __init__(self):
        super().__init__()

        # --- Layer 3 (High Freq) ---
        # WGN V3 输出: 1024 * 3 = 3072 通道
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Sequential(
            # 修改点 1: 输入通道从 1024 改为 3072 (1024*3)
            nn.Conv2d(3072, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # --- Layer 2 (High Freq) ---
        # WGN V3 输出: 512 * 3 = 1536 通道
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = nn.Sequential(
            # 修改点 2: 拼接后通道数 = 上层下来的(512) + 本层WGN的(1536) = 2048
            nn.Conv2d(512 + 1536, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # --- Layer 1 (High Freq) ---
        # WGN V3 输出: 256 * 3 = 768 通道
        self.up3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 直接上采样4倍
        self.conv3 = nn.Sequential(
            # 修改点 3: 拼接后通道数 = 上层下来的(256) + 本层WGN的(768) = 1024
            nn.Conv2d(256 + 768, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 最终输出
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x5_h, x4_h, x3_h):
        # x5_h: [B, 3072, 16, 16]
        x = self.conv1(self.up1(x5_h))  # -> [512, 32, 32]

        # x4_h: [B, 1536, 32, 32]
        x = torch.cat([x, x4_h], dim=1)  # -> [2048, 32, 32]
        x = self.conv2(self.up2(x))  # -> [256, 64, 64]

        # x3_h: [B, 768, 64, 64]
        x = torch.cat([x, x3_h], dim=1)  # -> [1024, 64, 64]
        x = self.conv3(x)  # -> [64, 64, 64]

        # 最终上采样回原图
        x = self.up3(x)  # -> [64, 256, 256]
        return self.final_conv(x)
# ---------------------------------------------------------
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, use_advanced_cafm=False, use_resnet_encoder=False, use_wgn_enhancement=False, wgn_orders=None):
        super(UNet, self).__init__()
        # n_channels: 输入图像的通道数（RGB=3）
        # n_classes: 输出类别数（语义分割的类别数量；二分类=2 或 1 视实现而定）
        # bilinear: 上采样方式，True=双线性插值，False=反卷积（ConvTranspose2d）
        # use_advanced_cafm: 是否在瓶颈层使用 Advanced_CAFM 模块进行特征增强
        # use_resnet_encoder: 是否使用ResNet50作为编码器
        # use_wgn_enhancement: 是否在ResNet50编码器各层后添加WGN增强
        # wgn_orders: WGN块的order配置，格式为{'layer1': (order_low, order_high), ...}
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
            import torch.nn.functional as F
            
            # 设置默认的WGN order配置
            if wgn_orders is None:
                wgn_orders = {
                    'layer1': (3, 2),  # 256通道，较小的order
                    'layer2': (4, 3),  # 512通道，中等order  
                    'layer3': (5, 4)   # 1024通道，较大的order
                }
            
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            
            # 提取ResNet50各层（舍弃layer4）
            self.conv1 = resnet.conv1      # 输出: 64通道, stride=2
            self.bn1 = resnet.bn1
            self.relu = resnet.relu
            self.maxpool = resnet.maxpool   # stride=2
            self.layer1 = resnet.layer1     # 输出: 256通道, stride=1
            self.layer2 = resnet.layer2     # 输出: 512通道, stride=2
            self.layer3 = resnet.layer3     # 输出: 1024通道, stride=2
            # layer4不使用
            
            # ========== WGN增强模块（可选）==========
            if use_wgn_enhancement:
                # 直接从 wgn 包导入 (默认指向 __init__.py 里定义的那个)
                from wgn import Wg_nConv_Block
                self.wgn_enhance1 = Wg_nConv_Block(256, *wgn_orders['layer1'])  # layer1后增强
                self.wgn_enhance2 = Wg_nConv_Block(512, *wgn_orders['layer2'])  # layer2后增强  
                self.wgn_enhance3 = Wg_nConv_Block(1024, *wgn_orders['layer3']) # layer3后增强
                # 🔥【新增】初始化边缘解码器
                self.edge_decoder = EdgeDecoder()
            # ========== 输入特征分支（用于最后的跳跃连接）==========
            self.input_branch = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
            
            # ========== 瓶颈层处理模块（开关控制）==========
            # use_advanced_cafm=True: 使用CAFM注意力增强
            if use_advanced_cafm:
                self.cafm = Advanced_CAFM(n_feat=1024, n_head=8)
            # use_advanced_cafm=False: 使用传统卷积块（基线对比）
            self.bottleneck_conv = DoubleConv(1024, 1024)
            
            # ========== 解码器（标准UNet对称结构，转置卷积版本）==========
            # up1: 16×16 → 32×32，拼接layer2输出(512通道)
            self.up1 = Up(
                in_channels=1024,    # 瓶颈层输出1024通道
                out_channels=512,
                bilinear=bilinear,
                skip_channels=512    # layer2(x4)的通道数
            )
            
            # up2: 32×32 → 64×64，拼接layer1输出(256通道)
            self.up2 = Up(
                in_channels=512,
                out_channels=256,
                bilinear=bilinear,
                skip_channels=256    # layer1(x3)的通道数
            )
            
            # up3: 64×64 → 128×128，拼接conv1输出(64通道) ⭐新设计！
            self.up3 = Up(
                in_channels=256,
                out_channels=64,
                bilinear=bilinear,
                skip_channels=64     # conv1(x1)的通道数
            )
            
            # up4: 128×128 → 256×256，拼接input_branch输出(64通道)
            self.up4 = Up(
                in_channels=64,
                out_channels=64,
                bilinear=bilinear,
                skip_channels=64     # input_branch的通道数
            )
            
            # ========== 输出层 ==========
            self.outc = OutConv(64, n_classes)
            
        else:
            # ========== 原始U-Net编码器 ==========
            # 编码器（下采样）部分：每走一层，特征通道数增加，空间分辨率减半
            self.inc = (DoubleConv(n_channels, 64))# 第一层：输入通道 -> 64，做两次(Conv-BN-ReLU)
            self.down1 = (Down(64, 128))# 下采样到 1/2，通道 64->128
            self.down2 = (Down(128, 256)) # 下采样到 1/4，通道 128->256
            self.down3 = (Down(256, 512)) # 下采样到 1/8，通道 256->512
            factor = 2 if bilinear else 1  # 若用双线性上采样，为了保持参数量，通道减半
            self.down4 = (Down(512, 1024 // factor))  # 下采样到 1/16，通道 512->(1024//factor)
            
            # 条件性地创建 Advanced_CAFM 模块用于瓶颈层增强
            if self.use_advanced_cafm:
                bottleneck_channels = 1024 // factor
                self.advanced_cafm_bottleneck = Advanced_CAFM(n_feat=bottleneck_channels)
            
            # 解码器（上采样）部分：每走一层，上采样 + 拼接跳连（skip-connection）+ 双卷积
            self.up1 = (Up(1024, 512 // factor, bilinear))# 由最底部向上，通道合并后再降到 512//factor
            self.up2 = (Up(512, 256 // factor, bilinear))# 再向上：512 -> 256//factor
            self.up3 = (Up(256, 128 // factor, bilinear))# 再向上：256 -> 128//factor
            self.up4 = (Up(128, 64, bilinear))# 再向上：128 -> 64
            self.outc = (OutConv(64, n_classes)) # 最后一层 1x1 卷积，把通道数变成类别数

    def forward(self, x):
        if self.use_resnet_encoder:
            # ========== ResNet50编码器前向传播 ==========
            import torch.nn.functional as F
            
            # 输入特征提取（用于最后的跳跃连接）
            input_features = self.input_branch(x)  # [B, 64, H, W] (256×256)
            
            # ResNet50编码器
            x1 = self.relu(self.bn1(self.conv1(x)))  # [B, 64, H/2, W/2] (128×128)
            x2 = self.maxpool(x1)                     # [B, 64, H/4, W/4] (64×64)
            x3 = self.layer1(x2)                      # [B, 256, H/4, W/4] (64×64)
            x3_high = None  # 占位符

            # WGN增强layer1输出（如果启用）
            if self.use_wgn_enhancement:
                x3, x3_high= self.wgn_enhance1(x3)           # [B, 256, H/4, W/4] (64×64) WGN增强
            
            x4 = self.layer2(x3)                      # [B, 512, H/8, W/8] (32×32)
            
            # WGN增强layer2输出（如果启用）
            if self.use_wgn_enhancement:
                x4, x4_high = self.wgn_enhance2(x4)           # [B, 512, H/8, W/8] (32×32) WGN增强
            
            x5 = self.layer3(x4)                      # [B, 1024, H/16, W/16] (16×16)
            x5_high = None

            # WGN增强layer3输出（如果启用）
            if self.use_wgn_enhancement:
                x5, x5_high = self.wgn_enhance3(x5)           # [B, 1024, H/16, W/16] (16×16) WGN增强
            
            # ========== 瓶颈层处理（开关控制）==========
            if self.use_advanced_cafm:
                # 使用CAFM注意力增强
                x5 = self.cafm(x5)  # [B, 1024, H/16, W/16] (16×16)
            else:
                # 使用传统卷积块（基线对比）
                x5 = self.bottleneck_conv(x5)  # [B, 1024, H/16, W/16] (16×16)
            
            # ========== 解码器（标准UNet对称结构）==========
            # up1: 上采样到32×32，拼接layer2输出(x4)
            x = self.up1(x5, x4)  # [B, 512, H/8, W/8] (32×32)
            
            # up2: 上采样到64×64，拼接layer1输出(x3)
            x = self.up2(x, x3)  # [B, 256, H/4, W/4] (64×64)
            
            # up3: 上采样到128×128，拼接conv1输出(x1) ⭐新设计！
            x = self.up3(x, x1)  # [B, 64, H/2, W/2] (128×128)
            
            # up4: 上采样到256×256，拼接input_branch输出
            x = self.up4(x, input_features)  # [B, 64, H, W] (256×256)
            
            # ========== 输出 ==========
            logits = self.outc(x)  # [B, n_classes, H, W] (256×256)
            # 🔥【修改】如果是训练模式且开了WGN，返回双结果
            if self.use_wgn_enhancement and self.training:
                logits_edge = self.edge_decoder(x5_high, x4_high, x3_high)
                return logits, logits_edge
            else:
                return logits
        
        else:
            # ========== 原始U-Net前向传播 ==========
            # 编码路径：一路下采样并保存中间结果用于跳连
            if self.checkpointing:
                # 使用 gradient checkpointing 节省显存
                x1 = torch.utils.checkpoint.checkpoint(self.inc, x, use_reentrant=False)
                x2 = torch.utils.checkpoint.checkpoint(self.down1, x1, use_reentrant=False)
                x3 = torch.utils.checkpoint.checkpoint(self.down2, x2, use_reentrant=False)
                x4 = torch.utils.checkpoint.checkpoint(self.down3, x3, use_reentrant=False)
                x5 = torch.utils.checkpoint.checkpoint(self.down4, x4, use_reentrant=False)
                
                # 如果启用了 Advanced_CAFM，对瓶颈层特征进行增强
                if self.use_advanced_cafm:
                    x5 = torch.utils.checkpoint.checkpoint(self.advanced_cafm_bottleneck, x5, use_reentrant=False)
                
                # 解码路径：上采样 + 与对应编码层的特征图拼接（U 形结构的"跳连"）
                x = torch.utils.checkpoint.checkpoint(self.up1, x5, x4, use_reentrant=False)
                x = torch.utils.checkpoint.checkpoint(self.up2, x, x3, use_reentrant=False)
                x = torch.utils.checkpoint.checkpoint(self.up3, x, x2, use_reentrant=False)
                x = torch.utils.checkpoint.checkpoint(self.up4, x, x1, use_reentrant=False)
                logits = torch.utils.checkpoint.checkpoint(self.outc, x, use_reentrant=False)
            else:
                # 正常前向传播
                x1 = self.inc(x)# 尺寸不变，通道 64
                x2 = self.down1(x1)# 空间 1/2，通道 128
                x3 = self.down2(x2) # 空间 1/4，通道 256
                x4 = self.down3(x3)# 空间 1/8，通道 512
                x5 = self.down4(x4)# 空间 1/16，通道 1024/factor（瓶颈层）
                
                # 如果启用了 Advanced_CAFM，对瓶颈层特征进行增强
                if self.use_advanced_cafm:
                    x5 = self.advanced_cafm_bottleneck(x5)
                
                # 解码路径：上采样 + 与对应编码层的特征图拼接（U 形结构的"跳连"）
                x = self.up1(x5, x4)# 用 x4 做跳连
                x = self.up2(x, x3) # 用 x3 做跳连
                x = self.up3(x, x2)# 用 x2 做跳连
                x = self.up4(x, x1)# 用 x1 做跳连
                logits = self.outc(x)# 1x1 卷积：把通道数变成类别数（每个像素输出各类的得分）
            return logits# 返回网络输出（未经过激活；训练时交给 Loss/后处理）

    def use_checkpointing(self):
        """启用梯度检查点：用计算换显存，适合显存受限的情况"""
        self.checkpointing = True
        
    def disable_checkpointing(self):
        """禁用梯度检查点：正常前向传播，更快但占用更多显存"""
        self.checkpointing = False