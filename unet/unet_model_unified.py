"""
unet_model_unified.py (Channel Fix Version)
修复：
1. 修正了 self.outc 的输入通道数错误 (从 32 改为 64)，解决了 RuntimeError。
2. 保持了之前所有的双流架构、STRG、WGN 等逻辑。
"""

from .unet_parts import *
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm 

sys.path.insert(0, str(Path(__file__).parent.parent))

# ================================================================
# 1. 动态导入所有可选模块
# ================================================================

# PHD 解码器核心块
try: from decoder.hybrid_decoder import PHD_DecoderBlock
except ImportError: PHD_DecoderBlock = None

# 双流增强模块 (STRG)
try: from .dual_enhance import STRG_Block
except ImportError: STRG_Block = None

# 显式边界流 (Boundary Stream)
try: from .boundary_stream import BoundaryStream
except ImportError: BoundaryStream = None

# WGN (Wavelet Group Norm) - 🔥 保留用于 Encoder 增强
try: from .wgn_module import WGN
except ImportError: WGN = None

# CAFM (Content-Aware Feature Modulation)
try: from .cafm_module import CAFM
except ImportError: CAFM = None


# ================================================================
# 2. 适配器：Up_PHD
# ================================================================
class Up_PHD(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, skip_channels=0, 
                 use_dcn=False, use_dubm=False, use_strg=False):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            conv_in_channels = in_channels + skip_channels
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            conv_in_channels = (in_channels // 2) + skip_channels

        # STRG 模块
        self.use_strg = use_strg and (STRG_Block is not None)
        if self.use_strg and skip_channels > 0:
            self.strg_enhance = STRG_Block(skip_channels=skip_channels, deep_channels=in_channels)

        self.conv = PHD_DecoderBlock(in_channels=conv_in_channels, out_channels=out_channels, use_dcn=use_dcn, use_dubm=use_dubm)

    def forward(self, x1, x2=None, edge_prior=None):
        x1 = self.up(x1)
        
        if x2 is not None:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            if diffX != 0 or diffY != 0:
                x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            
            if self.use_strg:
                x2 = self.strg_enhance(x_skip=x2, x_deep=x1)
            
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        
        return self.conv(x, edge_prior=edge_prior)


# ================================================================
# 3. 统一主模型 UNet
# ================================================================
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, 
                 encoder_name='resnet', decoder_name='phd', cnext_type='convnextv2_tiny', 
                 use_wgn_enhancement=False, use_cafm=False, use_edge_loss=False, wgn_orders=None,
                 use_dcn_in_phd=False, use_dubm=False, use_strg=False,
                 use_dual_stream=False):
        
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        
        self.use_cafm = use_cafm and (CAFM is not None)
        self.use_dual_stream = use_dual_stream and (BoundaryStream is not None)

        # --------------------------------------------------------
        # A. Encoder 初始化
        # --------------------------------------------------------
        self.channels = [] 
        if encoder_name == 'resnet':
            from torchvision.models import resnet50, ResNet50_Weights
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.enc_stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4
            self.channels = [256, 512, 1024, 2048]
            
        elif encoder_name == 'cnextv2':
            self.enc_model = timm.create_model(cnext_type, pretrained=True, features_only=True, out_indices=[0, 1, 2, 3], in_chans=n_channels)
            self.channels = self.enc_model.feature_info.channels()
            
        else:
            self.inc = DoubleConv(n_channels, 64)
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)
            factor = 2 if bilinear else 1
            self.down4 = Down(512, 1024 // factor)
            self.channels = [128, 256, 512, 1024 // factor]

        c1, c2, c3, c4 = self.channels

        # --------------------------------------------------------
        # B. WGN 初始化 (Encoder 增强保留)
        # --------------------------------------------------------
        if use_wgn_enhancement and wgn_orders is not None and WGN is not None:
            print("   ✨ Applying WGN Enhancement (Encoder)...")
            
            def replace_bn_with_wgn(module, order):
                for name, child in module.named_children():
                    if isinstance(child, (nn.BatchNorm2d, nn.GroupNorm)):
                        num_features = child.num_features if isinstance(child, nn.BatchNorm2d) else child.num_channels
                        setattr(module, name, WGN(num_features, order=order))
                    else:
                        replace_bn_with_wgn(child, order)

            if encoder_name == 'resnet':
                replace_bn_with_wgn(self.layer1, wgn_orders['layer1'][0])
                replace_bn_with_wgn(self.layer2, wgn_orders['layer2'][0])
                replace_bn_with_wgn(self.layer3, wgn_orders['layer3'][0])

        # --------------------------------------------------------
        # C. CAFM 初始化
        # --------------------------------------------------------
        if self.use_cafm:
            print("   ✨ Applying CAFM...")
            self.cafm1 = CAFM(c1)
            self.cafm2 = CAFM(c2)
            self.cafm3 = CAFM(c3)
            self.cafm4 = CAFM(c4)

        # --------------------------------------------------------
        # D. 双流架构：边界流初始化
        # --------------------------------------------------------
        if self.use_dual_stream:
            print("   🌊 [Dual-Stream] Initializing Boundary Stream (Explicit Edge)...")
            self.boundary_stream = BoundaryStream(in_channels=c1)

        # --------------------------------------------------------
        # E. Decoder 初始化
        # --------------------------------------------------------
        UpBlock = Up_PHD if decoder_name == 'phd' else Up
        
        if decoder_name == 'phd':
            self.up1 = UpBlock(c4, c3, bilinear, skip_channels=c3, use_dcn=use_dcn_in_phd, use_dubm=use_dubm, use_strg=use_strg)
            self.up2 = UpBlock(c3, c2, bilinear, skip_channels=c2, use_dcn=use_dcn_in_phd, use_dubm=use_dubm, use_strg=use_strg)
            self.up3 = UpBlock(c2, c1, bilinear, skip_channels=c1, use_dcn=use_dcn_in_phd, use_dubm=use_dubm, use_strg=use_strg)
        else:
            self.up1 = UpBlock(c4, c3, bilinear, skip_channels=c3)
            self.up2 = UpBlock(c3, c2, bilinear, skip_channels=c2)
            self.up3 = UpBlock(c2, c1, bilinear, skip_channels=c1)

        # 最后一层上采样：这里输出了 64 通道
        if bilinear:
            self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), DoubleConv(c1, 64))
        else:
            self.up4 = nn.Sequential(nn.ConvTranspose2d(c1, c1 // 2, kernel_size=2, stride=2), DoubleConv(c1 // 2, 64))
        
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 🔥🔥🔥 [修正] 无论是什么 Decoder，up4 都输出了 64 通道，所以这里统一为 64
        self.outc = OutConv(64, n_classes) 

    def forward(self, x):
        # 1. Encoder
        if self.encoder_name == 'cnextv2':
            feats = self.enc_model(x)
            s1, s2, s3, x4 = feats[0], feats[1], feats[2], feats[3]
        elif self.encoder_name == 'resnet':
            x0 = self.enc_stem(x)
            s1 = self.layer1(x0)
            s2 = self.layer2(s1)
            s3 = self.layer3(s2)
            x4 = self.layer4(s3)
        else:
            x1 = self.inc(x)
            s1 = self.down1(x1)
            s2 = self.down2(s1)
            s3 = self.down3(s2)
            x4 = self.down4(s3)

        # 2. CAFM
        if self.use_cafm:
            s1 = self.cafm1(s1)
            s2 = self.cafm2(s2)
            s3 = self.cafm3(s3)
            x4 = self.cafm4(x4)

        # 3. 双流架构
        boundary_logits = None
        if self.use_dual_stream:
            boundary_logits = self.boundary_stream(s1)
        # 🔥🔥🔥 核心修改：截断梯度 🔥🔥🔥
            # 我们传给 Decoder 的时候，加上 .detach()
            # 这样 Decoder 的梯度就不会回传给 boundary_stream
            # boundary_stream 只依靠最后的 edge_loss 来更新自己
            edge_prior_for_dubm = boundary_logits.detach()
        # 4. Decoder
        if self.decoder_name == 'phd':
            d1 = self.up1(x4, s3, edge_prior=boundary_logits)
            d2 = self.up2(d1, s2, edge_prior=boundary_logits)
            d3 = self.up3(d2, s1, edge_prior=boundary_logits)
        else:
            d1 = self.up1(x4, s3)
            d2 = self.up2(d1, s2)
            d3 = self.up3(d2, s1)
            
        d4 = self.up4(d3)
        d5 = self.final_up(d4)
        
        logits = self.outc(d5)

        # 5. 返回逻辑
        if self.training and self.use_dual_stream:
            return logits, boundary_logits
        else:
            return logits