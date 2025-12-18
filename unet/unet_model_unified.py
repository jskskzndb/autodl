"""
unet_model_unified.py (DSIS Skip-Channel Fix)
修复说明：
1. 修正了 up2 和 up3 初始化时的 skip_channels 参数，使其使用经过 DSIS 判断后的 skip_c2/skip_c1，
   而不是原始的 c2/c1。解决了 'expected 512 channels, but got 320' 的报错。
2. 完整保留了双流、STRG、CAFM 等逻辑。
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

# WGN (Wavelet Group Norm)
try: from .wgn_module import WGN
except ImportError: WGN = None

# CAFM (Content-Aware Feature Modulation)
try: from .cafm_module import CAFM
except ImportError: CAFM = None

# DSIS (Dual-Stream Interactive Skip)
try: from .dsis_module import DSIS_Module
except ImportError: DSIS_Module = None


class Up_PHD_3Plus(nn.Module):
    """
    ConvNeXt + UNet 3+ + PHD 完美结合版
    流程: 
    1. Aggregator: 收集 [s1,s2,s3,x4] + prev_dec -> 统一拼接 (320ch)
    2. PHD Block: 对 320ch 特征进行 Mamba/DCN 精修
    """
    def __init__(self, current_level, total_levels, enc_ch_list, prev_dec_ch, 
                 out_channels, use_dcn=False, use_dubm=False):
        super().__init__()
        
        # 1. 聚合器 (UNet 3+ 核心)
        # 假设 4层Encoder + 1层Decoder，拼接后通道数 = 5 * 64 = 320
        cat_channels = 64
        self.aggregator = UNet3P_Aggregator(current_level, total_levels, enc_ch_list, prev_dec_ch, cat_channels)
        
        agg_channels = (len(enc_ch_list) + 1) * cat_channels # 320
        
        # 2. PHD 解码器 (处理聚合后的特征)
        # 注意: PHD Block 的输入是 agg_channels (320)，输出是 out_channels
        self.phd_block = PHD_DecoderBlock(in_channels=agg_channels, out_channels=out_channels, 
                                          use_dcn=use_dcn, use_dubm=use_dubm)

    def forward(self, prev_dec_feat, enc_feats_list, edge_prior=None):
        # Step 1: 全尺度聚合
        x_agg = self.aggregator(prev_dec_feat, enc_feats_list)
        
        # Step 2: PHD 精修
        # PHD Block 期望接收 (x, edge_prior)。我们将 x_agg 视为输入 x
        x_out = self.phd_block(x_agg, edge_prior=edge_prior)
        
        return x_out
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
            # Padding 对齐
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
                 use_dcn_in_phd=False, use_dsis=False, use_dubm=False, use_strg=False,
                 use_dual_stream=False,
                 use_unet3p=False):
        
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        
        # 初始化各个模块的开关
        self.use_dsis = use_dsis and (DSIS_Module is not None)
        self.use_cafm = use_cafm and (CAFM is not None)
        self.use_dual_stream = use_dual_stream and (BoundaryStream is not None)
        self.use_unet3p = use_unet3p  # 🔥🔥🔥 加上这一行！
        self.use_dsis = use_dsis and (DSIS_Module is not None)
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
        # B. WGN 初始化
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
        # D. DSIS 初始化 (设置 skip_c1 和 skip_c2)
        # --------------------------------------------------------
        if self.use_dsis:
            print("   🔗 Applying DSIS (Dual-Stream Interactive Skip)...")
            dsis_channels = 64 # DSIS 输出固定为 64 通道
            self.dsis_module = DSIS_Module(c1_in=c1, c2_in=c2, c_base=dsis_channels)
            
            # 🔥 这里的计算逻辑是正确的
            skip_c1 = dsis_channels
            skip_c2 = dsis_channels
        else:
            skip_c1 = c1
            skip_c2 = c2

        # --------------------------------------------------------
        # E. 双流架构：边界流初始化
        # --------------------------------------------------------
        if self.use_dual_stream:
            print("   🌊 [Dual-Stream] Initializing Boundary Stream (Explicit Edge)...")
            self.boundary_stream = BoundaryStream(in_channels=c1)

        # --------------------------------------------------------
        # F. Decoder 初始化
        # --------------------------------------------------------
        if self.use_unet3p:
            print("   🌟 [Architecture] Enabled UNet 3+ Full-Scale Skip Connections (Perfect Mode)")
            # UNet 3+ Mode
            # Encoder List: [s1, s2, s3, x4] -> 对应 Channel [c1, c2, c3, c4]
            enc_ch_list = [c1, c2, c3, c4]
            total_levels = 4
            
            # --- Decoder Node 1 (对应 s3 分辨率, Level 2) ---
            # Input: Prev_Decoder(x4/c4), All Encoders
            # Output channels: 随意定义，通常还是保持 c3 或减半。PHD 内部会降维。
            # 这里我们设定输出为 c3 (384 for tiny)，方便后续传递
            self.up1 = Up_PHD_3Plus(current_level=2, total_levels=4, enc_ch_list=enc_ch_list, 
                                    prev_dec_ch=c4, out_channels=c3, 
                                    use_dcn=use_dcn_in_phd, use_dubm=use_dubm)
                                    
            # --- Decoder Node 2 (对应 s2 分辨率, Level 1) ---
            # Input: Prev_Decoder(up1 output, c3), All Encoders
            self.up2 = Up_PHD_3Plus(current_level=1, total_levels=4, enc_ch_list=enc_ch_list, 
                                    prev_dec_ch=c3, out_channels=c2, 
                                    use_dcn=use_dcn_in_phd, use_dubm=use_dubm)
                                    
            # --- Decoder Node 3 (对应 s1 分辨率, Level 0) ---
            # Input: Prev_Decoder(up2 output, c2), All Encoders
            self.up3 = Up_PHD_3Plus(current_level=0, total_levels=4, enc_ch_list=enc_ch_list, 
                                    prev_dec_ch=c2, out_channels=c1, 
                                    use_dcn=use_dcn_in_phd, use_dubm=use_dubm)
                                    
            # UNet 3+ 最终输出的是 c1 通道 (s1 尺寸)，需要再上采样一次回原图
            # 同样使用 DoubleConv 整理
            if bilinear:
                self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), DoubleConv(c1, 64))
            else:
                self.up4 = nn.Sequential(nn.ConvTranspose2d(c1, c1 // 2, kernel_size=2, stride=2), DoubleConv(c1 // 2, 64))

        else:
            UpBlock = Up_PHD if decoder_name == 'phd' else Up
        
            if decoder_name == 'phd':
            # Up1 接收 s3 (c3)，DSIS 不处理 c3，所以 skip 仍为 c3
                self.up1 = UpBlock(c4, c3, bilinear, skip_channels=c3, use_dcn=use_dcn_in_phd, use_dubm=use_dubm, use_strg=use_strg)
            
            # 🔥🔥🔥 [关键修复]：这里必须用 skip_c2，而不是 c2
                self.up2 = UpBlock(c3, c2, bilinear, skip_channels=skip_c2, use_dcn=use_dcn_in_phd, use_dubm=use_dubm, use_strg=use_strg)
            
            # 🔥🔥🔥 [关键修复]：这里必须用 skip_c1，而不是 c1
                self.up3 = UpBlock(c2, c1, bilinear, skip_channels=skip_c1, use_dcn=use_dcn_in_phd, use_dubm=use_dubm, use_strg=use_strg)
            else:
                self.up1 = UpBlock(c4, c3, bilinear, skip_channels=c3)
            # 这里的标准 Decoder 最好也适配一下，虽然你现在主要用 PHD
                self.up2 = UpBlock(c3, c2, bilinear, skip_channels=skip_c2)
                self.up3 = UpBlock(c2, c1, bilinear, skip_channels=skip_c1)

        # 最后一层
            if bilinear:
                self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), DoubleConv(c1, 64))
            else:
                self.up4 = nn.Sequential(nn.ConvTranspose2d(c1, c1 // 2, kernel_size=2, stride=2), DoubleConv(c1 // 2, 64))
        
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
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

        # 3. DSIS (注意：DSIS 和 UNet3+ 通道逻辑可能冲突，UNet3+ 时建议关闭 DSIS)
        if self.use_dsis:
            s1, s2 = self.dsis_module(s1, s2)

        # 4. 双流
        boundary_logits = None
        edge_prior = None
        if self.use_dual_stream:
            boundary_logits = self.boundary_stream(s1)
            edge_prior = boundary_logits.detach()

        # 5. Decoder (核心修复点：增加分支判断)
        if self.use_unet3p:
            # === 🔥 UNet 3+ 专用路径 (全尺度聚合) ===
            # 将所有特征打包成列表: [Scale0(s1), Scale1(s2), Scale2(s3), Scale3(x4)]
            enc_list = [s1, s2, s3, x4]
            
            # Decoder 1: 恢复到 s3 尺度
            d1 = self.up1(prev_dec_feat=x4, enc_feats_list=enc_list, edge_prior=edge_prior)
            
            # Decoder 2: 恢复到 s2 尺度
            d2 = self.up2(prev_dec_feat=d1, enc_feats_list=enc_list, edge_prior=edge_prior)
            
            # Decoder 3: 恢复到 s1 尺度
            d3 = self.up3(prev_dec_feat=d2, enc_feats_list=enc_list, edge_prior=edge_prior)
            
            # Final Up
            d4 = self.up4(d3)
            d5 = self.final_up(d4)
            logits = self.outc(d5)
            
        else:
            # === 普通路径 (级联解码) ===
            if self.decoder_name == 'phd':
                # 注意：如果 use_dual_stream 是 False，boundary_logits 就是 None
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

        # 6. 返回逻辑
        if self.training and self.use_dual_stream:
            return logits, boundary_logits
        else:
            return logits

      