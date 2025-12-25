"""
unet_model_unified.py (DSIS Skip-Channel Fix + Wavelet Denoising)
修复与增强说明：
1. [新增] 集成 NoiseCleaningSkipBlock (基于小波变换的跳跃连接去噪)。
2. [保留] 修正了 DSIS 通道逻辑，解决了通道不匹配报错。
3. [保留] 完整支持 UNet 3+、PHD Decoder、双流边界流等 SOTA 模块。
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


# ================================================================
# 2. [新增] 小波变换与去噪模块
# ================================================================

class HaarWaveletTransform(nn.Module):
    """ Haar 小波变换工具类 (无需训练) """
    def __init__(self):
        super().__init__()
        pass

    def dwt(self, x):
        # x: [B, C, H, W]
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        
        # Haar 分解公式
        ll = (x00 + x01 + x10 + x11) / 2
        lh = (x00 + x01 - x10 - x11) / 2
        hl = (x00 - x01 + x10 - x11) / 2
        hh = (x00 - x01 - x10 + x11) / 2
        return ll, lh, hl, hh

    def idwt(self, ll, lh, hl, hh):
        # Haar 重构公式
        x00 = (ll + lh + hl + hh) / 2
        x01 = (ll + lh - hl - hh) / 2
        x10 = (ll - lh + hl - hh) / 2
        x11 = (ll - lh - hl + hh) / 2
        
        b, c, h, w = ll.shape
        # 这里的 h, w 是下采样后的尺寸，输出应该是 2h, 2w
        out = torch.zeros(b, c, h * 2, w * 2, device=ll.device, dtype=ll.dtype)
        out[:, :, 0::2, 0::2] = x00
        out[:, :, 0::2, 1::2] = x01
        out[:, :, 1::2, 0::2] = x10
        out[:, :, 1::2, 1::2] = x11
        return out

class NoiseCleaningSkipBlock(nn.Module):
    """
    [满血版] 基于小波变换的跳跃连接去噪模块
    特性：
    1. 不压缩通道 (mid = in_channels) -> 保留最大信息量
    2. 使用标准卷积 (无 groups) -> 全通道交互，去噪更智能
    """
    def __init__(self, in_channels):
        super().__init__()
        self.dwt = HaarWaveletTransform()
        
        # 🔥 修改 1: 不省显存了，直接拉满
        mid = in_channels 
        
        # A. 低频保护流
        self.ll_enhance = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, 1), 
            nn.Sigmoid() 
        )
        
        # B. 高频去噪流
        self.high_process = nn.Sequential(
            # 第一步：先把 LH+HL+HH (3*C) 融合并降维到 C
            nn.Conv2d(in_channels * 3, mid, 1), 
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            
            # 🔥 修改 2: 核心去噪层
            # 删除了 groups 参数 -> 变成标准卷积
            # 作用：利用周围像素 + 所有通道信息，共同判断哪里是噪点
            nn.Conv2d(mid, in_channels * 3, 3, padding=1), 
            
            nn.Sigmoid() # 生成 0~1 的门控
        )
        
        # 融合层
        self.fusion = nn.Conv2d(in_channels * 4, in_channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. 小波分解
        ll, lh, hl, hh = self.dwt.dwt(x)
        
        # 2. 低频增强
        ll_weight = self.ll_enhance(ll)
        ll_clean = ll * (1 + ll_weight)
        
        # 3. 高频去噪
        high_raw = torch.cat([lh, hl, hh], dim=1)
        denoise_mask = self.high_process(high_raw) # 这一步现在是全通道感知的
        high_clean = high_raw * denoise_mask 
        
        # 4. 重组
        all_freqs = torch.cat([ll_clean, high_clean], dim=1)
        out_small = self.fusion(all_freqs)
        
        out = F.interpolate(out_small, size=(H, W), mode='bilinear', align_corners=True)
        
        return x + out


# ================================================================
# 3. UNet 3+ 适配器
# ================================================================
class Up_PHD_3Plus(nn.Module):
    """
    ConvNeXt + UNet 3+ + PHD 完美结合版
    """
    def __init__(self, current_level, total_levels, enc_ch_list, prev_dec_ch, 
                 out_channels, use_dcn=False, use_dubm=False):
        super().__init__()
        
        cat_channels = 64
        self.aggregator = UNet3P_Aggregator(current_level, total_levels, enc_ch_list, prev_dec_ch, cat_channels)
        
        agg_channels = (len(enc_ch_list) + 1) * cat_channels 
        
        self.phd_block = PHD_DecoderBlock(in_channels=agg_channels, out_channels=out_channels, 
                                          use_dcn=use_dcn, use_dubm=use_dubm)

    def forward(self, prev_dec_feat, enc_feats_list, edge_prior=None):
        # Step 1: 全尺度聚合
        x_agg = self.aggregator(prev_dec_feat, enc_feats_list)
        
        # Step 2: PHD 精修
        x_out = self.phd_block(x_agg, edge_prior=edge_prior)
        
        return x_out


# ================================================================
# 4. 标准/PHD 适配器
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
# 5. 统一主模型 UNet
# ================================================================
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, 
                 encoder_name='resnet', decoder_name='phd', cnext_type='convnextv2_tiny', 
                 use_wgn_enhancement=False, use_cafm=False, use_edge_loss=False, wgn_orders=None,
                 use_dcn_in_phd=False, use_dsis=False, use_dubm=False, use_strg=False,
                 use_dual_stream=False, use_unet3p=False, 
                 use_wavelet_denoise=False): # 🔥 [新增参数] 开启小波去噪
        
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
        self.use_unet3p = use_unet3p
        self.use_wavelet_denoise = use_wavelet_denoise # 🔥 保存开关

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
        # [新增] D. 小波去噪模块初始化 (NoiseCleaningSkipBlock)
        # --------------------------------------------------------
        if self.use_wavelet_denoise:
            print("   🌊 [Wavelet] Enabling Skip-Connection Denoising...")
            self.skip_clean1 = NoiseCleaningSkipBlock(c1)
            self.skip_clean2 = NoiseCleaningSkipBlock(c2)
            self.skip_clean3 = NoiseCleaningSkipBlock(c3)

        # --------------------------------------------------------
        # E. DSIS 初始化
        # --------------------------------------------------------
        if self.use_dsis:
            print("   🔗 Applying DSIS (Dual-Stream Interactive Skip)...")
            dsis_channels = 64
            self.dsis_module = DSIS_Module(c1_in=c1, c2_in=c2, c_base=dsis_channels)
            skip_c1 = dsis_channels
            skip_c2 = dsis_channels
        else:
            skip_c1 = c1
            skip_c2 = c2

        # --------------------------------------------------------
        # F. 双流架构：边界流初始化
        # --------------------------------------------------------
        if self.use_dual_stream:
            print("   🌊 [Dual-Stream] Initializing Boundary Stream (Explicit Edge)...")
            self.boundary_stream = BoundaryStream(in_channels=c1)

        # --------------------------------------------------------
        # G. Decoder 初始化
        # --------------------------------------------------------
        if self.use_unet3p:
            print("   🌟 [Architecture] Enabled UNet 3+ Full-Scale Skip Connections (Perfect Mode)")
            # Encoder List: [s1, s2, s3, x4] -> [c1, c2, c3, c4]
            enc_ch_list = [c1, c2, c3, c4]
            total_levels = 4
            
            # Decoder 1 (Level 2) -> Output c3
            self.up1 = Up_PHD_3Plus(current_level=2, total_levels=4, enc_ch_list=enc_ch_list, 
                                    prev_dec_ch=c4, out_channels=c3, 
                                    use_dcn=use_dcn_in_phd, use_dubm=use_dubm)
                                    
            # Decoder 2 (Level 1) -> Output c2
            self.up2 = Up_PHD_3Plus(current_level=1, total_levels=4, enc_ch_list=enc_ch_list, 
                                    prev_dec_ch=c3, out_channels=c2, 
                                    use_dcn=use_dcn_in_phd, use_dubm=use_dubm)
                                    
            # Decoder 3 (Level 0) -> Output c1
            self.up3 = Up_PHD_3Plus(current_level=0, total_levels=4, enc_ch_list=enc_ch_list, 
                                    prev_dec_ch=c2, out_channels=c1, 
                                    use_dcn=use_dcn_in_phd, use_dubm=use_dubm)
                                    
            if bilinear:
                self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), DoubleConv(c1, 64))
            else:
                self.up4 = nn.Sequential(nn.ConvTranspose2d(c1, c1 // 2, kernel_size=2, stride=2), DoubleConv(c1 // 2, 64))

        else:
            UpBlock = Up_PHD if decoder_name == 'phd' else Up
        
            if decoder_name == 'phd':
                self.up1 = UpBlock(c4, c3, bilinear, skip_channels=c3, use_dcn=use_dcn_in_phd, use_dubm=use_dubm, use_strg=use_strg)
                self.up2 = UpBlock(c3, c2, bilinear, skip_channels=skip_c2, use_dcn=use_dcn_in_phd, use_dubm=use_dubm, use_strg=use_strg)
                self.up3 = UpBlock(c2, c1, bilinear, skip_channels=skip_c1, use_dcn=use_dcn_in_phd, use_dubm=use_dubm, use_strg=use_strg)
            else:
                self.up1 = UpBlock(c4, c3, bilinear, skip_channels=c3)
                self.up2 = UpBlock(c3, c2, bilinear, skip_channels=skip_c2)
                self.up3 = UpBlock(c2, c1, bilinear, skip_channels=skip_c1)

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

        # 2. CAFM (可选增强)
        if self.use_cafm:
            s1 = self.cafm1(s1)
            s2 = self.cafm2(s2)
            s3 = self.cafm3(s3)
            x4 = self.cafm4(x4)

        # 🔥 [新增步骤] 3. 小波去噪 (Wavelet Denoising)
        # 放在 CAFM 之后，DSIS 之前，清洗特征
        if self.use_wavelet_denoise:
            s1 = self.skip_clean1(s1)
            s2 = self.skip_clean2(s2)
            s3 = self.skip_clean3(s3)

        # 4. DSIS (可选混合)
        if self.use_dsis:
            s1, s2 = self.dsis_module(s1, s2)

        # 5. 双流边界流 (可选)
        boundary_logits = None
        edge_prior = None
        if self.use_dual_stream:
            boundary_logits = self.boundary_stream(s1)
            edge_prior = boundary_logits.detach()

        # 6. Decoder (PHD / UNet 3+ / Standard)
        if self.use_unet3p:
            enc_list = [s1, s2, s3, x4]
            d1 = self.up1(prev_dec_feat=x4, enc_feats_list=enc_list, edge_prior=edge_prior)
            d2 = self.up2(prev_dec_feat=d1, enc_feats_list=enc_list, edge_prior=edge_prior)
            d3 = self.up3(prev_dec_feat=d2, enc_feats_list=enc_list, edge_prior=edge_prior)
            
            d4 = self.up4(d3)
            d5 = self.final_up(d4)
            logits = self.outc(d5)
            
        else:
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

        # 7. 返回结果
        if self.training and self.use_dual_stream:
            return logits, boundary_logits
        else:
            return logits