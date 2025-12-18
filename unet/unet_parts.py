""" Parts of the U-Net model """
# 本文件定义 UNet 的各个“零件”，供主干 unet_model.py 组装使用
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
   #两次：卷积 -> 批归一化 -> ReLU，经典小模块
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels# 如果没指定中间通道，就等于输出通道
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x) # 顺序执行两次(Conv-BN-ReLU)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    # 先最大池化把尺寸减半，再做一组 DoubleConv
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),# 尺寸 /2
            DoubleConv(in_channels, out_channels) # 通道从 in->out
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    # 先上采样（双线性或反卷积），与编码器对应层拼接，再做 DoubleConv
    def __init__(self, in_channels, out_channels, bilinear=True, skip_channels=None):
        """
        Args:
            in_channels: 解码器输入通道数（上一层的输出）
            out_channels: 本层最终输出通道数
            bilinear: 是否使用双线性插值（False则使用转置卷积）
            skip_channels: 跳跃连接的通道数（用于等通道拼接设计）
        """
        super().__init__()
        
        # 如果没有指定skip_channels，使用默认值（原始逻辑）
        if skip_channels is None:
            skip_channels = in_channels // 2
        
        # bilinear=True：用双线性插值上采样，随后用 1x1/3x3 的普通卷积降通道
        # bilinear=False：直接用反卷积(ConvTranspose2d)完成上采样与部分降通道
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # 如果使用双线性插值，需要1×1卷积降维到与skip相同
            self.channel_match = nn.Conv2d(in_channels, skip_channels, kernel_size=1, bias=False)
            # 拼接后通道数 = skip_channels * 2
            self.conv = DoubleConv(skip_channels * 2, out_channels)
        else:
            # 转置卷积：一步到位，上采样×2同时降维到与skip相同的通道数
            self.up = nn.ConvTranspose2d(in_channels, skip_channels, kernel_size=2, stride=2)
            self.channel_match = nn.Identity()  # 不需要额外操作
            # 拼接后通道数 = skip_channels * 2（等通道拼接）
            self.conv = DoubleConv(skip_channels * 2, out_channels)

    def forward(self, x1, x2):
        # x1：来自更深层的特征（小尺度）
        # x2：来自编码器的跳连特征（大尺度）
        x1 = self.up(x1) # 上采样 x1，使其与 x2 的空间尺寸一致（或接近）
        if x2 is None:
            return self.conv(x1)
        
        # input 是 CHW：计算两者的高宽差，做对称 padding 以精确对齐
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)# 通道维拼接（跳连）- 现在是等通道拼接
        return self.conv(x)# 拼接后再做 DoubleConv，融合信息


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)# 1x1 卷积：把通道数变成类别数

    def forward(self, x):
        return self.conv(x) # 输出每个像素的各类得分（logits）


# ==========================================
# UNet 3+ Core Module (Full-Scale Aggregator)
# ==========================================
class UNet3P_Aggregator(nn.Module):
    def __init__(self, current_level, total_levels, enc_channels_list, prev_dec_channels, cat_channels=64):
        """
        current_level: 当前解码层级索引 (0: s1分辨率, 1: s2分辨率, 2: s3分辨率...)
                       注意：为了配合 ConvNeXt 的 [s1, s2, s3, x4] 列表:
                       Level 0 对应 s1 (H/4)  <- Decoder 最终输出层
                       Level 1 对应 s2 (H/8)
                       Level 2 对应 s3 (H/16)
                       Level 3 对应 x4 (H/32) <- 这是 Bottleneck，不作为 Decoder 目标
        total_levels: 总层数 (通常为 4)
        enc_channels_list: Encoder 通道列表 [c1, c2, c3, c4]
        prev_dec_channels: 上一级 Decoder 输出的通道数
        cat_channels: UNet 3+ 统一压缩通道数 (默认 64)
        """
        super().__init__()
        self.current_level = current_level
        self.cat_channels = cat_channels
        
        # 1. 处理 Encoder 特征 (全尺度)
        self.enc_convs = nn.ModuleList()
        for i, c_in in enumerate(enc_channels_list):
            self.enc_convs.append(nn.Sequential(
                nn.Conv2d(c_in, cat_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(cat_channels),
                nn.ReLU(inplace=True)
            ))
            
        # 2. 处理 Previous Decoder 特征
        # 上一级 Decoder 肯定是更深层的 (Scale 更小)，所以需要 UpSample
        self.dec_conv = nn.Sequential(
            nn.Conv2d(prev_dec_channels, cat_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3. 最大池化层 (用于处理更浅层的 Encoder)
        self.max_pools = nn.ModuleList()
        for i in range(total_levels):
            if i < current_level:
                # 浅层 (大图) -> 深层 (小图): 需要下采样
                # 比如 i=0(s1), current=2(s3). s1->s3 需要 4倍下采样
                stride = 2 ** (current_level - i)
                self.max_pools.append(nn.MaxPool2d(kernel_size=stride, stride=stride, ceil_mode=True))
            else:
                self.max_pools.append(nn.Identity())

    def forward(self, prev_dec_feat, enc_feats_list):
        """
        prev_dec_feat: 来自上一级 Decoder 的特征
        enc_feats_list: [s1, s2, s3, x4]
        """
        target_h, target_w = enc_feats_list[self.current_level].shape[2:]
        feat_list = []
        
        # A. 遍历所有 Encoder 特征
        for i, feat in enumerate(enc_feats_list):
            # 1. 预处理 (Conv 统一通道)
            x = self.enc_convs[i](feat)
            
            # 2. 尺度对齐
            if i < self.current_level:
                # 来源是浅层 (s1) -> 目标是深层 (s3): 下采样
                x = self.max_pools[i](x)
            elif i > self.current_level:
                # 来源是深层 (x4) -> 目标是浅层 (s3): 上采样
                x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=True)
            # i == current_level: 直接使用，不做尺寸变换
            
            # 双重保险: 确保尺寸绝对对齐 (防止除不尽)
            if x.shape[2:] != (target_h, target_w):
                x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=True)
                
            feat_list.append(x)
            
        # B. 处理 Previous Decoder 特征 (深层 -> 浅层)
        x_dec = self.dec_conv(prev_dec_feat)
        x_dec = F.interpolate(x_dec, size=(target_h, target_w), mode='bilinear', align_corners=True)
        feat_list.append(x_dec)
        
        # C. 拼接 (UNet 3+ 核心)
        # 输出通道数 = (Encoder层数 + 1) * cat_channels
        return torch.cat(feat_list, dim=1)