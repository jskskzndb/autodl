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
