# advanced_cafm_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP_Generator(nn.Module):
    """
    您设计的、基于ASPP的多尺度Q/K/V生成器。
    结构: [并行 3x3 (d=1), 3x3 (d=2), 3x3 (d=4), 3x3 (d=8)] -> Concat -> 1x1 Conv
    并采用了瓶颈设计来提高效率。
    使用4个分支以确保通道数完美整除，同时提供更丰富的多尺度特征。
    """
    def __init__(self, in_channels, out_channels, atrous_rates=[1, 2, 4, 8]):
        super(ASPP_Generator, self).__init__()
        
        # 内部通道数规划
        # 1. 瓶颈降维
        bottleneck_channels = in_channels // 4
        # 2. 每个并行分支的输出通道数
        num_branches = len(atrous_rates)
        if bottleneck_channels % num_branches != 0:
            raise ValueError(f"bottleneck_channels ({bottleneck_channels}) must be divisible by num_branches ({num_branches})")
        branch_channels = bottleneck_channels // num_branches

        # 定义瓶颈层 1x1 卷积 (可选，但推荐用于降低计算量)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU()
        )
        
        # 定义四个并行的多尺度空洞卷积分支
        # padding = dilation, 以保持特征图尺寸不变
        # 4个分支提供更丰富的多尺度特征：感受野从 3×3 到 17×17
        self.branch1 = nn.Conv2d(bottleneck_channels, branch_channels, kernel_size=3, padding=atrous_rates[0], dilation=atrous_rates[0], bias=False)
        self.branch2 = nn.Conv2d(bottleneck_channels, branch_channels, kernel_size=3, padding=atrous_rates[1], dilation=atrous_rates[1], bias=False)
        self.branch3 = nn.Conv2d(bottleneck_channels, branch_channels, kernel_size=3, padding=atrous_rates[2], dilation=atrous_rates[2], bias=False)
        self.branch4 = nn.Conv2d(bottleneck_channels, branch_channels, kernel_size=3, padding=atrous_rates[3], dilation=atrous_rates[3], bias=False)
        
        # 最后的1x1卷积，用于信息融合和恢复维度
        total_branch_channels = branch_channels * num_branches
        self.fusion_conv = nn.Sequential(
            nn.BatchNorm2d(total_branch_channels),
            nn.ReLU(),
            nn.Conv2d(total_branch_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        # 先通过瓶颈层降维
        x = self.bottleneck(x)

        # 并行计算四个分支的输出
        # 分支1: dilation=1, 感受野 3×3 (局部细节)
        out_branch1 = self.branch1(x)
        # 分支2: dilation=2, 感受野 5×5 (近邻上下文)
        out_branch2 = self.branch2(x)
        # 分支3: dilation=4, 感受野 9×9 (中等上下文)
        out_branch3 = self.branch3(x)
        # 分支4: dilation=8, 感受野 17×17 (大范围上下文)
        out_branch4 = self.branch4(x)
        
        # 在通道维度上拼接所有结果，融合多尺度特征
        concat_out = torch.cat([out_branch1, out_branch2, out_branch3, out_branch4], dim=1)
        
        # 通过最终的1x1卷积进行融合
        final_out = self.fusion_conv(concat_out)
        
        return final_out


class Advanced_CAFM(nn.Module):
    """
    您设计的增强版 CAFM 模块，集成在一个单一文件中。
    融合了并行的局部分支、使用ASPP的全局分支、以及长残差连接。
    """
    def __init__(self, n_feat, n_head=8, d_k=None, shuffle_groups=8):
        super(Advanced_CAFM, self).__init__()
        
        if d_k is None:
            # 自动计算每个注意力头的维度
            if n_feat % n_head != 0:
                raise ValueError(f"n_feat ({n_feat}) must be divisible by n_head ({n_head})")
            d_k = n_feat // n_head

        # 保存 shuffle_groups 参数
        self.shuffle_groups = shuffle_groups
        if n_feat % shuffle_groups != 0:
            raise ValueError(f"n_feat ({n_feat}) must be divisible by shuffle_groups ({shuffle_groups})")

        # --- 1. 局部分支 (Local Branch) ---
        # 包含 Channel Shuffle 操作的局部特征提取分支
        self.local_conv1 = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=False),
            nn.BatchNorm2d(n_feat),
            nn.ReLU(inplace=True)
        )
        # Channel Shuffle 将在 forward 中手动调用
        self.local_conv2 = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n_feat)
        )

        # --- 2. 全局分支 (Global Branch) ---
        self.n_head = n_head
        self.d_k = d_k
        qkv_out_channels = n_head * d_k

        # 2.1 特征预处理 1x1 卷积
        # 作用：在生成QKV前，对输入特征进行一次提纯和非线性变换。
        self.conv1_global = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=False)
        
        # 2.2 ASPP-QKV 生成器（4分支多尺度版本）
        # 这是您的核心创新！为Q, K, V分别创建独立的ASPP模块实例。
        # 每个生成器使用4个并行分支（dilation=[1,2,4,8]），提供从局部到全局的多尺度特征
        self.q_generator = ASPP_Generator(in_channels=n_feat, out_channels=qkv_out_channels)
        self.k_generator = ASPP_Generator(in_channels=n_feat, out_channels=qkv_out_channels)
        self.v_generator = ASPP_Generator(in_channels=n_feat, out_channels=qkv_out_channels)
        
        # 2.3 自注意力计算的辅助层
        self.softmax = nn.Softmax(dim=-1)
        
        # 2.4 多头信息融合 1x1 卷积
        # 作用：将多头注意力的结果进行线性组合，融合成一个统一的特征表示。
        self.conv3_global = nn.Conv2d(qkv_out_channels, n_feat, kernel_size=1, bias=False)

    def channel_shuffle(self, x, groups):
        """
        Channel Shuffle 操作，用于在分组卷积后打乱通道顺序，促进不同组之间的信息交流。
        参数：
            x: 输入特征图 [B, C, H, W]
            groups: 分组数
        返回：
            打乱后的特征图 [B, C, H, W]
        """
        batch, channels, height, width = x.size()
        channels_per_group = channels // groups
        
        # reshape: [B, C, H, W] -> [B, groups, channels_per_group, H, W]
        x = x.view(batch, groups, channels_per_group, height, width)
        
        # transpose: [B, groups, channels_per_group, H, W] -> [B, channels_per_group, groups, H, W]
        x = x.transpose(1, 2).contiguous()
        
        # flatten: [B, channels_per_group, groups, H, W] -> [B, C, H, W]
        x = x.view(batch, channels, height, width)
        
        return x

    def forward(self, x):
        # 保存原始输入，用于最终的长残差连接
        input_residual = x
        
        # --- 1. 局部分支计算 ---
        # 第一层卷积
        local_out = self.local_conv1(x)
        # Channel Shuffle 操作
        local_out = self.channel_shuffle(local_out, self.shuffle_groups)
        # 第二层卷积
        local_out = self.local_conv2(local_out)
        
        # --- 2. 全局分支计算 ---
        # 2.1 预处理
        x_global_base = F.relu(self.conv1_global(x), inplace=True)
        
        # 2.2 使用ASPP生成Q, K, V
        q = self.q_generator(x_global_base)
        k = self.k_generator(x_global_base)
        v = self.v_generator(x_global_base)
        
        # 2.3 进行标准的自注意力计算
        b, c, h, w = v.shape
        
        # Reshape for multi-head attention: [B, C, H, W] -> [B, n_head, H*W, d_k]
        q = q.view(b, self.n_head, self.d_k, h * w).permute(0, 1, 3, 2)
        k = k.view(b, self.n_head, self.d_k, h * w)
        v = v.view(b, self.n_head, self.d_k, h * w).permute(0, 1, 3, 2)

        # 计算注意力图: (Q * K^T) / sqrt(d_k) -> Softmax
        attn_map = self.softmax(torch.matmul(q, k) / (self.d_k ** 0.5))

        # 用注意力图加权 V
        attn_out = torch.matmul(attn_map, v)
        
        # Reshape back to image format: [B, n_head, HW, d_k] -> [B, C, H, W]
        attn_out = attn_out.permute(0, 1, 3, 2).reshape(b, -1, h, w)
        
        # 2.4 融合多头信息
        global_out = self.conv3_global(attn_out)
        
        # --- 3. 融合与输出 ---
        # 3.1 应用您设计的长残差连接
        # 将全局分支的输出与最原始的输入相加。
        global_out_with_residual = global_out + input_residual
        
        # 3.2 最终，将局部分支的输出与增强后的全局分支输出相加。
        final_output = local_out + global_out_with_residual
        
        return final_output

if __name__ == '__main__':
    # --- 用于测试模块是否能正常运行的代码 ---
    
    print("=" * 60)
    print("Advanced CAFM Module Test (4-Branch ASPP Version)")
    print("=" * 60)
    
    # 测试用例1: 模拟 U-Net 瓶颈层（bilinear=True 情况）
    # batch_size=4, channels=512, height=32, width=32
    print("\n[Test 1] U-Net Bottleneck with bilinear=True (512 channels)")
    bottleneck_features_512 = torch.randn(4, 512, 32, 32)
    advanced_cafm_512 = Advanced_CAFM(n_feat=512, n_head=8)
    
    print(f"  输入特征图尺寸: {bottleneck_features_512.shape}")
    output_features_512 = advanced_cafm_512(bottleneck_features_512)
    print(f"  输出特征图尺寸: {output_features_512.shape}")
    assert bottleneck_features_512.shape == output_features_512.shape, "输入输出尺寸不匹配！"
    print("  ✅ 尺寸检查通过！")
    
    # 验证通道数整除性（512 → 128 → 32 per branch）
    print(f"  通道分配验证: 512 → bottleneck 128 → 4 branches × 32 = 128 ✅")
    
    # 测试用例2: 模拟 U-Net 瓶颈层（bilinear=False 情况）
    # batch_size=2, channels=1024, height=16, width=16
    print("\n[Test 2] U-Net Bottleneck with bilinear=False (1024 channels)")
    bottleneck_features_1024 = torch.randn(2, 1024, 16, 16)
    advanced_cafm_1024 = Advanced_CAFM(n_feat=1024, n_head=8)
    
    print(f"  输入特征图尺寸: {bottleneck_features_1024.shape}")
    output_features_1024 = advanced_cafm_1024(bottleneck_features_1024)
    print(f"  输出特征图尺寸: {output_features_1024.shape}")
    assert bottleneck_features_1024.shape == output_features_1024.shape, "输入输出尺寸不匹配！"
    print("  ✅ 尺寸检查通过！")
    
    # 验证通道数整除性（1024 → 256 → 64 per branch）
    print(f"  通道分配验证: 1024 → bottleneck 256 → 4 branches × 64 = 256 ✅")
    
    # 计算模型参数量
    num_params_512 = sum(p.numel() for p in advanced_cafm_512.parameters() if p.requires_grad)
    num_params_1024 = sum(p.numel() for p in advanced_cafm_1024.parameters() if p.requires_grad)
    
    print("\n" + "=" * 60)
    print("参数量统计:")
    print(f"  Advanced_CAFM (512 channels): {num_params_512 / 1e6:.2f} M")
    print(f"  Advanced_CAFM (1024 channels): {num_params_1024 / 1e6:.2f} M")
    print("=" * 60)
    print("✅ 所有测试通过！4分支ASPP版本工作正常。")
    print("=" * 60)