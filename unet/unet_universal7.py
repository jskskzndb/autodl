"""
unet/unet_universal1.py
[Universal Model] 全能型 UNet (Dynamic Memory-Augmented ProtoFormer Version)
架构特点:
  1. Spatial Encoder: ConvNeXt V2 (极致纯净的语义提取，关闭双流分支)
  2. Memory Engine: Non-parametric Feature Bank (非参数化记忆库，存储高质量建筑特征)
  3. Momentum Evolution: Teacher-Student 动量更新架构 (完美解决特征漂移与显存泄漏)
  4. L2 Normalization: 严格的单位超球面投影与余弦相似度检索 (提升训练初期的稳定性)
  5. Decoder: Dynamic ProtoFormer (基于检索的动态原型解码器)
  6. Return Logic: 统一返回 List 接口，适配 train02.py 训练循环
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import copy
import math
class DIAM(nn.Module):
    def __init__(self, query_channels, value_channels, K=4):
        """
        K: 每个像素发出的“探测触手”数量 (采样点数)，4 是效果和速度的最佳平衡点
        """
        super().__init__()
        self.K = K
        
        # 1. 偏移量预测头：预测 K 个点的 (x, y) 偏移，输出 2K 个通道
        self.offset_conv = nn.Sequential(
            nn.Conv2d(query_channels, query_channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(query_channels // 2, 2 * K, kernel_size=3, padding=1)
        )
        
        # 2. 注意力权重预测头：预测这 K 个点的重要性，输出 K 个通道
        self.weight_conv = nn.Sequential(
            nn.Conv2d(query_channels, query_channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(query_channels // 2, K, kernel_size=3, padding=1)
        )
        
        # 🔥 [终极防崩技巧] 零初始化 (Zero-Init)
        # 让偏移量和权重在训练初始阶段全为 0，等价于普通的逐像素相加，保护预训练权重不被破坏
        nn.init.zeros_(self.offset_conv[-1].weight)
        nn.init.zeros_(self.offset_conv[-1].bias)
        nn.init.zeros_(self.weight_conv[-1].weight)
        nn.init.zeros_(self.weight_conv[-1].bias)

    def forward(self, q, v):
        """
        q: 深层特征 (Query) [B, C_q, H, W] - 语义准，但边缘糊
        v: 浅层特征 (Value) [B, C_v, H, W] - 边缘准，但噪声大
        """
        B, C_v, H, W = v.shape
        
        # 1. 预测当前分辨率下的偏移量和权重
        offsets = self.offset_conv(q) # [B, 2K, H, W]
        weights = self.weight_conv(q) # [B, K, H, W]
        weights = torch.softmax(weights, dim=1) # 保证 K 个抓取点的权重和为 1
        
        # 2. 生成标准坐标网格 [-1, 1] (配合 grid_sample 的要求)
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, H, dtype=q.dtype, device=q.device),
            torch.linspace(-1, 1, W, dtype=q.dtype, device=q.device),
            indexing='ij'
        )
        base_grid = torch.stack([x_grid, y_grid], dim=-1) # [H, W, 2] 存储 (x,y)
        base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1) # [B, H, W, 2]
        
        out_v = torch.zeros_like(v)
        
        # 3. 动态抓取与融合
        for k in range(self.K):
            # 提取第 k 个探测点的偏移量
            offset_k = offsets[:, 2*k:2*k+2, :, :].permute(0, 2, 3, 1) # [B, H, W, 2]
            
            # 变形网格 = 基础网格 + 偏移量
            grid_k = base_grid + offset_k 
            
            # 使用双线性插值在浅层特征 v 上抓取亚像素
            v_sampled = F.grid_sample(v, grid_k, mode='bilinear', padding_mode='zeros', align_corners=True)
            
            # 乘以对应的注意力权重并累加
            weight_k = weights[:, k:k+1, :, :] # [B, 1, H, W]
            out_v += v_sampled * weight_k
            
        return out_v
# ================================================================
# 1. 核心引擎：非参数化记忆库 (Memory Bank)
# ================================================================

class FeatureMemoryBank(nn.Module):
    """
    存储高质量建筑特征的非参数化队列。
    capacity: 存储的特征向量总数 (建议 4096)
    dim: 特征维度 (统一降维至 128 维以兼顾计算效率与表征能力)
    """
    def __init__(self, capacity=4096, dim=128):
        super().__init__()
        self.capacity = capacity
        self.dim = dim
        
        # 🔥 [修改 B] 初始化时进行 L2 归一化，保证初始状态的向量也在单位超球面上
        init_tensor = F.normalize(torch.randn(capacity, dim), p=2, dim=1)
        
        # 注册为 buffer，随模型权重保存，但切断梯度图
        self.register_buffer("queue", init_tensor)
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def update(self, new_features):
        """
        new_features: [N, dim] 传入的已经是 L2 归一化后的高置信度特征
        """
        batch_size = new_features.shape[0]
        if batch_size == 0:
            return
            
        ptr = int(self.ptr)
        
        # 防止单次入库量超过总容量
        if batch_size > self.capacity:
            new_features = new_features[:self.capacity]
            batch_size = self.capacity

        # 环形队列替换逻辑
        if ptr + batch_size <= self.capacity:
            self.queue[ptr:ptr + batch_size] = new_features
        else:
            rem = self.capacity - ptr
            self.queue[ptr:] = new_features[:rem]
            self.queue[:batch_size - rem] = new_features[rem:]
            
        self.ptr[0] = (ptr + batch_size) % self.capacity

    def forward(self):
        # 🔥 [关键修复] 返回克隆副本并 detach，保护计算图不被后续的 in-place 覆盖破坏
        return self.queue.detach().clone()


# ================================================================
# 2. 动态原型交互模块 (基于检索)
# ================================================================

class DynamicPrototypeInteraction(nn.Module):
    def __init__(self, channels, memory_bank):
        super().__init__()
        self.channels = channels
        self.memory_bank = memory_bank
        
        # 检索键维度 (必须与 memory_bank.dim 保持一致)
        self.key_dim = memory_bank.dim
        
        # QKV 投影层
        self.q_proj = nn.Conv2d(channels, self.key_dim, 1)
        self.k_proj = nn.Linear(self.key_dim, self.key_dim)
        self.v_proj = nn.Linear(self.key_dim, self.key_dim)
        
        self.out_proj = nn.Conv2d(self.key_dim, channels, 1)
        self.norm = nn.BatchNorm2d(channels)
        
        # 局部特征补充
        self.local_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False), 
            nn.BatchNorm2d(channels), 
            nn.GELU()
        )
        # Learnable scale (Zero-init 策略，保证初始状态等价于恒等映射)
        self.gamma = nn.Parameter(torch.ones(channels) * 1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        
        # 1. 提取当前 Query 特征 
        q = self.q_proj(x)
        # 🔥 [关键修改] 检索前：对 Query 进行 L2 归一化，使其与 Memory Bank 处于同一超球面
        q = F.normalize(q, p=2, dim=1)
        q = q.flatten(2).transpose(1, 2) # [B, N, 128]
        
        # 2. 从记忆库获取全局特征作为 Key/Value
        m_features = self.memory_bank() # [Capacity, 128], 已经是单位向量
        
        # 经过 k_proj 后可能会改变模长，因此再次 L2 归一化保证严格的余弦相似度
        k = self.k_proj(m_features)
        k = F.normalize(k, p=2, dim=-1).unsqueeze(0).repeat(B, 1, 1) # [B, Cap, 128]
        
        v = self.v_proj(m_features).unsqueeze(0).repeat(B, 1, 1) # [B, Cap, 128]
        
        # 3. 执行基于检索的交叉注意力 (FP32 Safe 保证数值稳定)
        with torch.cuda.amp.autocast(enabled=False):
            q_32, k_32, v_32 = q.float(), k.float(), v.float()
            
            # 🔥 [关键修改] 因为已经归一化，点积即为余弦相似度，使用 Temperature 放缩 (通常设为 0.07)
            temperature = 0.07
            attn_logits = (q_32 @ k_32.transpose(-2, -1)) / temperature
            
            # 物理截断防止 Softmax 爆炸溢出
            attn_logits = torch.clamp(attn_logits, min=-30, max=30)
            attn = attn_logits.softmax(dim=-1)
            
            # 融合检索到的动态原型
            out = attn @ v_32 
            
        out = out.to(x.dtype).transpose(1, 2).reshape(B, self.key_dim, H, W)
        out = self.out_proj(out)
        out = out + self.local_conv(out)
        
        return self.norm(residual + out * self.gamma.view(1, -1, 1, 1))


# ================================================================
# 3. 完整解码器架构
# ================================================================

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)


class PHD_DecoderBlock_Pro(nn.Module):
    """
    重构后的 PHD Decoder: 引入 Memory Bank 动态检索
    """
    def __init__(self, in_channels, out_channels, memory_bank, depth=2): 
        super().__init__()
        self.align = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True)
        )
        self.layers = nn.ModuleList([])
        self.gamma_ffn = nn.Parameter(torch.ones(depth, out_channels) * 1e-5)
        
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                DynamicPrototypeInteraction(out_channels, memory_bank),
                FeedForward(out_channels, out_channels * 4)
            ]))

    def forward(self, x):
        x = self.align(x)
        for i, (proto_block, ffn) in enumerate(self.layers):
            x = proto_block(x)
            gamma = self.gamma_ffn[i].view(1, -1, 1, 1)
            x = x + gamma * ffn(x)
        return x


class Up_Universal(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, memory_bank):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 🔥 [新增] 实例化 DIAM 模块
        # in_channels 是深层上采样后的通道数，skip_channels 是浅层跳跃连接的通道数
        self.diam = DIAM(query_channels=in_channels, value_channels=skip_channels, K=4)
        # 🔥 [找回丢失的代码] 必须要有这个卷积块，否则 return self.conv(x) 会找不到对象！
        conv_in = in_channels + skip_channels
        self.conv = PHD_DecoderBlock_Pro(conv_in, out_channels, memory_bank, depth=2)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 🔥 标准的 UNet 奇数分辨率对齐逻辑，极度严谨，保证边缘不模糊
        diffY, diffX = x2.size()[2] - x1.size()[2], x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        # 🔥 [核心手术] 使用 DIAM 让浅层特征 x2 主动形变，对齐深层特征 x1
        x2_aligned = self.diam(q=x1, v=x2)
        
        # 通道拼接 (此时 x2_aligned 的边缘已经被完全“拉扯”到了正确的语义边界上)
        x = torch.cat([x2_aligned, x1], dim=1)
        return self.conv(x)


# ================================================================
# 4. 主模型: UniversalUNet (Teacher-Student Memory Edition)
# ================================================================

class UniversalUNet(nn.Module):
    def __init__(self, 
                 n_classes=1, 
                 cnext_type='convnextv2_tiny', 
                 pretrained=True,
                 **kwargs):
        super().__init__()
        self.n_classes = n_classes
        
        print(f"🤖 [Universal Model] Initialized with L2-Normalized Memory Bank & Momentum Update")
        print(f"   - Encoder: {cnext_type} (Pure Spatial)")
        print(f"   - Decoder: Dynamic ProtoFormer (Memory-Augmented)")
        
        # --------------------------------------------------------
        # [A] Student 架构 (用于梯度更新与前向推理)
        # --------------------------------------------------------
        self.spatial_encoder = timm.create_model(cnext_type, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3), drop_path_rate=0.0)
        s_dims = self.spatial_encoder.feature_info.channels() 
        
        self.memory_bank = FeatureMemoryBank(capacity=4096, dim=128)
        self.feat_to_mem = nn.Conv2d(s_dims[3], 128, 1)

        # --------------------------------------------------------
        # [B] Teacher 架构 (影子模型，用于提供稳定的入库特征)
        # --------------------------------------------------------
        self.teacher_encoder = copy.deepcopy(self.spatial_encoder)
        self.teacher_feat_to_mem = copy.deepcopy(self.feat_to_mem)
        
        # 彻底冻结 Teacher 梯度
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False
        for param in self.teacher_feat_to_mem.parameters():
            param.requires_grad = False

        # --------------------------------------------------------
        # [C] 解码器 (传入 memory_bank)
        # --------------------------------------------------------
        self.up1 = Up_Universal(in_channels=s_dims[3], skip_channels=s_dims[2], out_channels=s_dims[2], memory_bank=self.memory_bank)
        self.up2 = Up_Universal(in_channels=s_dims[2], skip_channels=s_dims[1], out_channels=s_dims[1], memory_bank=self.memory_bank)
        self.up3 = Up_Universal(in_channels=s_dims[1], skip_channels=s_dims[0], out_channels=s_dims[0], memory_bank=self.memory_bank)
        
        # 恢复原图尺寸的 4 倍上采样 (结合前面的 8 倍，完美闭环 ConvNeXt 的 32 倍下采样)
        self.final_up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.outc = nn.Conv2d(s_dims[0], n_classes, kernel_size=1)
        # ==========================================================
            # 👇 👇 👇 就是加在这里！ 👇 👇 👇
            # ==========================================================
            # 🔥 [新增] 为解码器的浅层加装输出头，直接使用动态通道数 s_dims
        


        self.use_csaf_loss = False
        


    def forward(self, x, mask=None):
        """
        x: [B, 3, H, W] 图像
        mask: [B, 1, H, W] 真值掩码 (仅在训练时用于指引特征入库)
        """
        # 1. Student Encoder 提取特征
        s_feats = self.spatial_encoder(x)
        
        # 2. Decoder 路径
        d1 = self.up1(s_feats[3], s_feats[2])
        d2 = self.up2(d1, s_feats[1])
        d3 = self.up3(d2, s_feats[0])
        
        d3_up = self.final_up(d3)
        
        # 🔥 [变量名修复]: 这里统一命名为 out_main
        out_main = self.outc(d3_up)
        
        # 3. 训练期间调用 Teacher 提取特征并更新记忆库
        if self.training and mask is not None:
            self._update_memory(x, mask)
            
            
            
            return [out_main]
            
        # 4. 推理和验证时，只返回主预测图
        if self.training:
            return [out_main]
        return out_main


    @torch.no_grad()
    def _update_memory(self, x, mask):
        """
        Teacher 模型提取平滑特征入库，防止特征漂移和显存泄漏，保证 L2 单位超球面映射
        """
        # 1. Teacher 前向传播 (极其稳定)
        t_feats = self.teacher_encoder(x)
        t_bottleneck = t_feats[3]
        
        # 2. 投影并显式切断计算图 (🔥 修改 A：彻底断绝显存泄漏 OOM 隐患)
        feat_low = self.teacher_feat_to_mem(t_bottleneck).detach().clone()
        
        # 3. 🔥 [修改 A 补充] 入库前：对特征维度 (dim=1) 进行 L2 归一化
        feat_low = F.normalize(feat_low, p=2, dim=1)
        
        # 4. 提取建筑正样本特征
        B, C, H, W = feat_low.shape
        # 🔥 [修复: 维度安全防护] 确保 mask 是 4D 张量 [B, 1, H, W]
        mask_float = mask.float()
        if mask_float.ndim == 3:
            mask_float = mask_float.unsqueeze(1)
            
        mask_small = F.interpolate(mask_float, size=(H, W), mode='nearest')
        
        flat_feat = feat_low.permute(0, 2, 3, 1).reshape(-1, 128) # [N, 128]
        flat_mask = mask_small.reshape(-1)
        
        # 筛选像素值为 1 (建筑物) 的索引
        building_indices = (flat_mask > 0.5).nonzero(as_tuple=True)[0]
        
        if building_indices.numel() > 0:
            # 随机采样，防止当前 Batch 的图像霸占整个队列 (设单张图最多贡献 128 个特征)
            max_samples = min(building_indices.numel(), 128)
            perm = torch.randperm(building_indices.numel(), device=feat_low.device)[:max_samples]
            samples = flat_feat[building_indices[perm]]
            
            # 存入记忆库
            self.memory_bank.update(samples)


    @torch.no_grad()
    def momentum_update_teacher(self, current_momentum=0.999):
        """
        在每个 Step (optimizer.step() 之后) 调用此方法，执行 EMA 更新。
        Teacher = m * Teacher + (1 - m) * Student
        """
        # 更新 Encoder
        for param_s, param_t in zip(self.spatial_encoder.parameters(), self.teacher_encoder.parameters()):
            param_t.data = param_t.data * current_momentum + param_s.data * (1. - current_momentum)
            
        # 更新 Projector
        for param_s, param_t in zip(self.feat_to_mem.parameters(), self.teacher_feat_to_mem.parameters()):
            param_t.data = param_t.data * current_momentum + param_s.data * (1. - current_momentum)


# ================================================================
# 5. 训练辅助工具 (供 train02.py 调用)
# ================================================================

def adjust_momentum(epoch, max_epochs, base_m=0.996):
    """
    余弦退火动量调度器 (Cosine Annealing Momentum Scheduler)
    用法: 
      在每个 epoch 开始前: m = adjust_momentum(epoch, max_epochs)
      在 optimizer.step() 后: model.momentum_update_teacher(m)
    """
    return 1. - 0.5 * (1. + math.cos(math.pi * epoch / max_epochs)) * (1. - base_m)

def compute_lightweight_contrastive_loss(proj_feat, mask, memory_bank, num_samples=256):
    """
    🔥 [轻量级超球面对比损失] 极低显存消耗，强行分离建筑与背景特征
    proj_feat: [B, 128, H, W] 当前图像的归一化特征
    mask: [B, 1, H, W] 真值掩码
    memory_bank: FeatureMemoryBank 实例
    """
    B, C, H, W = proj_feat.shape
    
    # 1. 拿出一份记忆库的独立分身 (全部是高纯度建筑特征)
    keys = memory_bank().detach().clone() # [4096, 128]
    
    # 2. 展平特征和真值掩码
    flat_feat = proj_feat.permute(0, 2, 3, 1).reshape(-1, C) # [B*H*W, 128]
    flat_mask = mask.reshape(-1)
    
    # 3. 剥离出 建筑像素 和 背景像素 的索引
    fg_indices = (flat_mask > 0.5).nonzero(as_tuple=True)[0]
    bg_indices = (flat_mask <= 0.5).nonzero(as_tuple=True)[0]
    
    loss_pos = torch.tensor(0.0, device=proj_feat.device)
    loss_neg = torch.tensor(0.0, device=proj_feat.device)
    
    # 4. [物理拉力] 建筑像素必须靠近记忆库
    if fg_indices.numel() > 0:
        # 随机抽样 256 个点，显存消耗近乎为 0
        fg_samples = min(fg_indices.numel(), num_samples)
        perm_fg = torch.randperm(fg_indices.numel(), device=proj_feat.device)[:fg_samples]
        q_fg = flat_feat[fg_indices[perm_fg]] # [256, 128]
        
        sim_fg = torch.mm(q_fg, keys.T) # 算余弦相似度 [-1, 1]
        max_sim_fg = sim_fg.max(dim=1)[0] # 找到字典里最像的那个特征
        loss_pos = (1.0 - max_sim_fg).mean() # 强迫最高相似度逼近 1
        
    # 5. [物理推力] 背景像素必须远离记忆库
    if bg_indices.numel() > 0:
        bg_samples = min(bg_indices.numel(), num_samples)
        perm_bg = torch.randperm(bg_indices.numel(), device=proj_feat.device)[:bg_samples]
        q_bg = flat_feat[bg_indices[perm_bg]] # [256, 128]
        
        sim_bg = torch.mm(q_bg, keys.T) 
        max_sim_bg = sim_bg.max(dim=1)[0]
        # 强迫背景与任何建筑特征的相似度都小于 0 (夹角大于90度)，如果大于 0 就惩罚
        loss_neg = torch.clamp(max_sim_bg - 0.0, min=0.0).mean() 
        
    return loss_pos + loss_neg

def edge_aware_loss(pred_probs, target):
    """
    🔥 FDENet 同款：空间梯度边缘感知损失 (Edge-Aware Loss)
    直接计算水平和垂直方向的像素差分，逼迫模型预测极其锐利的直角边缘。
    pred_probs: 模型预测的概率图 (经过 Sigmoid, 取值 0~1)
    target: 真实标签掩码 (0或1)
    """
    # 1. 计算 X 方向梯度 (水平差异)
    pred_dx = torch.abs(pred_probs[:, :, :, 1:] - pred_probs[:, :, :, :-1])
    target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
    
    # 2. 计算 Y 方向梯度 (垂直差异)
    pred_dy = torch.abs(pred_probs[:, :, 1:, :] - pred_probs[:, :, :-1, :])
    target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
    
    # 3. 边缘填充对齐维度 (补齐最右边和最下边的1个像素)
    pred_dx = F.pad(pred_dx, (0, 1, 0, 0))
    target_dx = F.pad(target_dx, (0, 1, 0, 0))
    pred_dy = F.pad(pred_dy, (0, 0, 0, 1))
    target_dy = F.pad(target_dy, (0, 0, 0, 1))
    
    # 4. 算 L1 绝对值误差
    loss_dx = F.l1_loss(pred_dx, target_dx)
    loss_dy = F.l1_loss(pred_dy, target_dy)
    
    return loss_dx + loss_dy