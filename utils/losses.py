# utils/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    一个更健壮、注释更清晰的 Focal Loss 实现，用于二分类分割任务。
    这个损失函数是 BCEWithLogitsLoss 的一个增强，通过降低对易分类样本的权重，
    帮助模型专注于学习困难样本（如边界、小目标），从而解决类别不平衡问题。
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean', epsilon: float = 1e-7):
        """
        Focal Loss 的初始化函数。

        Args:
            alpha (float): 类别权重因子。用于平衡正负样本的重要性。默认值0.25通常用于目标检测，
                           对于分割任务，特别是前景像素较少时，可能需要设为 > 0.5 的值。
            gamma (float): 聚焦参数。gamma > 0 会降低易分类样本的权重，使其专注于困难样本。
                           通常设为2.0。
            reduction (str): 指定应用于输出的规约方法: 'none' | 'mean' | 'sum'。
                             'mean': 输出的loss会是所有样本loss的平均值。
            epsilon (float): 一个极小的数，用于 clamp 操作，防止 p_t 严格等于0或1，增加数值稳定性。
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        前向传播计算 Focal Loss。

        Args:
            inputs (torch.Tensor): 模型的原始输出 (logits)，未经 sigmoid 激活。
                                   期望形状: [B, H, W] 或 [B, 1, H, W]。
            targets (torch.Tensor): 真实标签 (Ground Truth)。
                                    期望形状: [B, H, W] 或 [B, 1, H, W]。
        """
        # --- 建议 1: 确保 targets 的类型正确 ---
        # 内部强制转换为 float 类型，增强模块的健壮性，避免因输入类型错误导致崩溃。
        targets = targets.float()

        # 如果 inputs 的形状是 [B, 1, H, W]，则去掉 channel 维度以匹配 targets
        if inputs.dim() == 4 and inputs.shape[1] == 1:
            inputs = inputs.squeeze(1)

        # --- 核心计算 ---
        # 使用 BCEWithLogitsLoss 来获取基础的二元交叉熵损失，这是数值最稳定的做法。
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 计算概率 p，并通过 sigmoid 将 logits 转换为概率
        p = torch.sigmoid(inputs)
        
        # 计算 p_t，即模型对正确类别的预测概率
        # 如果真实标签是1, p_t = p; 如果真实标签是0, p_t = 1-p
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # --- 建议 2: 增加数值稳定性 ---
        # 使用 clamp 防止 p_t 过于接近0或1，避免 log(0) 或 (1-p_t) 幂运算的数值问题。
        p_t = torch.clamp(p_t, self.epsilon, 1.0 - self.epsilon)
        
        # 计算 Focal Loss 的核心：调制因子 (1 - p_t)^gamma
        loss_modulating_factor = (1.0 - p_t).pow(self.gamma)
        
        # 计算 alpha 类别权重因子
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # 计算最终的 Focal Loss
        focal_loss = alpha_t * loss_modulating_factor * bce_loss
        
        # --- 应用 reduction ---
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else: # 'none'
            return focal_loss


class DiceLossOnly(nn.Module):
    """
    独立的Dice损失函数类，用于单独使用Dice损失
    """
    def __init__(self, epsilon: float = 1e-6):
        super(DiceLossOnly, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        计算Dice损失
        
        Args:
            inputs: 模型输出的logits，会自动应用sigmoid
            targets: 真实标签
        """
        # 对logits应用sigmoid得到概率
        probs = torch.sigmoid(inputs)
        
        # 如果inputs是4D，压缩channel维度
        if probs.dim() == 4 and probs.shape[1] == 1:
            probs = probs.squeeze(1)
        
        # 确保targets是float类型
        targets = targets.float()
        
        # 计算Dice系数
        intersection = 2.0 * (probs * targets).sum()
        union = probs.sum() + targets.sum()
        
        # 计算Dice损失 (1 - Dice系数)
        dice_score = (intersection + self.epsilon) / (union + self.epsilon)
        return 1.0 - dice_score


class CombinedLoss(nn.Module):
    """
    组合损失函数，支持多种损失的加权组合
    """
    def __init__(self, loss_types: list, weights: list, focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        """
        初始化组合损失函数
        
        Args:
            loss_types: 损失函数类型列表，如 ['bce', 'dice']
            weights: 对应的权重列表，如 [1.0, 1.0]
            focal_alpha: Focal Loss的alpha参数
            focal_gamma: Focal Loss的gamma参数
        """
        super(CombinedLoss, self).__init__()
        self.loss_types = loss_types
        self.weights = weights
        
        # 初始化各种损失函数
        self.losses = nn.ModuleDict()
        for loss_type in loss_types:
            if loss_type == 'bce':
                self.losses[loss_type] = nn.BCEWithLogitsLoss()
            elif loss_type == 'focal':
                self.losses[loss_type] = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            elif loss_type == 'dice':
                self.losses[loss_type] = DiceLossOnly()
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        前向传播计算组合损失
        
        Args:
            inputs: 模型输出的logits
            targets: 真实标签
        """
        total_loss = 0
        for i, loss_type in enumerate(self.loss_types):
            loss_val = self.losses[loss_type](inputs, targets)
            total_loss += self.weights[i] * loss_val
        return total_loss