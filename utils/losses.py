# utils/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeLoss(nn.Module):
    """
    [新增] 边缘感知损失 (Edge-Aware Loss)
    来源: FDENet (Frequency-Guided Dual-Encoder Network)
    作用: 计算预测图梯度与真值梯度之间的 L1 差异，强迫模型输出锐利边缘。
    """
    def __init__(self, device='cuda'):
        super(EdgeLoss, self).__init__()
        # 定义 Sobel 算子 (用于计算梯度/边缘)
        # X方向梯度算子
        self.kernel_x = torch.tensor([[-1, 0, 1], 
                                      [-2, 0, 2], 
                                      [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        # Y方向梯度算子
        self.kernel_y = torch.tensor([[-1, -2, -1], 
                                      [0, 0, 0], 
                                      [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    def forward(self, pred, target):
        """
        pred: 模型的预测结果 (Logits 或 Prob), 形状 [B, 1, H, W]
        target: 真实的标签 Mask, 形状 [B, 1, H, W]
        """
        # 确保输入是 Sigmoid 后的概率图，以便计算平滑梯度
        # 如果传入的是 Logits (通常 < 0 或 > 1)，先做 Sigmoid
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
            
        # 计算预测图的梯度
        pred_grad_x = F.conv2d(pred, self.kernel_x, padding=1)
        pred_grad_y = F.conv2d(pred, self.kernel_y, padding=1)
        
        # 计算真值图的梯度
        # target 需要是 float 类型
        target = target.float()
        target_grad_x = F.conv2d(target, self.kernel_x, padding=1)
        target_grad_y = F.conv2d(target, self.kernel_y, padding=1)
        
        # 计算 L1 Loss (预测梯度 - 真值梯度)
        loss = torch.abs(pred_grad_x - target_grad_x).mean() + \
               torch.abs(pred_grad_y - target_grad_y).mean()
               
        return loss

class FocalLoss(nn.Module):
    """
    一个更健壮、注释更清晰的 Focal Loss 实现，用于二分类分割任务。
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean', epsilon: float = 1e-7):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        # 内部强制转换为 float 类型
        targets = targets.float()

        # 如果 inputs 的形状是 [B, 1, H, W]，则去掉 channel 维度以匹配 targets
        if inputs.dim() == 4 and inputs.shape[1] == 1:
            inputs = inputs.squeeze(1)

        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)
        p_t = torch.clamp(p_t, self.epsilon, 1.0 - self.epsilon)
        loss_modulating_factor = (1.0 - p_t).pow(self.gamma)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * loss_modulating_factor * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLossOnly(nn.Module):
    """
    独立的Dice损失函数类
    """
    def __init__(self, epsilon: float = 1e-6):
        super(DiceLossOnly, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        probs = torch.sigmoid(inputs)
        if probs.dim() == 4 and probs.shape[1] == 1:
            probs = probs.squeeze(1)
        targets = targets.float()
        intersection = 2.0 * (probs * targets).sum()
        union = probs.sum() + targets.sum()
        dice_score = (intersection + self.epsilon) / (union + self.epsilon)
        return 1.0 - dice_score


class CombinedLoss(nn.Module):
    """
    组合损失函数，支持多种损失的加权组合
    """
    def __init__(self, loss_types: list, weights: list, focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        super(CombinedLoss, self).__init__()
        self.loss_types = loss_types
        self.weights = weights
        self.losses = nn.ModuleDict()
        for loss_type in loss_types:
            if loss_type == 'bce':
                self.losses[loss_type] = nn.BCEWithLogitsLoss()
            elif loss_type == 'focal':
                self.losses[loss_type] = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            elif loss_type == 'dice':
                self.losses[loss_type] = DiceLossOnly()
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        total_loss = 0
        for i, loss_type in enumerate(self.loss_types):
            loss_val = self.losses[loss_type](inputs, targets)
            total_loss += self.weights[i] * loss_val
        return total_loss
def compute_prototype_ortho_loss(model, device):
    """
    [新增] 原型正交 Loss
    强制所有原型向量互不相同，防止坍塌。
    """
    loss = torch.tensor(0.0, device=device)
    count = 0
    
    for name, param in model.named_parameters():
        # 自动扫描模型中所有的 'prototypes' 参数
        if 'prototypes' in name and param.requires_grad:
            # param shape 通常是 [1, N, C] 或 [N, C]
            P = param
            if P.dim() > 2: 
                P = P.squeeze(0) # 变成 [N, C]
            
            # 1. 归一化 (只约束方向)
            P_norm = F.normalize(P, p=2, dim=1)
            
            # 2. 计算相似度矩阵 (Gram Matrix) [N, N]
            gram_matrix = torch.mm(P_norm, P_norm.t())
            
            # 3. 目标: 单位矩阵 (对角线1，其他0)
            identity = torch.eye(P.shape[0], device=device)
            
            # 4. 计算 MSE
            loss += F.mse_loss(gram_matrix, identity)
            count += 1
            
    return loss if count > 0 else torch.tensor(0.0, device=device)