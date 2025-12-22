import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff

# 小常量：避免除零
_EPS = 1e-6

@torch.inference_mode()
def evaluate(net, dataloader, device, amp, criterion=None, verbose=False):
    """
    评估模型在验证集上的表现，返回一个包含多种指标的字典。
    
    采用全局指标计算法（与 test.py 的 test_model 函数一致）：
    - 累加整个验证集的 total_tp, total_fp, total_fn
    - 最后使用全局累加值一次性计算所有指标
    - 这种方法比批次平均法更精确，不受批次大小影响
    
    指标包含：
      - dice:      Dice 系数（F1分数的另一种表达形式）
      - iou:       交并比（IoU / Jaccard Index）
      - precision: 精确率（Precision）
      - recall:    召回率（Recall / Sensitivity）
      - f1:        F1 分数（Precision和Recall的调和平均）
    
    Args:
        net: 模型
        dataloader: 验证数据加载器
        device: 计算设备
        amp: 是否使用混合精度
        verbose: 是否在控制台打印详细指标（默认False）
    
    Returns:
        dict: 包含上述5个指标的字典
    """
    net.eval()
    num_val_batches = len(dataloader)

    # 用于累加整个验证集的 TP, FP, FN（全局指标计算法✨）
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_val_loss = 0.0  # <--- 新增：累计 Loss
    # 遍历验证集
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # 把图片/标签放到正确设备和数据类型
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # 前向预测
            mask_pred = net(image)
            if isinstance(mask_pred, tuple): mask_pred = mask_pred[0]
            # 裁剪logits防止数值问题（与训练时保持一致）
            mask_pred = torch.clamp(mask_pred, min=-50, max=50)
            # --- 2. 新增：计算 Validation Loss ---
            if criterion is not None:
                # 【严谨逻辑】维度对齐与类型转换
                # 必须与 train.py 中 calc_loss 的处理逻辑保持严格一致：
                # 1. 如果预测值包含 Channel 维度 (B, 1, H, W)，必须 squeeze 掉，变成 (B, H, W)
                #    这是因为 PyTorch 的 BCEWithLogitsLoss 和我们定义的 CombinedLoss 均期望输入与 Target 形状完全一致
                pred_for_loss = mask_pred.squeeze(1) if mask_pred.dim() == 4 else mask_pred
                
                # 2. 真实标签必须转为 float 类型以匹配 Loss 函数的输入要求
                true_for_loss = mask_true.float()
                
                # 3. 计算并累加 Loss
                loss_val = criterion(pred_for_loss, true_for_loss)
                total_val_loss += loss_val.item()
            # ---------- 二分类情况 ----------
            # 要求真值是 0/1（如果不是，需要你预处理成 0/1）
            assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'

            # 预测：sigmoid 后阈值成 0/1
            prob = torch.sigmoid(mask_pred).squeeze(1)
            pred_bin = (prob > 0.5).float()
            true_bin = mask_true.float()

            # 展平到 (N*H*W,)
            p = pred_bin.reshape(-1)
            t = true_bin.reshape(-1)

            # 累加当前批次的 TP/FP/FN 到全局统计量
            total_tp += (p * t).sum()
            total_fp += (p * (1.0 - t)).sum()
            total_fn += ((1.0 - p) * t).sum()

    # 还原 train 模式
    net.train()

    # --- 使用全局累加值一次性计算所有指标（与 test.py 完全一致）---
    dice = (2 * total_tp + _EPS) / (2 * total_tp + total_fp + total_fn + _EPS)
    iou = (total_tp + _EPS) / (total_tp + total_fp + total_fn + _EPS)
    precision = (total_tp + _EPS) / (total_tp + total_fp + _EPS)
    recall = (total_tp + _EPS) / (total_tp + total_fn + _EPS)
    f1 = (2 * precision * recall + _EPS) / (precision + recall + _EPS)
    # 🔥🔥🔥 【补上这一行！】 🔥🔥🔥
    # 计算平均 Loss：总 Loss / 总批次
    # 如果 num_val_batches 为 0 (防除零)，就除以 1
    avg_loss = total_val_loss / max(num_val_batches, 1)
    metrics = {
        'dice': float(dice),
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'loss': float(avg_loss)
    }
    
    # 如果启用详细输出，在控制台打印指标
    if verbose:
        print("\n" + "=" * 50)
        print("         Validation Metrics (Global Method)")
        print("=" * 50)
        print(f"  Total TP: {float(total_tp):.0f}")
        print(f"  Total FP: {float(total_fp):.0f}")
        print(f"  Total FN: {float(total_fn):.0f}")
        print("-" * 50)
        for metric_name, value in metrics.items():
            print(f"  - {metric_name.upper():10s}: {value:.6f}")
        print("=" * 50 + "\n")
    
    return metrics


@torch.inference_mode()
def threshold_scan_evaluate(net, dataloader, device, amp, threshold_range=(0.3, 0.8), threshold_step=0.005):
    """
    阈值扫描评估函数：在指定阈值范围内扫描，找到最佳阈值及对应的Dice和F1分数
    
    Args:
        net: 神经网络模型
        dataloader: 验证数据加载器
        device: 设备（cuda或cpu）
        amp: 是否使用混合精度
        threshold_range: 阈值扫描范围，默认(0.3, 0.8)
        threshold_step: 阈值扫描步长，默认0.005
    
    Returns:
        dict: 包含最佳阈值、最佳Dice、最佳F1等信息的字典
    """
    net.eval()
    
    # 生成阈值列表
    thresholds = torch.arange(threshold_range[0], threshold_range[1] + threshold_step, threshold_step)
    
    # 存储所有预测概率和真实标签，用于后续阈值扫描
    all_probs = []
    all_true_masks = []
    
    # 首先收集所有预测概率和真实标签
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, desc='Collecting predictions for threshold scan', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            
            # 将数据移到设备
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            
            # 前向预测
            mask_pred = net(image)
            if isinstance(mask_pred, tuple): mask_pred = mask_pred[0]
            # 裁剪logits防止数值问题（与训练时保持一致）
            mask_pred = torch.clamp(mask_pred, min=-50, max=50)
            
            # 二分类：获取sigmoid概率
            prob = torch.sigmoid(mask_pred).squeeze(1)  # [B, H, W]
            all_probs.append(prob.cpu())
            all_true_masks.append(mask_true.cpu())
    
    # 拼接所有批次的数据
    all_probs = torch.cat(all_probs, dim=0)  # [N, H, W]
    all_true_masks = torch.cat(all_true_masks, dim=0)  # [N, H, W]
    
    # 对每个阈值计算Dice和F1分数
    best_dice = 0.0
    best_f1 = 0.0
    best_threshold_dice = threshold_range[0]
    best_threshold_f1 = threshold_range[0]
    
    threshold_results = {}
    
    for threshold in thresholds:
        # 根据当前阈值生成二值预测
        pred_binary = (all_probs > threshold).float()
        true_binary = all_true_masks.float()
        
        # 计算Dice系数
        dice = dice_coeff(pred_binary, true_binary, reduce_batch_first=False)
        
        # 计算F1分数（基于TP、FP、FN）
        pred_flat = pred_binary.reshape(-1)
        true_flat = true_binary.reshape(-1)
        
        tp = (pred_flat * true_flat).sum()
        fp = (pred_flat * (1.0 - true_flat)).sum()
        fn = ((1.0 - pred_flat) * true_flat).sum()
        
        precision = (tp + _EPS) / (tp + fp + _EPS)
        recall = (tp + _EPS) / (tp + fn + _EPS)
        f1 = (2 * precision * recall + _EPS) / (precision + recall + _EPS)
        
        # 存储当前阈值的结果
        threshold_results[float(threshold)] = {
            'dice': float(dice),
            'f1': float(f1),
            'precision': float(precision),
            'recall': float(recall)
        }
        
        # 更新最佳Dice
        if dice > best_dice:
            best_dice = float(dice)
            best_threshold_dice = float(threshold)
        
        # 更新最佳F1
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold_f1 = float(threshold)
    
    # 还原训练模式
    net.train()
    
    return {
        'best_dice': best_dice,
        'best_threshold_dice': best_threshold_dice,
        'best_f1': best_f1,
        'best_threshold_f1': best_threshold_f1,
        'threshold_results': threshold_results
    }
