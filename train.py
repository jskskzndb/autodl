import argparse
import logging
import os
import random
import sys
import numpy as np  # <--- 添加这一行
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from evaluate import evaluate, threshold_scan_evaluate
from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss


from utils.losses import FocalLoss, CombinedLoss, DiceLossOnly, EdgeLoss
from utils.utils import log_grad_stats

from unet import UNet

import random

def log_best_visuals(model, val_loader, device, num_samples=5):
    """
    将 原图、预测掩码、真值掩码 并排展示在 WandB 表格中。
    自动处理反标准化，防止原图变黑。
    """
    model.eval()
    
    # 1. 定义 ImageNet 的均值和方差 (用于反标准化)
    # 如果你在 dataset 里用了其他数值，请在这里修改
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    # 2. 创建 WandB 表格，定义三列
    columns = ["Input Image (原图)", "Prediction (预测)", "Ground Truth (真值)"]
    test_table = wandb.Table(columns=columns)

    print(f"✨ 正在生成 {num_samples} 组可视化样本...")

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if len(test_table.data) >= num_samples: break
            
            imgs = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # 推理
            outputs = model(imgs)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            for j in range(imgs.shape[0]):
                if len(test_table.data) >= num_samples: break
                
                # --- A. 修复原图 (防止全黑的核心步骤) ---
                # 1. 反标准化: image = image * std + mean
                img_tensor = imgs[j] * std + mean
                # 2. 限制数值范围在 0-1 之间 (消除计算误差导致的越界)
                img_tensor = torch.clamp(img_tensor, 0, 1)
                # 3. 转换维度 [C,H,W] -> [H,W,C] 并转为 numpy
                img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
                # 4. 乘以 255 并转为整数 (变成标准的 RGB 图片)
                img_np = (img_np * 255).astype(np.uint8)

                # --- B. 处理掩码 (变成黑白图) ---
                # 1. 取出单张掩码
                pred_mask = preds[j].squeeze().cpu().numpy()
                true_mask = masks[j].squeeze().cpu().numpy()
                
                # 2. 乘以 255！(非常重要：0变成黑，1变成白)
                pred_mask = (pred_mask * 255).astype(np.uint8)
                true_mask = (true_mask * 255).astype(np.uint8)
                
                # --- C. 创建 WandB 图片对象 ---
                input_img_log = wandb.Image(img_np)
                pred_img_log = wandb.Image(pred_mask)
                true_img_log = wandb.Image(true_mask)
                
                # --- D. 添加到表格的一行中 ---
                test_table.add_data(input_img_log, pred_img_log, true_img_log)

    # 3. 上传表格
    wandb.log({"Visual Results Table": test_table}, commit=False)
    print("✅ 可视化表格已上传！")
    
    model.train() # 恢复训练模式

# ================= 配置路径 =================
dir_img = Path('./data/train/imgs/')
dir_mask = Path('./data/train/masks/')
val_dir_img = Path('./data/val/imgs/')
val_dir_mask = Path('./data/val/masks/')
dir_checkpoint = Path('./data/checkpoints/')

# ================= 辅助函数 =================

def generate_edge_tensor(mask):
    """
    [保留] 实时将 Segmentation Mask 转为 Edge GT (高效 Sobel 算子)
    mask: [B, H, W] (LongTensor)
    return: [B, 1, H, W] (FloatTensor)
    """
    # 转换为 Float 并增加 Channel 维
    mask = mask.unsqueeze(1).float()
    
    # 定义 Sobel 算子
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=mask.device).float().view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=mask.device).float().view(1, 1, 3, 3)
    
    # 计算梯度
    edge_x = F.conv2d(mask, sobel_x, padding=1)
    edge_y = F.conv2d(mask, sobel_y, padding=1)
    
    # 梯度幅值
    edge = torch.sqrt(edge_x**2 + edge_y**2)
    
    # 二值化 (只要有梯度就是边缘)
    edge = (edge > 0.1).float()
    return edge

# [已删除] generate_edge_label (旧版膨胀腐蚀算法，已按要求移除)

def train_model(
        model,
        device,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 1.0,
        amp: bool = True,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        start_epoch: int = 1,
        checkpoint_to_load: dict = None,
        loss_combination: str = 'focal+dice',
        loss_weights: str = '1.0,1.0',
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        optimizer_type: str = 'adamw',
        backbone_lr_scale: float = 0.1,
        lambda_edge: float = 20.0,
        lambda_body: float = 1.0,
        
):
    # 1. 数据准备
    train_dataset = BasicDataset(dir_img, dir_mask, img_scale, mask_suffix='', augment=True)
    val_dataset = BasicDataset(val_dir_img, val_dir_mask, img_scale, mask_suffix='', augment=False)
    n_train = len(train_dataset)
    n_val = len(val_dataset)

    # 2. DataLoader
    num_workers = min(4, os.cpu_count()) if os.name == 'nt' else min(8, os.cpu_count())
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_args)

    # 3. WandB 初始化 (保留原有配置)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        img_scale=img_scale, amp=amp, backbone_lr=backbone_lr_scale
    ), allow_val_change=True)
    
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. 优化器与差分学习率
    backbone_params_ids = []
    use_differential_lr = False
    
    if backbone_lr_scale < 1.0:
        if hasattr(model, 'encoder_name') and model.encoder_name in ['resnet', 'cnextv2']:
            use_differential_lr = True
        elif hasattr(model, 'use_resnet_encoder') and model.use_resnet_encoder:
            use_differential_lr = True

    if use_differential_lr:
        logging.info(f'✨ 启用差分学习率策略: Backbone Scale = {backbone_lr_scale}')
        backbone_names = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 
                          'enc_stem', 'enc_model'] 
        
        for name, module in model.named_children():
            if name in backbone_names:
                for param in module.parameters():
                    backbone_params_ids.append(id(param))
        
        backbone_params = filter(lambda p: id(p) in backbone_params_ids, model.parameters())
        base_params = filter(lambda p: id(p) not in backbone_params_ids, model.parameters())
        
        param_groups = [
            {'params': base_params, 'lr': learning_rate}, 
            {'params': backbone_params, 'lr': learning_rate * backbone_lr_scale}
        ]
    else:
        logging.info('使用统一学习率 (无差分)')
        param_groups = model.parameters()

    if optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(param_groups, lr=learning_rate, weight_decay=weight_decay)
        logging.info('✅ Using AdamW optimizer')
    else:
        optimizer = optim.RMSprop(param_groups, lr=learning_rate, weight_decay=weight_decay,
                                  momentum=momentum, foreach=True)
        logging.info('✅ Using RMSprop optimizer')

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-9)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # 损失函数
    if model.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        loss_parts = loss_combination.split('+')
        weights = [float(w) for w in loss_weights.split(',')] if ',' in loss_weights else [1.0]*len(loss_parts)
        
        if loss_combination == 'bce': criterion = nn.BCEWithLogitsLoss()
        elif loss_combination == 'focal': criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        elif loss_combination == 'dice': criterion = DiceLossOnly()
        else: criterion = CombinedLoss(loss_parts, weights, focal_alpha, focal_gamma)
        logging.info(f'✅ Using Loss: {loss_combination}')
    # 🔥 [新增] 初始化 EdgeLoss (必须放在这里)
    edge_criterion = EdgeLoss(device=device)
    global_step = 0

    # 恢复 Checkpoint
    if checkpoint_to_load is not None:
        try:
            optimizer.load_state_dict(checkpoint_to_load['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint_to_load['scheduler_state_dict'])
            if 'grad_scaler_state_dict' in checkpoint_to_load and amp:
                grad_scaler.load_state_dict(checkpoint_to_load['grad_scaler_state_dict'])
            logging.info('✅ 训练状态完全恢复')
        except Exception as e:
            logging.warning(f'⚠️ 恢复优化器状态失败: {e}')

    # ============================================================
    # 5. 训练循环
    # ============================================================
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_grad_norms = []
        batch_count = 0
        
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    output = model(images)
                    
                   # -----------------------------------------------------------
                    # 🔥 修复后的逻辑：自适应处理 2输出 或 3输出
                    # -----------------------------------------------------------
                    if isinstance(output, tuple):
                        # 1. 准备边缘真值 (所有双流模式都需要)
                        true_edges = generate_edge_tensor(true_masks)

                        if len(output) == 3:
                            # === [模式 A] MDBES-Net (Seg, Body, Edge) ===
                            masks_pred, body_pred, edge_pred = output
                            
                            # Body GT 计算
                            true_masks_float = true_masks.unsqueeze(1).float()
                            true_body = torch.clamp(true_masks_float - true_edges, 0, 1)

                            # 尺寸对齐
                            if edge_pred.shape[2:] != true_edges.shape[2:]:
                                edge_pred = F.interpolate(edge_pred, size=true_edges.shape[2:], mode='bilinear', align_corners=True)
                                body_pred = F.interpolate(body_pred, size=true_edges.shape[2:], mode='bilinear', align_corners=True)

                            # Loss 计算
                            l_seg = calc_loss(masks_pred, true_masks, loss_combination, focal_alpha, focal_gamma)
                            l_body = F.binary_cross_entropy_with_logits(body_pred, true_body)
                            l_edge = F.binary_cross_entropy_with_logits(edge_pred, true_edges, pos_weight=torch.tensor([5.0], device=device))
                            
                            loss = l_seg + (lambda_body * l_body) + (lambda_edge * l_edge)

                        elif len(output) == 2:
                            # === [模式 B] S-DMFNet (Seg, Aux_Edge) ===
                            # 🔥 这是你现在需要的逻辑
                            masks_pred, edge_pred = output
                            
                            # 尺寸对齐
                            if edge_pred.shape[2:] != true_edges.shape[2:]:
                                edge_pred = F.interpolate(edge_pred, size=true_edges.shape[2:], mode='bilinear', align_corners=True)
                            
                            # Loss 计算
                            # 1. 主分割 Loss (BCE/Dice/Focal)
                            l_seg = calc_loss(masks_pred, true_masks, loss_combination, focal_alpha, focal_gamma)
                            
                            # 2. 辅助边缘 Loss (BCE With Logits)
                            # 使用辅助头预测的 edge_pred 和生成的 true_edges 进行比较
                            # pos_weight=5.0 是为了解决边缘像素过少的不平衡问题
                            l_edge = F.binary_cross_entropy_with_logits(
                                edge_pred, true_edges, pos_weight=torch.tensor([5.0], device=device)
                            )
                            
                            # 3. 总 Loss
                            loss = l_seg + (lambda_edge * l_edge)

                    else:
                        # === [模式 C] 单输出模式 (Seg only) ===
                        masks_pred = output
                        loss = calc_loss(masks_pred, true_masks, loss_combination, focal_alpha, focal_gamma)
                        
                        # 🔥 隐式边缘监督 (Gradient-based Edge Loss)
                        # 如果没有辅助头，就强迫主分割图的梯度要锐利
                        if lambda_edge > 0:
                            loss_e = edge_criterion(masks_pred, true_masks)
                            loss += lambda_edge * loss_e

                # 异常检测
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.error(f'Loss NaN/Inf detected: {loss.item()}. Skipping batch.')
                    optimizer.zero_grad()
                    continue
                # 反向传播
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                epoch_grad_norms.append(grad_norm.item())

                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                batch_count += 1
                epoch_loss += loss.item()
                
                # WandB 实时日志 (保留你的原配置)
                experiment.log({
                    'train/loss_batch': loss.item(), 
                    'train/grad_norm': grad_norm.item(), 
                    'global_step': global_step
                })
                pbar.set_postfix(**{'loss': loss.item(), 'grad': grad_norm.item()})

        # ====== 验证与评估 ======
        avg_epoch_loss = epoch_loss / max(batch_count, 1)
        avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms) if epoch_grad_norms else 0.0
        # 🔴 [修改 1] 传入 criterion
        # 注意：这里我们使用定义好的 criterion 计算 loss
        val_metrics = evaluate(model, val_loader, device, amp, criterion=criterion)
        
        
        # 2. 🔥 [关键修改] 禁用阈值扫描，直接复用 0.5 阈值的结果
        # logging.info('Starting threshold scanning...')
        # threshold_res = threshold_scan_evaluate(...) # <--- 注释掉这一行
        
        # 🔥 手动构造结果字典，保持变量名兼容，防止后面报错
        threshold_res = {
            'best_dice': val_metrics['dice'],      # 直接用 0.5 的 Dice
            'best_f1': val_metrics['f1'],          # 直接用 0.5 的 F1
            'best_threshold_dice': 0.5,            # 固定显示 0.5
            'best_threshold_f1': 0.5               # 固定显示 0.5
        }
        
        logging.info('⏩ Skipping threshold scan. Using fixed threshold 0.5.')

        scheduler.step()
        
        # 4. 详细控制台输出
        logging.info(
            f'Epoch {epoch}/{epochs} completed - '
            f'Train Loss: {avg_epoch_loss:.4f}, '
            f'Val Loss: {val_metrics["loss"]:.4f}, '
            f'Avg Grad Norm: {avg_grad_norm:.4f}, '
            f'Val Dice: {val_metrics["dice"]:.4f}, '
            f'Val IoU: {val_metrics["iou"]:.4f}, '
            f'Val F1: {val_metrics["f1"]:.4f}, '
            f'Val Precision: {val_metrics["precision"]:.4f}, '
            f'Val Recall: {val_metrics["recall"]:.4f}, '
            f'Best Dice: {threshold_res["best_dice"]:.4f} (threshold: {threshold_res["best_threshold_dice"]:.2f}), '
            f'Best F1: {threshold_res["best_f1"]:.4f} (threshold: {threshold_res["best_threshold_f1"]:.2f})'
        )

        # 5. 上传 WandB 日志
        current_lr = optimizer.param_groups[0]['lr']
        experiment.log({
            'train/epoch_loss': avg_epoch_loss,
            'val/loss': val_metrics['loss'],       # <--- 关键！添加这一行！
            'train/avg_grad_norm': avg_grad_norm,
            'val/dice': val_metrics['dice'],
            'val/iou': val_metrics['iou'],
            'val/f1': val_metrics['f1'],
            'val/precision': val_metrics['precision'],
            'val/recall': val_metrics['recall'],
            'val/best_dice': threshold_res['best_dice'],
            'val/best_f1': threshold_res['best_f1'],
            'epoch': epoch,                        # 🔥 新增: 当前轮次
            'train/learning_rate': current_lr      # 🔥 新增: 当前学习率曲线
        })

        

        # ====== 保存 Checkpoint ======
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'grad_scaler_state_dict': grad_scaler.state_dict(),
                'loss': avg_epoch_loss,
                'val_dice': float(val_metrics['dice']),
                'val_best_f1': float(threshold_res['best_f1']),
                # ... 保留你的其他键值
            }
            
            # Latest
            torch.save(checkpoint, str(dir_checkpoint / 'checkpoint_latest.pth'))
            # 2. 🔥 [修改点 2] 30轮以后，每一轮都额外保存一个文件
            if epoch > 30:
                # 文件名例如: checkpoint_epoch_31.pth, checkpoint_epoch_32.pth ...
                epoch_path = str(dir_checkpoint / f'checkpoint_epoch_{epoch}.pth')
                torch.save(checkpoint, epoch_path)
                logging.info(f'💾 [备份] 已保存第 {epoch} 轮权重: {epoch_path}')
            # Best
            best_path = str(dir_checkpoint / 'checkpoint_best.pth')
            current_dice = val_metrics['dice']
            save_best = False
            
            if not Path(best_path).exists():
                save_best = True
                logging.info(f'   🌟 首次创建最佳模型 (Dice: {current_dice:.4f})')
            else:
                try:
                    prev_best = torch.load(best_path, map_location='cpu', weights_only=False).get('val_dice', 0.0)
                    if current_dice > prev_best:
                        save_best = True
                        logging.info(f'   🏆 刷新最佳记录! ({prev_best:.4f} -> {current_dice:.4f})')
                    else:
                        # 🔥🔥🔥 这一行是你要求的关键日志 🔥🔥🔥
                        logging.info(f'   (当前 Dice {current_dice:.4f} 未超过最佳 {prev_best:.4f})')
                except:
                    save_best = True
            # 🔥 如果是最佳模型：保存权重 + 上传高清图片
            if save_best:
                torch.save(checkpoint, best_path)
                try:
                    # 调用我们写好的可视化函数
                    log_best_visuals(model, val_loader, device, num_samples=5)
                except Exception as e:
                    logging.warning(f"⚠️ 可视化上传失败: {e}")

            
    wandb.finish()

# 计算 Loss 辅助函数 (保持不变)
def calc_loss(masks_pred, true_masks, loss_combination, focal_alpha, focal_gamma):
    if loss_combination == 'bce':
        criterion = nn.BCEWithLogitsLoss()
        return criterion(masks_pred.squeeze(1), true_masks.float())
    elif loss_combination == 'focal':
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        return criterion(masks_pred.squeeze(1), true_masks.float())
    elif loss_combination == 'dice':
        return dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
    else:
        # Combined Loss (假设你在 utils.losses 里定义了)
        # 这里为了稳健性，如果找不到 CombinedLoss 类，回退到手动相加
        try:
            criterion = CombinedLoss(loss_combination.split('+'), [1.0, 1.0], focal_alpha, focal_gamma)
            return criterion(masks_pred.squeeze(1), true_masks.float())
        except:
             # 简单的 fallback
             bce = nn.BCEWithLogitsLoss()(masks_pred.squeeze(1), true_masks.float())
             dice = dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
             return bce + dice

def get_args():
    parser = argparse.ArgumentParser(description='Train the Unified UNet')
    
    # 基础参数
    parser.add_argument('--epochs', '-e', type=int, default=20)
    parser.add_argument('--batch-size', '-b', type=int, default=8)
    parser.add_argument('--learning-rate', '-l', type=float, default=1e-4)
    parser.add_argument('--load', '-f', type=str, default=False)
    parser.add_argument('--scale', '-s', type=float, default=1.0)
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0)
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--bilinear', action='store_true', default=False)
    parser.add_argument('--classes', '-c', type=int, default=1)
    parser.add_argument('--start-epoch', type=int, default=1)
    
    # 架构参数
    parser.add_argument('--encoder', type=str, default='resnet', choices=['resnet', 'cnextv2', 'standard'])
    parser.add_argument('--decoder', type=str, default='phd', choices=['phd', 'standard'])
    parser.add_argument('--cnext-type', type=str, default='convnextv2_tiny')
    
    # SOTA 模块开关
    parser.add_argument('--use-dcn', action='store_true', default=False, help='Enable standard DCNv3')
    parser.add_argument('--use-dubm', action='store_true', default=False, help='Enable D-UBM (SOTA)')
    parser.add_argument('--use-strg', action='store_true', default=False, help='Enable STRG Skip Enhancement')
    parser.add_argument('--use-dual-stream', action='store_true', default=False, help='Enable Dual-Stream Boundary Architecture')
    parser.add_argument('--use-wavelet-denoise', action='store_true', default=False, help='Enable Wavelet Denoising on Skip Connections')
    parser.add_argument('--use-dsis', action='store_true', default=False, help='Enable Dual-Stream Interactive Skip Module')
    parser.add_argument('--use-unet3p', action='store_true', default=False, help='Enable UNet 3+ Full-Scale Skip Connections')
    # [新增] MDBES-Net 相关参数
    parser.add_argument('--use_decouple', action='store_true', default=False, help='Enable MDBES-Net explicit decoupling supervision')
    parser.add_argument('--lambda_edge', type=float, default=20.0, help='Weight for the Edge loss (default: 20.0)')
    parser.add_argument('--lambda_body', type=float, default=1.0, help='Weight for the Body loss (default: 1.0)')
    
    # 其他增强模块 (保持原有开关定义，但移除了旧版 Edge Logic 的执行)
    parser.add_argument('--use-wgn-enhancement', action='store_true', default=False)
    parser.add_argument('--use-cafm', action='store_true', default=False)
    parser.add_argument('--use-edge-loss', action='store_true', default=False, help='Legacy WGN Edge Loss (Deprecated logic removed)')
    parser.add_argument('--use-fme', action='store_true', default=False, 
                        help='Enable Frequency-Mamba Enhancement (FME) module')
    # WGN 参数
    parser.add_argument('--wgn-base-order', type=int, default=3)
    parser.add_argument('--wgn-orders', type=str, default=None)

    # 优化参数
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'rmsprop'])
    parser.add_argument('--loss-combination', type=str, default='focal+dice')
    parser.add_argument('--loss-weights', type=str, default='1.0,1.0')
    parser.add_argument('--focal-alpha', type=float, default=0.25)
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--weight-decay', type=float, default=1e-8)
    parser.add_argument('--momentum', type=float, default=0.999)
    parser.add_argument('--gradient-clipping', type=float, default=1.0)
    parser.add_argument('--backbone-lr-scale', type=float, default=0.1)

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # WGN Orders 处理
    wgn_orders = None
    if args.use_wgn_enhancement:
        if args.wgn_orders:
            orders_list = [int(x) for x in args.wgn_orders.split(',')]
            wgn_orders = {'layer1': (orders_list[0], orders_list[1]), 'layer2': (orders_list[2], orders_list[3]), 'layer3': (orders_list[4], orders_list[5])}
        else:
            base = args.wgn_base_order
            wgn_orders = {'layer1': (base, base-1), 'layer2': (base+1, base), 'layer3': (base+2, base+1)}

    # 实例化模型
    logging.info(f"🚀 Building Model: Encoder={args.encoder}, Decoder={args.decoder}")
    model = UNet(
        n_channels=3,
        n_classes=args.classes,
        bilinear=args.bilinear,
        encoder_name=args.encoder,
        decoder_name=args.decoder,
        cnext_type=args.cnext_type,
        use_wgn_enhancement=args.use_wgn_enhancement,
        use_cafm=args.use_cafm,
        # 注意: 即使传入 use_edge_loss=True, train loop 中已移除了处理它的逻辑
        use_edge_loss=args.use_edge_loss, 
        wgn_orders=wgn_orders,
        use_dcn_in_phd=args.use_dcn,
        use_dubm=args.use_dubm,
        use_strg=args.use_strg,
        use_dual_stream=args.use_dual_stream, # 🔥 新增双流
        use_dsis=args.use_dsis, # 🔥 传入参数
        use_unet3p=args.use_unet3p, # 🔥 传入参数
        use_wavelet_denoise=args.use_wavelet_denoise  # 👈 传入这个参数
          # 🔥 传入 MDBES-Net 解耦参数
    )
    
    model = model.to(memory_format=torch.channels_last)
    model.to(device=device)

    # 加载权重
    checkpoint_to_load = None
    if args.load:
        try:
            ckpt = torch.load(args.load, map_location=device, weights_only=False)
            if 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
                checkpoint_to_load = ckpt
                # 🔥🔥🔥 [新增] 自动读取断点轮数，实现无缝续训 🔥🔥🔥
                if 'epoch' in ckpt:
                    args.start_epoch = ckpt['epoch'] + 1
                    logging.info(f"🔄 自动检测到断点 (Epoch {ckpt['epoch']})，将从 Epoch {args.start_epoch} 继续训练！")
            else:
                model.load_state_dict(ckpt)
            logging.info(f'Model loaded from {args.load}')
        except Exception as e:
            logging.error(f"Load failed: {e}")
            sys.exit(1)

    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            start_epoch=args.start_epoch,
            checkpoint_to_load=checkpoint_to_load,
            backbone_lr_scale=args.backbone_lr_scale,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            gradient_clipping=args.gradient_clipping,
            loss_combination=args.loss_combination,
            loss_weights=args.loss_weights,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            optimizer_type=args.optimizer,
            # 🔥 [新增] 把权重传给训练函数
            lambda_edge=args.lambda_edge,
            lambda_body=args.lambda_body
        )
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt checkpoint')