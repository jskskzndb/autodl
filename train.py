import argparse
import logging
import os
import random
import sys
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
from utils.losses import FocalLoss, CombinedLoss, DiceLossOnly
from utils.utils import log_grad_stats

from unet import UNet

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
    train_dataset = BasicDataset(dir_img, dir_mask, img_scale, mask_suffix='')
    val_dataset = BasicDataset(val_dir_img, val_dir_mask, img_scale, mask_suffix='')
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

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
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
                    # 🔥 新版逻辑：只支持 3输出 (MDBES) 或 1输出 (Baseline)
                    # -----------------------------------------------------------
                    if isinstance(output, tuple):
                        # === [模式 A] MDBES-Net 解耦模式 (Seg, Body, Edge) ===
                        # 只要是 tuple，就默认一定是 3 个输出
                        masks_pred, body_pred, edge_pred = output
                        
                        # 1. 准备真值 (GT)
                        true_edges = generate_edge_tensor(true_masks) 
                        
                        # Body GT = Mask - Edge (利用广播机制)
                        true_masks_float = true_masks.unsqueeze(1).float()
                        true_body = true_masks_float - true_edges
                        true_body = torch.clamp(true_body, 0, 1)

                        # 2. 尺寸对齐 (防止预测图和真值尺寸不一致)
                        if edge_pred.shape[2:] != true_edges.shape[2:]:
                            edge_pred = F.interpolate(edge_pred, size=true_edges.shape[2:], mode='bilinear', align_corners=True)
                            body_pred = F.interpolate(body_pred, size=true_edges.shape[2:], mode='bilinear', align_corners=True)

                        # 3. 计算 Loss (三合一)
                        
                        # (1) 主分割 Loss
                        l_seg = calc_loss(masks_pred, true_masks, loss_combination, focal_alpha, focal_gamma)
                        
                        # (2) 主体 Loss (BCE)
                        l_body = F.binary_cross_entropy_with_logits(body_pred, true_body)
                        
                        # (3) 边缘 Loss (BCE + 权重)
                        l_edge = F.binary_cross_entropy_with_logits(
                            edge_pred, true_edges, pos_weight=torch.tensor([5.0], device=device)
                        )
                        
                        # 4. 总 Loss 加权 (使用传入的 lambda 参数)
                        loss = l_seg + (lambda_body * l_body) + (lambda_edge * l_edge)

                    else:
                        # === [模式 B] 普通 Baseline 模式 (1个输出) ===
                        masks_pred = output
                        loss = calc_loss(masks_pred, true_masks, loss_combination, focal_alpha, focal_gamma)
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
        
        # 1. 常规验证
        val_metrics = evaluate(model, val_loader, device, amp)
        
        # 2. 阈值扫描
        logging.info('Starting threshold scanning...')
        threshold_res = threshold_scan_evaluate(model, val_loader, device, amp, (0.3, 0.8), 0.01)
        
        logging.info(f'Threshold scan completed - Best Dice: {threshold_res["best_dice"]:.4f} '
                     f'at threshold {threshold_res["best_threshold_dice"]:.2f}, '
                     f'Best F1: {threshold_res["best_f1"]:.4f} '
                     f'at threshold {threshold_res["best_threshold_f1"]:.2f}')
        
        scheduler.step()
        
        # 4. 详细控制台输出
        logging.info(
            f'Epoch {epoch}/{epochs} completed - '
            f'Avg Loss: {avg_epoch_loss:.4f}, '
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

        # 6. WandB 图片可视化 (适配双流输出)
        with torch.no_grad():
            val_loader_list = list(val_loader)
            num_samples = min(5, len(val_loader_list))
            selected_batch_indices = random.sample(range(len(val_loader_list)), num_samples)
            
            comparison_images = []
            for idx in selected_batch_indices:
                batch = val_loader_list[idx]
                imgs, masks = batch['image'], batch['mask']
                imgs = imgs.to(device, dtype=torch.float32, memory_format=torch.channels_last)
                masks = masks.to(device, dtype=torch.long)
                
                single_img = imgs[0:1]
                
                with torch.cuda.amp.autocast(enabled=amp):
                    output = model(single_img)
                    
                    # 适配可视化
                    if isinstance(output, tuple):
                        pred_mask = output[0]
                        pred_edge = output[1]
                    else:
                        pred_mask = output
                        # 如果没有边缘输出，生成全黑图占位，防止报错
                        pred_edge = torch.zeros_like(pred_mask) - 100.0 
                
                pred_binary = (torch.sigmoid(pred_mask) > 0.5).to(torch.uint8)
                edge_vis = torch.sigmoid(pred_edge)[0].detach().cpu()
                
                comparison_images.append({
                    'input': single_img[0].detach().cpu(),
                    'true_mask': masks[0].detach().cpu().numpy().astype('uint8'),
                    'pred_mask': pred_binary[0].detach().cpu().numpy().astype('uint8'),
                    'pred_edge': edge_vis.numpy().astype('uint8')
                })
        
        for i, img_data in enumerate(comparison_images):
            experiment.log({
                f'examples/sample_{i+1}_input': wandb.Image(img_data['input']),
                f'examples/sample_{i+1}_true_mask': wandb.Image(img_data['true_mask']),
                f'examples/sample_{i+1}_pred_mask': wandb.Image(img_data['pred_mask']),
                f'examples/sample_{i+1}_pred_edge': wandb.Image(img_data['pred_edge']),
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

            if save_best:
                torch.save(checkpoint, best_path)

            
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
        use_fme=args.use_fme,
        use_decouple=args.use_decouple,  # 🔥 传入 MDBES-Net 解耦参数
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