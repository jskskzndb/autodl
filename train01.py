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

# 🔥 【重要】导入统一模型类
# 请确保 unet/unet_model_unified.py 文件存在，或者你在 unet/__init__.py 里做好了映射
# 为了方便，这里假设你把上面的代码保存为了 unet/unet_model_unified.py
from unet.unet_model_unified import UNet

# 训练集路径
dir_img = Path('./data/train/imgs/')
dir_mask = Path('./data/train/masks/')
# 验证集路径
val_dir_img = Path('./data/val/imgs/')
val_dir_mask = Path('./data/val/masks/')
# 模型权重保存路径
dir_checkpoint = Path('./data/checkpoints/')

# 边缘生成函数
def generate_edge_label(mask_tensor):
    """ 现场生成边缘标签: Dilate - Erode """
    with torch.no_grad():
        mask_float = mask_tensor.unsqueeze(1).float()
        kernel_size = 3
        dilated = F.max_pool2d(mask_float, kernel_size, stride=1, padding=1)
        eroded = -F.max_pool2d(-mask_float, kernel_size, stride=1, padding=1)
        edge = dilated - eroded
        return edge.squeeze(1)

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
):
    # 1. 创建数据集
    train_dataset = BasicDataset('./data/train/imgs/', './data/train/masks/', img_scale, mask_suffix='')
    val_dataset = BasicDataset('./data/val/imgs/', './data/val/masks/', img_scale, mask_suffix='')
    n_train = len(train_dataset)
    n_val = len(val_dataset)

    # 2. 创建加载器
    num_workers = min(4, os.cpu_count()) if os.name == 'nt' else min(8, os.cpu_count())
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_args)

    # 3. WandB
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  img_scale=img_scale, amp=amp))
    
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    # 4. 优化器准备 (差分学习率逻辑)
    # ==============================================================================
    # 针对 ResNet/ConvNeXt 等骨干网络的参数筛选
    # 这里做一个简单的判断：如果模型有 'enc_model' (ConvNeXt) 或 'layer1' (ResNet) 属性，则视为有 backbone
    # ==============================================================================
    backbone_params_ids = []
    
    # 判断是否启用差分学习率
    use_differential_lr = False
    if backbone_lr_scale < 1.0:
        # 检查是否使用了预训练骨干 (根据新模型的属性)
        if hasattr(model, 'encoder_name') and model.encoder_name in ['resnet', 'cnextv2']:
            use_differential_lr = True
        # 兼容旧代码逻辑
        elif hasattr(model, 'use_resnet_encoder') and model.use_resnet_encoder:
            use_differential_lr = True

    if use_differential_lr:
        logging.info(f'✨ 启用差分学习率策略: Backbone Scale = {backbone_lr_scale}')
        
        # 收集骨干参数 ID
        # 策略：遍历模型的子模块，找到常见的 backbone 名字
        backbone_names = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', # ResNet
                          'enc_stem', 'enc_model'] # ConvNeXt / New Unified Model
        
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

    # 初始化优化器
    if optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(param_groups, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'rmsprop':
        optimizer = optim.RMSprop(param_groups, lr=learning_rate, weight_decay=weight_decay,
                                  momentum=momentum, foreach=True)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    # 去掉 'cuda' 参数，改为 torch.cuda.amp
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

    # 5. 训练循环
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    output = model(images)

                    # 处理多输出 (Mask + Edge)
                    if isinstance(output, tuple):
                        masks_pred, edges_pred = output
                        masks_pred_clipped = torch.clamp(masks_pred, min=-50, max=50)
                        loss_mask = criterion(masks_pred_clipped.squeeze(1), true_masks.float())
                        
                        # Edge Loss
                        true_edges = generate_edge_label(true_masks)
                        edges_pred = F.interpolate(edges_pred, size=true_edges.shape[-2:], mode='bilinear', align_corners=True)
                        pos_weight = torch.tensor([5.0], device=device)
                        loss_edge = F.binary_cross_entropy_with_logits(edges_pred.squeeze(1), true_edges, pos_weight=pos_weight)
                        
                        loss = loss_mask + 0.5 * loss_edge
                        masks_pred = masks_pred_clipped # For logging
                    else:
                        masks_pred = output
                        masks_pred_clipped = torch.clamp(masks_pred, min=-50, max=50)
                        loss = criterion(masks_pred_clipped.squeeze(1), true_masks.float())

                # 异常检测
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.error(f'Loss NaN/Inf detected: {loss.item()}. Skipping batch.')
                    optimizer.zero_grad()
                    continue

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                batch_count += 1
                epoch_loss += loss.item()
                
                experiment.log({'train/loss_batch': loss.item(), 'epoch': epoch})
                pbar.set_postfix(**{'loss': loss.item()})

        # 验证
        avg_epoch_loss = epoch_loss / max(batch_count, 1)
        val_metrics = evaluate(model, val_loader, device, amp)
        
        # 阈值扫描 (简化版日志)
        threshold_res = threshold_scan_evaluate(model, val_loader, device, amp, (0.3, 0.8), 0.05)
        
        scheduler.step()
        
        logging.info(f'Epoch {epoch} | Loss: {avg_epoch_loss:.4f} | Val Dice: {val_metrics["dice"]:.4f} | Best Dice: {threshold_res["best_dice"]:.4f}')
        
        experiment.log({
            'train/epoch_loss': avg_epoch_loss,
            'val/dice': val_metrics['dice'],
            'val/iou': val_metrics['iou'],
            'val/best_dice': threshold_res['best_dice']
        })

        # 保存 Checkpoint
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_dice': val_metrics['dice']
            }
            torch.save(state, str(dir_checkpoint / 'checkpoint_latest.pth'))
            
            # 保存最佳
            best_path = str(dir_checkpoint / 'checkpoint_best.pth')
            if not Path(best_path).exists():
                torch.save(state, best_path)
            else:
                prev = torch.load(best_path).get('val_dice', 0)
                if val_metrics['dice'] > prev:
                    torch.save(state, best_path)
                    logging.info(f'🏆 New Best Dice: {val_metrics["dice"]:.4f}')

    wandb.finish()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    
    # 基础参数
    parser.add_argument('--epochs', '-e', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0, help='Validation percentage')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    parser.add_argument('--start-epoch', type=int, default=1, help='Start epoch')
    
    # 🔥🔥🔥 核心架构选择参数 🔥🔥🔥
    parser.add_argument('--encoder', type=str, default='resnet', 
                        choices=['resnet', 'cnextv2', 'standard'],
                        help='Choose encoder backbone')
    
    parser.add_argument('--decoder', type=str, default='phd', 
                        choices=['phd', 'standard'],
                        help='Choose decoder type')
    
    parser.add_argument('--cnext-type', type=str, default='convnextv2_tiny',
                        help='Specific ConvNeXt model type (e.g., convnextv2_tiny, convnextv2_base)')
    
    # 功能开关
    parser.add_argument('--use-wgn', action='store_true', default=False, help='Enable WGN enhancement')
    parser.add_argument('--use-cafm', action='store_true', default=False, help='Enable CAFM bottleneck')
    parser.add_argument('--use-edge-loss', action='store_true', default=False, help='Enable auxiliary edge loss')
    
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
    
    # WGN 参数 (保留兼容性)
    parser.add_argument('--wgn-base-order', type=int, default=3)
    parser.add_argument('--wgn-orders', type=str, default=None)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # 处理 WGN orders
    wgn_orders = None
    if args.use_wgn:
        if args.wgn_orders:
            orders_list = [int(x) for x in args.wgn_orders.split(',')]
            wgn_orders = {
                'layer1': (orders_list[0], orders_list[1]),
                'layer2': (orders_list[2], orders_list[3]),
                'layer3': (orders_list[4], orders_list[5])
            }
        else:
            base = args.wgn_base_order
            wgn_orders = {
                'layer1': (base, base-1),
                'layer2': (base+1, base),
                'layer3': (base+2, base+1)
            }

    # 🔥🔥🔥 实例化统一模型 🔥🔥🔥
    logging.info(f"🚀 Building Model: Encoder={args.encoder.upper()}, Decoder={args.decoder.upper()}")
    if args.encoder == 'cnextv2':
        logging.info(f"   Model Type: {args.cnext_type}")
        
    model = UNet(
        n_channels=3,
        n_classes=args.classes,
        bilinear=args.bilinear,
        # 架构选择
        encoder_name=args.encoder,
        decoder_name=args.decoder,
        cnext_type=args.cnext_type,
        # 功能模块
        use_wgn=args.use_wgn,
        use_cafm=args.use_cafm,
        use_edge_loss=args.use_edge_loss,
        wgn_orders=wgn_orders
    )
    
    model = model.to(memory_format=torch.channels_last)
    model.to(device=device)

    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"🔥 Total Parameters: {total_params/1e6:.2f}M")

    # 加载权重
    checkpoint_to_load = None
    if args.load:
        ckpt = torch.load(args.load, map_location=device)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            checkpoint_to_load = ckpt
        else:
            model.load_state_dict(ckpt)
        logging.info(f'Model loaded from {args.load}')

    # 开始训练
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
            optimizer_type=args.optimizer
        )
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt checkpoint')