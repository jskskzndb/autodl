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
import cv2
from evaluate import evaluate, threshold_scan_evaluate
from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss
import csv
from timm.scheduler import CosineLRScheduler  # 🔥 [新增 1] 添加这一行
from utils.losses import FocalLoss, CombinedLoss, DiceLossOnly, EdgeLoss, compute_prototype_ortho_loss
from utils.utils import log_grad_stats

# 🔥 确保引用的是刚才修改过的文件
from unet.unet_universal3 import UniversalUNet as UNet

import random
import csv
import os

class MetricLogger:
    def __init__(self, save_path):
        self.save_path = save_path
        # 初始化 CSV 文件，如果文件不存在则写入表头
        if not os.path.exists(self.save_path):
            with open(self.save_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Dice', 'Precision', 'Recall', 'F1', 'IoU', 'LR'])

    def log(self, epoch, train_loss, val_metrics, lr):
        with open(self.save_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.4f}",
                f"{val_metrics['loss']:.4f}",
                f"{val_metrics['dice']:.4f}",
                f"{val_metrics['precision']:.4f}",
                f"{val_metrics['recall']:.4f}",
                f"{val_metrics['f1']:.4f}",
                f"{val_metrics['iou']:.4f}",
                f"{lr:.8f}"
            ])
def setup_seed(seed):
    import random
    import numpy as np
    import os  # 👈 确保导入 os
    os.environ['PYTHONHASHSEED'] = str(seed) # 👈 新增：控制哈希算法随机性
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)       # 👈 修正：单卡使用这个
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 保证算法结果确定
    torch.backends.cudnn.benchmark = False   # 建议注释掉。设为False会变慢，通常不值得
# 在 setup_seed 下方添加这个函数
def worker_init_fn(worker_id):
    import random
    import numpy as np
    # 获取 PyTorch 传下来的种子
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
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
            
            # 兼容多输出的情况，只取第一个主输出进行可视化
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
                
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

# 🔥 [新增] 实时生成 Body 和 Edge 的真值 (GPU版)
# 利用 MaxPool 实现形态学操作，速度极快。
def generate_body_edge_targets(masks, edge_width=5, device='cuda'):
    if masks.ndim == 3:
        masks = masks.unsqueeze(1)
    
    masks_float = masks.float()
    padding = edge_width // 2
    
    # 膨胀 (Dilation)
    dilated = F.max_pool2d(masks_float, kernel_size=edge_width, stride=1, padding=padding)
    # 腐蚀 (Erosion)
    eroded = -F.max_pool2d(-masks_float, kernel_size=edge_width, stride=1, padding=padding)
    
    # Edge = 膨胀 - 腐蚀
    edge = dilated - eroded
    # Body = 腐蚀
    body = eroded
    
    # 二值化保护
    body = (body > 0.5).float()
    edge = (edge > 0.5).float()
    
    return body.to(device), edge.to(device)

def generate_edge_tensor(mask, edge_width=3):
    """
    [旧版] 形态学边缘生成 (OpenCV)
    保留此函数以兼容旧的 EdgeLoss 调用（如果不使用 Decouple）
    """
    # 1. 记录原始所在的设备 (CPU or CUDA)
    target_device = mask.device
    
    # 2. 转换为 Numpy (CPU) 进行 OpenCV 处理
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = mask
        
    # 如果维度是 [B, 1, H, W]，压缩为 [B, H, W]
    if mask_np.ndim == 4:
        mask_np = mask_np.squeeze(1)
        
    B, H, W = mask_np.shape
    edges = []
    
    # 定义结构元素 (决定边缘宽度)
    # cv2.MORPH_RECT 表示矩形结构
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_width, edge_width))
    
    for i in range(B):
        # 确保是 uint8 类型
        m = mask_np[i].astype(np.uint8)
        
        # 膨胀 - 腐蚀
        dilated = cv2.dilate(m, kernel, iterations=1)
        eroded = cv2.erode(m, kernel, iterations=1)
        
        # 相减得到边缘
        edge = dilated - eroded
        
        # 二值化保护 (大于0的都算边缘)
        edge[edge > 0] = 1
        edges.append(edge)
    
    # 3. 堆叠并转回 Tensor
    edges = np.stack(edges, axis=0) # [B, H, W]
    edges_tensor = torch.from_numpy(edges).float().unsqueeze(1) # [B, 1, H, W]
    
    # 4. 移回原来的设备 (GPU)
    return edges_tensor.to(target_device)

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
        accumulation_steps: int = 1,  # <--- 🔥 必须加上这一行！
        use_decouple: bool = False # 🔥 [新增] 传入是否开启解耦
        
):
    # 1. 数据准备
    train_dataset = BasicDataset(dir_img, dir_mask, img_scale, mask_suffix='', augment=True)
    val_dataset = BasicDataset(val_dir_img, val_dir_mask, img_scale, mask_suffix='', augment=False)
    n_train = len(train_dataset)
    n_val = len(val_dataset)

    # 2. DataLoader
    num_workers = min(4, os.cpu_count()) if os.name == 'nt' else min(8, os.cpu_count())
    # 🔥🔥🔥 [新增] 定义一个随机数生成器，并设定种子
    g = torch.Generator()
    g.manual_seed(42)  # 这里填你想要的种子，建议和全局种子保持一致
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, worker_init_fn=worker_init_fn, generator=g, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True, worker_init_fn=worker_init_fn, generator=g, **loader_args)

    # 3. WandB 初始化 (保留原有配置)
    # 🔥 [新增] 必须先定义 run_id，否则后面会报错
    run_id = None
    if checkpoint_to_load is not None and 'wandb_id' in checkpoint_to_load:
        run_id = checkpoint_to_load['wandb_id']
        logging.info(f"🔗 检测到 WandB ID: {run_id}，正在恢复连接...")
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must', id=run_id)
    # 🔥 [新增] 初始化本地 CSV Logger
# 建议保存在 checkpoints 目录下，或者你指定的目录
    csv_log_path = dir_checkpoint / "training_metrics.csv"
    csv_logger = MetricLogger(csv_log_path)
    logging.info(f"📊 本地指标记录器已就绪: {csv_log_path}")
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
        Decoupling:      {use_decouple}
    ''')

    # 4. 优化器与差分学习率
    backbone_params_ids = []
    use_differential_lr = False
    
    # 🔥 修改 1: 只要你设置了倍率 < 1.0，就无条件开启，不要去检查 model.encoder_name
    if backbone_lr_scale < 1.0:
        use_differential_lr = True

    if use_differential_lr:
        logging.info(f'✨ 启用差分学习率策略: Backbone Scale = {backbone_lr_scale}')
        
        
        backbone_names = ['spatial_encoder'] 
        
        # 寻找骨干参数
        found_backbone_layer = False
        for name, module in model.named_children():
            if name in backbone_names:
                found_backbone_layer = True
                # logging.info(f"   -> 捕获骨干层: {name}") # 调试时可以解开这行
                for param in module.parameters():
                    backbone_params_ids.append(id(param))
        
        # 🔥 修改 2: 安全检查
        # 如果找了一圈没找到骨干层 (名字不对)，就自动回退到统一学习率，防止报错
        if not found_backbone_layer or len(backbone_params_ids) == 0:
            logging.warning("⚠️ 警告: 启用了差分学习率，但没在模型里找到名为 'enc_model' 或 'encoder' 的层！")
            logging.warning("   -> 将自动回退到统一学习率。请检查 UNet 代码中骨干网的变量名。")
            param_groups = model.parameters()
        else:
            # 正常分离参数
            backbone_params = filter(lambda p: id(p) in backbone_params_ids, model.parameters())
            base_params = filter(lambda p: id(p) not in backbone_params_ids, model.parameters())
            
            param_groups = [
                {'params': base_params, 'lr': learning_rate}, 
                {'params': backbone_params, 'lr': learning_rate * backbone_lr_scale}
            ]
            logging.info(f"   -> 成功分离参数: 骨干网将使用 lr={learning_rate * backbone_lr_scale:.2e}")
            
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

    # -------------------------------------------------------------------------
    # 🔥 [新增 3] 使用 timm 的 CosineLRScheduler 实现 Warmup
    # -------------------------------------------------------------------------
    # 自动设定 Warmup 轮数 (总轮数的 10%，最少 5 轮)
    warmup_epochs = int(epochs * 0.1)
    if warmup_epochs < 5: warmup_epochs = 5
    
    logging.info(f"📅 学习率调度: 总轮数 {epochs}, Warmup {warmup_epochs} 轮")

    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=epochs,            # 周期长度 (总 Epoch)
        lr_min=1e-6,                 # 最小学习率 (训练结束时降到多少)
        warmup_t=warmup_epochs,      # Warmup 持续多少个 Epoch
        warmup_lr_init=1e-6,         # Warmup 初始学习率 (从这个值线性升到目标 lr)
        warmup_prefix=False          # 设为 True，表示 warmup 过程包含在总周期计算逻辑中
    )
    
    # 🔥 初始化一下，确保第 1 个 Epoch 就开始使用 Warmup 学习率
    scheduler.step(0)
    # -------------------------------------------------------------------------
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
    # 🔥 [核心修改] 恢复 global_step
    if checkpoint_to_load is not None and 'global_step' in checkpoint_to_load:
        global_step = checkpoint_to_load['global_step']
        logging.info(f"📉 继续从 Global Step {global_step} 开始记录日志")
    else:
        global_step = 0

    # 恢复 Checkpoint
    if checkpoint_to_load is not None:
        try:
            optimizer.load_state_dict(checkpoint_to_load['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint_to_load['scheduler_state_dict'])
            if 'grad_scaler_state_dict' in checkpoint_to_load and amp:
                grad_scaler.load_state_dict(checkpoint_to_load['grad_scaler_state_dict'])
            
            # 🔥🔥🔥 [修改这里] 加上 .cpu() 🔥🔥🔥
            if 'cpu_rng_state' in checkpoint_to_load:
                # 1. 恢复 CPU 随机数 (必须是 CPU Tensor)
                torch.set_rng_state(checkpoint_to_load['cpu_rng_state'].cpu())
                
                # 2. 恢复 CUDA 随机数 (通常 set_rng_state 也偏好 CPU tensor 作为输入来设置 GPU 状态)
                try:
                    if 'cuda_rng_state' in checkpoint_to_load:
                        torch.cuda.set_rng_state(checkpoint_to_load['cuda_rng_state'].cpu())
                except Exception as e:
                    logging.warning(f"⚠️ 无法恢复 CUDA 随机状态 (可能显卡数量不一致): {e}")

                # 3. 恢复 Numpy 和 Python 随机数 (这些不受 map_location 影响)
                if 'numpy_rng_state' in checkpoint_to_load:
                    np.random.set_state(checkpoint_to_load['numpy_rng_state'])
                if 'py_rng_state' in checkpoint_to_load:
                    random.setstate(checkpoint_to_load['py_rng_state'])
                
                logging.info("🎲 随机数生成器状态已完美恢复！")

            logging.info('✅ 训练状态完全恢复')
        except Exception as e:
            logging.warning(f'⚠️ 恢复状态失败: {e}')
    # ============================================================
    # 5. 训练循环
    # ============================================================
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_grad_norms = []
        batch_count = 0
        
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for i, batch in enumerate(train_loader):
                images, true_masks = batch['image'], batch['mask']
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    output = model(images)
                    
                    # 🔥🔥🔥 [关键修复] 定义数值截断函数 🔥🔥🔥
                    # 防止模型输出过大导致 BCE Loss 计算出 NaN
                    def clamp_logits(x):
                        return torch.clamp(x, min=-50, max=50)

                    loss = 0.0
                    
                    # ==========================================================
                    # 🔥 [核心修改] 统一的输出解析逻辑 (兼容 DS 和 Decouple)
                    # ==========================================================
                    if isinstance(output, list):
                        # 1. 主分割 Loss (列表第一个永远是 Final Output)
                        pred_final = clamp_logits(output[0])
                        l_main = calc_loss(pred_final, true_masks, loss_combination, focal_alpha, focal_gamma)
                        loss += l_main
                        
                        idx = 1 # 指针，指向下一个要处理的输出
                        
                        # 2. 处理 Deep Supervision (Aux Head)
                        if model.use_deep_supervision:
                            # 假设有两个 aux head
                            if idx + 1 < len(output):
                                pred_aux2 = clamp_logits(output[idx])
                                pred_aux3 = clamp_logits(output[idx+1])
                                idx += 2
                                # 上采样对齐
                                pred_aux2 = F.interpolate(pred_aux2, size=true_masks.shape[1:], mode='bilinear', align_corners=True)
                                pred_aux3 = F.interpolate(pred_aux3, size=true_masks.shape[1:], mode='bilinear', align_corners=True)
                                
                                l_aux2 = calc_loss(pred_aux2, true_masks, loss_combination, focal_alpha, focal_gamma)
                                l_aux3 = calc_loss(pred_aux3, true_masks, loss_combination, focal_alpha, focal_gamma)
                                loss += 0.5 * l_aux2 + 0.4 * l_aux3

                        # 3. 🔥 处理 Decouple (Body + Edge)
                        # 判断条件：模型开启了解耦 且 列表里还有足够元素
                        # (注意：兼容 DataParallel，用 getattr 获取 use_decouple)
                        is_decouple = getattr(model, 'use_decouple', False) or getattr(model.module, 'use_decouple', False) if hasattr(model, 'module') else False
                        
                        if is_decouple and idx + 1 < len(output):
                            pred_body = clamp_logits(output[idx])
                            pred_edge = clamp_logits(output[idx+1])
                            idx += 2
                            
                            # A. 实时生成 Body/Edge 真值
                            target_body, target_edge = generate_body_edge_targets(true_masks, edge_width=5, device=device)
                            
                            # B. Body Loss (Dice + BCE)
                            # 对齐尺寸(防止模型中有下采样未恢复)
                            if pred_body.shape[2:] != target_body.shape[2:]:
                                pred_body = F.interpolate(pred_body, size=target_body.shape[2:], mode='bilinear')
                            
                            l_body = F.binary_cross_entropy_with_logits(pred_body, target_body) + \
                                     dice_loss(torch.sigmoid(pred_body), target_body)
                                     
                            # C. Edge Loss (Weighted BCE)
                            if pred_edge.shape[2:] != target_edge.shape[2:]:
                                pred_edge = F.interpolate(pred_edge, size=target_edge.shape[2:], mode='bilinear')
                                
                            pos_weight = torch.tensor([5.0], device=device) # 边缘正样本加权
                            l_edge = F.binary_cross_entropy_with_logits(pred_edge, target_edge, pos_weight=pos_weight)
                            
                            # D. 累加 Loss
                            loss += lambda_body * l_body + lambda_edge * l_edge

                        # 4. 处理旧的 SFDA Edge (如果还有剩余)
                        if idx < len(output):
                            # 这里可以忽略，或者计算旧的边缘 loss
                            pass
                            
                    else:
                        # 单输出情况
                        pred_main = clamp_logits(output)
                        loss = calc_loss(pred_main, true_masks, loss_combination, focal_alpha, focal_gamma)
                        
                        # 隐式边缘监督 (Sobel Edge Loss)
                        if lambda_edge > 0 and not use_decouple:
                            try:
                                loss_e = edge_criterion(pred_main.float(), true_masks.float())
                                loss += lambda_edge * loss_e
                            except NameError:
                                pass 
                    
                    # -----------------------------------------------------------
                    # 原型正交 Loss (如果有)
                    # -----------------------------------------------------------
                    lambda_ortho = 0.0 # 🔥 [修正] 建议开启为 0.01，原代码为 0.0 会导致不计算
                    if lambda_ortho > 0:
                         ortho_loss = compute_prototype_ortho_loss(model, device=device)
                         loss += lambda_ortho * ortho_loss

                    # -----------------------------------------------------------
                    # Loss 归一化 (用于梯度累积)
                    # -----------------------------------------------------------
                    loss = loss / accumulation_steps
                
                # 异常检测
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.error(f'Loss NaN/Inf detected: {loss.item()}. Skipping batch.')
                    optimizer.zero_grad()
                    continue
                
                # 反向传播
                grad_scaler.scale(loss).backward()
                
                # 🔥 [修改] 只有达到累计步数，或 epoch 结束时才更新
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                    grad_scaler.unscale_(optimizer)
                      # 🔥 [新增] 严格的梯度检查：如果梯度有 inf/nan，直接跳过这一步更新
                    grad_is_valid = True
                    for param in model.parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                grad_is_valid = False
                                break
                    
                    if not grad_is_valid:
                        logging.warning(f"⚠️ Epoch {epoch} Step {i}: Gradient Explosion detected (Inf/NaN). Skipping step.")
                        optimizer.zero_grad() # 丢弃这次的梯度
                        # 不做 step，也不做 update
                    else:
                        # 2.2 梯度裁剪 (Clip Gradient Norm)
                        # 🔥 [建议] 把 max_norm 从 1.0 降低到 0.5 或 0.1，对 Transformer 结构更稳
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                        
                        epoch_grad_norms.append(grad_norm.item())

                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    
                    # 更新完后才清零
                    optimizer.zero_grad(set_to_none=True)

                    # WandB 实时日志 (还原 loss 数值用于显示)
                    experiment.log({
                        'train/loss_batch': loss.item() * accumulation_steps, 
                        'train/grad_norm': grad_norm.item(), 
                        'global_step': global_step
                    })
                    pbar.set_postfix(**{'loss': loss.item() * accumulation_steps, 'grad': grad_norm.item()})
                    global_step += 1

                pbar.update(images.shape[0])
                batch_count += 1
                # 累加 Loss 用于显示 Epoch 平均值 (还原数值)
                epoch_loss += loss.item() * accumulation_steps

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

        # 🔥 [修改] 更新学习率 (必须传入当前 epoch 数值)
        # 注意: timm 的 step 需要传入 epoch 索引
        scheduler.step(epoch)
        
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
        # 🔥 [新增] 同时写入本地 CSV 文件
        csv_logger.log(epoch, avg_epoch_loss, val_metrics, current_lr)
        

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
                # 🔥 [核心修改] 新增这两行
                'wandb_id': experiment.id,  # 保存身份证号
                'global_step': global_step, # 保存当前步数
                # ... 保留你的其他键值
                # 🔥🔥🔥 [核心补充] 必须保存这 4 个随机状态 🔥🔥🔥
                'cpu_rng_state': torch.get_rng_state(),              # PyTorch CPU 随机状态
                'cuda_rng_state': torch.cuda.get_rng_state(),        # PyTorch GPU 随机状态 (单卡用这个)
                # 'cuda_rng_state': torch.cuda.get_rng_state_all(),  # 如果你是多卡训练，请用这行替换上一行
                'numpy_rng_state': np.random.get_state(),            # Numpy 随机状态 (影响数据增强)
                'py_rng_state': random.getstate(),                   # Python 原生随机状态
            }
            
            # Latest
            torch.save(checkpoint, str(dir_checkpoint / 'checkpoint_latest.pth'))
            # 2. 🔥 [修改点 2] 30轮以后，每一轮都额外保存一个文件
            if epoch > 50:
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
    """
    🔥 [最终完整版] 
    1. 强制 FP32 计算 (V100 保命)
    2. 保留了原本的 try-except 兼容逻辑
    """
    # -----------------------------------------------------------
    # 1. 全局强制转换：进门先安检，统统变 float32
    # -----------------------------------------------------------
    masks_pred = masks_pred.float()
    true_masks = true_masks.float()

    # -----------------------------------------------------------
    # 2. 分情况计算 (注意：下面都不需要再写 .float() 了)
    # -----------------------------------------------------------
    if loss_combination == 'bce':
        criterion = nn.BCEWithLogitsLoss()
        return criterion(masks_pred.squeeze(1), true_masks)
    
    elif loss_combination == 'focal':
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        return criterion(masks_pred.squeeze(1), true_masks)
    
    elif loss_combination == 'dice':
        # 确保 sigmoid 也在 float32 下进行
        return dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks, multiclass=False)
    
    else:
        # -----------------------------------------------------------
        # 3. 组合 Loss (保留你原来的 try-except 逻辑)
        # -----------------------------------------------------------
        try:
            # 尝试调用封装好的类
            criterion = CombinedLoss(loss_combination.split('+'), [1.0, 1.0], focal_alpha, focal_gamma)
            return criterion(masks_pred.squeeze(1), true_masks)
        except:
            # 🔥 Fallback: 手动相加
            # 这里的 input 已经是 float32 了，计算非常安全
            bce = nn.BCEWithLogitsLoss()(masks_pred.squeeze(1), true_masks)
            
            # Dice 需要 sigmoid 后的概率
            dice = dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks, multiclass=False)
            
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
    parser.add_argument('--cnext-type', type=str, default='convnextv2_base')
    
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
    parser.add_argument('--lambda_edge', type=float, default=1.0, help='Weight for the Edge loss (default: 2.0)')
    parser.add_argument('--lambda_body', type=float, default=1.0, help='Weight for the Body loss (default: 1.0)')
    
    # 其他增强模块 (保持原有开关定义，但移除了旧版 Edge Logic 的执行)
    parser.add_argument('--use-wgn-enhancement', action='store_true', default=False)
    parser.add_argument('--use-cafm', action='store_true', default=False)
    parser.add_argument('--use-edge-loss', action='store_true', default=False, help='Legacy WGN Edge Loss (Deprecated logic removed)')
    parser.add_argument('--use-fme', action='store_true', default=False, 
                        help='Enable Frequency-Mamba Enhancement (FME) module')
    parser.add_argument('--no-mfam', action='store_true', help='Disable MFAM for ablation study')
    # WGN 参数
    parser.add_argument('--wgn-base-order', type=int, default=3)
    parser.add_argument('--wgn-orders', type=str, default=None)

    # 优化参数
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'rmsprop'])
    parser.add_argument('--loss-combination', type=str, default='focal+dice')
    parser.add_argument('--loss-weights', type=str, default='1.0,1.0')
    parser.add_argument('--focal-alpha', type=float, default=0.25)
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--momentum', type=float, default=0.999)
    parser.add_argument('--gradient-clipping', type=float, default=1.0)
    parser.add_argument('--backbone-lr-scale', type=float, default=0.1)
    # 🔥 [新增] 梯度累计步数，默认1表示不累计
    parser.add_argument('--accumulation-steps', type=int, default=1, help='Gradient accumulation steps')
# 🔥 [新增 2] 添加预训练权重开关 (1=加载, 0=不加载)
    parser.add_argument('--pretrained', type=int, default=1, help='Load ImageNet weights? 1=Yes, 0=No')
    parser.add_argument('--use-deep-supervision', action='store_true', default=False, help='Enable Deep Supervision')
    return parser.parse_args()
 
if __name__ == '__main__':
    # 🔥🔥🔥 在这里调用，数字随便填（比如 42, 3407, 2023）
    setup_seed(42)
    
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
        decoder_type=args.decoder,
        cnext_type=args.cnext_type,
        # 🔥 [新增 4] 传入预训练参数 (转为布尔值)
        pretrained=(args.pretrained == 1),
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
        use_wavelet_denoise=args.use_wavelet_denoise,  # 👈 传入这个参数
        use_mfam=not args.no_mfam, # 注意这里：如果命令行加了 --no-mfam，则 use_mfam=False
        use_deep_supervision=args.use_deep_supervision, # 🔥 传入参数
        # 🔥 传入 MDBES-Net 解耦参数
        use_decouple=args.use_decouple # 👈 确保这里传入了参数！
    )
    
    model = model.to(memory_format=torch.channels_last)
    model.to(device=device)
    # =================================================================================
    # 🔥 [新增代码] 计算并打印模型参数量
    # =================================================================================
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logging.info(f"""
    📊 Model Summary:
        Total Parameters:     {total_params / 1e6:.2f} M
        Trainable Parameters: {trainable_params / 1e6:.2f} M
        Frozen Parameters:    {(total_params - trainable_params) / 1e6:.2f} M
    """)
    # =================================================================================
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
            lambda_body=args.lambda_body,
            accumulation_steps=args.accumulation_steps, # <--- 🔥 加上这一行！
            use_decouple=args.use_decouple # 👈 确保传给训练函数
        )
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt checkpoint')