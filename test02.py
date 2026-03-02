# test01.py (已集成 TTA 翻转增强功能)
from utils.metrics_distance import compute_hd95, compute_asd
import argparse
import logging
import os
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import numpy as np

# 🔥 导入统一版 UNet
from unet import UNet
from utils.data_loading import BasicDataset

_EPS = 1e-6

def test_model(
        net, device, test_loader, threshold, amp=False, save_predictions=False, 
        output_dir='data/test/predictions', use_tta=False  # 🔥 新增 TTA 参数
):
    net.eval()
    num_test_batches = len(test_loader)
    
    # 计数器
    total_tp = 0; total_fp = 0; total_fn = 0
    
    # 用于存储每张图的距离指标
    hd95_scores = []
    asd_scores = []

    if save_predictions:
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f'Prediction masks will be saved to {output_dir}')

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=amp):
            for i, batch in enumerate(tqdm(test_loader, total=num_test_batches, desc='Testing' if not use_tta else 'Testing with TTA', unit='batch')):
                images, true_masks = batch['image'], batch['mask']
                images = images.to(device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device, dtype=torch.long)

                # --- 1. 原始图像推理 ---
                output = net(images)
                masks_pred = output[0] if isinstance(output, (tuple, list)) else output
                masks_pred = torch.clamp(masks_pred, min=-50, max=50)
                
                if net.n_classes == 1:
                    # 获取原始概率图
                    pred_probs = torch.sigmoid(masks_pred)

                    # --- 🔥 2. TTA 逻辑 (仅当开启时执行) ---
                    if use_tta:
                        # A. 水平翻转增强
                        images_hf = torch.flip(images, dims=[3]) # [B, C, H, W] 的 W 轴翻转
                        out_hf = net(images_hf)
                        logits_hf = out_hf[0] if isinstance(out_hf, (tuple, list)) else out_hf
                        # 翻转回来并累加概率
                        pred_probs += torch.flip(torch.sigmoid(torch.clamp(logits_hf, -50, 50)), dims=[3])

                        # B. 垂直翻转增强
                        images_vf = torch.flip(images, dims=[2]) # [B, C, H, W] 的 H 轴翻转
                        out_vf = net(images_vf)
                        logits_vf = out_vf[0] if isinstance(out_vf, (tuple, list)) else out_vf
                        # 翻转回来并累加概率
                        pred_probs += torch.flip(torch.sigmoid(torch.clamp(logits_vf, -50, 50)), dims=[2])

                        # C. 取平均值 (原图 + 水平翻转 + 垂直翻转 = 3个样本)
                        pred_probs /= 3.0

                    # --- 3. 后续二值化逻辑 (完全继承原代码) ---
                    pred_binary = (pred_probs > threshold).float()
                    true_binary = true_masks.float()

                    # 计算 Dice/IoU 累加
                    p_flat = pred_binary.view(-1)
                    t_flat = true_binary.view(-1)
                    total_tp += (p_flat * t_flat).sum()
                    total_fp += (p_flat * (1 - t_flat)).sum()
                    total_fn += ((1 - p_flat) * t_flat).sum()

                    # 逐张计算 HD95 和 ASD
                    batch_size = pred_binary.shape[0]
                    for b in range(batch_size):
                        pred_np = pred_binary[b].squeeze().cpu().numpy().astype(np.uint8)
                        gt_np = true_binary[b].squeeze().cpu().numpy().astype(np.uint8)
                        
                        hd95_val = compute_hd95(pred_np, gt_np)
                        asd_val = compute_asd(pred_np, gt_np)
                        
                        if not np.isnan(hd95_val):
                            hd95_scores.append(hd95_val)
                        if not np.isnan(asd_val):
                            asd_scores.append(asd_val)

                    # 保存图片
                    if save_predictions:
                        start_idx = i * test_loader.batch_size
                        for j in range(pred_binary.shape[0]):
                            idx = start_idx + j
                            if idx < len(test_loader.dataset.ids):
                                file_id = test_loader.dataset.ids[idx]
                                pred_mask_np = pred_binary[j].squeeze().cpu().numpy().astype(np.uint8) * 255
                                pred_mask_img = Image.fromarray(pred_mask_np)
                                pred_mask_img.save(os.path.join(output_dir, f'{file_id}_pred.png'))

    # --- 汇总结果 ---
    dice = (2 * total_tp + _EPS) / (2 * total_tp + total_fp + total_fn + _EPS)
    iou = (total_tp + _EPS) / (total_tp + total_fp + total_fn + _EPS)
    precision = (total_tp + _EPS) / (total_tp + total_fp + _EPS)
    recall = (total_tp + _EPS) / (total_tp + total_fn + _EPS)
    f1 = (2 * precision * recall + _EPS) / (precision + recall + _EPS)

    avg_hd95 = np.mean(hd95_scores) if len(hd95_scores) > 0 else 0.0
    avg_asd = np.mean(asd_scores) if len(asd_scores) > 0 else 0.0

    return {
        'dice': float(dice), 
        'iou': float(iou), 
        'precision': float(precision), 
        'recall': float(recall), 
        'f1': float(f1),
        'hd95': float(avg_hd95),
        'asd': float(avg_asd)
    }

def get_args():
    parser = argparse.ArgumentParser(description='Test Unified UNet')
    
    # 基础参数
    parser.add_argument('--model', '-m', type=str, required=True, help='Path to .pth file')
    parser.add_argument('--test-img-dir', type=str, default='data/test/imgs/')
    parser.add_argument('--test-mask-dir', type=str, default='data/test/masks/')
    parser.add_argument('--scale', '-s', type=float, default=1.0)
    parser.add_argument('--threshold', '-t', type=float, default=None)
    parser.add_argument('--batch-size', '-b', type=int, default=1)
    parser.add_argument('--save-preds', action='store_true', default=False)
    parser.add_argument('--output-dir', type=str, default='data/test/predictions')
    
    # 架构参数
    parser.add_argument('--encoder', type=str, default='resnet', choices=['resnet', 'cnextv2', 'standard', 'swin'])
    parser.add_argument('--decoder', type=str, default='phd', choices=['phd', 'standard'])
    parser.add_argument('--cnext-type', type=str, default='convnextv2_base')
    parser.add_argument('--bilinear', action='store_true', default=False)
    parser.add_argument('--classes', '-c', type=int, default=1)
    
    # 功能开关
    parser.add_argument('--use-wgn-enhancement', action='store_true', default=False)
    parser.add_argument('--use-cafm', action='store_true', default=False)
    parser.add_argument('--use-edge-loss', action='store_true', default=False)
    parser.add_argument('--use-wavelet-denoise', action='store_true', default=False)
    parser.add_argument('--use-dcn', action='store_true', default=False)
    parser.add_argument('--use-dubm', action='store_true', default=False)
    parser.add_argument('--use-strg', action='store_true', default=False)
    parser.add_argument('--use-dual-stream', action='store_true', default=False)
    parser.add_argument('--use-dsis', action='store_true', default=False)
    parser.add_argument('--use-deep-supervision', action='store_true', default=False)
    parser.add_argument('--use-unet3p', action='store_true', default=False)
    parser.add_argument('--wgn-base-order', type=int, default=3)
    parser.add_argument('--wgn-orders', type=str, default=None)
    parser.add_argument('--use-sparse-skip', action='store_true', default=False)
    
    # 🔥🔥🔥 [新增] TTA 参数 🔥🔥🔥
    parser.add_argument('--use-tta', action='store_true', help='Enable Test-Time Augmentation (H+V Flip)')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # WGN Orders
    wgn_orders = None
    if args.use_wgn_enhancement:
        if args.wgn_orders:
            orders_list = [int(x) for x in args.wgn_orders.split(',')]
            wgn_orders = {'layer1': (orders_list[0], orders_list[1]), 'layer2': (orders_list[2], orders_list[3]), 'layer3': (orders_list[4], orders_list[5])}
        else:
            base = args.wgn_base_order
            wgn_orders = {'layer1': (base, base-1), 'layer2': (base+1, base), 'layer3': (base+2, base+1)}

    # 1. 模型实例化
    model = UNet(
        n_channels=3, n_classes=args.classes, bilinear=args.bilinear,
        encoder_name=args.encoder, decoder_type=args.decoder, cnext_type=args.cnext_type,
        use_wgn_enhancement=args.use_wgn_enhancement, use_cafm=args.use_cafm,
        use_edge_loss=args.use_edge_loss, wgn_orders=wgn_orders,
        use_dcn_in_phd=args.use_dcn, use_dsis=args.use_dsis, use_dubm=args.use_dubm,
        use_strg=args.use_strg, use_dual_stream=args.use_dual_stream,
        use_unet3p=args.use_unet3p, use_wavelet_denoise=args.use_wavelet_denoise,
        use_deep_supervision=args.use_deep_supervision, use_sparse_skip=args.use_sparse_skip
    )

    # 2. 加载权重
    try:
        checkpoint = torch.load(args.model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        logging.info('✅ Loaded weights.')
    except Exception as e:
        logging.error(f"❌ Load Error: {e}")
        sys.exit(1)

    model.to(device)

    # 3. 数据集
    test_dataset = BasicDataset(args.test_img_dir, args.test_mask_dir, args.scale)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # 4. 阈值逻辑
    best_threshold = args.threshold if args.threshold is not None else (checkpoint.get('best_threshold_f1', 0.5) if isinstance(checkpoint, dict) else 0.5)

    # 5. 测试 (传入 use_tta)
    final_metrics = test_model(
        model, device, test_loader, threshold=best_threshold, 
        use_tta=args.use_tta,  # 🔥 传入 TTA 开关
        save_predictions=args.save_preds, output_dir=args.output_dir
    )

    # 6. 打印
    print("\n" + "=" * 50)
    print(f" Final Test Report (TTA={'ON' if args.use_tta else 'OFF'})")
    print("-" * 50)
    for k, v in final_metrics.items():
        print(f"  - {k.upper()}: {v:.4f}")
    print("=" * 50)