# test.py

import argparse
import logging
import os
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import numpy as np

# --- 导入您项目中的核心模块 ---
from unet import UNet
from utils.data_loading import BasicDataset
from utils.dice_score import dice_coeff

# 一个小的常量，用于数值稳定
_EPS = 1e-6


def test_model(
        net,
        device,
        test_loader,
        threshold: float,
        amp: bool = False,
        save_predictions: bool = False,
        output_dir: str = 'data/test/predictions'
):
    """
    在测试集上评估模型的最终性能。
    """
    net.eval()
    num_test_batches = len(test_loader)

    total_tp = 0
    total_fp = 0
    total_fn = 0

    if save_predictions:
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f'Prediction masks will be saved to {output_dir}')

    with torch.no_grad():
        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
            for i, batch in enumerate(tqdm(test_loader, total=num_test_batches, desc='Testing round', unit='batch')):
                images, true_masks = batch['image'], batch['mask']

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                # 🔥【修改 1】获取输出并进行兼容性处理
                output = net(images)

                # 虽然 eval 模式通常只返回 Mask，但为了防止代码逻辑变动导致报错，加一个判断更稳健
                if isinstance(output, tuple):
                    masks_pred = output[0]  # 只取 Mask，丢弃 Edge
                else:
                    masks_pred = output

                # 裁剪logits以保证一致性
                masks_pred_clipped = torch.clamp(masks_pred, min=-50, max=50)

                if net.n_classes == 1:
                    # 二分类任务
                    pred_probs = torch.sigmoid(masks_pred_clipped)
                    pred_binary = (pred_probs > threshold).float()
                    true_binary = true_masks.float()

                    # 展平以便计算
                    p_flat = pred_binary.view(-1)
                    t_flat = true_binary.view(-1)

                    # 累加 TP, FP, FN
                    total_tp += (p_flat * t_flat).sum()
                    total_fp += (p_flat * (1 - t_flat)).sum()
                    total_fn += ((1 - p_flat) * t_flat).sum()

                    # 保存预测图像
                    if save_predictions:
                        start_idx = i * test_loader.batch_size
                        for j in range(pred_binary.shape[0]):
                            idx = start_idx + j
                            if idx < len(test_loader.dataset.ids):
                                file_id = test_loader.dataset.ids[idx]
                                pred_mask_np = pred_binary[j].squeeze().cpu().numpy().astype(np.uint8) * 255
                                pred_mask_img = Image.fromarray(pred_mask_np)
                                pred_mask_img.save(os.path.join(output_dir, f'{file_id}_pred.png'))
                else:
                    logging.warning("Multi-class evaluation not implemented in this test script.")
                    pass

    # --- 计算全局指标 ---
    dice = (2 * total_tp + _EPS) / (2 * total_tp + total_fp + total_fn + _EPS)
    iou = (total_tp + _EPS) / (total_tp + total_fp + total_fn + _EPS)
    precision = (total_tp + _EPS) / (total_tp + total_fp + _EPS)
    recall = (total_tp + _EPS) / (total_tp + total_fn + _EPS)
    f1 = (2 * precision * recall + _EPS) / (precision + recall + _EPS)

    metrics = {
        'dice': float(dice),
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }

    return metrics


def get_args():
    parser = argparse.ArgumentParser(description='Test a trained U-Net model on a test set')
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Path to the trained model checkpoint (.pth file)')
    parser.add_argument('--test-img-dir', type=str, default='data/test/imgs/',
                        help='Directory of test images')
    parser.add_argument('--test-mask-dir', type=str, default='data/test/masks/',
                        help='Directory of test masks')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images (must match training)')
    parser.add_argument('--threshold', '-t', type=float, default=None,
                        help='Fixed threshold. If not set, use best_threshold from checkpoint.')
    parser.add_argument('--batch-size', '-b', type=int, default=8, help='Batch size for testing')
    parser.add_argument('--save-preds', action='store_true', default=False, help='Save prediction masks')
    parser.add_argument('--output-dir', type=str, default='data/test/predictions',
                        help='Directory to save prediction masks')

    # --- 🔥【修改 2】补全模型结构参数 (必须与 train.py 一致) ---
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    parser.add_argument('--use-cafm', action='store_true', default=False, help='Enable Advanced CAFM module')
    parser.add_argument('--use-resnet-encoder', action='store_true', default=False, help='Use ResNet50 encoder')

    # WGN 相关参数 (如果不加这些，加载权重时会报错)
    parser.add_argument('--use-wgn-enhancement', action='store_true', default=False, help='Use WGN enhancement')
    parser.add_argument('--wgn-base-order', type=int, default=3, help='Base order for WGN blocks')
    parser.add_argument('--wgn-orders', type=str, default=None, help='Custom WGN orders')
 # 在 test.py 的 get_args() 函数里添加：
    parser.add_argument('--use-edge-loss', action='store_true', default=False,
                        help='Enable auxiliary edge decoder (needed for loading weights)')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # 🔥【修改 3】处理 WGN Orders (逻辑复制自 train.py)
    wgn_orders = None
    if args.use_wgn_enhancement and args.wgn_orders:
        orders_list = [int(x) for x in args.wgn_orders.split(',')]
        if len(orders_list) == 6:
            wgn_orders = {
                'layer1': (orders_list[0], orders_list[1]),
                'layer2': (orders_list[2], orders_list[3]),
                'layer3': (orders_list[4], orders_list[5])
            }
    elif args.use_wgn_enhancement:
        base = args.wgn_base_order
        wgn_orders = {
            'layer1': (base, base - 1),
            'layer2': (base + 1, base),
            'layer3': (base + 2, base + 1)
        }

    # 1. 创建模型 (必须传入 wgn_orders 和 use_wgn_enhancement)
    model = UNet(n_channels=3,
                 n_classes=args.classes,
                 bilinear=args.bilinear,
                 use_advanced_cafm=args.use_cafm,
                 use_resnet_encoder=args.use_resnet_encoder,
                 use_wgn_enhancement=args.use_wgn_enhancement,  # 🔥 传入参数
                 wgn_orders=wgn_orders,
                 use_edge_loss=args.use_edge_loss)  # 🔥 传入参数

    # 2. 加载模型权重
    logging.info(f'Loading model from {args.model}')
    try:
        checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    except FileNotFoundError:
        logging.error(f"Checkpoint file not found at {args.model}")
        sys.exit(1)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info('Loaded model weights from a full checkpoint.')
        if 'val_dice' in checkpoint:
            logging.info(f'Best Val Dice: {checkpoint["val_dice"]:.4f}')
    else:
        model.load_state_dict(checkpoint)
        logging.warning('Loaded a legacy checkpoint.')

    model.to(device=device)

    # 3. 创建测试数据集
    try:
        test_dataset = BasicDataset(args.test_img_dir, args.test_mask_dir, args.scale)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=max(1, os.cpu_count() // 2), pin_memory=True, drop_last=False
        )
    except Exception as e:
        logging.error(f"Dataset Error: {e}")
        sys.exit(1)

    # 4. 确定阈值
    if args.threshold is not None:
        best_threshold = args.threshold
    elif 'best_threshold_f1' in checkpoint:
        best_threshold = checkpoint['best_threshold_f1']
        logging.info(f"Using best F1 threshold: {best_threshold:.4f}")
    else:
        best_threshold = 0.5

    # 5. 执行测试
    final_metrics = test_model(
        model, device, test_loader,
        threshold=best_threshold, amp=False,
        save_predictions=args.save_preds, output_dir=args.output_dir
    )

    # 6. 打印结果
    print("\n" + "=" * 50)
    print("         Final Test Set Evaluation Report")
    print("-" * 50)
    for k, v in final_metrics.items():
        print(f"  - Global {k.upper()}: {v:.4f}")
    print("=" * 50)