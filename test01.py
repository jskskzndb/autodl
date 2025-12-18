# test01.py (适配 DCNv3 / D-UBM)

import argparse
import logging
import os
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import numpy as np

# 🔥 导入统一版 UNet (确保路径正确)
from unet import UNet
from utils.data_loading import BasicDataset

# 这里的 Dice 计算函数，如果你 utils 里没有，可以注释掉下面这行
# from utils.dice_score import dice_coeff 

_EPS = 1e-6

def test_model(
        net, device, test_loader, threshold, amp=False, save_predictions=False, output_dir='data/test/predictions'
):
    net.eval()
    num_test_batches = len(test_loader)
    total_tp = 0; total_fp = 0; total_fn = 0

    if save_predictions:
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f'Prediction masks will be saved to {output_dir}')

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=amp):
            for i, batch in enumerate(tqdm(test_loader, total=num_test_batches, desc='Testing', unit='batch')):
                images, true_masks = batch['image'], batch['mask']
                images = images.to(device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device, dtype=torch.long)

                # 🔥 兼容多输出 (训练时 D-UBM 可能返回 tuple，测试时只取第一个 logits)
                output = net(images)
                if isinstance(output, tuple):
                    masks_pred = output[0]
                else:
                    masks_pred = output

                masks_pred = torch.clamp(masks_pred, min=-50, max=50)

                if net.n_classes == 1:
                    pred_probs = torch.sigmoid(masks_pred)
                    pred_binary = (pred_probs > threshold).float()
                    true_binary = true_masks.float()

                    p_flat = pred_binary.view(-1)
                    t_flat = true_binary.view(-1)
                    
                    total_tp += (p_flat * t_flat).sum()
                    total_fp += (p_flat * (1 - t_flat)).sum()
                    total_fn += ((1 - p_flat) * t_flat).sum()

                    if save_predictions:
                        start_idx = i * test_loader.batch_size
                        for j in range(pred_binary.shape[0]):
                            idx = start_idx + j
                            if idx < len(test_loader.dataset.ids):
                                file_id = test_loader.dataset.ids[idx]
                                pred_mask_np = pred_binary[j].squeeze().cpu().numpy().astype(np.uint8) * 255
                                pred_mask_img = Image.fromarray(pred_mask_np)
                                pred_mask_img.save(os.path.join(output_dir, f'{file_id}_pred.png'))

    # 计算指标
    dice = (2 * total_tp + _EPS) / (2 * total_tp + total_fp + total_fn + _EPS)
    iou = (total_tp + _EPS) / (total_tp + total_fp + total_fn + _EPS)
    precision = (total_tp + _EPS) / (total_tp + total_fp + _EPS)
    recall = (total_tp + _EPS) / (total_tp + total_fn + _EPS)
    f1 = (2 * precision * recall + _EPS) / (precision + recall + _EPS)

    return {'dice': float(dice), 'iou': float(iou), 'precision': float(precision), 'recall': float(recall), 'f1': float(f1)}

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
    
    # 架构参数 (必须与训练时一致)
    parser.add_argument('--encoder', type=str, default='resnet', choices=['resnet', 'cnextv2', 'standard'])
    parser.add_argument('--decoder', type=str, default='phd', choices=['phd', 'standard'])
    parser.add_argument('--cnext-type', type=str, default='convnextv2_tiny')
    parser.add_argument('--bilinear', action='store_true', default=False)
    parser.add_argument('--classes', '-c', type=int, default=1)
    
    # 功能开关
    parser.add_argument('--use-wgn-enhancement', action='store_true', default=False)
    parser.add_argument('--use-cafm', action='store_true', default=False)
    parser.add_argument('--use-edge-loss', action='store_true', default=False)
    
    # 🔥🔥🔥 [关键修改 1] 新增 DCN/D-UBM 开关 🔥🔥🔥
    parser.add_argument('--use-dcn', action='store_true', default=False, help='Enable DCNv3')
    parser.add_argument('--use-dubm', action='store_true', default=False, help='Enable D-UBM (SOTA)')
    parser.add_argument('--use-strg', action='store_true', default=False)
    parser.add_argument('--use-dual-stream', action='store_true', default=False) # 🔥 修复报错的关键
    parser.add_argument('--use-dsis', action='store_true', default=False, help='Enable Dual-Stream Interactive Skip Module')
    # WGN 参数
    parser.add_argument('--wgn-base-order', type=int, default=3)
    parser.add_argument('--wgn-orders', type=str, default=None)

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # WGN Orders 构建
    wgn_orders = None
    if args.use_wgn_enhancement:
        if args.wgn_orders:
            orders_list = [int(x) for x in args.wgn_orders.split(',')]
            wgn_orders = {'layer1': (orders_list[0], orders_list[1]), 'layer2': (orders_list[2], orders_list[3]), 'layer3': (orders_list[4], orders_list[5])}
        else:
            base = args.wgn_base_order
            wgn_orders = {'layer1': (base, base-1), 'layer2': (base+1, base), 'layer3': (base+2, base+1)}

    # 🔥 1. 实例化模型 (结构必须与训练时完全一致)
    logging.info(f"🚀 Building Model: Encoder={args.encoder.upper()}, Decoder={args.decoder.upper()}")
    
    if args.use_dubm:
        logging.info("   ✨ Mode: D-UBM Enabled (SOTA 2)")
    elif args.use_dcn:
        logging.info("   ✨ Mode: DCNv3 Enabled (SOTA 1)")

    model = UNet(
        n_channels=3,
        n_classes=args.classes,
        bilinear=args.bilinear,
        encoder_name=args.encoder,
        decoder_name=args.decoder,
        cnext_type=args.cnext_type,
        use_wgn_enhancement=args.use_wgn_enhancement,
        use_cafm=args.use_cafm,
        use_edge_loss=args.use_edge_loss,
        wgn_orders=wgn_orders,
        # 🔥🔥🔥 [关键修改 2] 传入参数 🔥🔥🔥
        use_dcn_in_phd=args.use_dcn,
        use_dsis=args.use_dsis,
        use_dubm=args.use_dubm,
        use_dual_stream=args.use_dual_stream # 🔥 传入双流开关 
    )

    # 2. 加载权重
    logging.info(f'Loading model from {args.model}')
    try:
        checkpoint = torch.load(args.model, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info('✅ Loaded model weights successfully.')
        else:
            model.load_state_dict(checkpoint)
            logging.warning('⚠️ Loaded legacy weights structure.')
    except Exception as e:
        logging.error(f"❌ Checkpoint Load Error: {e}")
        logging.error("💡 Hint: Did you forget to add '--use-dubm' or '--use-dcn' in arguments?")
        sys.exit(1)

    model.to(device)

    # 3. 数据集
    try:
        test_dataset = BasicDataset(args.test_img_dir, args.test_mask_dir, args.scale)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=max(1, os.cpu_count() // 2), pin_memory=True, drop_last=False
        )
    except Exception as e:
        logging.error(f"Dataset Error: {e}")
        sys.exit(1)

    # 4. 阈值逻辑
    if args.threshold is not None:
        best_threshold = args.threshold
        logging.info(f"Using manual threshold: {best_threshold}")
    elif 'best_threshold_f1' in checkpoint: 
        best_threshold = checkpoint['best_threshold_f1']
        logging.info(f"Using optimal threshold from training: {best_threshold:.4f}")
    else:
        best_threshold = 0.5
        logging.info(f"Using default threshold: {best_threshold}")

    # 5. 测试
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