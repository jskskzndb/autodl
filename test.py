import argparse
import logging
import os
import sys
import re
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# 引入你的项目模块
from unet import UNet
from utils.data_loading import BasicDataset

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

_EPS = 1e-6

def test_model_silent(net, device, test_loader, threshold=0.5, amp=False):
    """
    静默版测试函数，只返回指标字典，不保存图片，不打印繁杂信息
    """
    net.eval()
    num_test_batches = len(test_loader)
    total_tp = 0; total_fp = 0; total_fn = 0

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=amp):
            for batch in tqdm(test_loader, total=num_test_batches, desc='  evaluating', unit='batch', leave=False):
                images, true_masks = batch['image'], batch['mask']
                images = images.to(device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device, dtype=torch.long)

                # 推理
                output = net(images)
                
                # 兼容性处理
                if isinstance(output, tuple):
                    masks_pred = output[0]
                else:
                    masks_pred = output

                masks_pred = torch.clamp(masks_pred, min=-50, max=50)

                # 二分类指标计算
                pred_probs = torch.sigmoid(masks_pred)
                pred_binary = (pred_probs > threshold).float()
                true_binary = true_masks.float()

                p_flat = pred_binary.view(-1)
                t_flat = true_binary.view(-1)
                
                total_tp += (p_flat * t_flat).sum()
                total_fp += (p_flat * (1 - t_flat)).sum()
                total_fn += ((1 - p_flat) * t_flat).sum()

    # 计算全局指标
    dice = (2 * total_tp + _EPS) / (2 * total_tp + total_fp + total_fn + _EPS)
    iou = (total_tp + _EPS) / (total_tp + total_fp + total_fn + _EPS)
    precision = (total_tp + _EPS) / (total_tp + total_fp + _EPS)
    recall = (total_tp + _EPS) / (total_tp + total_fn + _EPS)
    f1 = (2 * precision * recall + _EPS) / (precision + recall + _EPS)

    return {
        'Dice': float(dice), 
        'IoU': float(iou), 
        'F1': float(f1), 
        'Precision': float(precision), 
        'Recall': float(recall)
    }

def get_args():
    parser = argparse.ArgumentParser(description='Batch Test Checkpoints')
    
    # === 🔥 核心控制参数 ===
    parser.add_argument('--checkpoint-dir', '-d', type=str, required=True, help='存放 .pth 的文件夹')
    parser.add_argument('--start-epoch', type=int, default=0, help='测试起始轮次 (包含)')
    parser.add_argument('--end-epoch', type=int, default=1000, help='测试结束轮次 (包含)')
    parser.add_argument('--include-best', action='store_true', default=False, help='是否同时也测试 checkpoint_best.pth')
    parser.add_argument('--output-file', type=str, default='batch_test_results.csv', help='结果保存路径')
    
    # === 数据集参数 ===
    parser.add_argument('--test-img-dir', type=str, default='data/test/imgs/')
    parser.add_argument('--test-mask-dir', type=str, default='data/test/masks/')
    parser.add_argument('--scale', '-s', type=float, default=1.0)
    parser.add_argument('--batch-size', '-b', type=int, default=1)
    
    # === 架构参数 (必须与 train.py / test01.py 一致) ===
    parser.add_argument('--encoder', type=str, default='resnet', choices=['resnet', 'cnextv2', 'standard'])
    parser.add_argument('--decoder', type=str, default='phd', choices=['phd', 'standard'])
    parser.add_argument('--cnext-type', type=str, default='convnextv2_base')
    parser.add_argument('--bilinear', action='store_true', default=False)
    parser.add_argument('--classes', '-c', type=int, default=1)
    
    # === SOTA 模块开关 ===
    parser.add_argument('--use-dcn', action='store_true', default=False, help='Enable DCNv3')
    parser.add_argument('--use-dubm', action='store_true', default=False, help='Enable D-UBM')
    parser.add_argument('--use-dual-stream', action='store_true', default=False, help='Enable Dual Stream')
    parser.add_argument('--use-wavelet-denoise', action='store_true', default=False, help='Enable Wavelet')
    parser.add_argument('--use-dsis', action='store_true', default=False)
    parser.add_argument('--use-strg', action='store_true', default=False)
    parser.add_argument('--use-unet3p', action='store_true', default=False)
    parser.add_argument('--use-deep-supervision', action='store_true', default=False, help='Enable Deep Supervision (matches training)')
    # 其他增强参数
    parser.add_argument('--use-wgn-enhancement', action='store_true', default=False)
    parser.add_argument('--use-cafm', action='store_true', default=False)
    parser.add_argument('--use-edge-loss', action='store_true', default=False)
    parser.add_argument('--wgn-base-order', type=int, default=3)
    parser.add_argument('--wgn-orders', type=str, default=None)

    return parser.parse_args()

def extract_epoch(filename):
    """从文件名提取 epoch 数字"""
    match = re.search(r'epoch_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 扫描并筛选 Checkpoints
    ckpt_dir = Path(args.checkpoint_dir)
    all_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
    
    tasks = [] # 存储 (filename, epoch_num, is_special)
    
    for f in all_files:
        # 处理 checkpoint_epoch_XX.pth
        ep = extract_epoch(f)
        if ep is not None:
            # 筛选范围
            if args.start_epoch <= ep <= args.end_epoch:
                tasks.append((f, ep, False))
        
        # 处理 checkpoint_best.pth
        elif f == 'checkpoint_best.pth' and args.include_best:
            tasks.append((f, 999999, True)) # 999999 只是为了排序放在最后
    
    # 按 epoch 排序
    tasks.sort(key=lambda x: x[1])
    
    if not tasks:
        logging.error(f"No checkpoints found in range [{args.start_epoch}, {args.end_epoch}] in {ckpt_dir}")
        sys.exit(1)

    logging.info(f"📋 Found {len(tasks)} checkpoints to test.")

    # 2. 准备数据 (只加载一次)
    try:
        test_dataset = BasicDataset(args.test_img_dir, args.test_mask_dir, args.scale)
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=max(1, os.cpu_count() // 2), pin_memory=True, drop_last=False
        )
        logging.info(f"✅ Loaded dataset: {len(test_dataset)} images")
    except Exception as e:
        logging.error(f"Dataset Error: {e}")
        sys.exit(1)

    # 3. 构建模型 (只构建一次)
    # WGN Orders 处理
    wgn_orders = None
    if args.use_wgn_enhancement:
        if args.wgn_orders:
            orders_list = [int(x) for x in args.wgn_orders.split(',')]
            wgn_orders = {'layer1': (orders_list[0], orders_list[1]), 'layer2': (orders_list[2], orders_list[3]), 'layer3': (orders_list[4], orders_list[5])}
        else:
            base = args.wgn_base_order
            wgn_orders = {'layer1': (base, base-1), 'layer2': (base+1, base), 'layer3': (base+2, base+1)}

    logging.info(f"🏗️ Building Model... Encoder: {args.cnext_type}")
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
        use_dcn_in_phd=args.use_dcn,
        use_dsis=args.use_dsis,
        use_dubm=args.use_dubm,
        use_strg=args.use_strg,
        use_dual_stream=args.use_dual_stream,
        use_unet3p=args.use_unet3p,
        use_wavelet_denoise=args.use_wavelet_denoise,
        use_deep_supervision=args.use_deep_supervision
    )
    model.to(device)

    # 4. 批量测试循环
    results = []
    
    print("\n" + "="*90)
    print(f"{'Checkpoint':<30} | {'Dice':<8} | {'IoU':<8} | {'F1':<8} | {'Pre':<8} | {'Rec':<8}")
    print("-" * 90)

    for fname, epoch, is_special in tasks:
        ckpt_path = ckpt_dir / fname
        
        try:
            # 加载权重
            checkpoint = torch.load(ckpt_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            # 运行测试
            metrics = test_model_silent(model, device, test_loader, threshold=0.5, amp=False)
            
            # 记录数据
            row = {
                'Checkpoint': fname,
                'Epoch': epoch if not is_special else 'Best',
                **metrics
            }
            results.append(row)
            
            # 实时打印
            print(f"{fname:<30} | {metrics['Dice']:.4f}   | {metrics['IoU']:.4f}   | {metrics['F1']:.4f}   | {metrics['Precision']:.4f}   | {metrics['Recall']:.4f}")

        except Exception as e:
            logging.error(f"Error testing {fname}: {e}")

    print("=" * 90)

    # 5. 保存结果
    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.output_file, index=False)
        
        # 找出最佳轮次 (基于 Dice)
        best_row = df.loc[df['Dice'].idxmax()]
        print(f"\n🏆 Best Checkpoint in Batch: {best_row['Checkpoint']}")
        print(f"   Dice: {best_row['Dice']:.4f} | IoU: {best_row['IoU']:.4f} | F1: {best_row['F1']:.4f}")
        print(f"\n💾 Results saved to: {args.output_file}")

if __name__ == '__main__':
    main()