import argparse
import logging
import os
import sys
import re
import copy  # 🔥 新增
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# 🔥 请确保这里导入的是你最新的模型定义文件
# 如果你的文件名是 unet_universal3.py，请修改这里
from unet.unet_universal3 import UniversalUNet as UNet 
from utils.data_loading import BasicDataset

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

_EPS = 1e-6

def test_model_silent(net, device, test_loader, threshold=0.5, amp=False):
    """
    静默版测试函数
    """
    net.eval()
    num_test_batches = len(test_loader)
    total_tp = 0; total_fp = 0; total_fn = 0

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=amp):
            for batch in tqdm(test_loader, total=num_test_batches, desc='  Evaluating SWA Model', unit='batch'):
                images, true_masks = batch['image'], batch['mask']
                images = images.to(device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device, dtype=torch.long)

                # 推理
                output = net(images)
                
                # 兼容性处理
                if isinstance(output, tuple):
                    masks_pred = output[0]
                elif isinstance(output, list):
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
    parser = argparse.ArgumentParser(description='SWA (Weight Averaging) Test')
    
    # === 🔥 核心控制参数 ===
    parser.add_argument('--checkpoint-dir', '-d', type=str, required=True, help='存放 .pth 的文件夹')
    parser.add_argument('--start-epoch', type=int, default=80, help='平均起始轮次 (包含)')
    parser.add_argument('--end-epoch', type=int, default=100, help='平均结束轮次 (包含)')
    parser.add_argument('--save-swa-path', type=str, default='swa_model.pth', help='保存平均后模型的路径')
    
    # === 数据集参数 ===
    parser.add_argument('--test-img-dir', type=str, default='data/test/imgs/')
    parser.add_argument('--test-mask-dir', type=str, default='data/test/masks/')
    parser.add_argument('--scale', '-s', type=float, default=1.0)
    parser.add_argument('--batch-size', '-b', type=int, default=1)
    
    # === 架构参数 (必须与 train.py 一致) ===
    # 注意：这里的默认值应与你训练时的最佳配置一致
    parser.add_argument('--cnext-type', type=str, default='convnextv2_tiny')
    parser.add_argument('--classes', '-c', type=int, default=1)
    
    # === SOTA 模块开关 (请根据训练时的设置开启) ===
    parser.add_argument('--use-dual-stream', action='store_true', default=False)
    parser.add_argument('--use-deep-supervision', action='store_true', default=False)
    parser.add_argument('--use-unet3p', action='store_true', default=False)
    
    # 其他可能用到的参数 (保持兼容性)
    parser.add_argument('--use-wgn-enhancement', action='store_true', default=False)
    parser.add_argument('--use-cafm', action='store_true', default=False)
    parser.add_argument('--use-edge-loss', action='store_true', default=False)
    parser.add_argument('--wgn-base-order', type=int, default=3)
    parser.add_argument('--wgn-orders', type=str, default=None)
    parser.add_argument('--encoder', type=str, default='cnextv2')
    parser.add_argument('--decoder', type=str, default='phd')

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
    
    target_files = []
    for f in all_files:
        ep = extract_epoch(f)
        if ep is not None:
            if args.start_epoch <= ep <= args.end_epoch:
                target_files.append((ep, f))
    
    # 按 epoch 排序
    target_files.sort(key=lambda x: x[0])
    
    if not target_files:
        logging.error(f"❌ 没有在 {ckpt_dir} 找到范围 [{args.start_epoch}, {args.end_epoch}] 内的权重文件！")
        sys.exit(1)

    print(f"\n🔮 SWA 准备开始：将融合以下 {len(target_files)} 个模型的权重：")
    print(f"   Range: Epoch {target_files[0][0]} -> Epoch {target_files[-1][0]}")
    
    # 2. 🔥 核心逻辑：权重平均 (在 CPU 上进行以节省显存)
    avg_state_dict = None
    count = 0

    for ep, fname in tqdm(target_files, desc="Processing Weights"):
        path = ckpt_dir / fname
        # 加载到 CPU
        checkpoint = torch.load(path, map_location='cpu')
        
        # 处理不同保存格式 (有的包含 optimizer，有的直接是 dict)
        if 'model_state_dict' in checkpoint:
            curr_state_dict = checkpoint['model_state_dict']
        else:
            curr_state_dict = checkpoint
            
        if avg_state_dict is None:
            # 第一个模型，直接深拷贝
            avg_state_dict = copy.deepcopy(curr_state_dict)
        else:
            # 后续模型，累加参数
            for key in avg_state_dict:
                # 确保只累加浮点类型的参数 (排除 int/long 类型的 buffer，如果有的话)
                if avg_state_dict[key].is_floating_point():
                    avg_state_dict[key] += curr_state_dict[key]
        
        count += 1

    # 除以 N，取平均
    print(f"➗正在计算平均值 (N={count})...")
    for key in avg_state_dict:
        if avg_state_dict[key].is_floating_point():
            avg_state_dict[key] = avg_state_dict[key] / count
            
    print("✅ 权重融合完成！")

    # 3. 准备数据
    try:
        test_dataset = BasicDataset(args.test_img_dir, args.test_mask_dir, args.scale)
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=max(1, os.cpu_count() // 2), pin_memory=True, drop_last=False
        )
    except Exception as e:
        logging.error(f"Dataset Error: {e}")
        sys.exit(1)

    # 4. 构建模型并加载平均权重
    logging.info(f"🏗️ Building Model... Encoder: {args.cnext_type}")
    
    # WGN Orders (兼容旧代码逻辑)
    wgn_orders = None
    if args.use_wgn_enhancement:
        base = args.wgn_base_order
        wgn_orders = {'layer1': (base, base-1), 'layer2': (base+1, base), 'layer3': (base+2, base+1)}

    model = UNet(
        n_classes=args.classes,
        cnext_type=args.cnext_type,
        use_deep_supervision=args.use_deep_supervision,
        use_dual_stream=args.use_dual_stream,
        use_unet3p=args.use_unet3p,
        # 传入其他参数以防报错
        use_wgn_enhancement=args.use_wgn_enhancement,
        use_cafm=args.use_cafm,
        use_edge_loss=args.use_edge_loss,
        wgn_orders=wgn_orders,
        pretrained=False # 测试模式不需要预训练权重，直接加载 SWA 权重
    )
    
    # 加载平均后的权重
    try:
        model.load_state_dict(avg_state_dict)
        logging.info("🎉 成功加载 SWA 权重到模型！")
    except Exception as e:
        logging.error(f"加载权重失败，请检查模型结构参数是否与训练时一致: {e}")
        sys.exit(1)
        
    model.to(device)

    # 5. 测试融合后的模型
    print("\n" + "="*50)
    print("🚀 开始测试 SWA 模型...")
    metrics = test_model_silent(model, device, test_loader, threshold=0.5, amp=False)
    
    print("-" * 50)
    print(f"📊 SWA Final Results (Epoch {args.start_epoch}-{args.end_epoch}):")
    print(f"   Dice      : {metrics['Dice']:.4f}")
    print(f"   IoU       : {metrics['IoU']:.4f}")
    print(f"   F1-Score  : {metrics['F1']:.4f}")
    print(f"   Precision : {metrics['Precision']:.4f}")
    print(f"   Recall    : {metrics['Recall']:.4f}")
    print("=" * 50)

    # 6. 保存融合模型
    if args.save_swa_path:
        save_dict = {
            'model_state_dict': avg_state_dict,
            'epoch': 'swa',
            'desc': f'SWA average from epoch {args.start_epoch} to {args.end_epoch}'
        }
        torch.save(save_dict, args.save_swa_path)
        print(f"💾 SWA 模型已保存至: {args.save_swa_path}")

if __name__ == '__main__':
    main()