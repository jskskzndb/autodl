#!/usr/bin/env python3
"""
8GB显存优化版本的UNet训练脚本
专门为8GB显存的GPU（如RTX 3070、RTX 4060 Ti等）优化

使用方法：
python train_8gb.py --data-dir ./data --epochs 50
"""

import argparse
import logging
import os
import subprocess
import sys

def get_optimal_args_for_8gb():
    """返回8GB显存的最优参数"""
    return {
        'batch_size': 2,        # 小batch size减少显存占用
        'scale': 0.5,           # 图像缩放到50%减少显存
        'amp': False,           # 关闭混合精度以提高数值稳定性
        'bilinear': True,       # 双线性上采样比反卷积省显存
        'learning_rate': 3e-5,  # 降低学习率提高训练稳定性（从1e-4降低到3e-5）
        'classes': 1,           # 建筑物识别通常是二分类
    }

def main():
    parser = argparse.ArgumentParser(description='8GB显存优化的UNet训练')
    parser.add_argument('--data-dir', default='./data', help='数据目录路径')
    parser.add_argument('--epochs', '-e', type=int, default=50, help='训练轮数')
    parser.add_argument('--force-params', action='store_true', 
                       help='强制使用8GB优化参数（忽略用户自定义）')
    
    # 允许用户覆盖优化参数
    parser.add_argument('--batch-size', type=int, help='覆盖默认batch size')
    parser.add_argument('--scale', type=float, help='覆盖默认图像缩放')
    parser.add_argument('--learning-rate', type=float, help='覆盖默认学习率')
    
    args = parser.parse_args()
    
    # 获取优化参数
    optimal_params = get_optimal_args_for_8gb()
    
    # 构建训练命令
    cmd = [sys.executable, 'train.py']
    
    # 添加优化参数
    if args.force_params or args.batch_size is None:
        cmd.extend(['--batch-size', str(optimal_params['batch_size'])])
    else:
        cmd.extend(['--batch-size', str(args.batch_size)])
        
    if args.force_params or args.scale is None:
        cmd.extend(['--scale', str(optimal_params['scale'])])
    else:
        cmd.extend(['--scale', str(args.scale)])
        
    if args.force_params or args.learning_rate is None:
        cmd.extend(['--learning-rate', str(optimal_params['learning_rate'])])
    else:
        cmd.extend(['--learning-rate', str(args.learning_rate)])
    
    # 添加固定的优化参数
    cmd.extend([
        '--epochs', str(args.epochs),
        '--classes', str(optimal_params['classes']),
        '--bilinear'    # 使用双线性上采样
    ])
    
    # 根据配置决定是否启用混合精度
    if optimal_params['amp']:
        cmd.append('--amp')
    
    print("🚀 8GB显存优化训练启动")
    print("📋 优化配置:")
    print(f"   • Batch Size: {args.batch_size or optimal_params['batch_size']}")
    print(f"   • Image Scale: {args.scale or optimal_params['scale']}")
    print(f"   • Learning Rate: {args.learning_rate or optimal_params['learning_rate']}")
    print(f"   • Mixed Precision: {'✅' if optimal_params['amp'] else '❌ (关闭)'}")
    print(f"   • Bilinear Upsampling: ✅")
    print(f"   • Gradient Checkpointing: ✅ (自动启用)")
    print()
    print("💡 提示:")
    print("   • 如果仍然OOM，脚本会自动减半batch size")
    print("   • 监控显存使用：watch -n 1 nvidia-smi")
    print("   • 使用 --force-params 强制使用最保守设置")
    print()
    print(f"🎯 执行命令: {' '.join(cmd)}")
    print("=" * 60)
    
    # 执行训练
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败，退出码: {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n⏹️  训练被用户中断")
        sys.exit(0)

if __name__ == '__main__':
    main()


