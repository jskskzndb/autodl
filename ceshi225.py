import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
from unet import UNet
from utils.data_loading import BasicDataset

# 存储捕获到的 norm 数据
norms_report = {}

def get_hook(name):
    """ 创建一个 Hook 来捕获模块的输入输出 """
    def hook(module, input, output):
        # 针对 PrototypeInteractionBlock 捕获关键变量
        # 静态原型是 module.prototypes [1, N, C]
        # 动态偏移是 module.modulator 产生的中间值，这里我们直接从模块状态取
        if hasattr(module, 'prototypes') and hasattr(module, 'modulator'):
            static_p = module.prototypes.detach()
            # 为了获取当前输入的 offset，我们需要手动调用一次 modulator (或者直接 hook modulator)
            # 这里简单起见，我们假设 input[0] 是进入该层的特征图 x
            x = input[0].detach()
            # 执行一次调制器
            offset_p = module.modulator.seed_extractor(x).view(x.shape[0], module.num_prototypes, -1).detach()
            
            s_norm = torch.norm(static_p, p=2, dim=-1).mean().item()
            o_norm = torch.norm(offset_p, p=2, dim=-1).mean().item()
            
            if name not in norms_report:
                norms_report[name] = []
            norms_report[name].append((s_norm, o_norm))
    return hook

def run_diagnostic(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载模型
    model = UNet(
        n_classes=1, encoder_name='cnextv2', cnext_type='convnextv2_tiny', 
        decoder_type='phd', pretrained=False
    )
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint, strict=False)
    model.to(device).eval()

    # 2. 注册 Hooks (自动寻找所有的原型交互块)
    for name, module in model.named_modules():
        if 'PrototypeInteractionBlock' in str(type(module)):
            module.register_forward_hook(get_hook(name))

    # 3. 加载少量测试数据
    dataset = BasicDataset(args.test_img_dir, args.test_mask_dir, scale=1.0)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    print(f"🚀 开始诊断动态分支活跃度 (抽取 {args.num_samples} 张图)...")
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.num_samples: break
            model(batch['image'].to(device))

    # 4. 打印分析报告
    print("\n" + "="*60)
    print(f"{'层级名称':<35} | {'静态模长':<10} | {'动态偏移':<10} | {'活跃度 (Ratio)'}")
    print("-" * 60)
    
    for name, values in norms_report.items():
        avg_s = np.mean([v[0] for v in values])
        avg_o = np.mean([v[1] for v in values])
        ratio = (avg_o / avg_s) * 100
        
        # 颜色标记活跃度
        status = "🔥 活跃" if ratio > 10 else "💤 怠工"
        if ratio < 2: status = "💀 几乎消失"
        
        print(f"{name:<35} | {avg_s:>10.4f} | {avg_o:>10.4f} | {ratio:>11.2f}%  {status}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--test-img-dir', type=str, default='data/test/imgs/')
    parser.add_argument('--test-mask-dir', type=str, default='data/test/masks/')
    parser.add_argument('--num-samples', type=int, default=20)
    run_diagnostic(parser.parse_args())