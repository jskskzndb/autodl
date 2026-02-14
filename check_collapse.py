import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from unet import UNet

def check_prototypes(args, device='cuda'):
    print(f"🔍 正在加载模型权重: {args.model}")
    print(f"⚙️  模型配置: Decoder={args.decoder}, Dual-Stream={args.use_dual_stream}")
    
    # 1. 实例化模型 (根据命令行参数动态构建)
    model = UNet(
        n_channels=3, 
        n_classes=1, 
        encoder_name='cnextv2', 
        cnext_type='convnextv2_tiny', 
        # 🔥🔥🔥 [修改点 1] 使用传入的参数
        decoder_type=args.decoder,
        use_dual_stream=args.use_dual_stream,
        
        pretrained=False,
        # 其他模块默认关闭，或者你也可以加上参数控制
        use_fme=False,
        use_dsis=False,
        use_unet3p=False,
        use_wgn_enhancement=False,
        use_cafm=False 
    )
    
    # 2. 加载权重
    try:
        checkpoint = torch.load(args.model, map_location=device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # 🔥 strict=False 可以容忍一些不匹配 (比如多余的 head 权重)，防止脚本直接挂掉
        model.load_state_dict(state_dict, strict=False)
        print("✅ 权重加载成功 (Strict=False)")
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        return

    model.to(device)
    model.eval()
    
    # 3. 遍历查找所有的原型参数
    found_protos = False
    
    print("\n📊 原型相似度分析报告:")
    print("-" * 60)
    
    for name, param in model.named_parameters():
        if 'prototypes' in name:
            found_protos = True
            print(f"👉 发现原型层: {name}")
            
            P = param.detach()
            if P.dim() > 2:
                P = P.squeeze(0) # [N, C]
            
            num_protos = P.shape[0]
            dim = P.shape[1]
            print(f"   尺寸: {num_protos} 个原型, 维度 {dim}")
            
            # --- 计算相似度 ---
            P_norm = F.normalize(P, p=2, dim=1)
            similarity_matrix = torch.mm(P_norm, P_norm.t()).cpu().numpy()
            
            sim_no_diag = similarity_matrix.copy()
            np.fill_diagonal(sim_no_diag, np.nan)
            
            avg_sim = np.nanmean(sim_no_diag)
            max_sim = np.nanmax(sim_no_diag)
            
            print(f"   平均相似度: {avg_sim:.4f}")
            print(f"   最大相似度: {max_sim:.4f}")
            
            # --- 绘图 ---
            plt.figure(figsize=(8, 6))
            sns.heatmap(similarity_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title(f'Prototype Similarity: {name}\nAvg Sim: {avg_sim:.3f}')
            
            save_name = f"collapse_check_{name.replace('.','_')}.png"
            plt.savefig(save_name)
            print(f"   🖼️  热力图已保存: {save_name}")
            print("-" * 60)
            plt.close()

    if not found_protos:
        print("❌ 未在模型中找到 'prototypes' 参数。")
        print("   可能原因: 1. 模型加载的是 standard 解码器而非 phd。")
        print("   2. 权重文件本身就没有保存原型参数。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='data/checkpoints/checkpoint_best.pth', help='权重路径')
    
    # 🔥🔥🔥 [修改点 2] 添加缺失的参数定义
    parser.add_argument('--decoder', type=str, default='phd', choices=['phd', 'standard'], help='解码器类型')
    parser.add_argument('--use-dual-stream', action='store_true', help='开启双流架构')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    check_prototypes(args, device)