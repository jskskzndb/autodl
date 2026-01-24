import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from unet import UNet  # 确保能导入你的模型定义

def check_prototypes(model_path, device='cuda'):
    print(f"🔍 正在加载模型权重: {model_path}")
    
    # 1. 实例化模型 (必须与训练时参数一致)
    # 注意：这里只为了加载参数，encoder/decoder类型要对，pretrained无所谓
    model = UNet(
        n_channels=3, n_classes=1, 
        encoder_name='cnextv2', 
        cnext_type='convnextv2_tiny', 
        decoder_type='phd',
        pretrained=False,
        # === 显式关闭额外模块 ===
        use_dual_stream=False,      # 关闭双流 (对应 edge_head 报错)
        use_fme=False,              # 关闭频域增强 (对应 freq_layers 报错)
        use_dsis=False,             # 关闭交互跳跃连接
        use_unet3p=False,           # 关闭 UNet3+
        use_wgn_enhancement=False,  # 关闭 WGN
        use_cafm=False 
    )
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # 2. 遍历查找所有的原型参数
    found_protos = False
    
    print("\n📊 原型相似度分析报告:")
    print("-" * 60)
    
    for name, param in model.named_parameters():
        if 'prototypes' in name:
            found_protos = True
            print(f"👉 发现原型层: {name}")
            
            # param shape 通常是 [1, 16, C] 或 [16, C]
            P = param.detach()
            if P.dim() > 2:
                P = P.squeeze(0) # 变成 [N, C] (例如 [16, 128])
            
            num_protos = P.shape[0]
            dim = P.shape[1]
            print(f"   尺寸: {num_protos} 个原型, 维度 {dim}")
            
            # --- 计算相似度矩阵 ---
            # 1. 归一化 (Normalize) -> 变成单位向量
            P_norm = F.normalize(P, p=2, dim=1)
            
            # 2. 计算 Gram 矩阵 (P * P.T) -> 得到 N x N 的相似度矩阵
            # 结果在 -1 到 1 之间。1 表示完全一样，0 表示正交(完全不同)，-1 表示相反
            similarity_matrix = torch.mm(P_norm, P_norm.t()).cpu().numpy()
            
            # --- 统计指标 ---
            # 把对角线（自己和自己对比，永远是1）设为 NaN，以免影响平均值计算
            sim_no_diag = similarity_matrix.copy()
            np.fill_diagonal(sim_no_diag, np.nan)
            
            avg_sim = np.nanmean(sim_no_diag)
            max_sim = np.nanmax(sim_no_diag)
            min_sim = np.nanmin(sim_no_diag)
            
            print(f"   平均相似度: {avg_sim:.4f} (越接近 0 越好，接近 1 说明坍塌)")
            print(f"   最大相似度: {max_sim:.4f}")
            print(f"   最小相似度: {min_sim:.4f}")
            
            # --- 绘制热力图 ---
            plt.figure(figsize=(8, 6))
            sns.heatmap(similarity_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title(f'Prototype Similarity: {name}\nAvg Sim: {avg_sim:.3f}')
            plt.xlabel('Prototype Index')
            plt.ylabel('Prototype Index')
            
            save_name = f"collapse_check_{name.replace('.','_')}.png"
            plt.savefig(save_name)
            print(f"   🖼️  热力图已保存为: {save_name}")
            print("-" * 60)
            plt.close()

    if not found_protos:
        print("❌ 未在模型中找到名为 'prototypes' 的参数！请检查模型定义。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='data/checkpoints/checkpoint_best.pth', help='权重路径')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    check_prototypes(args.model, device)