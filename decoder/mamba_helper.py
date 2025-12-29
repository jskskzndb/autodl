import torch
import torch.nn as nn

# 尝试导入官方库
try:
    from mamba_ssm import Mamba
except ImportError:
    print("❌ 还没有安装 mamba-ssm！请运行: pip install causal-conv1d>=1.2.0 mamba-ssm")
    Mamba = None

class MambaLayer2D(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if Mamba is None:
            raise ImportError("Mamba module not found.")
            
        # 1. 这里的 Mamba 就是你直接调用的官方库
        self.mamba = Mamba(
            d_model=dim,      # 输入通道数
            d_state=d_state,  # 状态维度
            d_conv=d_conv,    # 局部卷积宽度
            expand=expand     # 扩张系数
        )
        
        # 2. LayerNorm 是 Mamba 的标配
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x 的形状是图片格式: [Batch, Channel, Height, Width]
        B, C, H, W = x.shape
        # 🔥 开启防爆盾：强制 FP32
        with torch.cuda.amp.autocast(enabled=False):
            
            # 1. 进门先转 FP32
            x = x.float()

            # 2. 变形 (FP32)
            x_seq = x.flatten(2).transpose(1, 2) 
            
            # 3. 归一化 + Mamba 处理 (FP32) <--- 这里是重点！
            x_seq = self.norm(x_seq)
            x_seq = self.mamba(x_seq) 
            
            # 4. 变回图片 (FP32)
            x_out = x_seq.transpose(1, 2).view(B, C, H, W)
        
        # 出了缩进再 return
        return x_out
         
        
        return x_out