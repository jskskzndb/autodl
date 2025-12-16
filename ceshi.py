import torch
# 把路径换成你那个 "ResUNet+cafm" 的最佳模型路径
ckpt = torch.load('data/checkpoints/checkpoint_best.pth')

# 如果是完整checkpoint
state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

# 检查是否有 'cafm' 开头的键
has_cafm = any(k.startswith('cafm') for k in state_dict.keys())
print(f"Contains CAFM weights? {has_cafm}")