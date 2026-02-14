import os
import torch
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def average_specific_epochs(ckpt_dir, epoch_list, output_path):
    """
    平均指定 Epoch 列表的权重
    """
    avg_state_dict = {}
    valid_files = []

    print(f"📊 [精英模式] 准备平均以下 {len(epoch_list)} 个后期高分模型 (Epoch >= 77):")
    
    # 1. 检查并加载所有模型
    for epoch in epoch_list:
        # 优先寻找 epoch 文件
        fname = f"checkpoint_epoch_{epoch}.pth"
        path = os.path.join(ckpt_dir, fname)
        
        # 特殊处理：如果 Epoch 77 的文件找不到，尝试用 checkpoint_best.pth 替代
        if not os.path.exists(path):
            if epoch == 77:
                logging.warning(f"⚠️  未找到 {fname}，尝试加载 checkpoint_best.pth...")
                path = os.path.join(ckpt_dir, "checkpoint_best.pth")
            else:
                logging.warning(f"⚠️  文件不存在: {path}，跳过此 Epoch。")
                continue
        
        if not os.path.exists(path):
             logging.error(f"❌ 依然无法找到 Epoch {epoch} 的权重文件。")
             continue

        print(f"   -> 加载 Epoch {epoch}: {os.path.basename(path)}")
        try:
            checkpoint = torch.load(path, map_location='cpu')
            
            # 兼容性处理
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            valid_files.append(state_dict)
        except Exception as e:
            logging.error(f"❌ 加载失败 {path}: {e}")

    if not valid_files:
        raise ValueError("❌ 没有找到任何有效的权重文件！")

    print(f"   (实际成功加载: {len(valid_files)} / {len(epoch_list)} 个)")

    # 2. 执行平均
    print(f"\n🔄 正在计算平均值...")
    first_state = valid_files[0]
    
    # 获取所有键
    keys = first_state.keys()
    
    for key in keys:
        # 仅平均浮点数参数
        if first_state[key].is_floating_point():
            sum_param = first_state[key].clone()
            for i in range(1, len(valid_files)):
                sum_param += valid_files[i][key]
            avg_state_dict[key] = sum_param / len(valid_files)
        else:
            # 非浮点参数（如 int64 的 buffer），保持第一个模型的值
            avg_state_dict[key] = first_state[key]

    # 3. 保存
    save_data = {'model_state_dict': avg_state_dict}
    torch.save(save_data, output_path)
    print(f"✅ [成功] 平均模型已保存至: {output_path}")

# ================= 配置区 =================
ckpt_dir = 'data/checkpoints'
output_path = os.path.join(ckpt_dir, 'checkpoint_avg_top10_late.pth')

# 🔥 严格筛选的 Epoch >= 77 的 Top 10
target_epochs = [
    77,   # 0.9599 (Best)
    89,   # 0.9595
    90,   # 0.9589
    81,   # 0.9588
    84,   # 0.9587
    85,   # 0.9586
    88,   # 0.9586
    91,   # 0.9586
    94,   # 0.9586
    103   # 0.9586
]

if __name__ == "__main__":
    try:
        average_specific_epochs(ckpt_dir, target_epochs, output_path)
    except Exception as e:
        print(f"❌ 运行失败: {e}")