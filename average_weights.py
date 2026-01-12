import torch
import os
import copy

def average_checkpoints(ckpt_dir, epoch_list, output_path):
    """
    Args:
        ckpt_dir: 存放权重的文件夹路径 (例如 './data/checkpoints')
        epoch_list: 需要平均的 epoch 数字列表 (例如 [23, 24, ..., 38])
        output_path: 结果保存的路径
    """
    avg_state_dict = None
    count = len(epoch_list)
    
    print(f"🔄 开始平均以下 {count} 个模型: {epoch_list}")
    print(f"📂 权重目录: {ckpt_dir}")

    for i, epoch in enumerate(epoch_list):
        # 拼接文件名，假设您的文件名格式是 checkpoint_epoch_XX.pth
        fname = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch}.pth")
        
        if not os.path.exists(fname):
            print(f"⚠️ 警告: 文件不存在，跳过: {fname}")
            count -= 1
            continue
            
        print(f"   [{i+1}/{len(epoch_list)}] Loading {fname}...")
        
        # 加载权重
        ckpt = torch.load(fname, map_location='cpu')
        
        # 兼容性处理：检查权重是在 'model_state_dict' 键下还是直接是字典
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            state_dict = ckpt
            
        # 初始化或累加
        if avg_state_dict is None:
            avg_state_dict = copy.deepcopy(state_dict)
        else:
            for key in avg_state_dict:
                # 只对浮点类型的参数进行平均 (跳过整数型的统计量，如 num_batches_tracked)
                if key in state_dict and torch.is_floating_point(avg_state_dict[key]):
                    avg_state_dict[key] += state_dict[key]
                # 对于整数参数 (如 BatchNorm 的 step)，通常保持第一个模型的即可，或者不需要平均
    
    # 执行除法，取平均值
    if avg_state_dict is not None:
        for key in avg_state_dict:
            if torch.is_floating_point(avg_state_dict[key]):
                avg_state_dict[key] /= count
        
        # 保存结果
        # 为了兼容您的 test.py，我们将结果包装在 'model_state_dict' 中 (如果原格式如此)
        # 这里为了通用，直接保存 state_dict，您的 test.py 应该能识别
        final_save = {'model_state_dict': avg_state_dict}
        torch.save(final_save, output_path)
        print(f"\n✅ 成功! 平均后的模型已保存至: {output_path}")
    else:
        print("❌ 错误: 没有加载到任何模型。")

if __name__ == "__main__":
    # ================= 🔧 修改配置区域 🔧 =================
    
    # 1. 您的权重文件夹路径
    CKPT_DIR = "./data/checkpoints"
    
    # 2. 【核心修改】您想要平均哪些轮次？
    # 场景 A: 手动指定几个 (例如 21, 23, 25)
    # EPOCHS_TO_AVG = [21, 23, 25]
    
    # 场景 B: 指定一个连续范围 (例如 23 到 38，包含 38)
    # range(start, end+1) -> range(23, 39) 代表 23,24,...,38
    EPOCHS_TO_AVG = list(range(23, 39)) 
    
    # 3. 输出文件的名字 (建议带上轮次范围，方便记忆)
    OUTPUT_FILE = "./data/checkpoints/checkpoint_swa_23_38.pth"
    
    # =======================================================
    
    average_checkpoints(CKPT_DIR, EPOCHS_TO_AVG, OUTPUT_FILE)