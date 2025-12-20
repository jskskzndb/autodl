import sys
import os
import torch
import numpy as np
from pathlib import Path

# 1. 导入 BasicDataset
# 确保 ceshi.py 和 utils 文件夹在同一级
try:
    from utils.data_loading import BasicDataset
except ImportError:
    print("❌ 报错：找不到 utils 模块。请确保 ceshi.py 在项目根目录下运行。")
    sys.exit(1)

# ==========================================
# 2. 根据你的截图修改路径
# 你的结构是: data -> train -> imgs
# ==========================================
# 使用相对路径 (推荐)
my_imgs_dir = './data/train/imgs/'
my_masks_dir = './data/train/masks/'

# 检查路径是否存在
if not os.path.exists(my_imgs_dir):
    print(f"❌ 错误：找不到图片路径: {my_imgs_dir}")
    print(f"   当前工作目录是: {os.getcwd()}")
    print("   请检查你的 ceshi.py 是否放在了和 'data' 文件夹同一级的地方。")
    sys.exit(1)

print(f"📂 读取数据路径: {my_imgs_dir}")
print("🚀 开始测试数据预处理...")

# 3. 初始化数据集
try:
    # 注意：scale=1.0 保持原图大小，或者改成 0.5 测试缩放
    dataset = BasicDataset(my_imgs_dir, my_masks_dir, scale=1.0)
    
    if len(dataset) == 0:
        print("❌ 错误：文件夹里没有找到图片！")
        sys.exit(1)

    # 4. 获取第一张图片
    first_item = dataset[0]
    sample_img = first_item['image'] # 获取图片张量

    # 5. 打印数值统计
    print("-" * 30)
    print(f"📊 图片张量形状: {sample_img.shape} (Channel, Height, Width)")
    print(f"MAX (最大值): {sample_img.max():.4f}")
    print(f"MIN (最小值): {sample_img.min():.4f}")
    print(f"MEAN (均值):  {sample_img.mean():.4f}")
    print("-" * 30)

    # 6. 自动判断结果
    if sample_img.min() < -1.0:
        print("✅ 验证成功！检测到负数 (MIN < -1)。")
        print("   ImageNet 标准化 (Z-Score) 已生效！")
        print("   现在的每一张图都符合 ConvNeXt V2 的“胃口”了。")
    elif sample_img.min() >= 0:
        print("⚠️ 警告：最小值仍然 >= 0 (通常是 0.0)。")
        print("   ❌ 标准化未生效！")
        print("   请检查 utils/data_loading.py 是否保存，或者代码逻辑是否有误。")
    else:
        print("❓ 结果存疑：有负数但数值不大，请确认是否只减了均值没除方差？")

except Exception as e:
    import traceback
    print(f"❌ 运行出错: {e}")
    traceback.print_exc()