import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from torchvision import transforms # 🔥 必须引入这个
from unet import UNet

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='data/checkpoints/checkpoint_best.pth',
                        metavar='FILE', help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', default='data/test/imgs/2_11.tif', # 你的测试图路径
                        metavar='INPUT', help='Filenames of input images')
    parser.add_argument('--output', '-o', default='debug_fix.png',
                        metavar='OUTPUT', help='Filenames of output images')
    return parser.parse_args()

def run_visual_check():
    args = get_args()
    
    # ================= 配置区 =================
    # 你的 4原型 + 0Loss + 预训练 配置 (保持与训练一致)
    model_config = dict(
        n_channels=3, n_classes=1, 
        encoder_name='cnextv2', cnext_type='convnextv2_tiny', 
        decoder_type='phd',
        pretrained=False, # 推理时这里False没关系，因为会加载权重
        use_dual_stream=False, 
        use_fme=False, use_dsis=False, use_unet3p=False, use_cafm=False
    )
    # =========================================

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 运行深度诊断...")
    print(f"📁 加载模型: {args.model}")
    print(f"🖼️ 测试图片: {args.input}")

    # 1. 加载模型
    try:
        model = UNet(**model_config)
        checkpoint = torch.load(args.model, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.to(device).eval()
        print("✅ 模型权重加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 2. 图片读取与预处理
    if not os.path.exists(args.input):
        print(f"❌ 找不到图片: {args.input}")
        return

    img_bgr = cv2.imread(args.input)
    if img_bgr is None:
        print("❌ 图片读取失败，可能是格式损坏或路径错误")
        return

    # 转换颜色 BGR -> RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # === 🔥 关键修复：添加与 BasicDataset 一致的归一化 ===
    # 你的 BasicDataset 第 145 行用了 mean=[0.485...], std=[0.229...]
    # 所以测试时必须完全一致！
    data_transform = transforms.Compose([
        transforms.ToTensor(), # 这一步会自动把 0-255 转为 0-1
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 生成输入 Tensor
    input_tensor = data_transform(img_rgb).unsqueeze(0).float().to(device)
    
    # 3. 推理
    print("⏳ 正在推理...")
    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, tuple): output = output[0]
        probs = torch.sigmoid(output)
        
        # === 核心诊断信息 ===
        max_val = probs.max().item()
        mean_val = probs.mean().item()
        print(f"\n📊 概率分布诊断:")
        print(f"   最大置信度 (Max Probability): {max_val:.6f}")
        print(f"   平均置信度 (Mean Probability): {mean_val:.6f}")
        
        # 动态阈值 (如果模型比较保守，自动降低门槛)
        threshold = 0.5
        if max_val < 0.5 and max_val > 0.1:
            print(f"⚠️ 警告：最大置信度 {max_val:.2f} 较低，自动降低阈值到 {max_val - 0.05:.2f}")
            threshold = max(0.1, max_val - 0.05)
            
        pred_mask = (probs > threshold).float().cpu().numpy()[0, 0]

    # 4. 绘图 (反归一化以便显示原图)
    plt.figure(figsize=(12, 4))
    
    # 为了显示好看，我们把原图归一化之前的样子还原出来
    display_img = img_rgb.astype(np.float32) / 255.0
    
    # 子图1: 原图
    plt.subplot(1, 3, 1); plt.imshow(display_img); plt.title("Original Input")
    plt.axis('off')
    
    # 子图2: 概率热力图
    plt.subplot(1, 3, 2); plt.imshow(probs.cpu().numpy()[0,0], cmap='jet'); plt.title(f"Prob Map (Max:{max_val:.2f})")
    plt.axis('off')
    
    # 子图3: 叠加图
    plt.subplot(1, 3, 3); plt.imshow(display_img)
    mask_visual = np.zeros((display_img.shape[0], display_img.shape[1], 4))
    mask_visual[pred_mask == 1] = [1, 0, 0, 0.5] # 红色，半透明
    plt.imshow(mask_visual); plt.title(f"Result (Thresh={threshold:.2f})")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"✅ 结果已保存为: {args.output}")

if __name__ == "__main__":
    run_visual_check()