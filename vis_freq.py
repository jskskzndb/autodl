import torch
from unet import UNet  # 这会自动加载 __init__.py 里定义的 UniversalUNet

def check_model_structure():
    print("🔍 正在检查模型结构...")
    
    # 初始化模型：指定使用 PHD 解码器
    model = UNet(
        n_classes=1,
        cnext_type='convnextv2_tiny',
        pretrained=False,
        use_dual_stream=True,  # 你的实验设置
        decoder_type='phd'     # 🔥 指定 PHD
    )
    
    # 打印其中一个解码器层 (up1)
    print("\n📦 Decoder Layer 1 Structure:")
    print(model.up1.conv)
    
    # === 自动检测特征 ===
    model_str = str(model.up1.conv)
    
    has_expand = "expand" in model_str or "Expand" in model_str
    has_ffn = "ffn" in model_str or "FFN" in model_str
    has_gelu = "GELU" in model_str
    
    print("\n" + "="*40)
    print("✅ 验证结果:")
    if has_ffn and has_gelu:
        print("🎉 恭喜！检测到 FFN 和 GELU，当前正在使用 [PHD Pro 增重版]！")
        print("   -> 这是一个 Inverted Bottleneck 结构，参数量已增强。")
    else:
        print("⚠️ 警告！未检测到 FFN/GELU，当前可能使用的是 [旧版 PHD] 或 [Standard]。")
        print("   -> 请检查 unet_universal.py 中的 Up_Universal 类。")
    print("="*40)

if __name__ == '__main__':
    check_model_structure()