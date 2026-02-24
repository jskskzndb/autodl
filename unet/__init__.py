
# =================================================================
# UNet 模型版本控制台
# 使用方法：取消注释你想用的那个版本，把其他的注释掉即可。
# =================================================================

# 🟢 选项 1: 默认/当前开发版 (unet_model.py)
# from .unet_model import UNet

# 🟢 选项 2: 完整消融实验版 (unet_model_complete.py)
# 推荐用于：跑 ResNet vs Standard 对比，或 WGN 有效性验证
# 特性：修复了 Standard U-Net 缺失 WGN 的问题，集成了所有功能
#from .unet_model_complete import UNet

# 🟢 选项 3: PHD Mamba 解码器版 (unet_model_phd.py)
# 推荐用于：测试 Mamba + StripConv 解码器的效果
# 特性：ResNet 编码器 + PHD 混合解码器
#from .unet_model_phd import UNet

#from .unet_model_unified import UNet 

#纯净版phd模型
#from .phd_unet import PHD_UNet as UNet

# =================================================================
# 历史版本 (归档)
# =================================================================
# V0: 原始备份
# from .unet_modelV0 import UNet

# V1: 早期截断版
# from .unet_modelV1 import UNet

# V2: 侧路增强版 (Standard ResUNet + WGN Skip)
# from .unet_modelV2 import UNet

#from .wvm_unet import WVM_UNet as UNet
#from .wvm_unet_copy import WVM_UNet as UNet
# from .unet_modelV3 import UNet
# 🔥 新的导入：指向 S-DMFNet
#from .unet_s_dmfnet import S_DMFNet as UNet
#from .unet_s_dmfnet2 import S_DMFNet as UNet
#from .unet_s_dmfnet3 import S_DMFNet as UNet

#from .unet_s_dmfnet_standard import S_DMFNet_Standard as UNet
#from .unet_cnext_standard import UNet_CNext_Standard as UNet
# 🟢 选项: PHD 消融模型 (ConvNeXt + PHD Decoder only)
#from .unet_cnext_phd import UNet_CNext_PHD as UNet
# 采用过拟合优化后的模型
#from .unet_cnext_phd2 import UNet_CNext_PHD as UNet
#from .unet_cnext_phd3 import UNet_CNext_PHD as UNet

#🟢 全能型 UNet
#功能: 一个类实现 4 种架构组合，通过参数控制。
#from .unet_universal1 import UniversalUNet as UNet

#🟢 全能型 UNet
#功能: 一个类实现 4 种架构组合，通过参数控制。+ASPP
#from .unet_universal2 import UniversalUNet as UNet
#🟢 全能型 UNet
#功能: 一个类实现 4 种架构组合，通过参数控制。PHD×2，16原型
#from .unet_universal3 import UniversalUNet as UNet

#🟢 全能型 UNet
#功能: 一个类实现 4 种架构组合，通过参数控制。PHD×2，8原型
#from .unet_universal4 import UniversalUNet as UNet

#🟢 全能型 UNet
#功能: 一个类实现 4 种架构组合，通过参数控制。PHD×2，8原型
#from .unet_universal5 import UniversalUNet as UNet

#🟢 全能型 UNet
#功能: 一个类实现 4 种架构组合，通过参数控制。PHD×2，8原型
#from .unet_universal6 import UniversalUNet as UNet

#🟢 全能型 UNet
#功能: 一个类实现 4 种架构组合，通过参数控制。PHD×2，8原型
#from .unet_universal7 import UniversalUNet as UNet

from .unet_universal8 import UniversalUNet as UNet