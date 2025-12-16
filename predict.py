import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F # 神经网络里的函数（如插值、激活等）
from PIL import Image# 读/写图片
from torchvision import transforms

from utils.data_loading import BasicDataset # 复用训练时的预处理函数
from unet import UNet
from utils.utils import plot_img_and_mask# 可视化：把原图与预测掩码画到一起
# ——— 核心：单张图片的前向预测函数 ———
def predict_img(net,# 已加载好权重的模型
                full_img,# PIL.Image 打开的原始图片
                device,# 设备（cuda 或 cpu）
                scale_factor=1,# 预测前的缩放比例（和训练一致）
                out_threshold=0.5):# 二分类时把概率>阈值的像素判为前景
    net.eval() # 评估模式：关闭 Dropout / BN 的动量更新等
    # 用数据集里的预处理，把 PIL 图片转成网络需要的张量（归一化/缩放/CHW）
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)# [C,H,W] -> [1,C,H,W]，因为模型按 batch 维度处理
    img = img.to(device=device, dtype=torch.float32)# 搬到正确设备，并用 float32

    with torch.no_grad(): # 预测时不需要反向传播，省内存/加速
        output = net(img).cpu() # 前向得到输出，并搬回 CPU 方便后处理
        if isinstance(output, tuple): output = output[0]
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')# 插值把输出还原到原图大小（双线性插值）
        # 二分类：先做 sigmoid 得到前景概率，再按阈值变成0/1
        mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()

# ——— 命令行参数定义：让脚本可在命令行使用 ———
def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--use-cafm', action='store_true', default=False, help='Use Advanced CAFM module')
    parser.add_argument('--use-resnet-encoder', action='store_true', default=False, help='Use ResNet50 encoder')
    
    return parser.parse_args()

# ——— 根据输入图片名生成输出文件名（默认在原名后加 _OUT.png） ———
def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'# 去掉扩展名，加 _OUT.png

    return args.output or list(map(_generate_name, args.input)) # 如果没提供 --output，就自动生成

# ——— 把类别掩码转成可保存的图片 ———
def mask_to_image(mask: np.ndarray, mask_values):
    # 根据 mask_values 的格式决定输出数组的形状/类型
    if isinstance(mask_values[0], list):
        # 多通道RGB调色板：输出(H,W,3)
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        # 二值布尔图：输出(H,W)，True/False
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        # 灰度伪色：输出(H,W)的uint8
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3: # 如果传入是 one-hot(3维)，先转成类别ID
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v# 把预测为第 i 类的像素，赋值为对应的颜色/数值

    return Image.fromarray(out)# 转成 PIL.Image 以便保存

# ——— 主程序入口：命令行运行时走这里 ———
if __name__ == '__main__':
    args = get_args()# 解析参数
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s') # 日志格式

    in_files = args.input# 输入图片列表
    out_files = get_output_filenames(args)# 输出图片列表（与输入一一对应）

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear,
               use_advanced_cafm=args.use_cafm, use_resnet_encoder=args.use_resnet_encoder)# 创建模型

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 选设备
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device) # 把模型搬到设备
    checkpoint = torch.load(args.model, map_location=device)# 读取 .pth
    
    # 处理新旧两种checkpoint格式
    if 'model_state_dict' in checkpoint:
        # 新格式：完整checkpoint
        net.load_state_dict(checkpoint['model_state_dict'])
        mask_values = checkpoint.get('mask_values', [0, 1])
        logging.info('Loaded model from full checkpoint')
    else:
        # 旧格式：只有模型权重
        mask_values = checkpoint.pop('mask_values', [0, 1])
        net.load_state_dict(checkpoint)
        logging.info('Loaded model from legacy checkpoint')

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):# 逐张图片预测
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename) # 用 PIL 打开图片

        mask = predict_img(net=net,# 调用上面的预测函数
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:# 如果不禁用保存
            out_filename = out_files[i]# 对应的输出名
            result = mask_to_image(mask, mask_values) # 把类别ID转成图片可视化
            result.save(out_filename) # 保存到磁盘
            logging.info(f'Mask saved to {out_filename}')

        if args.viz: # 如果要求可视化
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)# 弹窗显示原图+掩码
