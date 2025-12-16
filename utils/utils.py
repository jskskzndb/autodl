import matplotlib.pyplot as plt
import logging


def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()


def log_grad_stats(named_parameters):
    """
    梯度诊断辅助函数：遍历模型的所有参数，记录每层梯度的统计信息
    
    Args:
        named_parameters: 模型的 named_parameters() 迭代器
    """
    for name, param in named_parameters:
        if param.grad is not None:
            # 计算梯度的平均值和最大值
            avg_grad = param.grad.abs().mean().item()
            max_grad = param.grad.abs().max().item()
            # 记录到日志
            logging.info(f'Layer: {name} | Avg grad: {avg_grad:.2e} | Max grad: {max_grad:.2e}')
