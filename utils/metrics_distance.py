# 文件路径: utils/metrics_distance.py
import numpy as np
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, generate_binary_structure

def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    内部函数：计算两个二值掩码之间的表面距离图。
    """
    result = np.atleast_1d(result.astype(bool))
    reference = np.atleast_1d(reference.astype(bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()
    
    # 提取边界
    footprint = generate_binary_structure(result.ndim, connectivity)
    
    # 如果全黑或全白，返回空
    if 0 == np.count_nonzero(result): return np.array([]), np.array([])
    if 0 == np.count_nonzero(reference): return np.array([]), np.array([])
    
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)
    
    # 计算距离变换
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]
    
    dt = distance_transform_edt(~result_border, sampling=voxelspacing)
    sds2 = dt[reference_border]
    
    return sds, sds2

def compute_hd95(pred, gt, voxelspacing=None):
    """
    计算 95% 豪斯多夫距离 (HD95)
    pred, gt: numpy array (0/1), shape: [H, W]
    返回: float (数值越小越好)
    """
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0 # 都预测空，完美
    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan # 无法计算距离
    
    sds, sds2 = __surface_distances(pred, gt, voxelspacing)
    
    if len(sds) == 0 or len(sds2) == 0:
        return np.nan
        
    hd95 = np.percentile(np.hstack((sds, sds2)), 95)
    return hd95

def compute_asd(pred, gt, voxelspacing=None):
    """
    计算平均表面距离 (ASD)
    返回: float (数值越小越好)
    """
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan
        
    sds, sds2 = __surface_distances(pred, gt, voxelspacing)
    
    if len(sds) == 0 or len(sds2) == 0:
        return np.nan
        
    asd = np.mean(np.hstack((sds, sds2)))
    return asd