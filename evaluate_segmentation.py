#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
通用医学图像分割评估脚本
用于计算各种评估指标并生成高质量可视化图表
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm
from datetime import datetime
from medpy import metric
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from scipy.ndimage import binary_erosion, binary_dilation

# 设置matplotlib样式，使图表接近Nature Medicine风格
matplotlib.use('Agg')
try:
    # 尝试新版本的样式名称
    plt.style.use('seaborn-v0_8-whitegrid')  # 新版本Seaborn的命名方式
except:
    try:
        # 尝试旧版本的样式名称
        plt.style.use('seaborn-whitegrid')
    except:
        # 如果都不可用，使用默认样式
        plt.style.use('default')
        print("警告: 无法加载seaborn样式，使用默认样式")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

def setup_logger(output_dir):
    """设置日志记录器"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f'evaluation_log_{timestamp}.txt')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger()

def calculate_dice(pred, gt):
    """计算Dice系数"""
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)
    if union == 0:
        return 0.0
    return 2.0 * intersection / union

def calculate_roi_dice(pred, gt):
    """计算ROI区域的Dice系数（只考虑包含标签的切片）"""
    roi_slices = [i for i in range(gt.shape[0]) if np.sum(gt[i]) > 0]
    if not roi_slices:
        return 0.0
    roi_pred = pred[roi_slices]
    roi_gt = gt[roi_slices]
    return calculate_dice(roi_pred, roi_gt)

def calculate_precision_recall(pred, gt):
    """计算精确率和召回率"""
    true_positive = np.sum(pred * gt)
    false_positive = np.sum(pred) - true_positive
    false_negative = np.sum(gt) - true_positive
    
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    
    return precision, recall

def calculate_hd95(pred, gt, spacing=None):
    """计算95%Hausdorff距离"""
    if np.sum(pred) == 0 or np.sum(gt) == 0:
        return float('inf')
    
    try:
        if spacing is not None:
            hd95 = metric.binary.hd95(pred, gt, voxelspacing=spacing)
        else:
            hd95 = metric.binary.hd95(pred, gt)
        return hd95
    except Exception as e:
        logging.error(f"计算HD95时出错: {e}")
        return float('inf')

def calculate_iou(pred, gt):
    """计算IoU"""
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    return intersection / union if union > 0 else 0

def calculate_iou_accuracy(pred, gt, thresholds):
    """计算不同IoU阈值下的准确率"""
    results = {}
    iou = calculate_iou(pred, gt)
    
    for threshold in thresholds:
        results[threshold] = 1 if iou >= threshold else 0
    
    return results

def load_nifti(file_path):
    """加载NIFTI文件"""
    sitk_img = sitk.ReadImage(file_path)
    spacing = sitk_img.GetSpacing()
    spacing = (spacing[2], spacing[0], spacing[1])  # 转换为(z,y,x)格式
    array = sitk.GetArrayFromImage(sitk_img)
    return array, sitk_img, spacing

def evaluate_segmentation(pred_dir, gt_dir, output_dir, threshold=0.5, save_individual=False, prob_dir=None):
    """评估分割结果"""
    logger = setup_logger(output_dir)
    logger.info(f"开始评估分割结果: {pred_dir}")
    logger.info(f"真值目录: {gt_dir}")
    logger.info(f"输出目录: {output_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有预测文件
    pred_files = sorted(glob(os.path.join(pred_dir, "*.nii.gz")))
    logger.info(f"找到 {len(pred_files)} 个预测文件")
    
    # 初始化指标收集器
    all_metrics = {
        'case_id': [],
        'dice': [],
        'roi_dice': [],
        'precision': [],
        'recall': [],
        'hd95': [],
        'iou': [],
        'volume_pred': [],
        'volume_gt': []
    }
    
    # IoU阈值
    iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    iou_accuracies = []
    
    # 如果有概率图目录，准备ROC和PR曲线数据
    if prob_dir:
        all_probs = []
        all_labels = []
    
    # 处理每个预测文件
    for pred_file in tqdm(pred_files, desc="评估分割结果"):
        case_id = os.path.basename(pred_file)
        gt_file = os.path.join(gt_dir, case_id)
        
        # 检查真值文件是否存在
        if not os.path.exists(gt_file):
            logger.warning(f"找不到真值文件: {gt_file}")
            continue
        
        # 加载预测和真值
        pred_array, pred_sitk, spacing = load_nifti(pred_file)
        gt_array, _, _ = load_nifti(gt_file)
        
        # 确保二值化
        pred_binary = (pred_array > 0).astype(np.uint8)
        gt_binary = (gt_array > 0).astype(np.uint8)
        
        # 计算指标
        dice = calculate_dice(pred_binary, gt_binary)
        roi_dice = calculate_roi_dice(pred_binary, gt_binary)
        precision, recall = calculate_precision_recall(pred_binary, gt_binary)
        hd95 = calculate_hd95(pred_binary, gt_binary, spacing)
        iou = calculate_iou(pred_binary, gt_binary)
        iou_accuracy = calculate_iou_accuracy(pred_binary, gt_binary, iou_thresholds)
        iou_accuracies.append(iou_accuracy)
        
        # 计算体积
        volume_pred = np.sum(pred_binary) * spacing[0] * spacing[1] * spacing[2]
        volume_gt = np.sum(gt_binary) * spacing[0] * spacing[1] * spacing[2]
        
        # 收集指标
        all_metrics['case_id'].append(case_id)
        all_metrics['dice'].append(dice)
        all_metrics['roi_dice'].append(roi_dice)
        all_metrics['precision'].append(precision)
        all_metrics['recall'].append(recall)
        all_metrics['hd95'].append(hd95 if hd95 != float('inf') else np.nan)
        all_metrics['iou'].append(iou)
        all_metrics['volume_pred'].append(volume_pred)
        all_metrics['volume_gt'].append(volume_gt)
        
        # 如果有概率图目录，加载概率图并收集数据用于ROC和PR曲线
        if prob_dir:
            prob_file = os.path.join(prob_dir, case_id.replace('.nii.gz', '.npz'))
            if os.path.exists(prob_file):
                prob_data = np.load(prob_file)
                prob_map = prob_data['softmax']
                
                # 收集所有体素的概率和标签
                flat_prob = prob_map[1].flatten()  # 假设索引1是前景类别
                flat_gt = gt_binary.flatten()
                
                # 随机采样以减少数据量（对于大型3D图像）
                if len(flat_prob) > 1000000:
                    indices = np.random.choice(len(flat_prob), 1000000, replace=False)
                    flat_prob = flat_prob[indices]
                    flat_gt = flat_gt[indices]
                
                all_probs.extend(flat_prob)
                all_labels.extend(flat_gt)
        
        # 记录个例结果
        hd95_str = f"{hd95:.4f}" if hd95 != float('inf') else "inf"
        logger.info(f"案例 {case_id}: Dice={dice:.4f}, ROI Dice={roi_dice:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, HD95={hd95_str}, IoU={iou:.4f}")
        
        # 保存个例结果
        if save_individual:
            case_output_dir = os.path.join(output_dir, 'individual_cases')
            os.makedirs(case_output_dir, exist_ok=True)
            
            with open(os.path.join(case_output_dir, f"{case_id.replace('.nii.gz', '')}_metrics.txt"), 'w') as f:
                f.write(f"Dice: {dice:.4f}\n")
                f.write(f"ROI Dice: {roi_dice:.4f}\n")
                f.write(f"Precision: {precision:.4f}\n")
                f.write(f"Recall: {recall:.4f}\n")
                f.write(f"HD95: {hd95:.4f if hd95 != float('inf') else 'inf'}\n")
                f.write(f"IoU: {iou:.4f}\n")
                f.write(f"Volume (Pred): {volume_pred:.2f} mm³\n")
                f.write(f"Volume (GT): {volume_gt:.2f} mm³\n")
    
    # 创建DataFrame并保存
    df = pd.DataFrame(all_metrics)
    df.to_csv(os.path.join(output_dir, 'all_metrics.csv'), index=False)
    
    # 计算平均指标
    avg_metrics = {
        'dice': np.mean(all_metrics['dice']),
        'roi_dice': np.mean(all_metrics['roi_dice']),
        'precision': np.mean(all_metrics['precision']),
        'recall': np.mean(all_metrics['recall']),
        'hd95': np.nanmean([h for h in all_metrics['hd95'] if not np.isnan(h)]),
        'iou': np.mean(all_metrics['iou'])
    }
    
    # 保存平均指标
    with open(os.path.join(output_dir, 'average_metrics.txt'), 'w') as f:
        for metric, value in avg_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    # 计算不同IoU阈值下的准确率
    iou_results = {}
    for threshold in iou_thresholds:
        accuracy = sum(1 for acc in iou_accuracies if acc[threshold] == 1) / len(iou_accuracies)
        iou_results[threshold] = accuracy
    
    # 保存IoU准确率
    with open(os.path.join(output_dir, 'iou_accuracy.txt'), 'w') as f:
        f.write("=== 不同IoU阈值下的准确率 ===\n")
        for threshold, accuracy in iou_results.items():
            f.write(f"IoU阈值 {threshold}: {accuracy:.4f}\n")
    
    # 绘制IoU准确率曲线
    plt.figure(figsize=(10, 8))
    plt.plot(list(iou_results.keys()), list(iou_results.values()), marker='o', linewidth=2, markersize=8)
    plt.xlabel('IoU Threshold')
    plt.ylabel('Accuracy')
    plt.title('Threshold-Dependent Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'iou_accuracy_curve.png'))
    plt.close()
    
    # 绘制Dice分布直方图
    plt.figure(figsize=(10, 8))
    sns.histplot(all_metrics['dice'], bins=20, kde=True)
    plt.axvline(avg_metrics['dice'], color='r', linestyle='--', label=f'Mean Dice: {avg_metrics["dice"]:.4f}')
    plt.xlabel('Dice Coefficient')
    plt.ylabel('Frequency')
    plt.title('Distribution of Dice Coefficients')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'dice_distribution.png'))
    plt.close()
    
    # 如果有概率图目录，绘制ROC和PR曲线
    if prob_dir and all_probs:
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        
        # 绘制ROC曲线
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
        plt.close()
        
        # 计算PR曲线
        precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
        pr_auc = average_precision_score(all_labels, all_probs)
        
        # 绘制PR曲线
        plt.figure(figsize=(10, 8))
        plt.plot(recall_curve, precision_curve, color='blue', lw=2, label=f'PR curve (AP = {pr_auc:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, 'pr_curve.png'))
        plt.close()
        
        # 保存ROC和PR曲线数据
        np.savez(os.path.join(output_dir, 'roc_pr_data.npz'), 
                 fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                 precision=precision_curve, recall=recall_curve, pr_auc=pr_auc)
    
    # 绘制箱线图
    plt.figure(figsize=(12, 10))
    data = [all_metrics['dice'], all_metrics['roi_dice'], all_metrics['precision'], all_metrics['recall']]
    labels = ['Dice', 'ROI Dice', 'Precision', 'Recall']
    
    box = plt.boxplot(data, patch_artist=True, labels=labels)
    
    colors = ['lightblue', 'lightgreen', 'lightpink', 'lightyellow']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    for i, d in enumerate(data):
        plt.scatter([i+1] * len(d), d, alpha=0.5, color='navy')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('Distribution of Evaluation Metrics')
    plt.ylabel('Score')
    plt.savefig(os.path.join(output_dir, 'metrics_boxplot.png'))
    plt.close()
    
    # 绘制体积比较散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(all_metrics['volume_gt'], all_metrics['volume_pred'], alpha=0.7)
    
    # 添加对角线
    max_vol = max(max(all_metrics['volume_gt']), max(all_metrics['volume_pred']))
    plt.plot([0, max_vol], [0, max_vol], 'r--')
    
    plt.xlabel('Ground Truth Volume (mm³)')
    plt.ylabel('Predicted Volume (mm³)')
    plt.title('Volume Comparison')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'volume_comparison.png'))
    plt.close()
    
    logger.info("评估完成！结果已保存到 %s", output_dir)
    logger.info("平均指标:")
    for metric, value in avg_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    return avg_metrics

def main():
    parser = argparse.ArgumentParser(description='医学图像分割评估工具')
    parser.add_argument('--pred_dir', required=True, help='分割结果目录')
    parser.add_argument('--gt_dir', required=True, help='真值标签目录')
    parser.add_argument('--output_dir', required=True, help='输出目录')
    parser.add_argument('--prob_dir', default=None, help='概率图目录（用于ROC和PR曲线）')
    parser.add_argument('--threshold', type=float, default=0.5, help='分割阈值')
    parser.add_argument('--save_individual', action='store_true', help='保存每个案例的单独评估结果')
    
    args = parser.parse_args()
    
    evaluate_segmentation(
        args.pred_dir,
        args.gt_dir,
        args.output_dir,
        args.threshold,
        args.save_individual,
        args.prob_dir
    )

if __name__ == "__main__":
    main() 