#!/usr/bin/env python3
"""
计算完整校准指标：包含四种方法
1. ECE (absolute) - 绝对误差版本
2. ECE (squared) - 平方误差版本（论文方法）
3. ROC AUC - 标准ROC曲线下面积
4. Selective Accuracy AUC - 选择性准确率曲线下面积（论文方法）
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import os
import sys

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def calculate_ece_absolute(df, conf_col='confidence', acc_col='correct'):
    """
    计算Expected Calibration Error (Absolute版本)
    
    参数:
        df: DataFrame with 'confidence' (1-5) and 'correct' (0/1)
        conf_col: confidence列名
        acc_col: accuracy/correct列名
    
    返回:
        ece: ECE值（绝对误差）
        bin_stats: 每个bin的统计信息DataFrame
    """
    if len(df) == 0:
        return np.nan, pd.DataFrame()
    
    ece = 0
    n_total = len(df)
    bin_stats = []
    
    for conf_level in range(1, 6):
        bin_data = df[df[conf_col] == conf_level]
        if len(bin_data) == 0:
            continue
            
        # 计算bin内平均accuracy
        acc_bin = bin_data[acc_col].mean()
        
        # confidence归一化到[0,1]
        # confidence=1 -> 0.2, confidence=5 -> 1.0
        conf_normalized = 0.2 + (conf_level - 1) * 0.2
        
        # weight
        weight = len(bin_data) / n_total
        
        # ECE贡献（绝对误差）
        ece_contribution = abs(acc_bin - conf_normalized) * weight
        
        ece += ece_contribution
        
        bin_stats.append({
            'confidence': conf_level,
            'acc_bin': acc_bin,
            'conf_normalized': conf_normalized,
            'weight': weight,
            'n_samples': len(bin_data),
            'ece_contribution': ece_contribution
        })
    
    bin_stats_df = pd.DataFrame(bin_stats) if bin_stats else pd.DataFrame()
    
    return ece, bin_stats_df


def calculate_ece_squared(df, conf_col='confidence', acc_col='correct'):
    """
    计算Expected Calibration Error (Squared版本，论文方法)
    
    参数:
        df: DataFrame with 'confidence' (1-5) and 'correct' (0/1)
        conf_col: confidence列名
        acc_col: accuracy/correct列名
    
    返回:
        ece: ECE值（平方误差）
        bin_stats: 每个bin的统计信息DataFrame
    """
    if len(df) == 0:
        return np.nan, pd.DataFrame()
    
    ece = 0
    n_total = len(df)
    bin_stats = []
    
    for conf_level in range(1, 6):
        bin_data = df[df[conf_col] == conf_level]
        if len(bin_data) == 0:
            continue
            
        # 计算bin内平均accuracy
        acc_bin = bin_data[acc_col].mean()
        
        # confidence归一化到[0,1]
        conf_normalized = 0.2 + (conf_level - 1) * 0.2
        
        # weight
        weight = len(bin_data) / n_total
        
        # ECE贡献（平方误差，论文方法）
        ece_contribution = (acc_bin - conf_normalized) ** 2 * weight
        
        ece += ece_contribution
        
        bin_stats.append({
            'confidence': conf_level,
            'acc_bin': acc_bin,
            'conf_normalized': conf_normalized,
            'weight': weight,
            'n_samples': len(bin_data),
            'ece_contribution': ece_contribution
        })
    
    bin_stats_df = pd.DataFrame(bin_stats) if bin_stats else pd.DataFrame()
    
    return ece, bin_stats_df


def calculate_roc_auc(df, conf_col='confidence', acc_col='correct'):
    """
    计算ROC AUC（标准ROC曲线下面积）
    
    参数:
        df: DataFrame with 'confidence' (1-5) and 'correct' (0/1)
        conf_col: confidence列名
        acc_col: accuracy/correct列名
    
    返回:
        auc: ROC AUC值
    """
    if len(df) == 0:
        return np.nan
    
    # 将confidence转换为概率（归一化到[0,1]）
    # confidence=1 -> 0.2, confidence=5 -> 1.0
    y_pred = 0.2 + (df[conf_col].values - 1) * 0.2
    y_true = df[acc_col].values
    
    # 检查是否有variation
    if len(np.unique(y_true)) < 2:
        return np.nan
    
    # 计算AUC
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan
    
    return auc


def calculate_selective_accuracy_auc(df, conf_col='confidence', acc_col='correct'):
    """
    计算Selective Accuracy AUC（选择性准确率曲线下面积，论文方法）
    
    根据Geifman & El-Yaniv (2017)，计算选择性分类中的AUC
    
    参数:
        df: DataFrame with 'confidence' (1-5) and 'correct' (0/1)
        conf_col: confidence列名
        acc_col: accuracy/correct列名
    
    返回:
        auc: Selective Accuracy AUC值
        curve_data: 曲线数据（用于可视化）
    """
    if len(df) == 0:
        return np.nan, pd.DataFrame()
    
    # 按confidence从高到低排序
    df_sorted = df.sort_values(conf_col, ascending=False).reset_index(drop=True)
    
    # 计算累积准确率
    n_total = len(df_sorted)
    cumulative_correct = df_sorted[acc_col].cumsum()
    cumulative_count = np.arange(1, n_total + 1)
    
    # 覆盖率（coverage）：选择的样本比例
    coverage = cumulative_count / n_total
    
    # 选择性准确率（selective accuracy）：在给定覆盖率下的准确率
    selective_accuracy = cumulative_correct / cumulative_count
    
    # 计算AUC（使用梯形法则）
    try:
        auc = np.trapezoid(selective_accuracy, coverage)
    except Exception:
        return np.nan, pd.DataFrame()
    
    # 保存曲线数据
    curve_data = pd.DataFrame({
        'coverage': coverage,
        'selective_accuracy': selective_accuracy,
        'cumulative_correct': cumulative_correct.values,
        'cumulative_count': cumulative_count
    })
    
    return auc, curve_data


def calculate_all_metrics(df, groupby_cols=None):
    """
    计算所有四种校准指标
    
    参数:
        df: DataFrame with 'confidence', 'correct', 'model', 'task'
        groupby_cols: 分组列，如 ['method', 'task'] 或 ['method', 'task', 'model']
    
    返回:
        results_df: 包含所有组合的指标DataFrame
    """
    if groupby_cols is None:
        groupby_cols = []
    
    results = []
    
    if len(groupby_cols) == 0:
        # 只计算总体
        ece_abs, _ = calculate_ece_absolute(df)
        ece_sq, _ = calculate_ece_squared(df)
        roc_auc = calculate_roc_auc(df)
        sel_auc, _ = calculate_selective_accuracy_auc(df)
        
        results.append({
            'n_samples': len(df),
            'mean_confidence': df['confidence'].mean(),
            'mean_accuracy': df['correct'].mean(),
            'ECE_absolute': ece_abs,
            'ECE_squared': ece_sq,
            'ROC_AUC': roc_auc,
            'Selective_AUC': sel_auc
        })
    else:
        # 按分组计算
        for name, group in df.groupby(groupby_cols):
            if isinstance(name, tuple):
                group_dict = dict(zip(groupby_cols, name))
            else:
                group_dict = {groupby_cols[0]: name}
            
            ece_abs, _ = calculate_ece_absolute(group)
            ece_sq, _ = calculate_ece_squared(group)
            roc_auc = calculate_roc_auc(group)
            sel_auc, _ = calculate_selective_accuracy_auc(group)
            
            result = {
                **group_dict,
                'n_samples': len(group),
                'mean_confidence': group['confidence'].mean(),
                'mean_accuracy': group['correct'].mean(),
                'ECE_absolute': ece_abs,
                'ECE_squared': ece_sq,
                'ROC_AUC': roc_auc,
                'Selective_AUC': sel_auc
            }
            results.append(result)
    
    return pd.DataFrame(results)


def main():
    """主函数"""
    print("=" * 80)
    print("计算完整校准指标：ECE (absolute/squared) + AUC (ROC/Selective)")
    print("=" * 80)
    
    # 读取数据
    print("\n1. 读取数据...")
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 尝试多个可能的路径
    possible_paths = [
        os.path.join(base_dir, 'data', 'raw', 'self_report', 'self_report_9models_merged.csv'),
        os.path.join(base_dir, 'data', 'self_report', 'self_report_9models_merged.csv'),
    ]
    
    self_report_path = None
    for path in possible_paths:
        if os.path.exists(path):
            self_report_path = path
            break
    
    if self_report_path is None:
        print(f"   ❌ 文件不存在，尝试过的路径: {possible_paths}")
        return
    
    df = pd.read_csv(self_report_path)
    print(f"   数据形状: {df.shape}")
    
    # 确保有correct列
    if 'correct' not in df.columns:
        if 'accuracy' in df.columns:
            df['correct'] = df['accuracy'].astype(int)
        else:
            raise ValueError("数据中既没有'correct'也没有'accuracy'列")
    
    # 确保confidence是整数
    df['confidence'] = df['confidence'].astype(float).round().astype(int)
    df['correct'] = df['correct'].astype(int)
    
    # 计算不同层级的指标
    print("\n2. 计算校准指标...")
    
    output_dir = os.path.join(base_dir, 'results', 'calibration_9_models')
    os.makedirs(output_dir, exist_ok=True)
    
    # Level 1: Overall
    print("\n   Level 1: Overall")
    overall = calculate_all_metrics(df, groupby_cols=[])
    print(overall[['ECE_absolute', 'ECE_squared', 'ROC_AUC', 'Selective_AUC', 'n_samples']])
    overall.to_csv(os.path.join(output_dir, 'calibration_overall_full.csv'), index=False)
    
    # Level 2: By Task
    print("\n   Level 2: By Task")
    by_task = calculate_all_metrics(df, groupby_cols=['task'])
    by_task = by_task.sort_values('task')
    print(by_task[['task', 'ECE_absolute', 'ECE_squared', 'ROC_AUC', 'Selective_AUC']])
    by_task.to_csv(os.path.join(output_dir, 'calibration_by_task_full.csv'), index=False)
    
    # Level 3: By Model
    print("\n   Level 3: By Model")
    by_model = calculate_all_metrics(df, groupby_cols=['model'])
    by_model = by_model.sort_values('model')
    print(by_model[['model', 'ECE_absolute', 'ECE_squared', 'ROC_AUC', 'Selective_AUC']].head(10))
    by_model.to_csv(os.path.join(output_dir, 'calibration_by_model_full.csv'), index=False)
    
    # Level 4: Task × Model (完整数据)
    print("\n   Level 4: Task × Model (完整数据)")
    full_detail = calculate_all_metrics(df, groupby_cols=['task', 'model'])
    full_detail = full_detail.sort_values(['task', 'model'])
    print(f"   总组合数: {len(full_detail)}")
    print(full_detail[['task', 'model', 'ECE_absolute', 'ECE_squared', 'ROC_AUC', 'Selective_AUC']].head(10))
    full_detail.to_csv(os.path.join(output_dir, 'calibration_task_model_full.csv'), index=False)
    
    print("\n" + "=" * 80)
    print("✅ 完整校准指标计算完成！")
    print("=" * 80)
    print(f"\n输出目录: {output_dir}")
    print(f"\n输出文件:")
    print(f"  - calibration_overall_full.csv")
    print(f"  - calibration_by_task_full.csv")
    print(f"  - calibration_by_model_full.csv")
    print(f"  - calibration_task_model_full.csv")


if __name__ == '__main__':
    main()

