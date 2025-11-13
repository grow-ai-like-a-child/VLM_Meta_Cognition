#!/usr/bin/env python3
"""
计算校准指标：ECE 和 AUC
生成 calibration_summary_method_task.csv 和 task_calibration_metrics.csv
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import os

def calculate_ece(df, conf_col='confidence', acc_col='correct'):
    """
    计算 Expected Calibration Error (ECE)
    使用论文中的方法：按置信度级别（1-5）分bin，将置信度归一化到[0,1]
    """
    if len(df) == 0:
        return np.nan
    
    ece = 0
    n_total = len(df)
    
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
    
    return ece


def calculate_auc(df, conf_col='confidence', acc_col='correct'):
    """计算 ROC AUC"""
    if len(df) == 0:
        return np.nan
    
    try:
        # 将置信度归一化到0-1范围
        conf_normalized = (df[conf_col] - df[conf_col].min()) / (df[conf_col].max() - df[conf_col].min() + 1e-10)
        auc = roc_auc_score(df[acc_col], conf_normalized)
        return auc
    except:
        return np.nan


def calculate_slope(df, conf_col='confidence', acc_col='correct'):
    """计算线性回归斜率"""
    if len(df) < 2:
        return np.nan, np.nan, np.nan
    
    from scipy import stats
    
    # 按置信度分组，计算平均准确度
    grouped = df.groupby(conf_col)[acc_col].agg(['mean', 'count']).reset_index()
    grouped.columns = [conf_col, 'mean_acc', 'count']
    
    if len(grouped) < 2:
        return np.nan, np.nan, np.nan
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        grouped[conf_col].values, grouped['mean_acc'].values
    )
    
    return slope, intercept, r_value


def main():
    """主函数"""
    print("=" * 80)
    print("计算校准指标")
    print("=" * 80)
    
    # 获取项目根目录
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 读取数据
    print("\n1. 读取数据...")
    self_report_path = os.path.join(base_dir, 'data', 'raw', 'self_report', 'self_report_6models_merged.csv')
    logits_path = os.path.join(base_dir, 'data', 'raw', 'logits', 'logits_6models_merged.csv')
    
    self_report = pd.read_csv(self_report_path)
    logits = pd.read_csv(logits_path)
    
    self_report['method'] = 'Self-report'
    logits['method'] = 'Logits-based'
    
    print(f"   Self-report: {len(self_report)} 行")
    print(f"   Logits-based: {len(logits)} 行")
    
    # 合并数据
    all_data = pd.concat([self_report, logits], ignore_index=True)
    
    # 计算指标
    print("\n2. 计算校准指标...")
    tasks = ['Grid', 'Gabor', 'Brightness']
    methods = ['Self-report', 'Logits-based']
    
    summary_data = []
    slope_data = []
    
    for method in methods:
        method_data = all_data[all_data['method'] == method]
        
        for task in tasks:
            task_data = method_data[method_data['task'] == task]
            
            if len(task_data) > 0:
                # 计算 ECE
                ece = calculate_ece(task_data)
                
                # 计算 AUC
                auc = calculate_auc(task_data)
                
                # 计算斜率
                slope, intercept, r_value = calculate_slope(task_data)
                
                summary_data.append({
                    'method': method,
                    'task': task,
                    'ECE': ece,
                    'AUC': auc,
                    'n_samples': len(task_data)
                })
                
                slope_data.append({
                    'Task': task,
                    'Method': method,
                    'slope': slope,
                    'intercept': intercept,
                    'r_value': r_value
                })
                
                print(f"   {method} - {task}: ECE={ece:.3f}, AUC={auc:.3f}, slope={slope:.3f}")
    
    # 保存结果
    print("\n3. 保存结果...")
    output_dir = os.path.join(base_dir, 'results', 'calibration')
    os.makedirs(output_dir, exist_ok=True)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'calibration_summary_method_task.csv'), index=False)
    print(f"   ✅ calibration_summary_method_task.csv")
    
    slope_df = pd.DataFrame(slope_data)
    slope_df.to_csv(os.path.join(output_dir, 'task_calibration_metrics.csv'), index=False)
    print(f"   ✅ task_calibration_metrics.csv")
    
    print("\n" + "=" * 80)
    print("✅ 校准指标计算完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()

