#!/usr/bin/env python3
"""
Generate Figure 4: Calibration curves by task
每个任务一张图，每张图显示 Self-report 和 Logits-based 的校准曲线
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

def plot_task_calibration(ax, task, task_color, sr_data, logits_data):
    """绘制单个任务的校准曲线"""
    
    # 按confidence分组
    sr_grouped = sr_data.groupby('confidence')['correct'].agg(['count', 'mean', 'std']).reset_index()
    sr_grouped.columns = ['confidence', 'count', 'mean_acc', 'std_acc']
    sr_grouped['sem'] = sr_grouped['std_acc'] / np.sqrt(sr_grouped['count'])
    
    logits_grouped = logits_data.groupby('confidence')['correct'].agg(['count', 'mean', 'std']).reset_index()
    logits_grouped.columns = ['confidence', 'count', 'mean_acc', 'std_acc']
    logits_grouped['sem'] = logits_grouped['std_acc'] / np.sqrt(logits_grouped['count'])
    
    # 合并到完整levels
    all_levels = sorted(set(sr_grouped['confidence'].unique()) | set(logits_grouped['confidence'].unique()))
    
    sr_full = pd.DataFrame({'confidence': all_levels})
    sr_full = sr_full.merge(sr_grouped, on='confidence', how='left').fillna({'mean_acc': 0, 'sem': 0, 'count': 0})
    
    logits_full = pd.DataFrame({'confidence': all_levels})
    logits_full = logits_full.merge(logits_grouped, on='confidence', how='left').fillna({'mean_acc': 0, 'sem': 0, 'count': 0})
    
    # 绘制散点图
    x_sr = sr_full['confidence'].values
    y_sr = sr_full['mean_acc'].values
    sizes_sr = np.sqrt(sr_full['count'].values) * 20
    
    x_logits = logits_full['confidence'].values
    y_logits = logits_full['mean_acc'].values
    sizes_logits = np.sqrt(logits_full['count'].values) * 20
    
    # Self-report (蓝色)
    ax.scatter(x_sr, y_sr, s=sizes_sr, alpha=0.6, color='#D55E00', 
               edgecolors='black', linewidth=0.5, label='Self-report')
    
    # Logits-based (绿色)
    ax.scatter(x_logits, y_logits, s=sizes_logits, alpha=0.6, color='#0072B2',
               edgecolors='black', linewidth=0.5, marker='^', label='Logits-based')
    
    # 添加误差棒
    ax.errorbar(x_sr, y_sr, yerr=sr_full['sem'].values, fmt='none', 
                color='#D55E00', alpha=0.5, linewidth=1, capsize=3)
    ax.errorbar(x_logits, y_logits, yerr=logits_full['sem'].values, fmt='none',
                color='#0072B2', alpha=0.5, linewidth=1, capsize=3)
    
    # 对角线（完美校准）
    min_val = min(min(x_sr), min(y_sr), min(x_logits), min(y_logits))
    max_val = max(max(x_sr), max(y_sr), max(x_logits), max(y_logits))
    ax.plot([0.5, 5.5], [0, 1], 'k--', alpha=0.3, linewidth=1, label='Perfect calibration')
    
    # 设置
    ax.set_xlabel('Confidence Level', fontsize=12, weight='bold')
    ax.set_ylabel('Actual Accuracy', fontsize=12, weight='bold')
    ax.set_title(f'{task} Task', fontsize=14, weight='bold', color=task_color)
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0, 1)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    
    # 添加整体准确度标注
    overall_sr_acc = sr_data['correct'].mean()
    overall_logits_acc = logits_data['correct'].mean()
    
    ax.axhline(overall_sr_acc, color='#D55E00', linestyle=':', alpha=0.4, linewidth=1)
    ax.axhline(overall_logits_acc, color='#0072B2', linestyle=':', alpha=0.4, linewidth=1)
    
    textstr = f"SR:{overall_sr_acc:.2f}, Log:{overall_logits_acc:.2f}"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

def generate_calibration_figure4():
    """生成第四张校准图：按任务拆分"""
    
    print("=" * 80)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("生成 Figure 4: Task-Specific Calibration Curves")
    print("=" * 80)
    
    # 读取数据
    print("\n1. 读取数据...")
    self_report_path = os.path.join(base_dir, 'data', 'raw', 'self_report', 'self_report_6models_merged.csv')
    sr = pd.read_csv(self_report_path)
    logits_path = os.path.join(base_dir, 'data', 'raw', 'logits', 'logits_6models_merged.csv')
    logits = pd.read_csv(logits_path)
    
    # 创建图形
    tasks = ['Grid', 'Gabor', 'Brightness']
    colors = ['#D62728', '#D55E00', '#0072B2']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, task, color in zip(axes, tasks, colors):
        print(f"\n处理 {task} 任务...")
        
        task_sr = sr[sr['task'] == task]
        task_logits = logits[logits['task'] == task]
        
        # 打印统计
        print(f"   Self-report: {len(task_sr)} 试次, 准确度={task_sr['correct'].mean():.3f}")
        print(f"   Logits-based: {len(task_logits)} 试次, 准确度={task_logits['correct'].mean():.3f}")
        
        # 绘制
        plot_task_calibration(ax, task, color, task_sr, task_logits)
    
    plt.tight_layout()
    
    # 保存
    output_dir = os.path.join(base_dir, 'results', 'figures', 'calibration')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'figure4_task_calibration_curves.pdf'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'figure4_task_calibration_curves.png'), 
                dpi=300, bbox_inches='tight')
    
    print("\n✅ 图表已保存:")
    print("   - figure4_task_calibration_curves.pdf")
    print("   - figure4_task_calibration_curves.png")
    
    print("\n" + "=" * 80)
    print("✅ Figure 4 生成完成！")
    print("=" * 80)
    
    plt.close()

if __name__ == "__main__":
    generate_calibration_figure4()


