#!/usr/bin/env python3
"""
生成完整校准指标图：
1. 总图：9个模型合并，显示ECE/AUC + Calibration Curves
2. 分图：每个模型一张图，2×2子图显示4个指标
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10

# 任务颜色（统一使用）
TASK_COLORS = {
    'Grid': '#D62728',        # 红色
    'Gabor': '#D55E00',       # 橙色
    'Brightness': '#0072B2'   # 蓝色
}


def plot_calibration_curve(ax, task, task_data, task_color):
    """
    绘制单个任务的校准曲线（9个模型合并）
    
    参数:
        ax: matplotlib轴
        task: 任务名
        task_data: 该任务的数据（9个模型合并）
        task_color: 任务颜色
    """
    # 按confidence分组
    grouped = task_data.groupby('confidence')['correct'].agg(['count', 'mean', 'std']).reset_index()
    grouped.columns = ['confidence', 'count', 'mean_acc', 'std_acc']
    grouped['sem'] = grouped['std_acc'] / np.sqrt(grouped['count'])
    
    # 绘制散点图
    x = grouped['confidence'].values
    y = grouped['mean_acc'].values
    sizes = np.sqrt(grouped['count'].values) * 20
    
    ax.scatter(x, y, s=sizes, alpha=0.7, color=task_color, 
               edgecolors='black', linewidth=0.8, label=f'{task} Task')
    
    # 添加误差棒
    ax.errorbar(x, y, yerr=grouped['sem'].values, fmt='none', 
                color=task_color, alpha=0.5, linewidth=1.5, capsize=4)
    
    # 对角线（完美校准）
    ax.plot([0.5, 5.5], [0, 1], 'k--', alpha=0.4, linewidth=1.5, label='Perfect calibration')
    
    # 设置
    ax.set_xlabel('Confidence Level', fontsize=12, weight='bold')
    ax.set_ylabel('Actual Accuracy', fontsize=12, weight='bold')
    ax.set_title(f'{task} Task', fontsize=13, weight='bold', color=task_color)
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0, 1)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.grid(alpha=0.3, linestyle='--')
    
    # 添加整体准确度标注
    overall_acc = task_data['correct'].mean()
    ax.axhline(overall_acc, color=task_color, linestyle=':', alpha=0.5, linewidth=1.5)
    
    textstr = f'Overall Acc: {overall_acc:.3f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))


def generate_overall_summary_figure():
    """生成总图：9个模型合并的ECE/AUC + Calibration Curves"""
    
    print("\n" + "=" * 80)
    print("生成总图：Overall Summary（9个模型合并）")
    print("=" * 80)
    
    # 读取数据
    # 从 VLM_Meta_Cognition 目录作为根目录
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("\n1. 读取数据...")
    # 汇总数据
    summary_path = os.path.join(base_dir, 'results', 'calibration_9_models', 'calibration_by_task_full.csv')
    summary_df = pd.read_csv(summary_path)
    print(f"   汇总数据: {summary_df.shape}")
    
    # 原始数据（用于绘制曲线）
    raw_path = os.path.join(base_dir, 'data', 'raw', 'self_report', 'self_report_9models_merged.csv')
    raw_df = pd.read_csv(raw_path)
    print(f"   原始数据: {raw_df.shape}")
    
    # 创建图形
    print("\n2. 创建图表...")
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1.2], hspace=0.35, wspace=0.25,
                          left=0.08, right=0.98, top=0.95, bottom=0.08)
    
    tasks = ['Grid', 'Gabor', 'Brightness']
    
    # Panel A: ECE (absolute)
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(tasks))
    
    ece_values = [summary_df[summary_df['task'] == task]['ECE_absolute'].values[0] for task in tasks]
    colors = [TASK_COLORS[task] for task in tasks]
    
    bars = ax1.bar(x, ece_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.8)
    
    # 添加数值标签
    for i, val in enumerate(ece_values):
        ax1.text(i, val + 0.02, f'{val:.3f}', ha='center', va='bottom', fontsize=10, weight='bold')
    
    ax1.set_xlabel('Task', fontsize=13, weight='bold')
    ax1.set_ylabel('ECE (Absolute)', fontsize=13, weight='bold')
    ax1.set_title('A) Expected Calibration Error (Absolute)', fontsize=14, weight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim([0, max(ece_values) * 1.2])
    
    # Panel B: ROC AUC
    ax2 = fig.add_subplot(gs[0, 1])
    
    auc_values = [summary_df[summary_df['task'] == task]['ROC_AUC'].values[0] for task in tasks]
    
    bars = ax2.bar(x, auc_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.8)
    
    # 添加数值标签
    for i, val in enumerate(auc_values):
        ax2.text(i, val + 0.02, f'{val:.3f}', ha='center', va='bottom', fontsize=10, weight='bold')
    
    ax2.set_xlabel('Task', fontsize=13, weight='bold')
    ax2.set_ylabel('ROC AUC', fontsize=13, weight='bold')
    ax2.set_title('B) Area Under ROC Curve', fontsize=14, weight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tasks)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim([0, 1.0])
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Panel C: Summary stats
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    
    # 添加汇总统计文本
    summary_text = "Overall Statistics\n" + "=" * 30 + "\n\n"
    for task in tasks:
        task_summary = summary_df[summary_df['task'] == task].iloc[0]
        summary_text += f"{task}:\n"
        summary_text += f"  ECE (abs): {task_summary['ECE_absolute']:.3f}\n"
        summary_text += f"  ECE (sq):  {task_summary['ECE_squared']:.3f}\n"
        summary_text += f"  ROC AUC:   {task_summary['ROC_AUC']:.3f}\n"
        summary_text += f"  Sel AUC:   {task_summary['Selective_AUC']:.3f}\n"
        summary_text += f"  n={int(task_summary['n_samples'])}\n\n"
    
    ax3.text(0.1, 0.95, summary_text, transform=ax3.transAxes, fontsize=10,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    # Panel D-F: Calibration Curves（3个任务）
    for i, task in enumerate(tasks):
        ax = fig.add_subplot(gs[1, i])
        task_data = raw_df[raw_df['task'] == task]
        print(f"   绘制 {task} 校准曲线... ({len(task_data)} 样本)")
        plot_calibration_curve(ax, task, task_data, TASK_COLORS[task])
    
    # 保存
    print("\n3. 保存图表...")
    output_dir = os.path.join(base_dir, 'results', 'figures', 'calibration_9_models')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file_pdf = os.path.join(output_dir, 'figureA1_Calibration_metrics_and_curves_for_all_9_VLMs_including_closed-source_models.pdf')
    
    plt.savefig(output_file_pdf, bbox_inches='tight', dpi=300)
    
    print(f"   ✅ PDF: {output_file_pdf}")
    
    plt.close()


def generate_model_specific_figure(model_name, model_data):
    """
    生成单个模型的2×2子图
    
    参数:
        model_name: 模型名称
        model_data: 该模型的数据（3个任务）
    """
    print(f"\n   生成 {model_name} 的图表...")
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Calibration Metrics: {model_name}', fontsize=16, weight='bold', y=0.98)
    
    tasks = ['Grid', 'Gabor', 'Brightness']
    colors = [TASK_COLORS[task] for task in tasks]
    x = np.arange(len(tasks))
    
    # 准备数据
    ece_abs = [model_data[model_data['task'] == task]['ECE_absolute'].values[0] for task in tasks]
    ece_sq = [model_data[model_data['task'] == task]['ECE_squared'].values[0] for task in tasks]
    roc_auc = [model_data[model_data['task'] == task]['ROC_AUC'].values[0] for task in tasks]
    sel_auc = [model_data[model_data['task'] == task]['Selective_AUC'].values[0] for task in tasks]
    
    # 左上：ECE (absolute)
    ax = axes[0, 0]
    bars = ax.bar(x, ece_abs, color=colors, alpha=0.8, edgecolor='black', linewidth=0.8)
    for i, val in enumerate(ece_abs):
        ax.text(i, val + max(ece_abs) * 0.03, f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('ECE (Absolute)', fontsize=12, weight='bold')
    ax.set_title('ECE (Absolute)', fontsize=13, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, max(ece_abs) * 1.25])
    
    # 右上：ECE (squared)
    ax = axes[0, 1]
    bars = ax.bar(x, ece_sq, color=colors, alpha=0.8, edgecolor='black', linewidth=0.8)
    for i, val in enumerate(ece_sq):
        ax.text(i, val + max(ece_sq) * 0.03, f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('ECE (Squared)', fontsize=12, weight='bold')
    ax.set_title('ECE (Squared)', fontsize=13, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, max(ece_sq) * 1.25])
    
    # 左下：ROC AUC
    ax = axes[1, 0]
    bars = ax.bar(x, roc_auc, color=colors, alpha=0.8, edgecolor='black', linewidth=0.8)
    for i, val in enumerate(roc_auc):
        ax.text(i, val + 0.03, f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_xlabel('Task', fontsize=12, weight='bold')
    ax.set_ylabel('ROC AUC', fontsize=12, weight='bold')
    ax.set_title('ROC AUC', fontsize=13, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.0])
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # 右下：Selective AUC
    ax = axes[1, 1]
    bars = ax.bar(x, sel_auc, color=colors, alpha=0.8, edgecolor='black', linewidth=0.8)
    for i, val in enumerate(sel_auc):
        ax.text(i, val + 0.03, f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_xlabel('Task', fontsize=12, weight='bold')
    ax.set_ylabel('Selective AUC', fontsize=12, weight='bold')
    ax.set_title('Selective Accuracy AUC', fontsize=13, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    
    # 保存
    # 从 VLM_Meta_Cognition 目录作为根目录
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(base_dir, 'results', 'figures', 'calibration_9_models')
    
    # 清理模型名称用于文件名
    safe_model_name = model_name.replace('/', '_').replace(' ', '_')
    
    output_file_pdf = os.path.join(output_dir, f'figure_full_model_{safe_model_name}.pdf')
    output_file_png = os.path.join(output_dir, f'figure_full_model_{safe_model_name}.png')
    
    plt.savefig(output_file_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(output_file_png, bbox_inches='tight', dpi=300)
    
    plt.close()


def main():
    """主函数"""
    print("=" * 80)
    print("生成完整校准指标图")
    print("=" * 80)
    
    # 只生成总图
    generate_overall_summary_figure()
    
    print("\n" + "=" * 80)
    print("✅ 图表生成完成！")
    print("=" * 80)
    
    # 从 VLM_Meta_Cognition 目录作为根目录
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(base_dir, 'results', 'figures', 'calibration_9_models')
    print(f"\n输出目录: {output_dir}")
    print(f"  - 总图：figureA1_Calibration_metrics_and_curves_for_all_9_VLMs_including_closed-source_models.pdf")


if __name__ == '__main__':
    main()

