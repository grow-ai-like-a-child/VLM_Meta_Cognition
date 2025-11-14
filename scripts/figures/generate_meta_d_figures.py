#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成Meta-d'分析的可视化图表（6 vs 6公平对比 + 闭源模型单独）

生成主要图表：
1. meta_d_comparison_open_source.pdf - 6 vs 6公平对比图（开源模型）
2. meta_d_closed_source.pdf - 闭源模型单独图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import seaborn as sns
import os

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11

# 配色方案
COLORS = {
    'Self-report': '#D55E00',  # 橙色
    'Logits-based': '#0072B2'  # 蓝色
}

TASK_COLORS = {
    'Grid': '#D62728',        # 红色
    'Gabor': '#0072B2',       # 蓝色
    'Brightness': '#2CA02C'   # 绿色
}

MODEL_COLORS = {
    'claude-sonnet-4-5': '#E69F00',  # 橙色
    'gpt-4o': '#56B4E9',              # 浅蓝色
    'gpt-5': '#009E73'                # 绿色
}

MODEL_MARKERS = {
    'claude-sonnet-4-5': 'o',  # 圆形
    'gpt-4o': 's',             # 方形
    'gpt-5': '^'               # 三角形
}

TASK_MARKERS = {
    'Grid': 'o',        # 圆形
    'Gabor': 's',       # 方形
    'Brightness': '^'   # 三角形
}

def get_model_display_name(model):
    """获取模型的显示名称"""
    if model == 'claude-sonnet-4-5':
        return 'Claude'
    elif model == 'gpt-4o':
        return 'GPT-4o'
    elif model == 'gpt-5':
        return 'GPT-5'
    else:
        # 默认：使用第一个单词，首字母大写
        return model.split('-')[0].capitalize()


def plot_open_source_comparison(base_dir):
    """生成6 vs 6公平对比图（开源模型）- 精简版（2子图）"""
    print("   生成6 vs 6公平对比图（开源模型，精简版）...")
    
    # 读取数据
    overall_open = pd.read_csv(os.path.join(base_dir, 'results', 'meta_d', 'meta_d_overall_open_source.csv'))
    by_task_open = pd.read_csv(os.path.join(base_dir, 'results', 'meta_d', 'meta_d_by_task_open_source.csv'))
    
    # 创建图形 - 1行2列布局
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    
    # Panel A: M-ratio 柱状图（按方法）- 核心发现
    ax_a = fig.add_subplot(gs[0, 0])
    methods = ['Self-report', 'Logits-based']
    m_ratios = []
    m_ratio_stds = []
    
    # 读取详细数据用于计算标准差
    detailed_sr_open = pd.read_csv(os.path.join(base_dir, 'results', 'meta_d', 'meta_d_detailed_selfreport_open.csv'))
    detailed_logits = pd.read_csv(os.path.join(base_dir, 'results', 'meta_d', 'meta_d_detailed_logits.csv'))
    
    for method in methods:
        method_data = overall_open[overall_open['method'] == method]
        if len(method_data) > 0:
            m_ratios.append(method_data['M_ratio'].values[0])
            # 从详细数据计算标准差
            if method == 'Self-report':
                detailed_method = detailed_sr_open
            else:
                detailed_method = detailed_logits
            if len(detailed_method) > 0:
                m_ratio_stds.append(detailed_method['M_ratio'].std())
            else:
                m_ratio_stds.append(0)
        else:
            m_ratios.append(0)
            m_ratio_stds.append(0)
    
    bars = ax_a.bar(methods, m_ratios, color=[COLORS[m] for m in methods], 
                   alpha=0.8, edgecolor='black', linewidth=1.5, yerr=m_ratio_stds, capsize=10)
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, m_ratios)):
        ax_a.text(bar.get_x() + bar.get_width()/2, val + m_ratio_stds[i] + 0.15,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=14, weight='bold')
    
    ax_a.axhline(y=1, color='gray', linestyle='--', alpha=0.6, linewidth=2, label='Ideal (M_ratio=1)')
    ax_a.set_ylabel('M-ratio', fontsize=18, weight='bold')
    ax_a.set_xlabel('', fontsize=18, weight='bold')  # 无x轴标签
    ax_a.set_title("(A) M-ratio by Method", fontsize=17, weight='bold', pad=15)
    ax_a.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=12)
    ax_a.grid(axis='y', alpha=0.3, linestyle='--')
    ax_a.set_ylim([0, max(m_ratios) + max(m_ratio_stds) + 0.5])
    ax_a.tick_params(axis='both', labelsize=14)  # 放大刻度标签
    
    # Panel B: 按任务比较 - M-ratio
    ax_b = fig.add_subplot(gs[0, 1])
    tasks = ['Grid', 'Gabor', 'Brightness']
    x = np.arange(len(tasks))
    width = 0.35
    
    for i, method in enumerate(methods):
        method_data = by_task_open[by_task_open['method'] == method]
        values = [method_data[method_data['task'] == task]['M_ratio'].values[0] 
                 if len(method_data[method_data['task'] == task]) > 0 else 0 
                 for task in tasks]
        bars = ax_b.bar(x + (i - 0.5) * width, values, width, 
                       color=COLORS[method], alpha=0.8, 
                       edgecolor='black', linewidth=1.5, label=method)
        # 添加数值标签
        for j, (bar, val) in enumerate(zip(bars, values)):
            if val != 0:
                ax_b.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=11, weight='bold')
    
    ax_b.axhline(y=1, color='gray', linestyle='--', alpha=0.6, linewidth=2)
    ax_b.set_xlabel('', fontsize=18, weight='bold')  # 去掉"Task"标签
    ax_b.set_ylabel('M-ratio', fontsize=18, weight='bold')
    ax_b.set_title("(B) M-ratio by Task", fontsize=17, weight='bold', pad=15)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(tasks, fontsize=14)  # 放大x轴刻度标签
    ax_b.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=12)
    ax_b.grid(axis='y', alpha=0.3, linestyle='--')
    ax_b.tick_params(axis='both', labelsize=14)  # 放大刻度标签
    
    # 保存
    output_path = os.path.join(base_dir, 'results', 'figures', 'meta_d', 'meta_d_comparison_open_source.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      ✅ 已保存: {output_path}")


def plot_closed_source(base_dir):
    """生成闭源模型单独图（精简版：2子图）"""
    print("   生成闭源模型单独图（精简版）...")
    
    # 读取数据
    detailed_closed = pd.read_csv(os.path.join(base_dir, 'results', 'meta_d', 'meta_d_detailed_selfreport_closed.csv'))
    by_task_closed = pd.read_csv(os.path.join(base_dir, 'results', 'meta_d', 'meta_d_by_task_closed_source.csv'))
    
    # 创建图形 - 1行2列布局
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    
    # Panel A: meta-d' vs d' 散点图
    ax_a = fig.add_subplot(gs[0, 0])
    
    # 为每个模型-任务组合使用不同颜色（任务）和形状（模型）绘制散点
    for _, row in detailed_closed.iterrows():
        model = row['model']
        task = row['task']
        color = TASK_COLORS.get(task, COLORS['Self-report'])  # 任务用颜色
        marker = MODEL_MARKERS.get(model, 'o')  # 模型用形状
        
        ax_a.scatter(row['da'], row['meta_da'], 
                    color=color, marker=marker, alpha=0.7, s=100, 
                    edgecolors='black', linewidth=0.5)
    
    # 创建模型图例（形状）
    model_legend_elements = []
    for model in sorted(detailed_closed['model'].unique()):
        model_display = get_model_display_name(model)
        marker = MODEL_MARKERS.get(model, 'o')
        model_legend_elements.append(Line2D([0], [0], marker=marker, color='w', 
                                            markerfacecolor='gray', markersize=10,
                                            label=model_display, markeredgecolor='black', markeredgewidth=0.5))
    
    # 创建任务图例（颜色）
    task_legend_elements = []
    for task in ['Grid', 'Gabor', 'Brightness']:
        color = TASK_COLORS.get(task, COLORS['Self-report'])
        task_legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                          markerfacecolor=color, markersize=10,
                                          label=task, markeredgecolor='black', markeredgewidth=0.5))
    
    # 添加对角线
    max_val = max(detailed_closed[['da', 'meta_da']].max())
    min_val = min(detailed_closed[['da', 'meta_da']].min())
    ax_a.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1.5, label='M_ratio=1 (Ideal)')
    
    # 添加两个图例，并排显示
    legend1 = ax_a.legend(handles=model_legend_elements, loc='upper left', 
                         frameon=True, fancybox=True, shadow=True, fontsize=11, 
                         title='Model', title_fontsize=12)
    ax_a.add_artist(legend1)
    
    # 第二个图例紧贴第一个图例右侧（使用较小的x偏移）
    legend2 = ax_a.legend(handles=task_legend_elements, loc='upper left', 
                         frameon=True, fancybox=True, shadow=True, fontsize=11, 
                         title='Task', title_fontsize=12,
                         bbox_to_anchor=(0.28, 1.0))
    
    ax_a.set_xlabel("d' (Type 1 d')", fontsize=14, weight='bold')
    ax_a.set_ylabel("meta-d'", fontsize=18, weight='bold')
    ax_a.set_title("(A) Meta-d' vs d'", fontsize=17, weight='bold', pad=15)
    ax_a.grid(alpha=0.3, linestyle='--')
    ax_a.set_aspect('equal', adjustable='box')
    ax_a.tick_params(axis='both', labelsize=14)  # 放大刻度标签
    
    # Panel B: 按任务比较 - M-ratio
    ax_b = fig.add_subplot(gs[0, 1])
    tasks = ['Grid', 'Gabor', 'Brightness']
    x = np.arange(len(tasks))
    
    values = [by_task_closed[by_task_closed['task'] == task]['M_ratio'].values[0] 
             if len(by_task_closed[by_task_closed['task'] == task]) > 0 else 0 
             for task in tasks]
    
    # 为每个任务使用对应的颜色
    colors = [TASK_COLORS.get(task, COLORS['Self-report']) for task in tasks]
    bars = ax_b.bar(x, values, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, values)):
        if val != 0:
            if val >= 0:
                # 正值：标签显示在条形上方
                ax_b.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=12, weight='bold')
            else:
                # 负值：标签显示在条形下方
                ax_b.text(bar.get_x() + bar.get_width()/2, val - 0.02,
                         f'{val:.3f}', ha='center', va='top', fontsize=12, weight='bold')
    
    ax_b.axhline(y=1, color='gray', linestyle='--', alpha=0.6, linewidth=2)
    ax_b.set_xlabel('', fontsize=18, weight='bold')  # 去掉"Task"标签
    ax_b.set_ylabel('M-ratio', fontsize=18, weight='bold')
    ax_b.set_title("(B) M-ratio by Task", fontsize=17, weight='bold', pad=15)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(tasks, fontsize=14)  # 放大x轴刻度标签
    ax_b.grid(axis='y', alpha=0.3, linestyle='--')
    ax_b.tick_params(axis='both', labelsize=14)  # 放大刻度标签
    
    # 保存
    output_path = os.path.join(base_dir, 'results', 'figures', 'meta_d', 'meta_d_closed_source.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      ✅ 已保存: {output_path}")


def main():
    """主函数"""
    print("=" * 80)
    print("生成Meta-d'可视化图表（6 vs 6公平对比 + 闭源模型单独）")
    print("=" * 80)
    
    # 获取项目根目录
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 检查数据文件是否存在
    meta_d_dir = os.path.join(base_dir, 'results', 'meta_d')
    required_files = [
        'meta_d_overall_open_source.csv',
        'meta_d_overall_closed_source.csv',
        'meta_d_by_task_open_source.csv',
        'meta_d_by_task_closed_source.csv',
        'meta_d_detailed_selfreport_open.csv',
        'meta_d_detailed_selfreport_closed.csv',
        'meta_d_detailed_logits.csv'
    ]
    
    print("\n1. 检查数据文件...")
    for file in required_files:
        filepath = os.path.join(meta_d_dir, file)
        if os.path.exists(filepath):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} 不存在！")
            return
    
    # 创建输出目录
    output_dir = os.path.join(base_dir, 'results', 'figures', 'meta_d')
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成图表
    print("\n2. 生成图表...")
    plot_open_source_comparison(base_dir)
    plot_closed_source(base_dir)
    
    print("\n" + "=" * 80)
    print("✅ 所有图表生成完成！")
    print("=" * 80)
    print(f"\n输出目录: {output_dir}")
    print("\n生成的文件:")
    for file in sorted(os.listdir(output_dir)):
        if file.endswith(('.pdf', '.png')):
            print(f"  - {file}")


if __name__ == '__main__':
    main()
