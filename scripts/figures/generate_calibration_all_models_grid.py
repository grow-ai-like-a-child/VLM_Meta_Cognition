#!/usr/bin/env python3
"""
生成9×4大图：每行一个模型，每列一个指标
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# 任务颜色
TASK_COLORS = {
    'Grid': '#D62728',        # 红色
    'Gabor': '#D55E00',       # 橙色
    'Brightness': '#0072B2'   # 蓝色
}


def plot_metric_for_model(ax, model_name, model_data, metric_name, metric_col, 
                          is_first_row=False, is_last_row=False, is_first_col=False, 
                          ylim=None):
    """
    绘制单个模型的单个指标
    
    参数:
        ax: matplotlib轴
        model_name: 模型名称
        model_data: 该模型的数据
        metric_name: 指标显示名称
        metric_col: 数据列名
        is_first_row: 是否第一行（显示列标题）
        is_last_row: 是否最后一行（显示x轴标签）
        is_first_col: 是否第一列（显示y轴标签/模型名）
        ylim: y轴范围（统一所有子图）
    """
    tasks = ['Grid', 'Gabor', 'Brightness']
    colors = [TASK_COLORS[task] for task in tasks]
    x = np.arange(len(tasks))
    
    # 准备数据
    values = [model_data[model_data['task'] == task][metric_col].values[0] for task in tasks]
    
    # 绘制bar
    bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # 添加数值标签
    label_offset = (ylim[1] - ylim[0]) * 0.03 if ylim else max(values) * 0.05
    for i, val in enumerate(values):
        ax.text(i, val + label_offset, f'{val:.2f}', 
                ha='center', va='bottom', fontsize=8)
    
    # 设置y轴范围（使用统一范围）
    if ylim:
        ax.set_ylim(ylim)
    else:
        if 'AUC' in metric_col:
            ax.set_ylim([0, 1.0])
        else:
            ax.set_ylim([0, max(values) * 1.3])
    
    # ROC AUC添加0.5参考线
    if 'ROC' in metric_col:
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
    
    # 设置x轴（所有行都显示标签）
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=9)
    
    # 网格
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    
    # 第一行：显示列标题
    if is_first_row:
        ax.set_title(metric_name, fontsize=11, weight='bold', pad=8)
    
    # 第一列：显示模型名（作为y轴标签）
    if is_first_col:
        ax.set_ylabel(model_name, fontsize=10, weight='bold', rotation=0, 
                     ha='right', va='center', labelpad=40)
    
    # 调整刻度
    ax.tick_params(axis='both', which='major', labelsize=8)


def generate_all_models_grid_figure():
    """生成9×4大图"""
    
    print("\n" + "=" * 80)
    print("生成9×4大图：所有模型的4个指标对比")
    print("=" * 80)
    
    # 读取数据
    # 从 VLM_Meta_Cognition 目录作为根目录
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    detail_path = os.path.join(base_dir, 'results', 'calibration_9_models', 'calibration_task_model_full.csv')
    
    print("\n1. 读取数据...")
    df = pd.read_csv(detail_path)
    print(f"   数据形状: {df.shape}")
    
    # 获取所有模型
    models = sorted(df['model'].unique())
    print(f"   模型数量: {len(models)}")
    
    # 定义指标
    metrics = [
        ('ECE (Absolute)', 'ECE_absolute'),
        ('ECE (Squared)', 'ECE_squared'),
        ('ROC AUC', 'ROC_AUC'),
        ('Selective AUC', 'Selective_AUC')
    ]
    
    # 计算每个指标的全局最大值，用于统一y轴
    print("\n   计算全局y轴范围...")
    ylims = {}
    for metric_name, metric_col in metrics:
        if 'AUC' in metric_col:
            ylims[metric_col] = [0, 1.0]
        else:
            max_val = df[metric_col].max()
            ylims[metric_col] = [0, max_val * 1.15]  # 留15%空间
        print(f"     {metric_name}: {ylims[metric_col]}")
    
    # 创建图形
    print("\n2. 创建图表...")
    fig, axes = plt.subplots(len(models), len(metrics), 
                            figsize=(16, 2.2 * len(models)))
    
    # 调整子图间距
    plt.subplots_adjust(left=0.12, right=0.98, top=0.97, bottom=0.04, 
                       hspace=0.15, wspace=0.25)
    
    # 添加总标题
    fig.suptitle('Calibration Metrics Across All Models and Tasks', 
                fontsize=16, weight='bold', y=0.995)
    
    # 绘制每个子图
    print("\n3. 绘制子图...")
    gpt5_row_idx = None
    for i, model_name in enumerate(models):
        print(f"   处理 {model_name}...")
        model_data = df[df['model'] == model_name]
        
        # 记录gpt-5所在的行
        if model_name == 'gpt-5':
            gpt5_row_idx = i
        
        for j, (metric_name, metric_col) in enumerate(metrics):
            ax = axes[i, j] if len(models) > 1 else axes[j]
            
            plot_metric_for_model(
                ax, model_name, model_data, metric_name, metric_col,
                is_first_row=(i == 0),
                is_last_row=(i == len(models) - 1),
                is_first_col=(j == 0),
                ylim=ylims[metric_col]
            )
    
    # 为gpt-5那一行添加高亮框
    if gpt5_row_idx is not None:
        print("\n   为gpt-5添加高亮框...")
        from matplotlib.patches import Rectangle
        
        # 获取gpt-5那一行第一个和最后一个子图的位置
        ax_first = axes[gpt5_row_idx, 0]
        ax_last = axes[gpt5_row_idx, -1]
        
        # 在figure坐标系中添加矩形框
        # 获取第一个子图的左边界和最后一个子图的右边界
        bbox_first = ax_first.get_position()
        bbox_last = ax_last.get_position()
        
        # 创建一个跨越整行的矩形框
        rect = Rectangle(
            (bbox_first.x0 - 0.01, bbox_first.y0 - 0.01),  # 左下角
            bbox_last.x1 - bbox_first.x0 + 0.02,  # 宽度
            bbox_first.y1 - bbox_first.y0 + 0.02,  # 高度
            fill=False,
            edgecolor='red',
            linewidth=3,
            transform=fig.transFigure,
            zorder=100
        )
        fig.patches.append(rect)
    
    # 保存
    print("\n4. 保存图表...")
    output_dir = os.path.join(base_dir, 'results', 'figures', 'calibration_9_models')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file_pdf = os.path.join(output_dir, 'figureA2_Calibration_Metrics_Across_All_Models_and_Tasks.pdf')
    
    plt.savefig(output_file_pdf, bbox_inches='tight', dpi=300)
    
    print(f"   ✅ PDF: {output_file_pdf}")
    
    plt.close()
    
    print("\n" + "=" * 80)
    print("✅ 9×4大图生成完成！")
    print("=" * 80)


if __name__ == '__main__':
    generate_all_models_grid_figure()

