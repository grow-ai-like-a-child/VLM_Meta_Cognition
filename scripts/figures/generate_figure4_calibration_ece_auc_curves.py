#!/usr/bin/env python3
"""
Generate Figure 4: Calibration ECE AUC Curves
组合 Panel A (ECE 和 AUC 柱状图) 和 Panel B (校准曲线) 成完整的 Figure 4

Panel A: ECE 和 AUC 柱状图对比（左右两个子图）
Panel B: 三个任务的校准曲线（带斜率标注）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
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


def plot_panel_a_ece_auc(ax_ece, ax_auc, df):
    """绘制 Panel A: ECE 和 AUC 柱状图"""
    tasks = ['Grid', 'Gabor', 'Brightness']
    methods = ['Self-report', 'Logits-based']
    x = np.arange(len(tasks))
    width = 0.35
    
    # Panel A 左图: ECE
    for i, method in enumerate(methods):
        method_data = df[df['method'] == method]
        ece_values = [method_data[method_data['task'] == task]['ECE'].values[0] 
                     if len(method_data[method_data['task'] == task]) > 0 else np.nan
                     for task in tasks]
        
        bars = ax_ece.bar(x + (i - 0.5) * width, ece_values, width, 
                         label=method, color=COLORS[method], alpha=0.8, 
                         edgecolor='black', linewidth=0.5)
        
        # 添加数值标签
        for j, val in enumerate(ece_values):
            if not np.isnan(val):
                ax_ece.text(x[j] + (i - 0.5) * width, val + 0.02, f'{val:.3f}',
                           ha='center', va='bottom', fontsize=9)
    
    # 设置Y轴范围为0-1
    ax_ece.set_ylim([0.0, 1.0])
    
    ax_ece.set_xlabel('Task', fontsize=13, weight='bold')
    ax_ece.set_ylabel('ECE', fontsize=13, weight='bold')
    ax_ece.set_title('Expected Calibration Error(ECE)', fontsize=14, weight='bold', pad=10)
    ax_ece.set_xticks(x)
    ax_ece.set_xticklabels(tasks)
    ax_ece.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax_ece.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Panel A 右图: AUC
    for i, method in enumerate(methods):
        method_data = df[df['method'] == method]
        auc_values = [method_data[method_data['task'] == task]['AUC'].values[0] 
                     if len(method_data[method_data['task'] == task]) > 0 else np.nan
                     for task in tasks]
        
        bars = ax_auc.bar(x + (i - 0.5) * width, auc_values, width, 
                         label=method, color=COLORS[method], alpha=0.8, 
                         edgecolor='black', linewidth=0.5)
        
        # 添加数值标签
        for j, val in enumerate(auc_values):
            if not np.isnan(val):
                ax_auc.text(x[j] + (i - 0.5) * width, val + 0.01, f'{val:.3f}',
                           ha='center', va='bottom', fontsize=9)
    
    ax_auc.set_xlabel('Task', fontsize=13, weight='bold')
    ax_auc.set_ylabel('AUC', fontsize=13, weight='bold')
    ax_auc.set_title('Area under ROC Curve(AUC)', fontsize=14, weight='bold', pad=10)
    ax_auc.set_xticks(x)
    ax_auc.set_xticklabels(tasks)
    ax_auc.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax_auc.grid(axis='y', alpha=0.3, linestyle='--')
    ax_auc.set_ylim([0.0, 1.0])
    ax_auc.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)


def plot_panel_b_calibration_curves(axes, self_report, logits, slope_df):
    """绘制 Panel B: 三个任务的校准曲线（参考generate_calibration_figure4.py的风格）"""
    tasks = ['Grid', 'Gabor', 'Brightness']
    
    for idx, (ax, task) in enumerate(zip(axes, tasks)):
        # 获取该任务的数据
        sr_task = self_report[self_report['task'] == task]
        logits_task = logits[logits['task'] == task]
        
        # 处理Self-report（参考generate_calibration_figure4.py的方法）
        sr_grouped = sr_task.groupby('confidence')['correct'].agg(['count', 'mean', 'std']).reset_index()
        sr_grouped.columns = ['confidence', 'count', 'mean_acc', 'std_acc']
        sr_grouped['sem'] = sr_grouped['std_acc'] / np.sqrt(sr_grouped['count'])
        
        # 处理Logits-based
        logits_grouped = logits_task.groupby('confidence')['correct'].agg(['count', 'mean', 'std']).reset_index()
        logits_grouped.columns = ['confidence', 'count', 'mean_acc', 'std_acc']
        logits_grouped['sem'] = logits_grouped['std_acc'] / np.sqrt(logits_grouped['count'])
        
        # 合并到完整levels（参考generate_calibration_figure4.py）
        all_levels = sorted(set(sr_grouped['confidence'].unique()) | set(logits_grouped['confidence'].unique()))
        
        sr_full = pd.DataFrame({'confidence': all_levels})
        sr_full = sr_full.merge(sr_grouped, on='confidence', how='left').fillna({'mean_acc': 0, 'sem': 0, 'count': 0})
        
        logits_full = pd.DataFrame({'confidence': all_levels})
        logits_full = logits_full.merge(logits_grouped, on='confidence', how='left').fillna({'mean_acc': 0, 'sem': 0, 'count': 0})
        
        # 获取slope信息
        sr_slope_data = slope_df[(slope_df['Task'] == task) & (slope_df['Method'] == 'Self-report')]
        logits_slope_data = slope_df[(slope_df['Task'] == task) & (slope_df['Method'] == 'Logits-based')]
        
        # 绘制完美校准线（参考generate_calibration_figure4.py）
        ax.plot([1, 5], [0.2, 1.0], 'gray', linestyle='--', alpha=0.4, linewidth=2, 
                label='Perfect calibration', zorder=1)
        
        # 绘制Self-report（参考generate_calibration_figure4.py的样式）
        x_sr = sr_full['confidence'].values
        y_sr = sr_full['mean_acc'].values
        
        ax.scatter(x_sr, y_sr, s=60, alpha=0.8, color=COLORS['Self-report'],
                  edgecolors='white', linewidth=1.0, label='Self-report', zorder=3)
        ax.errorbar(x_sr, y_sr, yerr=sr_full['sem'].values, fmt='none',
                   color=COLORS['Self-report'], alpha=0.5, linewidth=1.2, capsize=3, zorder=2)
        
        # 添加Self-report回归线
        if len(x_sr) > 1 and len(sr_slope_data) > 0:
            slope_sr = sr_slope_data['slope'].values[0]
            intercept_sr = sr_slope_data['intercept'].values[0]
            x_line = np.array([1, 5])
            y_line = intercept_sr + slope_sr * x_line
            ax.plot(x_line, y_line, color=COLORS['Self-report'], linewidth=2.5, 
                   linestyle='--', alpha=0.8, zorder=2)
            # 添加slope标注 - 放在回归线的中点位置
            mid_x = 3.0
            mid_y = intercept_sr + slope_sr * mid_x
            # 确保标注在可见范围内，但不要重叠
            text_y = mid_y + 0.05 if mid_y < 0.5 else mid_y - 0.05
            text_y = max(0.1, min(0.9, text_y))
            ax.text(mid_x, text_y, f's={slope_sr:.3f}', 
                   fontsize=9, color=COLORS['Self-report'], weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                            edgecolor=COLORS['Self-report'], linewidth=1.0), zorder=4)
        
        # 绘制Logits-based（参考generate_calibration_figure4.py的样式）
        x_logits = logits_full['confidence'].values
        y_logits = logits_full['mean_acc'].values
        
        ax.scatter(x_logits, y_logits, s=60, alpha=0.8, color=COLORS['Logits-based'],
                  edgecolors='white', linewidth=1.0, marker='^', label='Logits-based', zorder=3)
        ax.errorbar(x_logits, y_logits, yerr=logits_full['sem'].values, fmt='none',
                   color=COLORS['Logits-based'], alpha=0.5, linewidth=1.2, capsize=3, zorder=2)
        
        # 添加Logits-based回归线
        if len(x_logits) > 1 and len(logits_slope_data) > 0:
            slope_logits = logits_slope_data['slope'].values[0]
            intercept_logits = logits_slope_data['intercept'].values[0]
            x_line = np.array([1, 5])
            y_line = intercept_logits + slope_logits * x_line
            ax.plot(x_line, y_line, color=COLORS['Logits-based'], linewidth=2.5,
                   linestyle='--', alpha=0.8, zorder=2)
            # 添加slope标注 - 放在回归线的中点位置，避免与Self-report重叠
            mid_x = 3.0
            mid_y = intercept_logits + slope_logits * mid_x
            # 确保标注在可见范围内，避免与Self-report重叠
            if len(sr_slope_data) > 0:
                sr_mid_y = intercept_sr + slope_sr * mid_x
                # 如果Logits的回归线在Self-report上方，标注放在下方；反之亦然
                if mid_y > sr_mid_y:
                    text_y = mid_y - 0.08
                else:
                    text_y = mid_y + 0.08
            else:
                text_y = mid_y + 0.05 if mid_y < 0.5 else mid_y - 0.05
            text_y = max(0.1, min(0.9, text_y))
            ax.text(mid_x, text_y, f's={slope_logits:.3f}', 
                   fontsize=9, color=COLORS['Logits-based'], weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                            edgecolor=COLORS['Logits-based'], linewidth=1.0), zorder=4)
        
        # 设置（参考generate_calibration_figure4.py）
        ax.set_xlabel('Confidence Level', fontsize=13, weight='bold')
        if idx == 0:
            ax.set_ylabel('Mean Accuracy', fontsize=13, weight='bold')
        ax.set_title(f'{task} Task', fontsize=15, weight='bold', color=TASK_COLORS[task], pad=12)
        ax.set_xlim(0.5, 5.5)
        ax.set_ylim(0, 1.05)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.grid(alpha=0.25, linestyle='--', linewidth=0.8, zorder=0)
        # 中间的子图（Gabor）将legend放在上方，其他放在下方
        legend_loc = 'upper right' if idx == 1 else 'lower right'
        ax.legend(fontsize=10, loc=legend_loc, frameon=True, fancybox=True, shadow=True, 
                 framealpha=0.95, borderpad=0.8)


def generate_figure4():
    """生成完整的 Figure 4: Calibration ECE AUC Curves"""
    
    print("=" * 80)
    print("生成 Figure 4: Calibration ECE AUC Curves")
    print("=" * 80)
    
    # 获取项目根目录
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 读取数据
    print("\n1. 读取数据...")
    
    # Panel A 数据: 校准指标汇总
    # 尝试多个可能的位置
    possible_paths = [
        os.path.join(base_dir, 'results', 'calibration', 'calibration_summary_method_task.csv'),
        os.path.join(base_dir, 'results', 'calibration_9_models', 'calibration_summary_method_task.csv'),
    ]
    
    summary_path = None
    for path in possible_paths:
        if os.path.exists(path):
            summary_path = path
            break
    
    if summary_path is None:
        print(f"❌ 校准指标汇总文件不存在")
        print("   请先运行: python scripts/analysis/calculate_calibration_metrics.py")
        return
    
    summary_df = pd.read_csv(summary_path)
    print(f"   ✅ 校准指标汇总: {summary_path} ({summary_df.shape})")
    
    # Panel B 数据: 原始数据
    # 尝试多个可能的数据路径
    possible_sr_paths = [
        os.path.join(base_dir, 'data', 'self_report', 'self_report_6models_merged.csv'),
        os.path.join(base_dir, 'data', 'raw', 'self_report', 'self_report_6models_merged.csv'),
    ]
    possible_logits_paths = [
        os.path.join(base_dir, 'data', 'logits', 'logits_6models_merged.csv'),
        os.path.join(base_dir, 'data', 'raw', 'logits', 'logits_6models_merged.csv'),
    ]
    
    self_report_path = None
    for path in possible_sr_paths:
        if os.path.exists(path):
            self_report_path = path
            break
    
    logits_path = None
    for path in possible_logits_paths:
        if os.path.exists(path):
            logits_path = path
            break
    
    if self_report_path is None:
        print(f"❌ Self-report数据文件不存在")
        print(f"   尝试过的路径: {possible_sr_paths}")
        return
    if logits_path is None:
        print(f"❌ Logits-based数据文件不存在")
        print(f"   尝试过的路径: {possible_logits_paths}")
        return
    
    self_report = pd.read_csv(self_report_path)
    logits = pd.read_csv(logits_path)
    print(f"   ✅ Self-report数据: {self_report_path} ({len(self_report)} 行)")
    print(f"   ✅ Logits-based数据: {logits_path} ({len(logits)} 行)")
    
    # Panel B 数据: 斜率数据
    # 尝试多个可能的位置
    possible_slope_paths = [
        os.path.join(base_dir, 'results', 'calibration', 'task_calibration_metrics.csv'),
        os.path.join(base_dir, 'results', 'calibration_9_models', 'task_calibration_metrics.csv'),
    ]
    
    slope_path = None
    for path in possible_slope_paths:
        if os.path.exists(path):
            slope_path = path
            break
    
    if slope_path and os.path.exists(slope_path):
        slope_df = pd.read_csv(slope_path)
        print(f"   ✅ 斜率数据: {slope_path}")
    else:
        print(f"⚠️  斜率数据文件不存在，将从数据中计算斜率...")
        # 计算斜率
        from scipy import stats
        slope_data = []
        tasks = ['Grid', 'Gabor', 'Brightness']
        methods = ['Self-report', 'Logits-based']
        
        for method in methods:
            method_data = self_report if method == 'Self-report' else logits
            for task in tasks:
                task_data = method_data[method_data['task'] == task]
                if len(task_data) > 0:
                    grouped = task_data.groupby('confidence')['correct'].agg(['mean', 'count']).reset_index()
                    grouped.columns = ['confidence', 'mean_acc', 'count']
                    if len(grouped) >= 2:
                        slope, intercept, r_value, _, _ = stats.linregress(
                            grouped['confidence'].values, grouped['mean_acc'].values
                        )
                        slope_data.append({
                            'Task': task,
                            'Method': method,
                            'slope': slope,
                            'intercept': intercept,
                            'r_value': r_value
                        })
        slope_df = pd.DataFrame(slope_data)
        print(f"   ✅ 已计算斜率数据")
    
    # 创建图形
    print("\n2. 创建图表...")
    # 使用gridspec创建2行布局：Panel A (1行2列) + Panel B (1行3列，高度降低)
    # 降低图片高度：从10改为8
    fig = plt.figure(figsize=(18, 8))
    # 调整高度比例，让下排图高度降低：从[1, 1.2]改为[1, 0.9]
    # 使用统一的列布局，让上下排的左右边对齐
    # 调整top参数，为A)标签留出空间
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.9],
                          hspace=0.35, left=0.08, right=0.98, top=0.96, bottom=0.1)
    
    # Panel A: ECE 和 AUC (上排，占据全部宽度，两图平分)
    # 使用GridSpecFromSubplotSpec让两图平分上排空间，左右边与下排对齐
    # 缩小两图之间的空白：wspace从0.25改为0.15
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0], 
                                               width_ratios=[1, 1], wspace=0.15)
    ax_ece = fig.add_subplot(gs_top[0, 0])
    ax_auc = fig.add_subplot(gs_top[0, 1])
    plot_panel_a_ece_auc(ax_ece, ax_auc, summary_df)
    
    # Panel B: 校准曲线 (下排，占据全部宽度，三列平分)
    # 下排使用全部宽度，三列平分，左右边与上排对齐
    # 缩小三图之间的空白：wspace从0.25改为0.15
    gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, 0], 
                                                  width_ratios=[1, 1, 1], wspace=0.15)
    ax_curves = [fig.add_subplot(gs_bottom[0, i]) for i in range(3)]
    plot_panel_b_calibration_curves(ax_curves, self_report, logits, slope_df)
    
    # 添加Panel标签：A) 在左上角，B) 在下排左上角
    # Panel A标签（上排左上角，顶到最上边）
    fig.text(0.05, 1.0, 'A)', fontsize=18, weight='bold', ha='left', va='top', transform=fig.transFigure)
    # Panel B标签（下排左上角）
    fig.text(0.05, 0.52, 'B)', fontsize=18, weight='bold', ha='left', va='top', transform=fig.transFigure)
    
    # 使用tight_layout调整布局，为顶部标签留出空间
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # 保存
    print("\n3. 保存图表...")
    output_dir = os.path.join(base_dir, 'results', 'figures', 'calibration')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file_pdf = os.path.join(output_dir, 'figure4_calibration_ece_auc_curves.pdf')
    output_file_png = os.path.join(output_dir, 'figure4_calibration_ece_auc_curves.png')
    
    plt.savefig(output_file_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(output_file_png, bbox_inches='tight', dpi=300)
    
    print(f"   ✅ PDF: {output_file_pdf}")
    print(f"   ✅ PNG: {output_file_png}")
    
    plt.close()
    
    print("\n" + "=" * 80)
    print("✅ Figure 4 生成完成！")
    print("=" * 80)


if __name__ == '__main__':
    generate_figure4()

