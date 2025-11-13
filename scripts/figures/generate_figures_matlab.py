#!/usr/bin/env python3
"""
使用MATLAB拟合结果生成图表
生成Figure3_matlab, Figure4_matlab, Figure5_matlab

基于MATLAB拟合结果（27个subjects self-report + 18个subjects logits）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import os

# 设置matplotlib参数
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 15
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['font.family'] = 'sans-serif'

def clip01(x):
    """Clip values to [1e-10, 1-1e-10]"""
    return np.clip(x, 1e-10, 1 - 1e-10)

def calculate_model_posterior_from_ic(ic_matrix):
    """
    从IC矩阵（AIC或BIC）计算模型后验概率
    
    参数:
        ic_matrix: shape (n_subjects, n_models) 的IC矩阵（AIC或BIC）
    
    返回:
        mean_posterior: shape (n_models,)，每个模型的平均后验概率
        subject_posteriors: shape (n_subjects, n_models)，每个被试的后验
    """
    # 计算delta IC (相对于每个被试的最佳模型)
    min_ic = np.min(ic_matrix, axis=1, keepdims=True)
    delta_ic = ic_matrix - min_ic
    
    # 转换为后验概率: p(m|y) ∝ exp(-ΔIC/2)
    log_posterior = -delta_ic / 2
    # 数值稳定性: 减去最大值
    log_posterior = log_posterior - np.max(log_posterior, axis=1, keepdims=True)
    posterior = np.exp(log_posterior)
    
    # 归一化
    posterior = posterior / np.sum(posterior, axis=1, keepdims=True)
    
    # 汇总：平均后验
    mean_posterior = np.mean(posterior, axis=0)
    
    return mean_posterior, posterior

def calculate_exceedance_probability(ic_matrix, n_samples=10000, alpha0=1.0, seed=0):
    """
    计算超越概率和模型概率（基于AIC/BIC）
    
    参数:
        ic_matrix: shape (n_subjects, n_models) 的IC矩阵
        n_samples: 采样次数
        alpha0: Dirichlet先验参数
        seed: 随机种子，确保可复现
    
    返回:
        p_exc: 超越概率
        p_model: 模型概率
        alpha: Dirichlet参数
    """
    np.random.seed(seed)
    n_subjects, n_models = ic_matrix.shape
    
    # 计算每个被试的模型后验（使用IC而非BIC）
    _, subject_posteriors = calculate_model_posterior_from_ic(ic_matrix)
    
    # 估计Dirichlet参数（使用VBA Toolbox方法）
    # 论文方法: α_m = α_0 + Σ_s q_{s,m}
    # 参考: Rigoux et al. (2014) Bayesian model selection for group studies
    alpha = alpha0 + np.sum(subject_posteriors, axis=0)
    
    # 计算p_model（模型频率，论文右列主图用的）
    # p_model = E[r] = α_m / Σ α_j
    p_model = alpha / np.sum(alpha)
    
    # 计算p_exc（超越概率，使用MCMC采样）
    # p_exc = P(r_m > r_j for all j ≠ m)
    p_exc = np.zeros(n_models)
    
    for _ in range(n_samples):
        # 从Dirichlet分布采样
        r = np.random.dirichlet(alpha)
        # 找到最大值的索引
        max_idx = np.argmax(r)
        p_exc[max_idx] += 1
    
    p_exc = p_exc / n_samples
    
    return p_exc, p_model, alpha

def prepare_aic_summary(fits_df, task):
    """
    准备AIC汇总数据（使用全局参照法，使最佳模型delta=0）
    
    方法：先求和所有VLM的AIC，然后相对于最小的sum_AIC计算delta
    这样最佳模型的delta永远是0，与参考文献一致
    
    返回:
        DataFrame with columns: model, sum_delta_AIC, mean_AIC, std_AIC, std_delta_AIC, n_vlms
    """
    task_data = fits_df[fits_df['task'] == task].copy()
    
    # 方法2（全局参照法）：先求和AIC，再计算delta
    # 1. 对每个模型，求和所有VLM的AIC
    sum_aic_by_model = task_data.groupby('model')['AIC'].sum()
    
    # 2. 找到最小的sum_AIC（全局最佳模型）
    min_sum_aic = sum_aic_by_model.min()
    
    # 3. 计算每个模型的sum_delta_AIC
    sum_delta_aic = sum_aic_by_model - min_sum_aic
    
    # 4. 计算其他统计量
    aic_stats = task_data.groupby('model')['AIC'].agg(['mean', 'std', 'count'])
    delta_aic_stats = task_data.groupby('model')['delta_AIC'].agg(['mean', 'std', 'count'])
    
    # 合并结果
    result = pd.DataFrame({
        'model': sum_delta_aic.index,
        'sum_delta_AIC': sum_delta_aic.values,
        'mean_AIC': aic_stats['mean'],
        'std_AIC': aic_stats['std'],
        'std_delta_AIC': delta_aic_stats['std'],
        'n_vlms': aic_stats['count']
    }).reset_index(drop=True)
    
    return result

def prepare_bic_summary(fits_df, task):
    """
    准备BIC汇总数据（使用全局参照法，使最佳模型delta=0）
    
    方法：先求和所有VLM的BIC，然后相对于最小的sum_BIC计算delta
    这样最佳模型的delta永远是0，与参考文献一致
    
    返回:
        DataFrame with columns: model, sum_delta_BIC, mean_BIC, std_BIC, std_delta_BIC, n_vlms
    """
    task_data = fits_df[fits_df['task'] == task].copy()
    
    # 方法2（全局参照法）：先求和BIC，再计算delta
    # 1. 对每个模型，求和所有VLM的BIC
    sum_bic_by_model = task_data.groupby('model')['BIC'].sum()
    
    # 2. 找到最小的sum_BIC（全局最佳模型）
    min_sum_bic = sum_bic_by_model.min()
    
    # 3. 计算每个模型的sum_delta_BIC
    sum_delta_bic = sum_bic_by_model - min_sum_bic
    
    # 4. 计算其他统计量
    bic_stats = task_data.groupby('model')['BIC'].agg(['mean', 'std', 'count'])
    delta_bic_stats = task_data.groupby('model')['delta_BIC'].agg(['mean', 'std', 'count'])
    
    # 合并结果
    result = pd.DataFrame({
        'model': sum_delta_bic.index,
        'sum_delta_BIC': sum_delta_bic.values,
        'mean_BIC': bic_stats['mean'],
        'std_BIC': bic_stats['std'],
        'std_delta_BIC': delta_bic_stats['std'],
        'n_vlms': bic_stats['count']
    }).reset_index(drop=True)
    
    return result

def calculate_pexc_for_task(fits_df, task, ic_type='AIC'):
    """
    为单个任务计算p_exc和p_model（基于AIC或BIC）
    
    参数:
        fits_df: 拟合结果DataFrame
        task: 任务名称
        ic_type: 'AIC' 或 'BIC'
    
    返回:
        DataFrame with columns: model, p_exc, p_model
    """
    task_data = fits_df[fits_df['task'] == task].copy()
    
    # 构建IC矩阵 (n_vlms, n_models)
    vlm_list = task_data['vlm'].unique()
    model_list = task_data['model'].unique()
    
    ic_matrix = np.zeros((len(vlm_list), len(model_list)))
    
    ic_column = ic_type  # 'AIC' 或 'BIC'
    
    for i, vlm in enumerate(vlm_list):
        for j, model in enumerate(model_list):
            vlm_model_data = task_data[(task_data['vlm'] == vlm) & (task_data['model'] == model)]
            if len(vlm_model_data) > 0:
                ic_matrix[i, j] = vlm_model_data[ic_column].iloc[0]
            else:
                ic_matrix[i, j] = np.inf  # 如果模型不存在，设为无穷大
    
    # 计算p_exc和p_model
    p_exc, p_model, alpha = calculate_exceedance_probability(ic_matrix, n_samples=10000, alpha0=1.0, seed=0)
    
    result = pd.DataFrame({
        'model': model_list,
        'p_exc': p_exc,
        'p_model': p_model
    })
    
    return result

def calculate_average_ranks(fits_df, data_type='self-report'):
    """
    计算每个任务的4个指标排名和平均排名（类似论文Figure 4）
    
    参数:
        fits_df: 拟合结果DataFrame
        data_type: 'self-report' 或 'logits'
    
    返回:
        DataFrame with columns: task, model, rank_AIC_fixed, rank_AIC_random, 
                                 rank_BIC_fixed, rank_BIC_random, avg_rank
    """
    print(f"\n计算 {data_type} 的平均排名...")
    
    tasks = ['Grid', 'Gabor', 'Brightness']
    all_rank_data = []
    
    for task in tasks:
        print(f"  处理任务: {task}")
        
        # 1. AIC fixed-effects: 基于sum_delta_AIC排名（越小越好）
        aic_summary = prepare_aic_summary(fits_df, task)
        aic_fixed_ranks = aic_summary.sort_values('sum_delta_AIC', ascending=True).reset_index(drop=True)
        aic_fixed_ranks['rank_AIC_fixed'] = range(1, len(aic_fixed_ranks) + 1)
        
        # 2. AIC random-effects: 基于p_model排名（越大越好）
        aic_pexc_result = calculate_pexc_for_task(fits_df, task, ic_type='AIC')
        aic_random_ranks = aic_pexc_result.sort_values('p_model', ascending=False).reset_index(drop=True)
        aic_random_ranks['rank_AIC_random'] = range(1, len(aic_random_ranks) + 1)
        
        # 3. BIC fixed-effects: 基于sum_delta_BIC排名（越小越好）
        bic_summary = prepare_bic_summary(fits_df, task)
        bic_fixed_ranks = bic_summary.sort_values('sum_delta_BIC', ascending=True).reset_index(drop=True)
        bic_fixed_ranks['rank_BIC_fixed'] = range(1, len(bic_fixed_ranks) + 1)
        
        # 4. BIC random-effects: 基于p_model排名（越大越好）
        bic_pexc_result = calculate_pexc_for_task(fits_df, task, ic_type='BIC')
        bic_random_ranks = bic_pexc_result.sort_values('p_model', ascending=False).reset_index(drop=True)
        bic_random_ranks['rank_BIC_random'] = range(1, len(bic_random_ranks) + 1)
        
        # 合并所有排名
        rank_data = aic_fixed_ranks[['model', 'rank_AIC_fixed']].merge(
            aic_random_ranks[['model', 'rank_AIC_random']], on='model', how='outer'
        ).merge(
            bic_fixed_ranks[['model', 'rank_BIC_fixed']], on='model', how='outer'
        ).merge(
            bic_random_ranks[['model', 'rank_BIC_random']], on='model', how='outer'
        )
        
        # 计算平均排名
        rank_data['avg_rank'] = (
            rank_data['rank_AIC_fixed'] + 
            rank_data['rank_AIC_random'] + 
            rank_data['rank_BIC_fixed'] + 
            rank_data['rank_BIC_random']
        ) / 4.0
        
        rank_data.insert(0, 'task', task)
        all_rank_data.append(rank_data)
    
    # 合并所有任务的数据
    full_rank_data = pd.concat(all_rank_data, ignore_index=True)
    
    return full_rank_data

def plot_average_rank_panel(ax, task, task_color, rank_data, data_type, show_xlabel=True):
    """
    绘制单个任务的平均排名面板
    
    参数:
        ax: Axes对象
        task: 任务名称
        task_color: 任务颜色
        rank_data: 包含排名数据的DataFrame
        data_type: 'Self-report' 或 'Logits'
        show_xlabel: 是否显示x轴标签
    """
    # 获取该任务的数据
    task_rank_data = rank_data[rank_data['task'] == task].copy()
    # 排序：先按avg_rank排序，如果avg_rank相同，SDT排在前面
    task_rank_data['sort_key'] = task_rank_data.apply(
        lambda row: (row['avg_rank'], 0 if row['model'] == 'SDT' else 1), 
        axis=1
    )
    task_rank_data = task_rank_data.sort_values('sort_key', ascending=True).drop('sort_key', axis=1)
    
    models = task_rank_data['model'].values
    avg_ranks = task_rank_data['avg_rank'].values
    n_models = len(models)
    
    # 设置颜色
    if data_type == 'Self-report':
        base_color = task_color
        edge_color = 'black'
    else:  # Logits
        base_color = task_color
        edge_color = 'gray'
    
    # 绘制bar图（rank越小越好，所以用反向y轴）
    y_pos = np.arange(n_models)[::-1]
    bars = ax.barh(y_pos, avg_ranks,
                  color=base_color, alpha=0.7, edgecolor=edge_color, 
                  linewidth=0.5)
    
    # Gradient color (dark to light)
    for j, bar in enumerate(bars):
        alpha = 0.9 - (j / n_models) * 0.6
        bar.set_alpha(alpha)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=11, weight='bold')
    
    # 设置x轴范围和刻度，确保显示到5（因为有5个模型）
    # 所有子图都显示x轴刻度1-5
    ax.set_xlim(0, 5.5)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

def create_average_rank_figure(rank_data, output_file, data_type='self-report'):
    """
    生成平均排名图（类似论文Figure 4）
    
    参数:
        rank_data: 包含排名数据的DataFrame
        output_file: 输出文件路径
        data_type: 'self-report' 或 'logits'
    """
    print("=" * 80)
    print(f"生成平均排名图 ({data_type}) - MATLAB版本")
    print("=" * 80)
    
    tasks = ['Grid', 'Gabor', 'Brightness']
    colors = ['#D62728', '#1F77B4', '#2CA02C']
    labels = ['a', 'b', 'c']
    
    # 创建图表
    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1],
                          hspace=0.3, wspace=0.3,
                          left=0.1, right=0.95, top=0.9, bottom=0.15)
    
    for i, (task, color, label) in enumerate(zip(tasks, colors, labels)):
        print(f"\n  处理任务: {task}")
        
        ax = fig.add_subplot(gs[0, i])
        plot_average_rank_panel(ax, task, color, rank_data, 
                               'Self-report' if data_type == 'self-report' else 'Logits',
                               show_xlabel=True)
        
        # 添加任务名称
        ax.text(0.5, 1.05, f'{label}  {task} Task', transform=ax.transAxes, 
                fontsize=18, weight='bold', va='bottom', ha='center')
        
        print(f"     ✅ {task} 任务完成")
    
    # 保存图表
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"\n   ✅ 图表已保存: {output_file}")
    plt.close()
    
    print("\n" + "=" * 80)
    print("✅ 平均排名图生成完成！")
    print("=" * 80)

def create_combined_average_rank_figure(selfreport_rank_data, logits_rank_data, output_dir):
    """
    生成合并的平均排名图（2行×3列，类似figure3_rank_combined）
    
    参数:
        selfreport_rank_data: Self-report排名数据
        logits_rank_data: Logits排名数据
        output_dir: 输出目录
    """
    print("=" * 80)
    print("生成合并的平均排名图 (2行×3列) - MATLAB版本")
    print("=" * 80)
    
    # 添加data_type列
    selfreport_rank_data['data_type'] = 'Self-report'
    logits_rank_data['data_type'] = 'Logits'
    
    combined_df = pd.concat([selfreport_rank_data, logits_rank_data], ignore_index=True)
    print(f"   数据形状: {combined_df.shape}")
    
    # 创建图表 - 2行×3列布局
    print("\n创建合并图表...")
    
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1],
                          hspace=0.4, wspace=0.15,
                          left=0.1, right=0.98, top=0.92, bottom=0.1)
    
    tasks = ['Grid', 'Gabor', 'Brightness']
    colors = ['#D62728', '#1F77B4', '#2CA02C']
    
    # 绘制6个子图
    row_labels = ['a', 'b']
    for row, data_type in enumerate(['Self-report', 'Logits']):
        print(f"\n   处理 {data_type} 数据...")
        data_subset = combined_df[combined_df['data_type'] == data_type].copy()
        
        for col, (task, color) in enumerate(zip(tasks, colors)):
            ax = fig.add_subplot(gs[row, col])
            
            # 绘制（所有子图都显示x轴刻度，不显示x轴标签）
            plot_average_rank_panel(ax, task, color, data_subset, data_type, show_xlabel=False)
            
            # 添加左边列的方法标签
            if col == 0:
                method_label = 'Self-report' if row == 0 else 'Logits-based'
                label = f'{row_labels[row]}) {method_label}'
                ax.text(-0.22, 1.02, label, transform=ax.transAxes, 
                        fontsize=13, weight='bold', va='bottom', ha='left')
            
            # 添加任务名称在子图上方（仅第一行）
            if row == 0:
                ax.text(0.5, 1.2, task, transform=ax.transAxes, 
                        fontsize=13, weight='bold', va='bottom', ha='center')
            
            print(f"       ✅ {task} 任务完成")
    
    # 保存图表
    output_file = output_dir / 'figure3_rank_combined.pdf'
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"\n   ✅ 图表已保存: {output_file}")
    plt.close()
    
    # PNG版本已禁用，只生成PDF
    # fig = plt.figure(figsize=(12, 5))
    # gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1],
    #                       hspace=0.4, wspace=0.15,
    #                       left=0.1, right=0.98, top=0.92, bottom=0.1)
    # 
    # row_labels = ['a', 'b']
    # for row, data_type in enumerate(['Self-report', 'Logits']):
    #     data_subset = combined_df[combined_df['data_type'] == data_type].copy()
    #     
    #     for col, (task, color) in enumerate(zip(tasks, colors)):
    #         ax = fig.add_subplot(gs[row, col])
    #         
    #         # 绘制（所有子图都显示x轴刻度，不显示x轴标签）
    #         plot_average_rank_panel(ax, task, color, data_subset, data_type, show_xlabel=False)
    #         
    #         if col == 0:
    #             method_label = 'Self-report' if row == 0 else 'Logits-based'
    #             label = f'{row_labels[row]}) {method_label}'
    #             ax.text(-0.22, 1.02, label, transform=ax.transAxes, 
    #                     fontsize=13, weight='bold', va='bottom', ha='left')
    #         
    #         if row == 0:
    #             ax.text(0.5, 1.2, task, transform=ax.transAxes, 
    #                     fontsize=13, weight='bold', va='bottom', ha='center')
    # 
    # output_file_png = output_dir / 'figure3_rank_combined.png'
    # plt.savefig(output_file_png, bbox_inches='tight', dpi=300)
    # print(f"   ✅ PNG图表已保存: {output_file_png}")
    # plt.close()
    
    print("\n" + "=" * 80)
    print("✅ 合并的平均排名图生成完成！")
    print("=" * 80)

def plot_task_rank_panel(ax, task, task_color, panel_label, rank_data, data_type, show_xlabel=True):
    """
    绘制单个任务的rank面板
    
    Parameters:
        ax: Axes对象
        task: 任务名称
        task_color: 任务颜色
        panel_label: 面板标签 (a, b, c)
        rank_data: 排名数据 (DataFrame with model, rank columns)
        data_type: 'Self-report' or 'Logits'
        show_xlabel: 是否显示x轴标签
    """
    # 按rank排序（rank越小越好）
    rank_sorted = rank_data.sort_values('rank', ascending=True).copy()
    
    models = rank_sorted['model'].values
    ranks = rank_sorted['rank'].values
    n_models = len(models)
    
    # Reverse y_pos so that best model (rank=1) is at top
    y_pos = np.arange(n_models)[::-1]
    
    # 根据data_type选择颜色风格
    if data_type == 'Self-report':
        base_color = task_color
        edge_color = 'black'
    else:  # Logits
        base_color = task_color
        edge_color = 'gray'
    
    # 绘制bar图
    bars = ax.barh(y_pos, ranks,
                   color=base_color, alpha=0.7, edgecolor=edge_color, 
                   linewidth=0.5)
    
    # Gradient color (dark to light)
    for i, bar in enumerate(bars):
        alpha = 0.9 - (i / n_models) * 0.6  # From 0.9 to 0.3
        bar.set_alpha(alpha)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=11, weight='bold')
    
    # 设置x轴刻度
    ax.set_xticks([1, 2, 3, 4, 5])
    
    ax.set_xlim(0, 5.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

def create_combined_rank_figure(figure3_data, figure4_data, output_dir):
    """生成合并的rank图（类似figure3_rank_combined）"""
    print("=" * 80)
    print("生成合并的 Rank 图 (2行×3列) - MATLAB版本")
    print("=" * 80)
    
    # 添加data_type列
    figure3_data['data_type'] = 'Self-report'
    figure4_data['data_type'] = 'Logits'
    
    combined_df = pd.concat([figure3_data, figure4_data], ignore_index=True)
    print(f"   数据形状: {combined_df.shape}")
    
    # 计算排名
    print("\n2. 计算排名...")
    for data_type in ['Self-report', 'Logits']:
        data_subset = combined_df[combined_df['data_type'] == data_type].copy()
        for task in ['Grid', 'Gabor', 'Brightness']:
            task_data = data_subset[data_subset['task'] == task].copy()
            task_data['rank'] = task_data['p_model'].rank(method='min', ascending=False).astype(int)
    
    # 创建图表 - 2行×3列布局
    print("\n3. 创建合并图表...")
    
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1],
                          hspace=0.4, wspace=0.15,
                          left=0.1, right=0.98, top=0.92, bottom=0.1)
    
    tasks = ['Grid', 'Gabor', 'Brightness']
    colors = ['#D62728', '#1F77B4', '#2CA02C']
    
    # 绘制6个子图
    row_labels = ['a', 'b']
    for row, data_type in enumerate(['Self-report', 'Logits']):
        print(f"\n   处理 {data_type} 数据...")
        data_subset = combined_df[combined_df['data_type'] == data_type].copy()
        
        for col, (task, color) in enumerate(zip(tasks, colors)):
            ax = fig.add_subplot(gs[row, col])
            
            # 获取该任务的数据
            task_data = data_subset[data_subset['task'] == task].copy()
            task_data['rank'] = task_data['p_model'].rank(method='min', ascending=False).astype(int)
            
            # 只在最底下行显示x轴标签
            show_xlabel = (row == 1) and (col == 0)  # 最左下角
            
            # 绘制
            plot_task_rank_panel(ax, task, color, '',  
                               task_data[['model', 'rank']], data_type, show_xlabel)
            
            # 添加左边列的方法标签
            if col == 0:
                method_label = 'Self-report' if row == 0 else 'Logits-based'
                label = f'{row_labels[row]}) {method_label}'
                ax.text(-0.22, 1.02, label, transform=ax.transAxes, 
                        fontsize=13, weight='bold', va='bottom', ha='left')
            
            # 添加任务名称在子图上方（仅第一行）
            if row == 0:
                ax.text(0.5, 1.2, task, transform=ax.transAxes, 
                        fontsize=13, weight='bold', va='bottom', ha='center')
            
            print(f"       ✅ {task} 任务完成")
    
    # 保存图表
    output_file = output_dir / 'figure3_rank_combined.pdf'
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"\n   ✅ 图表已保存: {output_file}")
    plt.close()
    
    # PNG版本已禁用，只生成PDF
    # fig = plt.figure(figsize=(12, 5))
    # gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1],
    #                       hspace=0.4, wspace=0.15,
    #                       left=0.1, right=0.98, top=0.92, bottom=0.1)
    # 
    # row_labels = ['a', 'b']
    # for row, data_type in enumerate(['Self-report', 'Logits']):
    #     data_subset = combined_df[combined_df['data_type'] == data_type].copy()
    #     
    #     for col, (task, color) in enumerate(zip(tasks, colors)):
    #         ax = fig.add_subplot(gs[row, col])
    #         
    #         task_data = data_subset[data_subset['task'] == task].copy()
    #         task_data['rank'] = task_data['p_model'].rank(method='min', ascending=False).astype(int)
    #         
    #         show_xlabel = (row == 1) and (col == 0)
    #         
    #         plot_task_rank_panel(ax, task, color, '', 
    #                            task_data[['model', 'rank']], data_type, show_xlabel)
    #         
    #         if col == 0:
    #             method_label = 'Self-report' if row == 0 else 'Logits-based'
    #             label = f'{row_labels[row]}) {method_label}'
    #             ax.text(-0.22, 1.02, label, transform=ax.transAxes, 
    #                     fontsize=13, weight='bold', va='bottom', ha='left')
    #         
    #         if row == 0:
    #             ax.text(0.5, 1.2, task, transform=ax.transAxes, 
    #                     fontsize=13, weight='bold', va='bottom', ha='center')
    # 
    # output_file_png = output_dir / 'figure3_rank_combined.png'
    # plt.savefig(output_file_png, bbox_inches='tight', dpi=300)
    # print(f"   ✅ PNG图表已保存: {output_file_png}")
    # plt.close()
    
    print("\n" + "=" * 80)
    print("✅ 合并的Rank图生成完成！")
    print("=" * 80)

def generate_figure_data(fits_df, data_type='self-report'):
    """
    生成figure数据文件（类似figure3_data.csv和figure4_data.csv）
    
    参数:
        fits_df: 拟合结果DataFrame
        data_type: 'self-report' 或 'logits'
    
    返回:
        DataFrame with columns: task, model, sum_delta_AIC, mean_AIC, std_AIC, p_exc, p_model, ...
    """
    print(f"\n生成 {data_type} 的figure数据...")
    
    tasks = ['Grid', 'Gabor', 'Brightness']
    all_task_data = []
    
    for task in tasks:
        print(f"  处理任务: {task}")
        
        # 准备AIC汇总
        aic_summary = prepare_aic_summary(fits_df, task)
        
        # 计算p_exc和p_model
        pexc_result = calculate_pexc_for_task(fits_df, task)
        
        # 合并数据
        task_data = aic_summary.merge(pexc_result, on='model', how='outer')
        task_data.insert(0, 'task', task)
        
        all_task_data.append(task_data)
    
    # 合并所有任务的数据
    full_data = pd.concat(all_task_data, ignore_index=True)
    
    return full_data

def plot_task_panel(ax_aic, ax_pmodel, task, task_color, panel_label, aic_summary, pexc_result):
    """
    绘制单个任务的面板（AIC + p_model）
    
    参数:
        ax_aic: AIC子图的Axes对象
        ax_pmodel: p_model子图的Axes对象
        task: 任务名称
        task_color: 任务颜色
        panel_label: 面板标签 (a, b, c)
        aic_summary: AIC汇总数据
        pexc_result: p_exc和p_model结果数据
    """
    # 合并数据
    combined = aic_summary.merge(pexc_result, on='model', how='outer')
    
    # =====================
    # Left plot: AIC analysis
    # =====================
    
    # Sort by sum_delta_AIC
    aic_sorted = combined.sort_values('sum_delta_AIC', ascending=True).copy()
    
    models = aic_sorted['model'].values
    sum_delta_aic = aic_sorted['sum_delta_AIC'].values
    n_models = len(models)
    
    # Plot main figure
    # Reverse y_pos so that best model (smallest AIC) is at top
    y_pos = np.arange(n_models)[::-1]
    
    # 计算标准误差
    std_delta_aic = aic_sorted['std_delta_AIC'].values
    n_vlms = aic_sorted['n_vlms'].values
    se_aic = std_delta_aic / np.sqrt(n_vlms)
    
    # 95%置信区间
    error_lower = 1.96 * se_aic
    error_upper = 1.96 * se_aic
    
    # 限制误差线长度
    max_error_ratio = 0.3
    error_lower = np.minimum(error_lower, sum_delta_aic * max_error_ratio)
    error_upper = np.minimum(error_upper, sum_delta_aic * max_error_ratio)
    
    # 最佳模型（delta=0）不应该有误差线
    error_lower = np.where(sum_delta_aic == 0, 0, error_lower)
    error_upper = np.where(sum_delta_aic == 0, 0, error_upper)
    
    bars_aic = ax_aic.barh(y_pos, sum_delta_aic, 
                          xerr=[error_lower, error_upper],
                          color=task_color, alpha=0.7, edgecolor='black', 
                          linewidth=0.5, error_kw={'linewidth': 1, 'ecolor': 'black'})
    
    # Gradient color (dark to light)
    for i, bar in enumerate(bars_aic):
        alpha = 0.9 - (i / n_models) * 0.6
        bar.set_alpha(alpha)
    
    ax_aic.set_yticks(y_pos)
    ax_aic.set_yticklabels(models, fontsize=14, weight='bold')
    ax_aic.set_xlabel('Sum (AIC$_{model}$ - AIC$_{best}$)', fontsize=16)
    ax_aic.set_xlim(0, max(sum_delta_aic) * 1.1)
    ax_aic.spines['top'].set_visible(False)
    ax_aic.spines['right'].set_visible(False)
    ax_aic.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add "Most/Least likely model" labels
    ax_aic.text(-0.18, 0.95, 'Most likely\nmodel', 
                transform=ax_aic.transAxes, ha='right', va='top', 
                fontsize=11, style='italic', color='black')
    ax_aic.text(-0.18, 0.05, 'Least likely\nmodel', 
                transform=ax_aic.transAxes, ha='right', va='bottom', 
                fontsize=11, style='italic', color='black')
    
    # Add double-headed arrow
    ax_aic.annotate('', xy=(-0.22, 0.80), xytext=(-0.22, 0.20),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    
    # Inset: Top 2 AIC details
    if n_models >= 2:
        inset_aic = ax_aic.inset_axes([0.58, 0.75, 0.28, 0.20])
        top2 = aic_sorted[aic_sorted['sum_delta_AIC'] > 0].head(2)
        top2_models = top2['model'].values
        top2_aic = top2['sum_delta_AIC'].values
        
        top2_std_delta_aic = top2['std_delta_AIC'].values
        top2_n_vlms = top2['n_vlms'].values
        top2_se_aic = top2_std_delta_aic / np.sqrt(top2_n_vlms)
        top2_error_lower = 1.96 * top2_se_aic
        top2_error_upper = 1.96 * top2_se_aic
        
        max_error_ratio = 0.3
        top2_error_lower = np.minimum(top2_error_lower, top2_aic * max_error_ratio)
        top2_error_upper = np.minimum(top2_error_upper, top2_aic * max_error_ratio)
        
        y_inset = np.arange(2)[::-1]
        inset_aic.barh(y_inset, top2_aic, xerr=[top2_error_lower, top2_error_upper], 
                      color=task_color, alpha=0.8, edgecolor='black', 
                      linewidth=0.5, error_kw={'linewidth': 1})
        inset_aic.set_yticks(y_inset)
        inset_aic.set_yticklabels(top2_models, fontsize=12)
        inset_aic.set_xlabel('Sum ΔAIC', fontsize=12)
        inset_aic.set_xlim(0, max(top2_aic) * 1.2)
        inset_aic.tick_params(labelsize=11)
        inset_aic.grid(axis='x', alpha=0.3)
        inset_aic.spines['top'].set_visible(False)
        inset_aic.spines['right'].set_visible(False)
        inset_aic.patch.set_facecolor('white')
        inset_aic.patch.set_alpha(0.9)
    
    # Add panel label with task name
    ax_aic.text(-0.35, 1.05, f'{panel_label}  {task} Task', transform=ax_aic.transAxes, 
                fontsize=18, weight='bold', va='top', ha='left')
    
    # =====================
    # Right plot: p_model analysis
    # =====================
    
    # Sort by p_model (descending)
    pmodel_sorted = combined.sort_values('p_model', ascending=False).copy()
    models_pmodel = pmodel_sorted['model'].values
    p_model_values = pmodel_sorted['p_model'].values
    n_models_pmodel = len(models_pmodel)
    
    # Reverse y_pos for p_model plot
    y_pos_pmodel = np.arange(n_models_pmodel)[::-1]
    
    # Set minimum visible width
    min_width = 0.01
    p_model_display = np.where(p_model_values < min_width, min_width, p_model_values)
    
    bars_pmodel = ax_pmodel.barh(y_pos_pmodel, p_model_display, 
                                 color=task_color, alpha=0.7, edgecolor='black', 
                                 linewidth=0.5)
    
    # Gradient color
    for i, bar in enumerate(bars_pmodel):
        alpha = 0.9 - (i / n_models_pmodel) * 0.6
        bar.set_alpha(alpha)
    
    ax_pmodel.set_yticks(y_pos_pmodel)
    ax_pmodel.set_yticklabels(models_pmodel, fontsize=14, weight='bold')
    ax_pmodel.set_xlabel('Model frequency ($p_{model}$)', fontsize=16)
    ax_pmodel.set_xlim(0, max(p_model_values) * 1.2)
    ax_pmodel.spines['top'].set_visible(False)
    ax_pmodel.spines['right'].set_visible(False)
    ax_pmodel.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Inset: p_exc details
    if 'p_exc' in pmodel_sorted.columns:
        # Create inset - positioned slightly lower but with distance from bottom
        # 与figure3保持一致的大小
        inset_pexc = ax_pmodel.inset_axes([0.65, 0.20, 0.42, 0.30])
        
        # Sort by p_exc, take top 5
        pexc_sorted = pmodel_sorted.sort_values('p_exc', ascending=False).head(5)
        pexc_models = pexc_sorted['model'].values
        pexc_values = pexc_sorted['p_exc'].values
        
        # Reverse y for inset too
        y_pexc = np.arange(len(pexc_models))[::-1]
        inset_pexc.barh(y_pexc, pexc_values, 
                       color=task_color, alpha=0.8, edgecolor='black', linewidth=0.5)
        inset_pexc.set_yticks(y_pexc)
        inset_pexc.set_yticklabels(pexc_models, fontsize=12)
        inset_pexc.set_xlabel('$p_{exc}$', fontsize=12)
        # 根据实际的最大p_exc值动态设置x轴范围
        max_pexc = pexc_values.max()
        inset_pexc.set_xlim(0, max(max_pexc * 1.1, 0.1))  # 至少显示到最大值的110%，最小0.1
        inset_pexc.tick_params(labelsize=11)
        inset_pexc.grid(axis='x', alpha=0.3)
        inset_pexc.spines['top'].set_visible(False)
        inset_pexc.spines['right'].set_visible(False)
        # Add white background to make it stand out
        inset_pexc.patch.set_facecolor('white')
        inset_pexc.patch.set_alpha(0.9)

def create_model_comparison_figure(figure_data, output_file, data_type='self-report'):
    """
    生成模型比较图（类似figure4_model_comparison或figure5_model_comparison_logits）
    
    参数:
        figure_data: figure数据DataFrame
        output_file: 输出文件路径
        data_type: 'self-report' 或 'logits'
    """
    print("=" * 80)
    print(f"生成模型比较图 ({data_type}) - MATLAB版本")
    print("=" * 80)
    
    tasks = ['Grid', 'Gabor', 'Brightness']
    colors = ['#D62728', '#1F77B4', '#2CA02C']
    labels = ['a', 'b', 'c']
    
    # 创建图表 - 3行×2列布局
    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1],
                          hspace=0.35, wspace=0.25,
                          left=0.12, right=0.95, top=0.95, bottom=0.05)
    
    for i, (task, color, label) in enumerate(zip(tasks, colors, labels)):
        print(f"  处理任务: {task}")
        
        # 获取任务数据
        task_data = figure_data[figure_data['task'] == task].copy()
        
        # 准备AIC汇总
        aic_summary = task_data[['model', 'sum_delta_AIC', 'mean_AIC', 'std_AIC', 'std_delta_AIC', 'n_vlms']].copy()
        
        # 准备p_exc和p_model
        pexc_result = task_data[['model', 'p_exc', 'p_model']].copy()
        
        # 创建子图
        ax_aic = fig.add_subplot(gs[i, 0])
        ax_pmodel = fig.add_subplot(gs[i, 1])
        
        # 绘制面板
        plot_task_panel(ax_aic, ax_pmodel, task, color, label, aic_summary, pexc_result)
        
        print(f"     ✅ {task} 任务完成")
    
    # 保存图表
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"\n   ✅ 图表已保存: {output_file}")
    plt.close()
    
    print("\n" + "=" * 80)
    print("✅ 模型比较图生成完成！")
    print("=" * 80)

def main():
    """主函数"""
    print("=" * 80)
    print("使用MATLAB拟合结果生成图表")
    print("=" * 80)
    
    # 项目根目录
    project_root = Path(__file__).parent.parent.parent
    
    # 读取MATLAB拟合结果
    print("\n1. 读取MATLAB拟合结果...")
    selfreport_fits_file = project_root / 'data' / 'processed' / 'model_fitting_matlab' / 'self_report_matlab_fits.csv'
    logits_fits_file = project_root / 'data' / 'processed' / 'model_fitting_matlab' / 'logits_matlab_fits.csv'
    
    selfreport_fits = pd.read_csv(selfreport_fits_file)
    logits_fits = pd.read_csv(logits_fits_file)
    
    # 只使用6个共同的开源模型，排除3个闭源模型（gpt-4o, gpt-5, claude-sonnet-4-5）
    closed_source_models = ['gpt-4o', 'gpt-5', 'claude-sonnet-4-5']
    print(f"\n过滤闭源模型: {closed_source_models}")
    print(f"过滤前 - Self-report: {len(selfreport_fits)} 行, {selfreport_fits['vlm'].nunique()} 个模型")
    selfreport_fits = selfreport_fits[~selfreport_fits['vlm'].isin(closed_source_models)].copy()
    print(f"过滤后 - Self-report: {len(selfreport_fits)} 行, {selfreport_fits['vlm'].nunique()} 个模型")
    print(f"Logits: {len(logits_fits)} 行, {logits_fits['vlm'].nunique()} 个模型")
    
    print(f"   Self-report数据: {len(selfreport_fits)} 行 ({selfreport_fits['vlm'].nunique()} 个模型)")
    print(f"   Logits数据: {len(logits_fits)} 行 ({logits_fits['vlm'].nunique()} 个模型)")
    
    # 生成figure数据
    print("\n2. 生成figure数据...")
    figure3_data = generate_figure_data(selfreport_fits, data_type='self-report')
    figure4_data = generate_figure_data(logits_fits, data_type='logits')
    
    # 保存figure数据文件
    figure_data_dir = project_root / 'data' / 'intermediate'
    figure_data_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存数据文件，每张图片一个CSV文件
    # figureA3_model_comparison 使用self-report数据（AIC汇总和p_model/p_exc）
    figureA3_data_file = figure_data_dir / 'figureA3_selfreport_comparison_data.csv'
    # figureA4_model_comparison_logits 使用logits数据（AIC汇总和p_model/p_exc）
    figureA4_data_file = figure_data_dir / 'figureA4_logits_comparison_data.csv'
    
    # 保存数据文件
    figure3_data.to_csv(figureA3_data_file, index=False)  # figureA3使用self-report数据
    figure4_data.to_csv(figureA4_data_file, index=False)  # figureA4使用logits数据
    
    print(f"\n   ✅ 已保存: {figureA3_data_file} (用于figureA3_model_comparison)")
    print(f"   ✅ 已保存: {figureA4_data_file} (用于figureA4_model_comparison_logits)")
    
    # 生成图表
    print("\n3. 生成图表...")
    output_dir = project_root / 'results' / 'figures' / 'Model_fitting'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成Figure3_rank_combined
    create_combined_rank_figure(figure3_data, figure4_data, output_dir)
    
    # 生成FigureA3_model_comparison (self-report)
    figureA3_output = output_dir / 'figureA3_model_comparison.pdf'
    create_model_comparison_figure(figure3_data, figureA3_output, data_type='self-report')
    
    # 生成FigureA4_model_comparison_logits (logits)
    # 注意：这里使用figure4_data，但保存为figureA4_data.csv
    figureA4_output = output_dir / 'figureA4_model_comparison_logits.pdf'
    create_model_comparison_figure(figure4_data, figureA4_output, data_type='logits')
    
    # 生成平均排名图（类似论文Figure 4）
    print("\n4. 生成平均排名图...")
    
    # Self-report数据的平均排名
    selfreport_rank_data = calculate_average_ranks(selfreport_fits, data_type='self-report')
    # 不再生成单独的self-report平均排名图
    # selfreport_rank_output = output_dir / 'figure4_average_rank_selfreport_matlab.pdf'
    # create_average_rank_figure(selfreport_rank_data, selfreport_rank_output, data_type='self-report')
    
    # Logits数据的平均排名
    logits_rank_data = calculate_average_ranks(logits_fits, data_type='logits')
    # 不再生成单独的logits平均排名图
    # logits_rank_output = output_dir / 'figure4_average_rank_logits_matlab.pdf'
    # create_average_rank_figure(logits_rank_data, logits_rank_output, data_type='logits')
    
    # 生成合并的平均排名图（2行×3列）
    create_combined_average_rank_figure(selfreport_rank_data, logits_rank_data, output_dir)
    
    # 保存figure3_rank_combined的数据（平均排名，四个统计）
    # 合并self-report和logits数据，添加data_type列
    selfreport_rank_data['data_type'] = 'Self-report'
    logits_rank_data['data_type'] = 'Logits'
    figure3_rank_data = pd.concat([selfreport_rank_data, logits_rank_data], ignore_index=True)
    
    figure3_data_file = figure_data_dir / 'figure3_rank_data.csv'
    figure3_rank_data.to_csv(figure3_data_file, index=False)
    
    print(f"\n   ✅ 已保存: {figure3_data_file} (用于figure3_rank_combined，包含平均排名数据)")
    
    print("\n" + "=" * 80)
    print("✅ 所有图表生成完成！")
    print("=" * 80)
    print(f"\n输出文件:")
    print(f"  - {figure3_data_file} (figure3_rank_combined的平均排名数据)")
    print(f"  - {figureA3_data_file} (figureA3_model_comparison的AIC汇总和p_model/p_exc数据)")
    print(f"  - {figureA4_data_file} (figureA4_model_comparison_logits的AIC汇总和p_model/p_exc数据)")
    print(f"  - {output_dir / 'figure3_rank_combined.pdf'}")
    print(f"  - {figureA3_output}")
    print(f"  - {figureA4_output}")

if __name__ == "__main__":
    main()

