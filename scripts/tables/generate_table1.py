#!/usr/bin/env python3
"""
Table 1: Best-fitting process models per VLM-task combination
显示每个VLM-任务组合的最佳拟合模型（基于AIC）
"""

import pandas as pd
import numpy as np
import os

def generate_table1():
    """生成Table 1: 每个VLM-任务组合的最佳拟合模型"""
    print("=" * 80)
    print("生成 Table 1: Best-fitting process models per VLM-task combination")
    print("=" * 80)
    
    # 获取项目根目录（VLM_Meta_Cognition）
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 读取拟合数据
    print("\n1. 读取数据...")
    self_report_fits_file = os.path.join(base_dir, 'data', 'processed', 'model_fitting_matlab', 'self_report_matlab_fits.csv')
    logits_fits_file = os.path.join(base_dir, 'data', 'processed', 'model_fitting_matlab', 'logits_matlab_fits.csv')
    
    if not os.path.exists(self_report_fits_file):
        print(f"❌ Self-report拟合文件不存在: {self_report_fits_file}")
        return None
    
    if not os.path.exists(logits_fits_file):
        print(f"❌ Logits拟合文件不存在: {logits_fits_file}")
        return None
    
    self_report_fits = pd.read_csv(self_report_fits_file)
    logits_fits = pd.read_csv(logits_fits_file)
    
    print(f"   ✅ Self-report数据: {len(self_report_fits)} 行")
    print(f"   ✅ Logits数据: {len(logits_fits)} 行")
    
    # 只使用6个开源VLM（排除闭源模型）
    closed_source_models = ['gpt-4o', 'gpt-5', 'claude-sonnet-4-5']
    open_source_models = ['qwen2.5-vl-7b', 'qwen2.5-vl-72b', 'ovis2_34b', 'kimi-vl-a3b', 'gemma3_27b', 'qwen2.5-vl-32b']
    
    self_report_fits = self_report_fits[~self_report_fits['vlm'].isin(closed_source_models)].copy()
    logits_fits = logits_fits[~logits_fits['vlm'].isin(closed_source_models)].copy()
    
    print(f"   过滤后 - Self-report: {len(self_report_fits)} 行 ({self_report_fits['vlm'].nunique()} 个模型)")
    print(f"   过滤后 - Logits: {len(logits_fits)} 行 ({logits_fits['vlm'].nunique()} 个模型)")
    
    # 计算每个VLM-task组合的最佳模型（基于AIC）
    def get_best_model(fits_df, vlm, task):
        """获取指定VLM-task组合的最佳模型（AIC最小）"""
        vlm_task_data = fits_df[(fits_df['vlm'] == vlm) & (fits_df['task'] == task)]
        if len(vlm_task_data) == 0:
            return None
        best_idx = vlm_task_data['AIC'].idxmin()
        return vlm_task_data.loc[best_idx, 'model']
    
    # 生成表格数据
    print("\n2. 生成表格数据...")
    tasks = ['Grid', 'Gabor', 'Brightness']
    methods = ['Self-report', 'Logits-based']
    
    table_data = []
    
    # 按论文中的顺序排列VLM
    vlm_order = ['qwen2.5-vl-7b', 'qwen2.5-vl-72b', 'ovis2_34b', 'kimi-vl-a3b', 'gemma3_27b', 'qwen2.5-vl-32b']
    
    for vlm in vlm_order:
        row = {'vlm': vlm}
        
        # Self-report的3个任务
        for task in tasks:
            best_model = get_best_model(self_report_fits, vlm, task)
            row[f'self_report_{task}'] = best_model if best_model else 'N/A'
        
        # Logits-based的3个任务
        for task in tasks:
            best_model = get_best_model(logits_fits, vlm, task)
            row[f'logits_{task}'] = best_model if best_model else 'N/A'
        
        table_data.append(row)
        print(f"   {vlm}: SR({row['self_report_Grid']}, {row['self_report_Gabor']}, {row['self_report_Brightness']}) | "
              f"Logits({row['logits_Grid']}, {row['logits_Gabor']}, {row['logits_Brightness']})")
    
    table_df = pd.DataFrame(table_data)
    
    # 计算主导模型统计
    print("\n3. 计算主导模型统计...")
    
    # Self-report统计
    sr_stats = {}
    for task in tasks:
        sr_stats[task] = {}
        for vlm in vlm_order:
            model = get_best_model(self_report_fits, vlm, task)
            if model:
                sr_stats[task][model] = sr_stats[task].get(model, 0) + 1
    
    # Logits-based统计
    logits_stats = {}
    for task in tasks:
        logits_stats[task] = {}
        for vlm in vlm_order:
            model = get_best_model(logits_fits, vlm, task)
            if model:
                logits_stats[task][model] = logits_stats[task].get(model, 0) + 1
    
    # 创建统计行
    all_models = ['PE', 'BCH', 'SDT', 'LogN', 'WEV']
    stats_rows = []
    
    for model in all_models:
        stats_row = {'vlm': model}
        # Self-report统计
        for task in tasks:
            stats_row[f'self_report_{task}'] = sr_stats[task].get(model, 0)
        # Logits-based统计
        for task in tasks:
            stats_row[f'logits_{task}'] = logits_stats[task].get(model, 0)
        stats_rows.append(stats_row)
    
    stats_df = pd.DataFrame(stats_rows)
    
    # 合并表格数据和统计
    final_table = pd.concat([table_df, stats_df], ignore_index=True)
    
    # 保存结果
    print("\n4. 保存结果...")
    output_dir = os.path.join(base_dir, 'results', 'tables')
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存完整表格
    output_file = os.path.join(output_dir, 'table1_model_fits_per_vlm_task.csv')
    final_table.to_csv(output_file, index=False)
    print(f"   ✅ {output_file}")
    
    # 保存仅VLM数据（不含统计行）
    vlm_only_file = os.path.join(output_dir, 'table1_vlm_task_models.csv')
    table_df.to_csv(vlm_only_file, index=False)
    print(f"   ✅ {vlm_only_file}")
    
    # 保存统计数据
    stats_file = os.path.join(output_dir, 'table1_dominant_model_counts.csv')
    stats_df.to_csv(stats_file, index=False)
    print(f"   ✅ {stats_file}")
    
    # 打印统计摘要
    print("\n主导模型统计:")
    print("\nSelf-report:")
    for task in tasks:
        print(f"  {task}: {dict(sr_stats[task])}")
    
    print("\nLogits-based:")
    for task in tasks:
        print(f"  {task}: {dict(logits_stats[task])}")
    
    print("\n" + "=" * 80)
    print("✅ Table 1 生成完成！")
    print("=" * 80)
    
    return table_df, stats_df

if __name__ == "__main__":
    generate_table1()
