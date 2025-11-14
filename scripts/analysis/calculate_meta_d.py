#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算meta-d'指标（6 vs 6公平对比 + 单独闭源模型）

生成所有层级的meta-d'分析结果：
- Level 1: 整体分析
  * Self-report开源(6) vs Logits-based(6) - 公平对比
  * Self-report闭源(3) - 单独展示
- Level 2: 按任务分析
- Level 3: 按模型分析
- Level 4: 按任务×模型分析
"""

import pandas as pd
import numpy as np
import os
from meta_d_functions import calculate_meta_d_for_group, validate_data, get_data_summary

# 定义模型分类
CLOSED_SOURCE_MODELS = ['claude-sonnet-4-5', 'gpt-4o', 'gpt-5']
OPEN_SOURCE_MODELS = ['gemma3_27b', 'kimi-vl-a3b', 'ovis2_34b', 
                      'qwen2.5-vl-32b', 'qwen2.5-vl-72b', 'qwen2.5-vl-7b']


def main():
    """主函数"""
    print("=" * 80)
    print("Meta-d' 分析")
    print("=" * 80)
    
    # 获取项目根目录
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 1. 读取数据
    print("\n1. 读取数据...")
    gt_path = os.path.join(base_dir, 'data', 'raw', 'data_with_ground_truth.csv')
    sr_path = os.path.join(base_dir, 'data', 'raw', 'self_report', 'self_report_9models_merged.csv')
    logits_path = os.path.join(base_dir, 'data', 'raw', 'logits', 'logits_6models_merged.csv')
    
    gt_df = pd.read_csv(gt_path)
    sr_df = pd.read_csv(sr_path)
    logits_df = pd.read_csv(logits_path)
    
    print(f"   Ground truth: {len(gt_df)} 个唯一qid")
    print(f"   Self-report: {len(sr_df)} 行, {sr_df['model'].nunique()} 个模型")
    print(f"   Logits-based: {len(logits_df)} 行, {logits_df['model'].nunique()} 个模型")
    
    # 2. 合并数据
    print("\n2. 合并数据...")
    sr_merged = sr_df.merge(gt_df, on='qid', how='left')
    logits_merged = logits_df.merge(gt_df, on='qid', how='left')
    
    # 添加method列
    sr_merged['method'] = 'Self-report'
    logits_merged['method'] = 'Logits-based'
    
    print(f"   合并后 - Self-report: {len(sr_merged)} 行")
    print(f"   合并后 - Logits-based: {len(logits_merged)} 行")
    
    # 检查数据质量
    print("\n3. 检查数据质量...")
    sr_valid, sr_msg = validate_data(sr_merged)
    logits_valid, logits_msg = validate_data(logits_merged)
    print(f"   Self-report: {sr_msg}")
    print(f"   Logits-based: {logits_msg}")
    
    if not sr_valid or not logits_valid:
        print("   ⚠️  数据质量检查未通过，但继续执行...")
    
    # 定义任务和模型列表
    tasks = ['Grid', 'Gabor', 'Brightness']
    
    # 分离开源和闭源模型
    print("\n3.5. 分离开源和闭源模型...")
    sr_open = sr_merged[sr_merged['model'].isin(OPEN_SOURCE_MODELS)].copy()
    sr_closed = sr_merged[sr_merged['model'].isin(CLOSED_SOURCE_MODELS)].copy()
    
    print(f"   Self-report开源: {len(sr_open)} 行, {sr_open['model'].nunique()} 个模型")
    print(f"   Self-report闭源: {len(sr_closed)} 行, {sr_closed['model'].nunique()} 个模型")
    print(f"   Logits-based: {len(logits_merged)} 行, {logits_merged['model'].nunique()} 个模型")
    
    # 3. Level 1: 整体分析（6 vs 6公平对比 + 单独闭源）
    print("\n4. Level 1: 整体分析...")
    results_overall = []
    results_overall_open = []  # 6 vs 6公平对比
    results_overall_closed = []  # 闭源模型单独
    
    # Self-report开源（6个）- 用于6 vs 6对比
    print("   计算 Self-report开源(6) 整体...")
    result_sr_open = calculate_meta_d_for_group(sr_open, 'Self-report开源整体', padCells=1)
    if result_sr_open['success']:
        results_overall_open.append({
            'method': 'Self-report',
            'model_group': 'Open-source',
            'meta_da': result_sr_open['meta_da'],
            'da': result_sr_open['da'],
            'M_ratio': result_sr_open['M_ratio'],
            'M_diff': result_sr_open['M_diff'],
            'logL': result_sr_open['logL'],
            'n_trials': result_sr_open['n_trials']
        })
        print(f"      meta_da={result_sr_open['meta_da']:.4f}, da={result_sr_open['da']:.4f}, M_ratio={result_sr_open['M_ratio']:.4f}")
    
    # Logits-based（6个开源）- 用于6 vs 6对比
    print("   计算 Logits-based(6) 整体...")
    result_logits = calculate_meta_d_for_group(logits_merged, 'Logits-based整体', padCells=1)
    if result_logits['success']:
        results_overall_open.append({
            'method': 'Logits-based',
            'model_group': 'Open-source',
            'meta_da': result_logits['meta_da'],
            'da': result_logits['da'],
            'M_ratio': result_logits['M_ratio'],
            'M_diff': result_logits['M_diff'],
            'logL': result_logits['logL'],
            'n_trials': result_logits['n_trials']
        })
        print(f"      meta_da={result_logits['meta_da']:.4f}, da={result_logits['da']:.4f}, M_ratio={result_logits['M_ratio']:.4f}")
    
    # Self-report闭源（3个）- 单独展示
    print("   计算 Self-report闭源(3) 整体...")
    result_sr_closed = calculate_meta_d_for_group(sr_closed, 'Self-report闭源整体', padCells=1)
    if result_sr_closed['success']:
        results_overall_closed.append({
            'method': 'Self-report',
            'model_group': 'Closed-source',
            'meta_da': result_sr_closed['meta_da'],
            'da': result_sr_closed['da'],
            'M_ratio': result_sr_closed['M_ratio'],
            'M_diff': result_sr_closed['M_diff'],
            'logL': result_sr_closed['logL'],
            'n_trials': result_sr_closed['n_trials']
        })
        print(f"      meta_da={result_sr_closed['meta_da']:.4f}, da={result_sr_closed['da']:.4f}, M_ratio={result_sr_closed['M_ratio']:.4f}")
    
    # 保留原有的overall结果（向后兼容）
    results_overall = results_overall_open + results_overall_closed
    
    # 4. Level 2: 按任务分析（6 vs 6 + 闭源）
    print("\n5. Level 2: 按任务分析...")
    results_by_task = []
    results_by_task_open = []  # 6 vs 6公平对比
    results_by_task_closed = []  # 闭源模型单独
    
    # Self-report开源（6个）按任务
    print("   Self-report开源 按任务...")
    for task in tasks:
        task_data = sr_open[sr_open['task'] == task]
        group_name = f'Self-report开源-{task}'
        print(f"   计算 {group_name}...")
        result = calculate_meta_d_for_group(task_data, group_name, padCells=1)
        if result['success']:
            results_by_task_open.append({
                'method': 'Self-report',
                'model_group': 'Open-source',
                'task': task,
                'meta_da': result['meta_da'],
                'da': result['da'],
                'M_ratio': result['M_ratio'],
                'M_diff': result['M_diff'],
                'logL': result['logL'],
                'n_trials': result['n_trials']
            })
            print(f"      meta_da={result['meta_da']:.4f}, da={result['da']:.4f}, M_ratio={result['M_ratio']:.4f}")
    
    # Logits-based（6个）按任务
    print("   Logits-based 按任务...")
    for task in tasks:
        task_data = logits_merged[logits_merged['task'] == task]
        group_name = f'Logits-based-{task}'
        print(f"   计算 {group_name}...")
        result = calculate_meta_d_for_group(task_data, group_name, padCells=1)
        if result['success']:
            results_by_task_open.append({
                'method': 'Logits-based',
                'model_group': 'Open-source',
                'task': task,
                'meta_da': result['meta_da'],
                'da': result['da'],
                'M_ratio': result['M_ratio'],
                'M_diff': result['M_diff'],
                'logL': result['logL'],
                'n_trials': result['n_trials']
            })
            print(f"      meta_da={result['meta_da']:.4f}, da={result['da']:.4f}, M_ratio={result['M_ratio']:.4f}")
    
    # Self-report闭源（3个）按任务
    print("   Self-report闭源 按任务...")
    for task in tasks:
        task_data = sr_closed[sr_closed['task'] == task]
        group_name = f'Self-report闭源-{task}'
        print(f"   计算 {group_name}...")
        result = calculate_meta_d_for_group(task_data, group_name, padCells=1)
        if result['success']:
            results_by_task_closed.append({
                'method': 'Self-report',
                'model_group': 'Closed-source',
                'task': task,
                'meta_da': result['meta_da'],
                'da': result['da'],
                'M_ratio': result['M_ratio'],
                'M_diff': result['M_diff'],
                'logL': result['logL'],
                'n_trials': result['n_trials']
            })
            print(f"      meta_da={result['meta_da']:.4f}, da={result['da']:.4f}, M_ratio={result['M_ratio']:.4f}")
    
    # 合并结果（向后兼容）
    results_by_task = results_by_task_open + results_by_task_closed
    
    # 5. Level 3: 按模型分析
    print("\n6. Level 3: 按模型分析...")
    results_by_model_sr = []
    results_by_model_logits = []
    
    # Self-report: 9个模型
    print("   Self-report 模型...")
    for model in sorted(sr_merged['model'].unique()):
        model_data = sr_merged[sr_merged['model'] == model]
        result = calculate_meta_d_for_group(model_data, f'Self-report-{model}', padCells=1)
        if result['success']:
            results_by_model_sr.append({
                'model': model,
                'meta_da': result['meta_da'],
                'da': result['da'],
                'M_ratio': result['M_ratio'],
                'M_diff': result['M_diff'],
                'logL': result['logL'],
                'n_trials': result['n_trials']
            })
            print(f"      {model}: meta_da={result['meta_da']:.4f}, M_ratio={result['M_ratio']:.4f}")
    
    # Logits-based: 6个模型
    print("   Logits-based 模型...")
    for model in sorted(logits_merged['model'].unique()):
        model_data = logits_merged[logits_merged['model'] == model]
        result = calculate_meta_d_for_group(model_data, f'Logits-based-{model}', padCells=1)
        if result['success']:
            results_by_model_logits.append({
                'model': model,
                'meta_da': result['meta_da'],
                'da': result['da'],
                'M_ratio': result['M_ratio'],
                'M_diff': result['M_diff'],
                'logL': result['logL'],
                'n_trials': result['n_trials']
            })
            print(f"      {model}: meta_da={result['meta_da']:.4f}, M_ratio={result['M_ratio']:.4f}")
    
    # 6. Level 4: 按任务×模型分析
    print("\n7. Level 4: 按任务×模型分析...")
    results_detailed_sr = []
    results_detailed_sr_open = []  # 开源详细
    results_detailed_sr_closed = []  # 闭源详细
    results_detailed_logits = []
    
    # Self-report开源: 6模型 × 3任务
    print("   Self-report开源 详细分析...")
    for model in sorted(sr_open['model'].unique()):
        for task in tasks:
            model_task_data = sr_open[(sr_open['model'] == model) & 
                                      (sr_open['task'] == task)]
            group_name = f'Self-report开源-{model}-{task}'
            result = calculate_meta_d_for_group(model_task_data, group_name, padCells=1)
            if result['success']:
                results_detailed_sr_open.append({
                    'model': model,
                    'task': task,
                    'meta_da': result['meta_da'],
                    'da': result['da'],
                    'M_ratio': result['M_ratio'],
                    'M_diff': result['M_diff'],
                    'logL': result['logL'],
                    'n_trials': result['n_trials']
                })
    
    print(f"      完成 {len(results_detailed_sr_open)} 个组合")
    
    # Self-report闭源: 3模型 × 3任务
    print("   Self-report闭源 详细分析...")
    for model in sorted(sr_closed['model'].unique()):
        for task in tasks:
            model_task_data = sr_closed[(sr_closed['model'] == model) & 
                                       (sr_closed['task'] == task)]
            group_name = f'Self-report闭源-{model}-{task}'
            result = calculate_meta_d_for_group(model_task_data, group_name, padCells=1)
            if result['success']:
                results_detailed_sr_closed.append({
                    'model': model,
                    'task': task,
                    'meta_da': result['meta_da'],
                    'da': result['da'],
                    'M_ratio': result['M_ratio'],
                    'M_diff': result['M_diff'],
                    'logL': result['logL'],
                    'n_trials': result['n_trials']
                })
    
    print(f"      完成 {len(results_detailed_sr_closed)} 个组合")
    
    # 合并Self-report详细结果（向后兼容）
    results_detailed_sr = results_detailed_sr_open + results_detailed_sr_closed
    
    # Logits-based: 6模型 × 3任务
    print("   Logits-based 详细分析...")
    for model in sorted(logits_merged['model'].unique()):
        for task in tasks:
            model_task_data = logits_merged[(logits_merged['model'] == model) & 
                                           (logits_merged['task'] == task)]
            group_name = f'Logits-based-{model}-{task}'
            result = calculate_meta_d_for_group(model_task_data, group_name, padCells=1)
            if result['success']:
                results_detailed_logits.append({
                    'model': model,
                    'task': task,
                    'meta_da': result['meta_da'],
                    'da': result['da'],
                    'M_ratio': result['M_ratio'],
                    'M_diff': result['M_diff'],
                    'logL': result['logL'],
                    'n_trials': result['n_trials']
                })
    
    print(f"      完成 {len(results_detailed_logits)} 个组合")
    
    # 7. 生成汇总统计
    print("\n8. 生成汇总统计...")
    summary_stats = []
    
    # 按方法汇总
    for method_name in ['Self-report', 'Logits-based']:
        if method_name == 'Self-report':
            data_list = results_detailed_sr
        else:
            data_list = results_detailed_logits
        
        if len(data_list) > 0:
            meta_da_values = [r['meta_da'] for r in data_list if not np.isnan(r['meta_da'])]
            da_values = [r['da'] for r in data_list if not np.isnan(r['da'])]
            m_ratio_values = [r['M_ratio'] for r in data_list if not np.isnan(r['M_ratio'])]
            
            if len(meta_da_values) > 0:
                summary_stats.append({
                    'metric': 'meta_da',
                    'method': method_name,
                    'mean': np.mean(meta_da_values),
                    'std': np.std(meta_da_values),
                    'min': np.min(meta_da_values),
                    'max': np.max(meta_da_values),
                    'median': np.median(meta_da_values),
                    'n': len(meta_da_values)
                })
                
                summary_stats.append({
                    'metric': 'M_ratio',
                    'method': method_name,
                    'mean': np.mean(m_ratio_values),
                    'std': np.std(m_ratio_values),
                    'min': np.min(m_ratio_values),
                    'max': np.max(m_ratio_values),
                    'median': np.median(m_ratio_values),
                    'n': len(m_ratio_values)
                })
    
    # 8. 保存结果
    print("\n9. 保存结果...")
    output_dir = os.path.join(base_dir, 'results', 'meta_d')
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存各个层级的分析结果
    # 整体结果（向后兼容）
    if results_overall:
        pd.DataFrame(results_overall).to_csv(
            os.path.join(output_dir, 'meta_d_overall.csv'), index=False)
        print(f"   ✅ meta_d_overall.csv ({len(results_overall)} 行)")
    
    # 6 vs 6公平对比（开源模型）
    if results_overall_open:
        pd.DataFrame(results_overall_open).to_csv(
            os.path.join(output_dir, 'meta_d_overall_open_source.csv'), index=False)
        print(f"   ✅ meta_d_overall_open_source.csv ({len(results_overall_open)} 行) - 6 vs 6公平对比")
    
    # 闭源模型单独
    if results_overall_closed:
        pd.DataFrame(results_overall_closed).to_csv(
            os.path.join(output_dir, 'meta_d_overall_closed_source.csv'), index=False)
        print(f"   ✅ meta_d_overall_closed_source.csv ({len(results_overall_closed)} 行) - 闭源模型")
    
    # 按任务结果（向后兼容）
    if results_by_task:
        pd.DataFrame(results_by_task).to_csv(
            os.path.join(output_dir, 'meta_d_by_task.csv'), index=False)
        print(f"   ✅ meta_d_by_task.csv ({len(results_by_task)} 行)")
    
    # 按任务结果（6 vs 6公平对比）
    if results_by_task_open:
        pd.DataFrame(results_by_task_open).to_csv(
            os.path.join(output_dir, 'meta_d_by_task_open_source.csv'), index=False)
        print(f"   ✅ meta_d_by_task_open_source.csv ({len(results_by_task_open)} 行) - 6 vs 6公平对比")
    
    # 按任务结果（闭源模型）
    if results_by_task_closed:
        pd.DataFrame(results_by_task_closed).to_csv(
            os.path.join(output_dir, 'meta_d_by_task_closed_source.csv'), index=False)
        print(f"   ✅ meta_d_by_task_closed_source.csv ({len(results_by_task_closed)} 行) - 闭源模型")
    
    if results_by_model_sr:
        pd.DataFrame(results_by_model_sr).to_csv(
            os.path.join(output_dir, 'meta_d_by_model_selfreport.csv'), index=False)
        print(f"   ✅ meta_d_by_model_selfreport.csv ({len(results_by_model_sr)} 行)")
    
    if results_by_model_logits:
        pd.DataFrame(results_by_model_logits).to_csv(
            os.path.join(output_dir, 'meta_d_by_model_logits.csv'), index=False)
        print(f"   ✅ meta_d_by_model_logits.csv ({len(results_by_model_logits)} 行)")
    
    # 详细结果（向后兼容）
    if results_detailed_sr:
        pd.DataFrame(results_detailed_sr).to_csv(
            os.path.join(output_dir, 'meta_d_detailed_selfreport.csv'), index=False)
        print(f"   ✅ meta_d_detailed_selfreport.csv ({len(results_detailed_sr)} 行)")
    
    # 详细结果（开源模型）
    if results_detailed_sr_open:
        pd.DataFrame(results_detailed_sr_open).to_csv(
            os.path.join(output_dir, 'meta_d_detailed_selfreport_open.csv'), index=False)
        print(f"   ✅ meta_d_detailed_selfreport_open.csv ({len(results_detailed_sr_open)} 行) - 开源模型")
    
    # 详细结果（闭源模型）
    if results_detailed_sr_closed:
        pd.DataFrame(results_detailed_sr_closed).to_csv(
            os.path.join(output_dir, 'meta_d_detailed_selfreport_closed.csv'), index=False)
        print(f"   ✅ meta_d_detailed_selfreport_closed.csv ({len(results_detailed_sr_closed)} 行) - 闭源模型")
    
    if results_detailed_logits:
        pd.DataFrame(results_detailed_logits).to_csv(
            os.path.join(output_dir, 'meta_d_detailed_logits.csv'), index=False)
        print(f"   ✅ meta_d_detailed_logits.csv ({len(results_detailed_logits)} 行)")
    
    if summary_stats:
        pd.DataFrame(summary_stats).to_csv(
            os.path.join(output_dir, 'meta_d_summary_statistics.csv'), index=False)
        print(f"   ✅ meta_d_summary_statistics.csv ({len(summary_stats)} 行)")
    
    print("\n" + "=" * 80)
    print("✅ Meta-d' 分析完成！")
    print("=" * 80)
    print(f"\n结果保存在: {output_dir}")
    print(f"\n生成的文件:")
    print(f"  【6 vs 6公平对比（开源模型）】")
    print(f"  - meta_d_overall_open_source.csv: {len(results_overall_open) if results_overall_open else 0} 行")
    print(f"  - meta_d_by_task_open_source.csv: {len(results_by_task_open) if results_by_task_open else 0} 行")
    print(f"  - meta_d_detailed_selfreport_open.csv: {len(results_detailed_sr_open) if results_detailed_sr_open else 0} 行")
    print(f"  - meta_d_detailed_logits.csv: {len(results_detailed_logits) if results_detailed_logits else 0} 行")
    print(f"\n  【闭源模型单独】")
    print(f"  - meta_d_overall_closed_source.csv: {len(results_overall_closed) if results_overall_closed else 0} 行")
    print(f"  - meta_d_by_task_closed_source.csv: {len(results_by_task_closed) if results_by_task_closed else 0} 行")
    print(f"  - meta_d_detailed_selfreport_closed.csv: {len(results_detailed_sr_closed) if results_detailed_sr_closed else 0} 行")
    print(f"\n  【向后兼容文件】")
    print(f"  - meta_d_overall.csv: {len(results_overall) if results_overall else 0} 行")
    print(f"  - meta_d_by_task.csv: {len(results_by_task) if results_by_task else 0} 行")
    print(f"  - meta_d_detailed_selfreport.csv: {len(results_detailed_sr) if results_detailed_sr else 0} 行")
    print(f"  - meta_d_by_model_selfreport.csv: {len(results_by_model_sr) if results_by_model_sr else 0} 行")
    print(f"  - meta_d_by_model_logits.csv: {len(results_by_model_logits) if results_by_model_logits else 0} 行")
    print(f"  - meta_d_summary_statistics.csv: {len(summary_stats) if summary_stats else 0} 行")


if __name__ == '__main__':
    main()

