#!/usr/bin/env python3
"""
根据choice和correct推断ground truth，生成包含ground_truth列的新CSV文件
"""

import pandas as pd
import numpy as np
import os

def infer_ground_truth(df):
    """
    根据choice和correct推断ground truth
    
    逻辑：
    - 如果 (choice=='A' and correct==1) 或 (choice=='B' and correct==0)，则ground_truth='A'
    - 如果 (choice=='B' and correct==1) 或 (choice=='A' and correct==0)，则ground_truth='B'
    """
    df = df.copy()
    
    # 推断ground truth
    df['ground_truth'] = np.where(
        ((df['choice'] == 'A') & (df['correct'] == 1)) | 
        ((df['choice'] == 'B') & (df['correct'] == 0)),
        'A', 'B'
    )
    
    return df

def main():
    """主函数"""
    print("=" * 80)
    print("生成包含ground truth的数据文件")
    print("=" * 80)
    
    # 获取项目根目录
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 读取数据
    print("\n1. 读取数据...")
    self_report_path = os.path.join(base_dir, 'data', 'raw', 'self_report', 'self_report_9models_merged.csv')
    logits_path = os.path.join(base_dir, 'data', 'raw', 'logits', 'logits_6models_merged.csv')
    
    self_report = pd.read_csv(self_report_path)
    logits = pd.read_csv(logits_path)
    
    print(f"   Self-report: {len(self_report)} 行, {self_report['model'].nunique()} 个模型")
    print(f"   Logits-based: {len(logits)} 行, {logits['model'].nunique()} 个模型")
    
    # 添加method列
    self_report['method'] = 'Self-report'
    logits['method'] = 'Logits-based'
    
    # 推断ground truth
    print("\n2. 推断ground truth...")
    self_report = infer_ground_truth(self_report)
    logits = infer_ground_truth(logits)
    
    # 验证ground truth推断
    print("\n3. 验证ground truth推断...")
    print(f"   Self-report - Ground truth分布:")
    print(f"      A: {(self_report['ground_truth'] == 'A').sum()} ({100*(self_report['ground_truth'] == 'A').mean():.1f}%)")
    print(f"      B: {(self_report['ground_truth'] == 'B').sum()} ({100*(self_report['ground_truth'] == 'B').mean():.1f}%)")
    
    print(f"   Logits-based - Ground truth分布:")
    print(f"      A: {(logits['ground_truth'] == 'A').sum()} ({100*(logits['ground_truth'] == 'A').mean():.1f}%)")
    print(f"      B: {(logits['ground_truth'] == 'B').sum()} ({100*(logits['ground_truth'] == 'B').mean():.1f}%)")
    
    # 验证推断逻辑的一致性
    print("\n4. 验证推断逻辑...")
    # 检查：当ground_truth='A'时，如果选择A应该correct=1，如果选择B应该correct=0
    sr_check = (
        ((self_report['ground_truth'] == 'A') & (self_report['choice'] == 'A') & (self_report['correct'] == 1)).sum() +
        ((self_report['ground_truth'] == 'A') & (self_report['choice'] == 'B') & (self_report['correct'] == 0)).sum() +
        ((self_report['ground_truth'] == 'B') & (self_report['choice'] == 'B') & (self_report['correct'] == 1)).sum() +
        ((self_report['ground_truth'] == 'B') & (self_report['choice'] == 'A') & (self_report['correct'] == 0)).sum()
    )
    print(f"   Self-report推断一致性: {sr_check}/{len(self_report)} ({100*sr_check/len(self_report):.1f}%)")
    
    logits_check = (
        ((logits['ground_truth'] == 'A') & (logits['choice'] == 'A') & (logits['correct'] == 1)).sum() +
        ((logits['ground_truth'] == 'A') & (logits['choice'] == 'B') & (logits['correct'] == 0)).sum() +
        ((logits['ground_truth'] == 'B') & (logits['choice'] == 'B') & (logits['correct'] == 1)).sum() +
        ((logits['ground_truth'] == 'B') & (logits['choice'] == 'A') & (logits['correct'] == 0)).sum()
    )
    print(f"   Logits-based推断一致性: {logits_check}/{len(logits)} ({100*logits_check/len(logits):.1f}%)")
    
    # 合并数据
    print("\n5. 合并数据...")
    # 统一列顺序
    common_cols = ['trial_id', 'qid', 'task', 'model', 'method', 'choice', 'correct', 'confidence', 'ground_truth']
    
    # Self-report可能没有difficulty_level和pA/pB，需要处理
    if 'difficulty_level' in self_report.columns:
        common_cols.insert(4, 'difficulty_level')
    if 'pA' in self_report.columns:
        common_cols.insert(-3, 'pA')
        common_cols.insert(-3, 'pB')
    
    # 确保所有列都存在
    for col in common_cols:
        if col not in self_report.columns:
            self_report[col] = np.nan
        if col not in logits.columns:
            logits[col] = np.nan
    
    # 选择共同列
    self_report_clean = self_report[common_cols].copy()
    logits_clean = logits[common_cols].copy()
    
    # 合并
    merged_data = pd.concat([self_report_clean, logits_clean], ignore_index=True)
    
    print(f"   合并后: {len(merged_data)} 行")
    print(f"   列: {merged_data.columns.tolist()}")
    
    # 保存文件
    print("\n6. 保存文件...")
    output_path = os.path.join(base_dir, 'data', 'raw', 'data_with_ground_truth.csv')
    merged_data.to_csv(output_path, index=False)
    print(f"   ✅ 已保存到: {output_path}")
    
    # 显示统计信息
    print("\n7. 数据统计...")
    print(f"   总行数: {len(merged_data)}")
    print(f"   方法分布:")
    print(merged_data['method'].value_counts())
    print(f"   任务分布:")
    print(merged_data['task'].value_counts())
    print(f"   模型数量: {merged_data['model'].nunique()}")
    print(f"   Ground truth分布:")
    print(merged_data['ground_truth'].value_counts())
    
    print("\n" + "=" * 80)
    print("✅ Ground truth数据文件生成完成！")
    print("=" * 80)

if __name__ == '__main__':
    main()

