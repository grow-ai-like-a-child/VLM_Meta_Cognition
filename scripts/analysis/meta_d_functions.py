#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meta-d' 分析核心函数库

提供数据转换和meta-d'计算的统一接口
"""

import pandas as pd
import numpy as np
from trials2counts import trials2counts
from meta_d_MLE import fit_meta_d_MLE


def convert_data_format(df):
    """
    将我们的数据格式转换为trials2counts需要的格式
    
    参数:
        df: DataFrame，必须包含以下列：
            - ground_truth: 'A' 或 'B'
            - choice: 'A' 或 'B'
            - confidence: 1-5 (整数)
    
    返回:
        stimID: 列表，0=S1, 1=S2
        response: 列表，0=R1, 1=R2
        rating: 列表，1-5
    """
    # 数据验证
    required_cols = ['ground_truth', 'choice', 'confidence']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"数据中缺少必需的列: {col}")
    
    # 过滤有效数据
    valid_data = df.dropna(subset=required_cols).copy()
    
    # 转换ground_truth: 'A'→0 (S1), 'B'→1 (S2)
    stimID = [0 if gt == 'A' else 1 for gt in valid_data['ground_truth']]
    
    # 转换choice: 'A'→0 (R1), 'B'→1 (R2)
    response = [0 if ch == 'A' else 1 for ch in valid_data['choice']]
    
    # 转换confidence: 确保是整数
    rating = valid_data['confidence'].astype(int).tolist()
    
    return stimID, response, rating


def calculate_meta_d_for_group(df, group_name='', padCells=1, nRatings=5, s=1):
    """
    为指定的数据组计算meta-d'
    
    参数:
        df: DataFrame，包含ground_truth, choice, confidence列
        group_name: 组名（用于日志和错误信息）
        padCells: 是否使用padding（默认1，推荐，避免零值问题）
        nRatings: 置信度级别数（默认5）
        s: SDT方差比例（默认1，等方差模型）
    
    返回:
        dict: 包含以下字段：
            - meta_da: meta-d'值（RMS单位）
            - da: Type 1 d'值（RMS单位）
            - M_ratio: meta_da / da（元认知效率）
            - M_diff: meta_da - da
            - logL: 对数似然值
            - n_trials: trial数量
            - success: 是否成功计算
    
    如果计算失败，返回包含NaN的字典
    """
    result = {
        'meta_da': np.nan,
        'da': np.nan,
        'M_ratio': np.nan,
        'M_diff': np.nan,
        'logL': np.nan,
        'n_trials': len(df),
        'success': False
    }
    
    # 数据验证
    if len(df) < 50:
        print(f"   ⚠️  {group_name}: 数据不足（{len(df)}个trials），跳过")
        return result
    
    try:
        # 1. 数据转换
        stimID, response, rating = convert_data_format(df)
        
        if len(stimID) < 50:
            print(f"   ⚠️  {group_name}: 有效数据不足（{len(stimID)}个trials），跳过")
            return result
        
        # 2. 转换为响应计数矩阵
        nR_S1, nR_S2 = trials2counts(stimID, response, rating, nRatings, 
                                     padCells=padCells, padAmount=None)
        
        # 3. 计算meta-d'
        fit = fit_meta_d_MLE(nR_S1, nR_S2, s=s)
        
        # 4. 提取结果
        result['meta_da'] = fit['meta_da']
        result['da'] = fit['da']
        result['M_ratio'] = fit['M_ratio']
        result['M_diff'] = fit['M_diff']
        result['logL'] = fit['logL']
        result['success'] = True
        
        return result
        
    except Exception as e:
        print(f"   ❌ {group_name}: 计算失败 - {str(e)}")
        return result


def validate_data(df, min_trials=50):
    """
    验证数据质量
    
    参数:
        df: DataFrame
        min_trials: 最小trial数量
    
    返回:
        tuple: (is_valid, message)
    """
    if df is None or len(df) == 0:
        return False, "数据为空"
    
    required_cols = ['ground_truth', 'choice', 'confidence']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return False, f"缺少必需的列: {missing_cols}"
    
    if len(df) < min_trials:
        return False, f"数据量不足（{len(df)} < {min_trials}）"
    
    # 检查数据范围
    if df['confidence'].min() < 1 or df['confidence'].max() > 5:
        return False, f"置信度超出范围（应为1-5）"
    
    valid_gt = df['ground_truth'].isin(['A', 'B']).sum()
    if valid_gt < len(df) * 0.9:  # 至少90%的数据有效
        return False, f"ground_truth数据质量不佳（{valid_gt}/{len(df)}有效）"
    
    valid_choice = df['choice'].isin(['A', 'B']).sum()
    if valid_choice < len(df) * 0.9:
        return False, f"choice数据质量不佳（{valid_choice}/{len(df)}有效）"
    
    return True, "数据验证通过"


def get_data_summary(df):
    """
    获取数据摘要信息
    
    参数:
        df: DataFrame
    
    返回:
        dict: 数据摘要
    """
    summary = {
        'n_trials': len(df),
        'n_valid': df.dropna(subset=['ground_truth', 'choice', 'confidence']).shape[0],
        'confidence_dist': df['confidence'].value_counts().sort_index().to_dict() if 'confidence' in df.columns else {},
        'ground_truth_dist': df['ground_truth'].value_counts().to_dict() if 'ground_truth' in df.columns else {},
        'choice_dist': df['choice'].value_counts().to_dict() if 'choice' in df.columns else {},
    }
    return summary

