#!/bin/bash
# VLM MetaCognition - 完整分析流程
# 一键运行所有分析：校准分析、图表生成和表格生成

echo "=========================================="
echo "VLM MetaCognition - 完整分析流程"
echo "生成所有图表、表格和分析结果"
echo "=========================================="
echo ""

# 检查是否在正确目录
if [ ! -f "scripts/figures/generate_figures_matlab.py" ]; then
    echo "❌ 错误: 请在 VLM_Meta_Cognition 目录下运行此脚本"
    echo "当前目录: $(pwd)"
    exit 1
fi

echo "✅ 当前目录: $(pwd)"
echo ""

# ============================================
# 步骤1: 生成校准分析结果
# ============================================
echo "📊 步骤1: 生成校准分析结果..."
echo ""

echo "  📈 计算基础校准指标（6个开源模型）..."
python3 scripts/analysis/calculate_calibration_metrics.py
echo ""

echo "  📈 计算完整校准指标（9个模型，包含闭源）..."
python3 scripts/analysis/calculate_calibration_metrics_full.py
echo ""

# ============================================
# 步骤2: 生成主文图表
# ============================================
echo "📊 步骤2: 生成主文图表..."
echo ""

echo "  📊 生成 Figure 3, A3, A4: 模型排名和比较..."
python3 scripts/figures/generate_figures_matlab.py
echo ""

echo "  📊 生成 Figure 4: 校准分析完整版..."
python3 scripts/figures/generate_figure4_calibration_ece_auc_curves.py
echo ""

# ============================================
# 步骤3: 生成附录图表
# ============================================
echo "📊 步骤3: 生成附录图表..."
echo ""

echo "  📊 生成 Figure A1: 9个模型的校准指标和曲线..."
python3 scripts/figures/generate_calibration_full_figures.py
echo ""

echo "  📊 生成 Figure A2: 9×4网格图..."
python3 scripts/figures/generate_calibration_all_models_grid.py
echo ""

# ============================================
# 步骤4: 生成表格
# ============================================
echo "📋 步骤4: 生成表格..."
echo ""

echo "  📋 生成 Table 1: Best-fitting process models per VLM-task combination..."
python3 scripts/tables/generate_table1.py
echo ""

echo "  📋 生成 Table 2: 模型拟合结果..."
python3 scripts/tables/generate_table2.py
echo ""

# ============================================
# 完成
# ============================================
echo "=========================================="
echo "✅ 所有图表和表格生成完成！"
echo "=========================================="
echo ""
echo "📁 结果保存在: results/ 目录下"
echo ""
echo "生成的图表:"
echo "  - results/figures/Model_fitting/figure3_rank_combined.pdf"
echo "  - results/figures/Model_fitting/figureA3_model_comparison.pdf"
echo "  - results/figures/Model_fitting/figureA4_model_comparison_logits.pdf"
echo "  - results/figures/calibration/figure4_calibration_ece_auc_curves.pdf"
echo "  - results/figures/calibration_9_models/figureA1_Calibration_metrics_and_curves_for_all_9_VLMs_including_closed-source_models.pdf"
echo "  - results/figures/calibration_9_models/figureA2_Calibration_Metrics_Across_All_Models_and_Tasks.pdf"
echo ""
echo "生成的表格:"
echo "  - results/tables/table1_*.csv"
echo "  - results/tables/table2_*.csv"
echo ""
