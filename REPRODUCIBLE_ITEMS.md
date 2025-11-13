# 可复现内容清单

本文档详细列出 `VLM_Meta_Cognition` 项目可以复现的所有分析结果、图表和表格。

---

## 📊 主文图表（Main Figures）

### ✅ Figure 3: Model Rankings
- **文件名**: `figure3_rank_combined.pdf`
- **位置**: `results/figures/Model_fitting/figure3_rank_combined.pdf`
- **生成脚本**: `scripts/figures/generate_figures_matlab.py`
- **所需数据**:
  - `data/processed/model_fitting_matlab/vlm_matlab_fits.csv` (Self-report拟合结果)
  - `data/processed/model_fitting_matlab/logits_matlab_fits.csv` (Logits拟合结果)
- **内容**: 2行×3列，显示6个开源模型在3个任务上的排名（Self-report和Logits-based）
- **运行命令**:
  ```bash
  python scripts/figures/generate_figures_matlab.py
  ```

### ✅ Figure 4: Calibration Analysis (完整版)
- **文件名**: `figure4_calibration_ece_auc_curves.pdf`
- **位置**: `results/figures/calibration/figure4_calibration_ece_auc_curves.pdf`
- **生成脚本**: `scripts/figures/generate_figure4_calibration_ece_auc_curves.py`
- **所需数据**:
  - `results/calibration/calibration_summary_method_task.csv` (需先运行分析脚本)
  - `data/raw/self_report/vlm_merged_models.csv`
  - `data/raw/logits/vlm_merged_models_logits_only.csv`
  - `results/calibration/task_calibration_metrics.csv` (斜率数据)
- **内容**: 
  - Panel A: ECE和AUC柱状图（2个图）
  - Panel B: 三个任务的校准曲线（3个图）
- **运行命令**:
  ```bash
  # 1. 先运行校准分析
  python scripts/analysis/calculate_calibration_metrics.py
  # 2. 生成图表
  python scripts/figures/generate_figure4_calibration_ece_auc_curves.py
  ```

### ✅ Figure 4: Calibration Curves (仅Panel B)
- **文件名**: `figure4_task_calibration_curves.pdf` (如果单独生成)
- **生成脚本**: `scripts/figures/generate_calibration_figure4.py`
- **内容**: 三个任务的校准曲线（Self-report和Logits-based对比）

---

## 📊 附录图表（Appendix Figures）

### ✅ Figure A1: Calibration Metrics and Curves for All 9 VLMs
- **文件名**: `figureA1_Calibration_metrics_and_curves_for_all_9_VLMs_including_closed-source_models.pdf`
- **位置**: `results/figures/calibration_9_models/figureA1_Calibration_metrics_and_curves_for_all_9_VLMs_including_closed-source_models.pdf`
- **生成脚本**: `scripts/figures/generate_calibration_full_figures.py`
- **所需数据**:
  - `results/calibration_9_models/calibration_by_task_full.csv` (需先运行分析脚本)
  - `data/raw/self_report/vlm_all_models_merged.csv` (9个模型，包含3个闭源)
- **内容**: 
  - Panel A: ECE (Absolute) 柱状图
  - Panel B: ROC AUC 柱状图
  - Panel C: 汇总统计表
  - Panel D-F: 三个任务的校准曲线（9个模型合并）
- **运行命令**:
  ```bash
  # 1. 先运行完整校准分析
  python scripts/analysis/calculate_calibration_metrics_full.py
  # 2. 生成图表
  python scripts/figures/generate_calibration_full_figures.py
  ```

### ✅ Figure A2: Calibration Metrics Across All Models and Tasks
- **文件名**: `figureA2_Calibration_Metrics_Across_All_Models_and_Tasks.pdf`
- **位置**: `results/figures/calibration_9_models/figureA2_Calibration_Metrics_Across_All_Models_and_Tasks.pdf`
- **生成脚本**: `scripts/figures/generate_calibration_all_models_grid.py`
- **所需数据**:
  - `results/calibration_9_models/calibration_task_model_full.csv` (需先运行分析脚本)
- **内容**: 9行×4列网格图
  - 每行：一个模型（共9个模型）
  - 每列：一个指标（ECE Absolute, ECE Squared, ROC AUC, Selective AUC）
  - 每个子图：该模型在该指标上三个任务的表现
  - gpt-5所在行有红色高亮框
- **运行命令**:
  ```bash
  # 1. 先运行完整校准分析
  python scripts/analysis/calculate_calibration_metrics_full.py
  # 2. 生成图表
  python scripts/figures/generate_calibration_all_models_grid.py
  ```

### ✅ Figure A3: Self-Report Model Comparison
- **文件名**: `figureA3_model_comparison.pdf`
- **位置**: `results/figures/Model_fitting/figureA3_model_comparison.pdf`
- **生成脚本**: `scripts/figures/generate_figures_matlab.py` (与Figure 3相同脚本)
- **所需数据**:
  - `data/processed/model_fitting_matlab/vlm_matlab_fits.csv` (Self-report拟合结果)
- **内容**: 2行×3列，显示6个开源模型在3个任务上的AIC和p_model比较（Self-report）
- **运行命令**:
  ```bash
  python scripts/figures/generate_figures_matlab.py
  ```

### ✅ Figure A4: Logits Model Comparison
- **文件名**: `figureA4_model_comparison_logits.pdf`
- **位置**: `results/figures/Model_fitting/figureA4_model_comparison_logits.pdf`
- **生成脚本**: `scripts/figures/generate_figures_matlab.py` (与Figure 3相同脚本)
- **所需数据**:
  - `data/processed/model_fitting_matlab/logits_matlab_fits.csv` (Logits拟合结果)
- **内容**: 2行×3列，显示6个开源模型在3个任务上的AIC和p_model比较（Logits-based）
- **运行命令**:
  ```bash
  python scripts/figures/generate_figures_matlab.py
  ```

---

## 📋 表格（Tables）

### ✅ Table 1: Best-fitting process models per VLM-task combination
- **文件名**: `table1_vlm_task_models.csv`
- **位置**: `results/tables/table1_vlm_task_models.csv`
- **生成脚本**: `scripts/tables/generate_table1.py`
- **所需数据**:
  - `data/processed/model_fitting_matlab/vlm_matlab_fits.csv` (Self-report拟合结果)
  - `data/processed/model_fitting_matlab/logits_matlab_fits.csv` (Logits拟合结果)
- **内容**: 
  - 6个开源VLM在3个任务上的最佳拟合模型（Self-report和Logits-based分别）
  - 主导模型统计计数（每个模型在不同任务中出现的次数）
- **运行命令**:
  ```bash
  python scripts/tables/generate_table1.py
  ```
- **输出文件**:
  - `table1_vlm_task_models.csv` - VLM-任务组合的最佳模型
  - `table1_dominant_model_counts.csv` - 主导模型统计
  - `table1_model_fits_per_vlm_task.csv` - 完整表格（包含统计行）

### ✅ Table 2: Model Fits
- **文件名**: `table2_model_fits.csv`
- **位置**: `results/tables/table2_model_fits.csv`
- **生成脚本**: `scripts/tables/generate_table2.py`
- **所需数据**:
  - `data/processed/model_fitting_matlab/vlm_matlab_fits.csv`
  - `data/processed/model_fitting_matlab/logits_matlab_fits.csv`
- **内容**: 模型拟合统计结果（AIC, BIC, 主导模型等）
- **运行命令**:
  ```bash
  python scripts/tables/generate_table2.py
  ```

### ✅ Table 2: Dominance Statistics
- **文件名**: `table2_dominance_statistics.csv`
- **位置**: `results/tables/table2_dominance_statistics.csv`
- **生成脚本**: `scripts/tables/generate_table2.py`
- **内容**: 主导模型统计

### ✅ Table 2: Detailed Dominant Models
- **文件名**: `table2_detailed_dominant_models.csv`
- **位置**: `results/tables/table2_detailed_dominant_models.csv`
- **生成脚本**: `scripts/tables/generate_table2.py`
- **内容**: 详细的主导模型数据

---

## 📊 校准分析结果（Calibration Analysis Results）

### ✅ 基础校准指标（6个开源模型）
- **位置**: `results/calibration/`
- **生成脚本**: `scripts/analysis/calculate_calibration_metrics.py`
- **输出文件**:
  - `calibration_summary_method_task.csv` - 按方法和任务汇总
  - `task_calibration_metrics.csv` - 任务校准指标（包含slope）

### ✅ 完整校准指标（9个模型，包含3个闭源）
- **位置**: `results/calibration_9_models/`
- **生成脚本**: `scripts/analysis/calculate_calibration_metrics_full.py`
- **输出文件**:
  - `calibration_by_task_full.csv` - 按任务汇总（3个任务）
  - `calibration_by_model_full.csv` - 按模型汇总（9个模型）
  - `calibration_task_model_full.csv` - 按任务×模型详细数据（27行）
  - `calibration_overall_full.csv` - 整体汇总

---

## 📊 中间数据文件（Intermediate Data）

### ✅ Figure 3 数据
- **文件名**: `figure3_rank_combined_data.csv`
- **位置**: `data/figure_data/figure3_rank_combined_data.csv`
- **内容**: 平均排名数据（自动生成）

### ✅ Figure A3 数据
- **文件名**: `figureA3_model_comparison_data.csv`
- **位置**: `data/figure_data/figureA3_model_comparison_data.csv`
- **内容**: AIC汇总和p_model/p_exc数据（自动生成）

### ✅ Figure A4 数据
- **文件名**: `figureA4_model_comparison_logits_data.csv`
- **位置**: `data/figure_data/figureA4_model_comparison_logits_data.csv`
- **内容**: AIC汇总和p_model/p_exc数据（自动生成）

---

## 📝 总结

### 可复现的图表（6个）
1. ✅ Figure 3: Model Rankings
2. ✅ Figure 4: Calibration Analysis (完整版)
3. ✅ Figure A1: Calibration Metrics and Curves for All 9 VLMs
4. ✅ Figure A2: Calibration Metrics Across All Models and Tasks
5. ✅ Figure A3: Self-Report Model Comparison
6. ✅ Figure A4: Logits Model Comparison

### 可复现的表格（5个）
1. ✅ Table 1: Best-fitting process models per VLM-task combination
2. ✅ Table 1: Dominant Model Counts
3. ✅ Table 2: Model Fits
4. ✅ Table 2: Dominance Statistics
5. ✅ Table 2: Detailed Dominant Models

### 可复现的分析结果（2组）
1. ✅ 基础校准指标（6个开源模型）
2. ✅ 完整校准指标（9个模型，包含3个闭源）

### 无法复现的内容
- ❌ Figure 1: Two-stage Paradigm (外部制作)
- ❌ Figure 2: Task Design (外部制作)
- ❌ Table 1: Confidence Distribution (需要手动提取)

---

## 🚀 一键生成所有内容

使用 `run_all.sh` 脚本可以生成大部分内容：

```bash
cd VLM_Meta_Cognition
bash run_all.sh
```

**注意**: 该脚本可能不包含所有步骤，建议按照上述清单逐个运行。

---

## 📊 完整运行顺序

1. **生成校准分析结果**:
   ```bash
   python scripts/analysis/calculate_calibration_metrics.py
   python scripts/analysis/calculate_calibration_metrics_full.py
   ```

2. **生成主文图表**:
   ```bash
   python scripts/figures/generate_figures_matlab.py  # Figure 3, A3, A4
   python scripts/figures/generate_figure4_calibration_ece_auc_curves.py  # Figure 4
   ```

3. **生成附录图表**:
   ```bash
   python scripts/figures/generate_calibration_full_figures.py  # Figure A1
   python scripts/figures/generate_calibration_all_models_grid.py  # Figure A2
   ```

4. **生成表格**:
   ```bash
   python scripts/tables/generate_table1.py  # Table 1
   python scripts/tables/generate_table2.py  # Table 2
   ```

---

**总计**: 可以复现 **6个图表** + **5个表格** + **2组分析结果** = **13个可复现项目**

