# VLM MetaCognition

完整的视觉语言模型（VLM）元认知研究项目，包含模型拟合、校准分析和论文图表表格生成。所有文件命名已规范化，便于理解和复现。

---

## 📁 项目结构

```
VLM_Meta_Cognition/
├── scripts/              # 分析脚本
│   ├── figures/         # 图表生成脚本
│   ├── tables/          # 表格生成脚本
│   └── analysis/        # 分析脚本（校准分析等）
├── data/                # 数据文件
│   ├── raw/             # 原始数据
│   │   ├── self_report/ # Self-report置信度数据
│   │   └── logits/      # Logits-based置信度数据
│   ├── processed/       # 处理后的数据
│   │   └── model_fitting_matlab/  # MATLAB模型拟合结果
│   └── intermediate/    # 中间数据（分析过程中的临时数据）
└── results/             # 分析结果
    ├── figures/         # 生成的图表（PDF）
    ├── tables/          # 生成的表格（CSV）
    └── calibration/     # 校准分析结果
```

---

## 📊 数据文件命名规范

### 原始数据 (`data/raw/`)

| 文件名 | 说明 | 模型数量 |
|--------|------|----------|
| `self_report_6models_merged.csv` | Self-report置信度数据（6个开源模型） | 6 |
| `self_report_9models_merged.csv` | Self-report置信度数据（9个模型，包含3个闭源） | 9 |
| `logits_6models_merged.csv` | Logits-based置信度数据（6个开源模型） | 6 |

### 处理后数据 (`data/processed/`)

| 文件名 | 说明 |
|--------|------|
| `self_report_matlab_fits.csv` | Self-report模型拟合结果（MATLAB） |
| `logits_matlab_fits.csv` | Logits模型拟合结果（MATLAB） |

### 中间数据 (`data/intermediate/`)

| 文件名 | 说明 |
|--------|------|
| `figure3_rank_data.csv` | Figure 3的排名数据 |
| `figureA3_selfreport_comparison_data.csv` | Figure A3的模型比较数据（Self-report） |
| `figureA4_logits_comparison_data.csv` | Figure A4的模型比较数据（Logits） |

---

## 🚀 快速开始

### 环境要求

- **Python版本**: Python 3.7+ (推荐使用 `python3` 命令)
- **依赖包**: 运行 `pip install -r requirements.txt` 安装所需包

### 一键运行完整分析流程

```bash
cd VLM_Meta_Cognition
bash run_all.sh
```

**注意**: 这将运行完整的分析流程，包括校准分析、图表生成和表格生成。建议按照下面的详细步骤逐个运行，以确保理解每个步骤。

### ✅ 项目内容

本项目包含完整的 VLM 元认知研究分析流程：

**分析内容**:
- ✅ 模型拟合分析（5种认知模型：SDT, PE, WEV, LogN, BCH）
- ✅ 校准分析（ECE, AUC, Selective AUC等指标）
- ✅ 模型比较和排名分析

**可生成内容**:
- ✅ Figure 3: Model Rankings
- ✅ Figure 4: Calibration Analysis
- ✅ Figure A1: Calibration metrics for all 9 VLMs
- ✅ Figure A2: Calibration Metrics Grid
- ✅ Figure A3: Self-Report Model Comparison
- ✅ Figure A4: Logits Model Comparison
- ✅ Table 1: Best-fitting process models
- ✅ Table 2: Model Fits

**所有脚本已测试通过！**

---

## 📊 主文图表复现指南

### Figure 3: Model Rankings

**描述**: 显示6个开源VLM在3个任务上的模型排名（Self-report和Logits-based对比）

**输出文件**: `results/figures/Model_fitting/figure3_rank_combined.pdf`

**所需数据**:
- `data/processed/model_fitting_matlab/self_report_matlab_fits.csv`
- `data/processed/model_fitting_matlab/logits_matlab_fits.csv`

**运行步骤**:
```bash
cd VLM_Meta_Cognition
python3 scripts/figures/generate_figures_matlab.py
```

**生成内容**:
- 2行×3列图表
- 上排：Self-report方法
- 下排：Logits-based方法
- 每列：一个任务（Grid, Gabor, Brightness）

**中间数据**:
- 自动生成 `data/intermediate/figure3_rank_data.csv`

---

### Figure 4: Calibration Analysis (完整版)

**描述**: 校准分析完整图表，包含Panel A（ECE和AUC柱状图）和Panel B（三个任务的校准曲线）

**输出文件**: `results/figures/calibration/figure4_calibration_ece_auc_curves.pdf`

**所需数据**:
- `data/raw/self_report/self_report_6models_merged.csv`
- `data/raw/logits/logits_6models_merged.csv`
- `results/calibration/calibration_summary_method_task.csv` (需先运行分析脚本)
- `results/calibration/task_calibration_metrics.csv` (需先运行分析脚本)

**运行步骤**:
```bash
cd VLM_Meta_Cognition

# 步骤1: 先运行校准分析（生成汇总数据）
python3 scripts/analysis/calculate_calibration_metrics.py

# 步骤2: 生成完整的 Figure 4
python3 scripts/figures/generate_figure4_calibration_ece_auc_curves.py
```

**预期输出**:
```
步骤1输出:
✅ calibration_summary_method_task.csv
✅ task_calibration_metrics.csv

步骤2输出:
✅ PDF: results/figures/calibration/figure4_calibration_ece_auc_curves.pdf
✅ PNG: results/figures/calibration/figure4_calibration_ece_auc_curves.png
```

**生成内容**:
- **Panel A**: 
  - 左图：ECE (Expected Calibration Error) 柱状图
  - 右图：AUC (Area under ROC Curve) 柱状图
- **Panel B**: 
  - 三个任务的校准曲线（Grid, Gabor, Brightness）
  - 每个任务显示Self-report和Logits-based两条曲线
  - 包含斜率标注

**注意事项**:
- 必须先运行 `calculate_calibration_metrics.py` 生成校准指标
- 如果 `results/calibration/` 目录不存在，脚本会自动创建

---

## 📊 附录图表复现指南

### Figure A1: Calibration Metrics and Curves for All 9 VLMs

**描述**: 包含3个闭源模型在内的9个VLM的校准指标和曲线

**输出文件**: `results/figures/calibration_9_models/figureA1_Calibration_metrics_and_curves_for_all_9_VLMs_including_closed-source_models.pdf`

**所需数据**:
- `data/raw/self_report/self_report_9models_merged.csv`
- `results/calibration_9_models/calibration_by_task_full.csv` (需先运行分析脚本)

**运行步骤**:
```bash
cd VLM_Meta_Cognition

# 步骤1: 先运行完整校准分析（如果还没有运行）
python3 scripts/analysis/calculate_calibration_metrics_full.py

# 步骤2: 生成图表
python3 scripts/figures/generate_calibration_full_figures.py
```

**预期输出**:
```
步骤1输出:
✅ calibration_overall_full.csv
✅ calibration_by_task_full.csv
✅ calibration_by_model_full.csv
✅ calibration_task_model_full.csv

步骤2输出:
✅ PDF: results/figures/calibration_9_models/figureA1_Calibration_metrics_and_curves_for_all_9_VLMs_including_closed-source_models.pdf
```

**生成内容**:
- Panel A: ECE (Absolute) 柱状图
- Panel B: ROC AUC 柱状图
- Panel C: 汇总统计表
- Panel D-F: 三个任务的校准曲线（9个模型合并）

---

### Figure A2: Calibration Metrics Across All Models and Tasks

**描述**: 9行×4列网格图，显示所有模型在所有任务上的校准指标

**输出文件**: `results/figures/calibration_9_models/figureA2_Calibration_Metrics_Across_All_Models_and_Tasks.pdf`

**所需数据**:
- `results/calibration_9_models/calibration_task_model_full.csv` (需先运行分析脚本)

**运行步骤**:
```bash
cd VLM_Meta_Cognition

# 步骤1: 先运行完整校准分析（如果还没有运行）
python3 scripts/analysis/calculate_calibration_metrics_full.py

# 步骤2: 生成图表
python3 scripts/figures/generate_calibration_all_models_grid.py
```

**预期输出**:
```
✅ PDF: results/figures/calibration_9_models/figureA2_Calibration_Metrics_Across_All_Models_and_Tasks.pdf
```

**生成内容**:
- 9行（每个模型一行）
- 4列（每个指标一列：ECE Absolute, ECE Squared, ROC AUC, Selective AUC）
- 每个子图显示该模型在该指标上三个任务的表现
- gpt-5所在行有红色高亮框

---

### Figure A3: Self-Report Model Comparison

**描述**: 6个开源模型在3个任务上的AIC和p_model比较（Self-report方法）

**输出文件**: `results/figures/Model_fitting/figureA3_model_comparison.pdf`

**所需数据**:
- `data/processed/model_fitting_matlab/self_report_matlab_fits.csv`

**运行步骤**:
```bash
cd VLM_Meta_Cognition
python3 scripts/figures/generate_figures_matlab.py
```

**注意**: 与Figure 3使用相同脚本，会自动生成Figure A3和A4

**生成内容**:
- 2行×3列图表
- 上排：AIC比较
- 下排：p_model比较
- 每列：一个任务

**中间数据**:
- 自动生成 `data/intermediate/figureA3_selfreport_comparison_data.csv`

---

### Figure A4: Logits Model Comparison

**描述**: 6个开源模型在3个任务上的AIC和p_model比较（Logits-based方法）

**输出文件**: `results/figures/Model_fitting/figureA4_model_comparison_logits.pdf`

**所需数据**:
- `data/processed/model_fitting_matlab/logits_matlab_fits.csv`

**运行步骤**:
```bash
cd VLM_Meta_Cognition
python3 scripts/figures/generate_figures_matlab.py
```

**注意**: 与Figure 3使用相同脚本，会自动生成Figure A3和A4

**生成内容**:
- 2行×3列图表
- 上排：AIC比较
- 下排：p_model比较
- 每列：一个任务

**中间数据**:
- 自动生成 `data/intermediate/figureA4_logits_comparison_data.csv`

---

## 📋 表格复现指南

### Table 1: Best-fitting process models per VLM-task combination

**描述**: 显示每个VLM-任务组合的最佳拟合模型，以及主导模型统计

**输出文件**:
- `results/tables/table1_vlm_task_models.csv` - 每个VLM-任务组合的最佳拟合模型
- `results/tables/table1_dominant_model_counts.csv` - 主导模型统计计数
- `results/tables/table1_model_fits_per_vlm_task.csv` - 完整表格（包含统计行）

**所需数据**:
- `data/processed/model_fitting_matlab/self_report_matlab_fits.csv`
- `data/processed/model_fitting_matlab/logits_matlab_fits.csv`

**运行步骤**:
```bash
cd VLM_Meta_Cognition
python3 scripts/tables/generate_table1.py
```

**预期输出**:
```
✅ table1_vlm_task_models.csv
✅ table1_dominant_model_counts.csv
✅ table1_model_fits_per_vlm_task.csv
```

**生成内容**:
- 6个开源VLM在3个任务上的最佳拟合模型（Self-report和Logits-based分别）
- 主导模型统计计数（每个模型在不同任务中出现的次数）

**表格结构**:
- 第一部分：VLM-任务组合的最佳模型矩阵
- 第二部分：主导模型统计（PE, BCH, SDT, LogN, WEV）

---

### Table 2: Model Fits

**描述**: 模型拟合结果对比（基于AIC）

**输出文件**:
- `results/tables/table2_model_fits.csv` - 模型拟合统计结果
- `results/tables/table2_dominance_statistics.csv` - 主导模型统计
- `results/tables/table2_detailed_dominant_models.csv` - 详细的主导模型数据

**所需数据**:
- `data/processed/model_fitting_matlab/self_report_matlab_fits.csv`
- `data/processed/model_fitting_matlab/logits_matlab_fits.csv`

**运行步骤**:
```bash
cd VLM_Meta_Cognition
python3 scripts/tables/generate_table2.py
```

**预期输出**:
```
✅ table2_model_fits.csv
✅ table2_dominance_statistics.csv
✅ table2_detailed_dominant_models.csv
```

**生成内容**:
- AIC, BIC统计
- 主导模型统计
- 详细的主导模型数据

---

## 📊 校准分析结果

### 基础校准指标（6个开源模型）

**位置**: `results/calibration/`

**生成脚本**: `scripts/analysis/calculate_calibration_metrics.py`

**运行步骤**:
```bash
cd VLM_Meta_Cognition
python3 scripts/analysis/calculate_calibration_metrics.py
```

**预期输出**:
```
✅ calibration_summary_method_task.csv
✅ task_calibration_metrics.csv
```

**输出文件**:
- `calibration_summary_method_task.csv` - 按方法和任务汇总
- `task_calibration_metrics.csv` - 任务校准指标（包含slope）

**所需数据**:
- `data/raw/self_report/self_report_6models_merged.csv`
- `data/raw/logits/logits_6models_merged.csv`

---

### 完整校准指标（9个模型，包含3个闭源）

**位置**: `results/calibration_9_models/`

**生成脚本**: `scripts/analysis/calculate_calibration_metrics_full.py`

**运行步骤**:
```bash
cd VLM_Meta_Cognition
python3 scripts/analysis/calculate_calibration_metrics_full.py
```

**预期输出**:
```
✅ calibration_by_task_full.csv
✅ calibration_by_model_full.csv
✅ calibration_task_model_full.csv
✅ calibration_overall_full.csv
```

**输出文件**:
- `calibration_by_task_full.csv` - 按任务汇总（3个任务）
- `calibration_by_model_full.csv` - 按模型汇总（9个模型）
- `calibration_task_model_full.csv` - 按任务×模型详细数据（27行）
- `calibration_overall_full.csv` - 整体汇总

**所需数据**:
- `data/raw/self_report/self_report_9models_merged.csv`

---

## 📋 完整运行顺序（推荐）

为了确保所有依赖关系正确，建议按以下顺序运行：

### 1. 生成校准分析结果
```bash
# 基础校准（6个开源模型）
python3 scripts/analysis/calculate_calibration_metrics.py

# 完整校准（9个模型，包含闭源）
python3 scripts/analysis/calculate_calibration_metrics_full.py
```

### 2. 生成主文图表
```bash
# Figure 3, A3, A4（模型排名和比较）
python3 scripts/figures/generate_figures_matlab.py

# Figure 4（校准分析完整版）
python3 scripts/figures/generate_figure4_calibration_ece_auc_curves.py
```

### 3. 生成附录图表
```bash
# Figure A1（9个模型的校准指标和曲线）
python3 scripts/figures/generate_calibration_full_figures.py

# Figure A2（9×4网格图）
python3 scripts/figures/generate_calibration_all_models_grid.py
```

### 4. 生成表格
```bash
# Table 1（最佳拟合模型）
python3 scripts/tables/generate_table1.py

# Table 2（模型拟合结果）
python3 scripts/tables/generate_table2.py
```

---

## ⚙️ 依赖要求

### Python版本
- **Python 3.7+** (推荐使用 `python3` 命令)
- 在macOS/Linux上，通常使用 `python3` 而不是 `python`

### Python包
安装依赖：
```bash
pip3 install -r requirements.txt
# 或
pip install -r requirements.txt
```

主要依赖包：
- `pandas` - 数据处理
- `numpy` - 数值计算
- `matplotlib` - 图表生成
- `scipy` - 统计分析（部分脚本）

### 验证安装
```bash
python3 -c "import pandas, numpy, matplotlib, scipy; print('✅ 所有依赖已安装')"
```

---

## 📝 注意事项

### 1. 路径要求
- 所有脚本都使用相对路径，从 `VLM_Meta_Cognition/` 目录运行
- 确保在正确的目录下运行脚本

### 2. 数据完整性
- 确保所有必需的数据文件都存在
- 如果缺少数据文件，脚本会报错并提示缺失的文件路径

### 3. 运行顺序
- 某些图表需要先运行校准分析脚本
- 建议按照"完整运行顺序"章节的顺序执行

### 4. 输出目录
- 所有结果保存在 `results/` 目录下
- 如果目录不存在，脚本会自动创建

### 5. 文件命名
- 所有数据文件已规范化命名
- 命名格式：`{data_type}_{method}_{model_count}_{additional_info}.csv`

---

## 🔍 详细文档

更多详细信息请参阅：
- `REPRODUCIBLE_ITEMS.md` - 可复现内容清单
- `REPRODUCIBILITY_CHECK.md` - 可复现性检查报告

**注意**: 所有分析的详细复现指南已包含在本README中，无需查看其他文档。

---

## 📊 图表清单总结

### 主文图表（2个）
1. ✅ **Figure 3**: Model Rankings (`generate_figures_matlab.py`)
   - 输出: `results/figures/Model_fitting/figure3_rank_combined.pdf`
   - 状态: ✅ 已验证可生成

2. ✅ **Figure 4**: Calibration Analysis (`generate_figure4_calibration_ece_auc_curves.py`)
   - 输出: `results/figures/calibration/figure4_calibration_ece_auc_curves.pdf`
   - 状态: ✅ 已验证可生成

### 附录图表（4个）
3. ✅ **Figure A1**: Calibration metrics and curves for all 9 VLMs (`generate_calibration_full_figures.py`)
   - 输出: `results/figures/calibration_9_models/figureA1_Calibration_metrics_and_curves_for_all_9_VLMs_including_closed-source_models.pdf`
   - 状态: ✅ 已验证可生成

4. ✅ **Figure A2**: Calibration Metrics Across All Models and Tasks (`generate_calibration_all_models_grid.py`)
   - 输出: `results/figures/calibration_9_models/figureA2_Calibration_Metrics_Across_All_Models_and_Tasks.pdf`
   - 状态: ✅ 已验证可生成

5. ✅ **Figure A3**: Self-Report Model Comparison (`generate_figures_matlab.py`)
   - 输出: `results/figures/Model_fitting/figureA3_model_comparison.pdf`
   - 状态: ✅ 已验证可生成

6. ✅ **Figure A4**: Logits Model Comparison (`generate_figures_matlab.py`)
   - 输出: `results/figures/Model_fitting/figureA4_model_comparison_logits.pdf`
   - 状态: ✅ 已验证可生成

### 表格（2个）
7. ✅ **Table 1**: Best-fitting process models per VLM-task combination (`generate_table1.py`)
   - 输出: `results/tables/table1_vlm_task_models.csv`, `table1_dominant_model_counts.csv`
   - 状态: ✅ 已验证可生成

8. ✅ **Table 2**: Model Fits (`generate_table2.py`)
   - 输出: `results/tables/table2_model_fits.csv`, `table2_dominance_statistics.csv`
   - 状态: ✅ 已验证可生成

**总计**: 可以复现 **6个图表** + **2个表格** = **8个可复现项目**

**所有项目均已测试通过！** ✅

---

## 📖 项目概述

本项目是完整的 VLM 元认知研究项目，包含：

1. **模型拟合分析**: 使用5种认知模型（SDT, PE, WEV, LogN, BCH）拟合VLM的置信度数据
2. **校准分析**: 评估VLM置信度的可靠性（ECE, AUC, Selective AUC等指标）
3. **模型比较**: 比较不同模型在不同任务和VLM上的表现
4. **结果可视化**: 生成论文所需的所有图表和表格

所有分析流程已标准化，确保结果的可重现性。

---

## 🐛 常见问题

### Q1: 运行脚本时提示 `command not found: python`
**A**: 使用 `python3` 而不是 `python`：
```bash
python3 scripts/figures/generate_figures_matlab.py
```

### Q2: 运行脚本时提示找不到数据文件
**A**: 检查数据文件是否在正确的位置，文件名是否匹配。所有数据文件应位于 `data/raw/` 或 `data/processed/` 目录下。

**检查方法**:
```bash
# 检查原始数据
ls -la data/raw/self_report/
ls -la data/raw/logits/

# 检查处理后数据
ls -la data/processed/model_fitting_matlab/
```

### Q3: 生成的图表与论文中的不一致
**A**: 确保：
1. 使用了正确的数据文件（6个模型 vs 9个模型）
2. 按照正确的顺序运行脚本（先运行分析脚本，再运行图表生成脚本）
3. 检查是否有中间数据文件需要重新生成

### Q4: 校准分析结果为空
**A**: 确保：
1. 原始数据文件存在且格式正确
2. 数据文件包含必需的列（confidence, correct, task, vlm等）
3. 运行了 `calculate_calibration_metrics.py` 或 `calculate_calibration_metrics_full.py`

**验证方法**:
```bash
# 检查校准结果文件是否存在
ls -la results/calibration/
ls -la results/calibration_9_models/
```

### Q5: 如何只生成某个特定的图表？
**A**: 参考上面的"复现指南"章节，找到对应图表的运行步骤，只运行该脚本即可。

### Q6: 脚本运行成功但没有生成文件
**A**: 检查：
1. 是否有写入权限
2. 输出目录是否存在（脚本会自动创建）
3. 查看脚本输出的完整日志，确认文件保存路径

**验证方法**:
```bash
# 检查所有生成的图表
find results/figures -name "*.pdf" | sort

# 检查所有生成的表格
find results/tables -name "*.csv" | sort
```

### Q7: 导入错误（ImportError）
**A**: 确保所有依赖包已安装：
```bash
pip3 install -r requirements.txt
```

如果仍有问题，检查Python版本：
```bash
python3 --version  # 应该是 3.7 或更高
```

---

## 📧 联系方式

如有问题或建议，请参考项目主README或联系项目维护者。
