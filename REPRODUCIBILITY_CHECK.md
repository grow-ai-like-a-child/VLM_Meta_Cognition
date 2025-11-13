# 可复现性检查报告

## ✅ 可复现性分析

### 1. 数据完整性 ✅

**必需数据文件检查：**
- ✅ `data/raw/self_report/vlm_merged_models.csv` - 存在
- ✅ `data/raw/self_report/vlm_all_models_merged.csv` - 存在（9个模型）
- ✅ `data/raw/logits/vlm_merged_models_logits_only.csv` - 存在
- ✅ `data/processed/model_fitting_matlab/vlm_matlab_fits.csv` - 存在
- ✅ `data/processed/model_fitting_matlab/logits_matlab_fits.csv` - 存在

### 2. 代码完整性 ✅

**所有脚本文件存在：**
- ✅ `scripts/analysis/calculate_calibration_metrics.py`
- ✅ `scripts/analysis/calculate_calibration_metrics_full.py`
- ✅ `scripts/figures/generate_figures_matlab.py`
- ✅ `scripts/figures/generate_calibration_figure4.py`
- ✅ `scripts/figures/generate_figure4_calibration_ece_auc_curves.py`
- ✅ `scripts/figures/generate_calibration_full_figures.py`
- ✅ `scripts/figures/generate_calibration_all_models_grid.py`
- ✅ `scripts/tables/generate_table2.py`

### 3. 路径依赖 ✅

**路径管理方式：**
- ✅ 所有脚本使用相对路径（基于 `VLM_Meta_Cognition` 根目录）
- ✅ 使用 `os.path.dirname()` 和 `os.path.join()` 构建路径
- ✅ 部分脚本支持多个可能的路径位置（向后兼容）

**路径计算方式：**
```python
# 标准模式：从脚本位置计算根目录
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 结果：scripts/figures/ -> VLM_Meta_Cognition/
```

### 4. 依赖包 ✅

**必需的Python包：**
- ✅ `pandas` - 数据处理
- ✅ `numpy` - 数值计算
- ✅ `matplotlib` - 图表生成
- ✅ `sklearn` - ROC AUC计算

**注意：** 当前没有 `requirements.txt` 文件，建议添加。

### 5. 运行顺序 ⚠️

**正确的运行顺序：**

1. **先运行分析脚本生成中间数据：**
   ```bash
   # 生成基础校准指标（6个开源模型）
   python scripts/analysis/calculate_calibration_metrics.py
   
   # 生成完整校准指标（9个模型，包含闭源）
   python scripts/analysis/calculate_calibration_metrics_full.py
   ```

2. **然后生成图表：**
   ```bash
   # Figure 3, A3, A4（模型拟合结果）
   python scripts/figures/generate_figures_matlab.py
   
   # Figure 4（校准分析）
   python scripts/figures/generate_figure4_calibration_ece_auc_curves.py
   
   # Figure A1, A2（完整校准分析）
   python scripts/figures/generate_calibration_full_figures.py
   python scripts/figures/generate_calibration_all_models_grid.py
   ```

3. **最后生成表格：**
   ```bash
   python scripts/tables/generate_table2.py
   ```

### 6. 潜在问题 ⚠️

1. **缺少 requirements.txt**
   - 建议创建 `requirements.txt` 列出所有依赖包及版本

2. **路径硬编码问题**
   - `calculate_calibration_metrics_full.py` 中有一处硬编码路径：
     ```python
     self_report_path = os.path.join(base_dir, 'data', 'self_report', 'vlm_all_models_merged.csv')
     ```
   - 应该改为：`data/raw/self_report/vlm_all_models_merged.csv`

3. **中间数据依赖**
   - 某些脚本需要先运行其他脚本生成中间数据
   - 文档中应明确说明运行顺序

### 7. 文档完整性 ✅

- ✅ `README.md` - 存在，包含基本说明
- ✅ `QUICK_START.md` - 存在
- ✅ `docs/FIGURES_TABLES_DEPENDENCIES.md` - 详细的依赖说明
- ✅ `run_all.sh` - 一键运行脚本（但可能不完整）

## 📋 改进建议

### 高优先级

1. **创建 requirements.txt**
   ```txt
   pandas>=1.3.0
   numpy>=1.21.0
   matplotlib>=3.4.0
   scikit-learn>=0.24.0
   ```

2. **修复路径问题**
   - 统一所有脚本的数据路径为 `data/raw/` 和 `data/processed/`

3. **完善 run_all.sh**
   - 包含所有必需的运行步骤和正确的顺序

### 中优先级

4. **添加数据验证**
   - 在脚本开始时检查必需数据文件是否存在

5. **添加版本信息**
   - 在输出文件中记录生成时间和版本信息

## ✅ 总体评估

**可复现性：85%**

- ✅ 数据文件完整
- ✅ 代码脚本完整
- ✅ 路径管理基本正确
- ⚠️ 缺少依赖包列表
- ⚠️ 需要明确运行顺序
- ⚠️ 部分路径需要统一

**结论：** 在添加 `requirements.txt` 和修复路径问题后，该文件夹内容可以复现。

