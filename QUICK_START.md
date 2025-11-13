# 快速开始指南

## 📍 正确的路径

`VLM_Meta_Cognition` 文件夹位于：
```
VLM_MetaCognition_restructured/VLM_Meta_Cognition/
```

## 🚀 运行步骤

### 方法1：从项目根目录进入
```bash
cd VLM_MetaCognition_restructured/VLM_Meta_Cognition
```

### 方法2：直接使用完整路径
```bash
cd /Users/maiwang/VLM_MetaCognition/VLM_MetaCognition_restructured/VLM_Meta_Cognition
```

## 📊 生成图表和表格

进入 `VLM_Meta_Cognition` 目录后，运行：

```bash
# 生成 Figure 3 (模型排名)
python3 scripts/figures/generate_figures_matlab.py

# 生成 Figure 4 (校准分析)
python3 scripts/figures/generate_calibration_figure4.py

# 生成表格
python3 scripts/tables/generate_table2.py
```

## ✅ 验证

运行以下命令确认你在正确的目录：
```bash
pwd
# 应该显示: .../VLM_MetaCognition_restructured/VLM_Meta_Cognition

ls scripts/figures/
# 应该看到: generate_calibration_figure4.py 等文件
```

## 📝 注意事项

- 确保使用 `python3` 而不是 `python`
- 所有脚本都从 `VLM_Meta_Cognition/` 目录运行
- 生成的结果会保存在 `results/` 目录下

