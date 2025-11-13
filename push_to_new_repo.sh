#!/bin/bash
# 推送 VLM_Meta_Cognition 到新仓库并创建 dev 分支

echo "=========================================="
echo "推送 VLM_Meta_Cognition 到新仓库"
echo "=========================================="
echo ""

# 进入项目目录
cd "$(dirname "$0")"
echo "当前目录: $(pwd)"
echo ""

# 初始化 git 仓库（如果还没有）
if [ ! -d ".git" ]; then
    echo "📦 初始化 Git 仓库..."
    git init
    echo ""
fi

# 添加远程仓库
REPO_URL="https://github.com/grow-ai-like-a-child/VLM_Meta_Cognition.git"
echo "🔗 设置远程仓库: $REPO_URL"
git remote remove origin 2>/dev/null
git remote add origin $REPO_URL
git remote -v
echo ""

# 添加所有文件
echo "📦 添加所有文件..."
git add .
echo ""

# 显示状态
echo "文件状态:"
git status --short | head -20
echo ""

# 提交
echo "💾 提交更改..."
git commit -m "Initial commit: VLM MetaCognition project

- Complete VLM metacognition research project
- Model fitting analysis (5 cognitive models: SDT, PE, WEV, LogN, BCH)
- Calibration analysis (ECE, AUC, Selective AUC metrics)
- Model comparison and ranking analysis
- All figures and tables generation scripts
- Standardized file naming
- All scripts tested and verified"
echo ""

# 推送到 main 分支
echo "🚀 推送到 main 分支..."
git branch -M main
git push -u origin main
echo ""

# 创建并推送 dev 分支
echo "🌿 创建并推送 dev 分支..."
git checkout -b dev
git push -u origin dev
echo ""

echo "=========================================="
echo "✅ 完成！"
echo "=========================================="
echo ""
echo "仓库链接:"
echo "  Main: https://github.com/grow-ai-like-a-child/VLM_Meta_Cognition"
echo "  Dev:  https://github.com/grow-ai-like-a-child/VLM_Meta_Cognition/tree/dev"
echo ""

