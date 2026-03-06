#!/bin/bash

# 获取当前日期（年月日）
DATE_STR=$(date '+%Y-%m-%d')

# 运行 Python 脚本并自动处理缺少的包
run_python() {
    while true; do
        # 尝试运行脚本
        ERR=$(python3 optim_post.py 2>&1)

        # 检查是否有缺少包的错误
        MISSING_PACKAGE=$(echo "$ERR" | grep -oP "ModuleNotFoundError: No module named '\K[^']+")

        if [ -z "$MISSING_PACKAGE" ]; then
            # 没有缺少包，正常退出循环
            break
        else
            echo "❌ 缺少 Python 包: $MISSING_PACKAGE，正在安装..."
            pip install "$MISSING_PACKAGE"
        fi
    done
}


run_python

CHANGED_FILES=$(git status --porcelain | awk '{print $2}' | tr '\n' ' ')

if [ -z "$CHANGED_FILES" ]; then
    echo "No changes to commit."
    exit 0
fi

COMMIT_MSG="$DATE_STR 修改: $CHANGED_FILES"

git add .
git commit -m "$COMMIT_MSG"
git push

echo "✅ 脚本运行并提交完成！"