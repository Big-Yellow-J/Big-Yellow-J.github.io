#!/bin/bash

# 获取当前日期（年月日）
DATE_STR=$(date '+%Y-%m-%d')

# 获取本次改动的文件列表（已修改、已新建、已删除）
CHANGED_FILES=$(git status --porcelain | awk '{print $2}' | tr '\n' ' ')

# 如果没有改动就退出
if [ -z "$CHANGED_FILES" ]; then
    echo "No changes to commit."
    exit 0
fi

# 拼接提交信息
COMMIT_MSG="$DATE_STR 修改: $CHANGED_FILES"

# 执行提交
git add .
git commit -m "$COMMIT_MSG"
git push