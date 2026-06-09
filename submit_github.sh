#!/bin/bash
# submit_github.sh — 一键预发与提交
#
# 用法：
#   ./submit_github.sh              optim_post.py → git add → commit → push
#   ./submit_github.sh --draft      只跑 optim_post.py（图片+摘要），不提交
#   ./submit_github.sh --check      跑预检（front matter + 死链），不改动 / 不提交
#   ./submit_github.sh --no-push    跑 optim + commit，但不 push
#   ./submit_github.sh --help

set -e

MODE="commit"
PUSH=1

for arg in "$@"; do
  case "$arg" in
    --draft)   MODE="draft" ;;
    --check)   MODE="check" ;;
    --no-push) PUSH=0 ;;
    --help|-h)
      grep -E '^#( |$)' "$0" | sed 's/^# \?//'
      exit 0 ;;
    *)
      echo "未知参数: $arg（--help 查看用法）" >&2
      exit 2 ;;
  esac
done

DATE_STR=$(date '+%Y-%m-%d')

# 跑 optim_post.py；缺包自动安装
run_python() {
    while true; do
        ERR=$(python3 optim_post.py 2>&1)
        echo "$ERR"
        MISSING=$(echo "$ERR" | grep -oP "ModuleNotFoundError: No module named '\K[^']+")
        [ -z "$MISSING" ] && break
        echo "❌ 缺少 Python 包: $MISSING，正在安装..."
        pip install "$MISSING"
    done
}

# --check 模式：front matter + 死链
run_check() {
    echo "🔍 1/2 校验 front matter ..."
    if [ -f scripts/validate_front_matter.py ]; then
        python3 scripts/validate_front_matter.py
    else
        echo "  (skip: scripts/validate_front_matter.py 不存在)"
    fi

    echo "🔍 2/2 死链巡检 ..."
    if command -v lychee >/dev/null 2>&1; then
        lychee --no-progress --max-concurrency 8 \
            --exclude '^https?://(localhost|127\.)' \
            _posts/ README.md || true
    else
        echo "  (skip: 未装 lychee，安装：cargo install lychee 或 apt install lychee)"
    fi
    echo "✅ 检查完成"
}

case "$MODE" in
  check)
    run_check
    ;;
  draft)
    run_python
    echo "✅ 草稿处理完成（未提交）"
    git status --short
    ;;
  commit)
    run_python
    CHANGED_FILES=$(git status --porcelain | awk '{print $2}' | tr '\n' ' ')
    if [ -z "$CHANGED_FILES" ]; then
        echo "No changes to commit."
        exit 0
    fi
    COMMIT_MSG="$DATE_STR 修改: $CHANGED_FILES"
    git add .
    git commit -m "$COMMIT_MSG"
    if [ "$PUSH" -eq 1 ]; then
        git push
        echo "✅ 已提交并推送"
    else
        echo "✅ 已提交（未 push）"
    fi
    ;;
esac
