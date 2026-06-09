#!/bin/bash
# blog.sh：本地开发 / 构建 / 部署入口
#   run [port]      预构建 + pagefind 索引 + jekyll serve --watch
#   build           生成 dist/ 并建 pagefind 索引
#   deploy          build + 上传 COS + 刷新 CDN
#   pagefind [dir]  仅重建 <dir>/pagefind 索引（dir 默认 dist）

set -e

run_pagefind() {
  local dir="$1"
  if ! command -v npx >/dev/null 2>&1; then
    echo "⚠️  未检测到 npx，跳过 Pagefind 索引（本地搜索不可用）"
    return 0
  fi
  echo "🔎 生成 Pagefind 索引：$dir/pagefind"
  npx -y pagefind --site "$dir" --output-path "$dir/pagefind" >/dev/null
}

case "$1" in
  run)
    port=${2:-8080}
    # 先一次性 build 到 _site/ 再生成 pagefind 索引；serve 复用产物并 watch 增量重建
    bundle exec jekyll build
    run_pagefind "_site"
    bundle exec jekyll serve --skip-initial-build --watch --host=0.0.0.0 --port="$port"
    ;;
  build)
    bundle exec jekyll build --destination=dist
    run_pagefind "dist"
    ;;
  deploy)
    bundle exec jekyll build --destination=dist
    run_pagefind "dist"
    cos-upload local:./dist blog:/
    curl -fL -u freshCDN "https://cloud.page404.cn/api/fresh-cdn/"
    ;;
  pagefind)
    target=${2:-dist}
    run_pagefind "$target"
    ;;
  *)
    echo "用法: $0 {run [port]|build|deploy|pagefind [dir]}"
    exit 1
    ;;
esac
