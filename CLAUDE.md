# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

这是一个基于 Jekyll 3.9.5 的个人技术博客（域名 `https://www.big-yellow-j.top`，CNAME 已配置），主题派生自 `TMaize/tmaize-blog`。仓库 = 内容（Markdown 文章）+ Jekyll 模板/布局 + 一套 Python 工具链（图片优化、AI 摘要生成、搜索引擎推送）。部署通过 `.github/workflows/jekyll-gh-pages.yml` 在 push 到 `master` 时自动构建并发布到 GitHub Pages。

## 常用命令

本地启动 / 构建（要求已按 README 配置好 Ruby、bundler 与镜像源）：

```bash
./blog.sh run [port]    # bundle exec jekyll serve --watch --host=0.0.0.0 --port=$port (默认 8080)
./blog.sh build         # bundle exec jekyll build --destination=dist
./blog.sh deploy        # build + cos-upload + 刷新 CDN
```

或直接：

```bash
bundle install
bundle exec jekyll serve --watch --host=127.0.0.1 --port=8080
bundle exec jekyll build --destination=dist
```

文章/草稿后处理与提交（一站式脚本）：

```bash
./submit_github.sh      # 运行 optim_post.py（按需 pip install 缺失依赖）→ git add . && commit "YYYY-MM-DD 修改: <files>" && push
python3 optim_post.py   # 仅运行图片转 webp + 上传 sm.ms + 调 LLM 生成 description 写回 YAML 头
python3 submit_url.py   # 读取 _site/static/xml/sitemap.xml，向 Google Indexing API + Bing IndexNow 推送 URL
```

`optim_post.py` 需要 `./API_KEY.env`（含 `SMMS_API_LIST`、`LLM_URL`、`API_KEY`），并维护 `DEAL-MD.json` 作为已处理记录。`submit_url.py` 需要 `./google_api.json` 服务账号凭证，并硬编码了本地代理 `127.0.0.1:7897`，本地无代理时需要先注释/调整环境变量。

## 内容目录约定

- `_posts/`：正式发布文章。文件名 `YYYY-MM-DD-slug.md`，permalink 由 `_config.yml` 指定为 `/posts/:year/:month/:day/:title.html`。
- `writing/`：草稿区，`_config.yml` 中通过 `include: writing` 让 Jekyll 也处理它（用于 TODO 预览）。`mypost.html` 检测 `page.url contains "TODO"` 会显示红色 "writing....." 标记。`preview_writing.html` 是手写的草稿预览页（静态文件名列表，需要手动同步）。
- `_jupyter/`：Notebook 源文件，配合 `code/` 下生成的 HTML（如 `code/LeNet.html`）通过 iframe 嵌入文章。
- `pages/`：顶级导航页（code/categories/cv/life/links/search/about）。
- `images/post_image/<post-slug>/`：`optim_post.py` 为每篇 md 创建的同名目录，存放下载并转 webp 后的图片，再上传到 `sm.ms` 图床并把 md 内 URL 替换为图床地址。
- `_framework/OpenRLHF/`、`code/Python/`：博客引用/讲解的第三方代码与示例（多数被 `.gitignore` 排除编译产物与权重文件）。

## 文章 YAML 头（README 已示例）

`mypost.html` 期望的常用字段：`layout: mypost`、`title`、`categories`、`address`、`tags`、`extMath`、`show_footer_image`、`show`（控制是否在首页列出）、`images`、`stickie`（置顶）、`special_tag`（如 "更新中"）、`description`（缺失时 `optim_post.py` 会自动用豆包/DeepSeek 等 LLM 生成 150–250 汉字摘要并写回）。

## Jekyll 布局结构

- 布局入口 `_layouts/mypost.html`：负责标题/字数/访问量/AI 摘要框/分享按钮/上下篇导航/评论（twikoo）/TOC/统计脚本，并对 `{{ content }}` 做正则替换为图片注入 `loading="lazy"`、`data-lightbox="gallery"`、`class="content-image"`。
- 公共片段在 `_includes/`（`head.html`、`header.html`、`footer.html`、`script.html`、`ext-mathjax.html`、`ext-serviceWorker.html` 等），首页/分类/搜索/CV/生活页位于 `pages/` 与根目录 `index.html`。
- 站点配置 `_config.yml`：分页 30/页，启用 `jekyll-minifier`（注释/空白/JS/CSS 压缩），数学渲染走 MathJax，代码高亮使用 prism 客户端（kramdown 关闭了内置 rouge 高亮 `syntax_highlighter_opts.disable: true`）。修改 `_config.yml` 后需要重启 `jekyll serve`。
- 安装新插件流程见 README：改 `Gemfile` → `bundle install` → 在 `_config.yml` 的 `plugins:` 下追加。

## 编辑/写作时的注意点

- 写新文章默认放 `_posts/`；若仍在编辑可放 `writing/` 或文件名带 `-TODO`（`optim_post.py` 在建立图片目录时会 `replace('-TODO','')`，部署后红色 writing 提示也依赖 URL 含 "TODO"）。
- 不要手工修改 `DEAL-MD.json`——它由 `optim_post.py` 维护，记录每篇 md 的旧/新 description 对比。
- `*.json`、`*.lock`、`_site/`、`dist/`、`.idea/`、`.obsidian/`、`.junie/`、模型权重（`*.pth`、`*.safetensors`、`*.gguf`、`*.onnx` 等）以及部分 `code/` 子树均在 `.gitignore` 中，提交前留意 `git status`。
- `submit_github.sh` 使用 `git add .` 全量暂存，提交信息固定为 `"YYYY-MM-DD 修改: <changed files>"`——如果你新增了不该入库的文件，先用 `.gitignore` 或手动 `git restore --staged` 排除，再跑脚本。
