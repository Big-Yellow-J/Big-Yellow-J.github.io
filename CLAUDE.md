# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

这是一个基于 Jekyll 3.9.5 的个人技术博客（域名 `https://www.big-yellow-j.top`，CNAME 已配置），主题派生自 `TMaize/tmaize-blog`。仓库 = 内容（Markdown 文章）+ Jekyll 模板/布局 + 一套 Python 工具链（图片优化、AI 摘要生成、搜索引擎推送）。部署通过 `.github/workflows/jekyll-gh-pages.yml` 在 push 到 `master` 时自动构建并发布到 GitHub Pages。

## 知识库结构（2026-06 重构后）

- 首页（`index.html`）按年份分组文章卡片，`paginate: 15`/页。
- 分类（`pages/categories.html`）：分类卡片网格 + 每分类完整文章列表，锚点 `#<category>`。
- 标签（`pages/tags.html`）：标签云（字号随频次）+ 每标签文章列表。
- 搜索（`pages/search.html` + `static/js/search.js`）：基于 `search.xml` 的客户端全文检索。
- 文章页（`_layouts/mypost.html`）：顶部面包屑 + AI 摘要 + 正文 + 文末"相关阅读"（同 `categories[0]` 取最近 5 篇）+ 分享 + 上下篇 + 评论。
- 阅读辅助：固定侧栏 TOC（`static/js/toc.js` + `post.css #toc`），移动端改抽屉（FAB 在**左下**避开 `.footer-btn` 右下三件套）；阅读进度条（`blog.js .read-progress`）。

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

- `_layouts/mypost.html`：面包屑（带 JSON-LD BreadcrumbList）→ 标题/字数 → AI 摘要 → 正文（图片注入 `loading="lazy"` `decoding="async"` `class="content-image"`，由 `gallery.js` 包装成 PhotoSwipe `<a data-pswp-width/height>`）→ 相关阅读 → 分享 → 上下篇 → 评论。
- `_includes/head.html` 已内置完整 SEO：OG / Twitter Card / canonical / robots / BlogPosting JSON-LD；preconnect 4 个 CDN + dns-prefetch 6 个上报域名；字体只在 `mypost` 才加载 Serif SC；PhotoSwipe CSS 仅文章页 `preload`；统一 Font Awesome 6.5.0 一处；title 模板自动加 `| {{ site.title }}`。
- `_includes/script.html` 已有按需 MathJax 逻辑：`site.extMath` 或 `page.extMath` 才挂 `ext-mathjax.html`。**优化建议**：把 `_config.yml extMath: true` 改为 `false`，让每篇有公式的文章自己声明 `extMath: true`，可省 1.7MB 给非数学文章访问者。
- `_includes/comment-twikoo.js` 改用 `IntersectionObserver` 监听 `#tcomment`，滚动到评论区前 400px 才加载 KaTeX + Twikoo（首屏省 ~800KB）。
- `static/css/`：`common.css`（全站）、`post.css`（文章页：正文/TOC/AI 摘要/相关阅读/面包屑/进度条/toast/暗色覆盖入口）、`page.css`（首页/分类/标签/搜索：列表+分类卡片网格+标签云）、`theme-dark.css`（暗色覆盖，新增组件都在其中补了暗色规则）、`code-light.css` / `code-dark.css`（Prism 主题切换）。
- 站点配置 `_config.yml`：分页 **15**/页，菜单含「首页 / Code / 归类 / 标签 / 搜索 / 生活 / CV」，启用 `jekyll-minifier`，代码用 prism 客户端高亮（kramdown rouge 已关）。修改 `_config.yml` 后需要重启 `jekyll serve`。
- 安装新插件流程见 README：改 `Gemfile` → `bundle install` → 在 `_config.yml` 的 `plugins:` 下追加。
- `manifest.json` 在根目录，让站点可"添加到主屏"（PWA）。
- `robots.txt` 在根目录，`Disallow: /writing/` 避免草稿被索引，并指向 `static/xml/sitemap.xml`。
- `static/xml/sitemap.xml` 的 `lastmod` 使用 `post.last_modified_at`（不是 `site.time`）。

## 编辑/写作时的注意点

- 写新文章默认放 `_posts/`；若仍在编辑可放 `writing/` 或文件名带 `-TODO`（`optim_post.py` 在建立图片目录时会 `replace('-TODO','')`，部署后红色 writing 提示也依赖 URL 含 "TODO"）。
- 建议每篇 YAML 头加 `image:` 字段指向首张图 URL——会被 `head.html` 用作 `og:image`，社交分享卡片更吸引人。
- 不要手工修改 `DEAL-MD.json`——它由 `optim_post.py` 维护，记录每篇 md 的旧/新 description 对比。
- 文章里的图片：跑过 `optim_post.py` 后会从 `![alt](url)` 改写成 `<img src="..." alt="..." width="W" height="H" loading="lazy" decoding="async" />` —— 这是为了让浏览器在图未下载时就占好位防 CLS，并且让 PhotoSwipe（`static/js/gallery.js`）能正确建立相册尺寸。**手写文章时也尽量用 HTML `<img>` 带宽高**。
- `optim_post.py` 已对图床地址（`s2.loli.net` / `sm.ms` / `s.ee`）有 `SKIP_HOSTS` 白名单跳过：不会重复上传，但仍会拉一次原图取尺寸写回。
- `submit_url.py` 通过 `BLOG_SUBMIT_PROXY` 环境变量读代理（不再硬编码 `127.0.0.1:7897`）；本地有代理：`export BLOG_SUBMIT_PROXY=http://127.0.0.1:7897`，无代理留空。
- `*.json`、`*.lock`、`_site/`、`dist/`、`.idea/`、`.obsidian/`、`.junie/`、模型权重（`*.pth`、`*.safetensors`、`*.gguf`、`*.onnx` 等）以及部分 `code/` 子树和 `writing/DistributeTraining/learning_distribute_training/outputs/`（训练产物）均在 `.gitignore` 中，提交前留意 `git status`。`manifest.json` 是 `*.json` 通配的例外。
- `submit_github.sh` 使用 `git add .` 全量暂存，提交信息固定为 `"YYYY-MM-DD 修改: <changed files>"`——如果你新增了不该入库的文件，先用 `.gitignore` 或手动 `git restore --staged` 排除，再跑脚本。
