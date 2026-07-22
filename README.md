# Big-Yellow-J Blog

基于 [Jekyll 3.9.5](https://jekyllrb.com/) 的个人技术博客，主题派生自 [TMaize/tmaize-blog](https://github.com/TMaize/tmaize-blog)。在线访问：<https://www.big-yellow-j.top>。

仓库内容 = Markdown 文章 + Jekyll 模板/布局 + 一套 Python 工具链（图片优化、AI 摘要、URL 推送、front matter 校验）。
---

## 目录

- [Big-Yellow-J Blog](#big-yellow-j-blog)
  - [仓库内容 = Markdown 文章 + Jekyll 模板/布局 + 一套 Python 工具链（图片优化、AI 摘要、URL 推送、front matter 校验）。](#仓库内容--markdown-文章--jekyll-模板布局--一套-python-工具链图片优化ai-摘要url-推送front-matter-校验)
  - [目录](#目录)
  - [写作格式](#写作格式)
  - [本地预览](#本地预览)
    - [Linux / WSL / macOS](#linux--wsl--macos)
    - [Windows](#windows)
  - [部署方式 A：GitHub Actions + GitHub Pages（推荐）](#部署方式-agithub-actions--github-pages推荐)
    - [一次性配置](#一次性配置)
  - [部署方式 B：阿里云 ECS / 轻量服务器](#部署方式-b阿里云-ecs--轻量服务器)
    - [系统要求](#系统要求)
    - [一次性安装](#一次性安装)
    - [Nginx 站点](#nginx-站点)
    - [自动拉新 + 重建](#自动拉新--重建)
    - [阿里云特别提醒](#阿里云特别提醒)
  - [部署方式 C：树莓派自托管](#部署方式-c树莓派自托管)
    - [硬件 / 系统](#硬件--系统)
    - [步骤](#步骤)
    - [公网访问（选一个）](#公网访问选一个)
    - [自动拉新](#自动拉新)
    - [树莓派注意点](#树莓派注意点)
  - [可选 Secrets / 环境变量](#可选-secrets--环境变量)
  - [常用脚本](#常用脚本)
  - [安装新 Jekyll 插件](#安装新-jekyll-插件)
  - [License \& 来源](#license--来源)

---

## 写作格式

文章放 `_posts/`，文件名 `YYYY-MM-DD-slug.md`，最简 YAML 头：

```yaml
---
layout: mypost
title: 深入浅出了解生成模型-3：Diffusion 模型原理以及代码
categories: 生成模型              # 必填，影响面包屑/相关阅读/分类页
tags: [cv-backbone, 生成模型, diffusion model]
address: 武汉🏯
extMath: true                      # 有 LaTeX 公式时填 true，启用 MathJax
show: true                         # 是否在首页列表显示
images: true                       # 是否启用图片渲染（PhotoSwipe 相册）
stickie: true                      # 置顶
special_tag: 更新中                # 角标
image: https://s2.loli.net/...     # 可选：用作 og:image 社交分享卡片
description: |                     # 缺失会被 optim_post.py 用 LLM 自动补
  日常使用比较多的生成模型……
---
```

草稿放 `writing/` 或文件名带 `-TODO`（部署后 subtitle 会显示红色 `writing.....`）。

提交前推荐先跑 `python3 scripts/validate_front_matter.py`，校验所有文章的 front matter（同样在 GitHub Actions 的 `lint` job 里强制跑）。

---

## 本地预览

### Linux / WSL / macOS

```bash
# 装系统依赖（Ubuntu/Debian）
sudo apt install ruby-full build-essential ruby-bundler

# 用国内镜像加速 gem（可选）
gem sources --add https://mirrors.tuna.tsinghua.edu.cn/rubygems/ --remove https://rubygems.org/
bundle config mirror.https://rubygems.org https://mirrors.tuna.tsinghua.edu.cn/rubygems

# 装项目依赖
bundle install

# 启动本地 server
./blog.sh run 8080          # 等价于 bundle exec jekyll serve --watch --host=0.0.0.0 --port=8080

# 仅构建静态产物
./blog.sh build             # 输出到 dist/
```

打开 <http://127.0.0.1:8080> 看效果。

### Windows

推荐 WSL2 Ubuntu，命令同上。原生 Windows 也可以装 [RubyInstaller](https://rubyinstaller.org/) + DevKit，但 jekyll 在 Windows 下偶发文件锁问题，不推荐。

---

## 部署方式 A：GitHub Actions + GitHub Pages（推荐）

当前生产部署。push 到 `master` 自动触发 `.github/workflows/jekyll-gh-pages.yml`：

```
push master
  ↓
lint    (校验 _posts 的 YAML 头，缺 title/categories 直接 fail)
  ↓
build   (jekyll build --trace，缓存 .jekyll-cache 加速增量构建)
  ↓
deploy  (上线到 https://<user>.github.io/ 或自定义域名)
  ↓
notify  (调 submit_url.py，自动 ping Google / Bing / Baidu；secret 缺失则跳过)
```

非内容文件（`CLAUDE.md` / `README.md` / `optim_post.py` / `blog.sh` / `writing/**` 等）改动不会触发部署，配置见 workflow 中 `paths-ignore`。

### 一次性配置

1. **GitHub repo**：fork 本仓库或 clone 到自己的 `<user>.github.io` repo。
2. **Pages 来源**：repo Settings → Pages → Build and deployment → Source 选 **GitHub Actions**。
3. **自定义域名**（可选）：根目录有 `CNAME` 文件，写入你的域名（一行）；DNS 把域名 CNAME 到 `<user>.github.io.`，等 GitHub 自动签 SSL（约 5 分钟）。
4. **secrets（可选，启用搜索引擎自动推送）**：repo Settings → Secrets and variables → Actions → New secret。详见 [可选 Secrets](#可选-secrets--环境变量) 一节。

GitHub Actions 与依赖自动跟版本：`.github/dependabot.yml` 每周扫一次 action / gem 升级并开 PR。

---

## 部署方式 B：阿里云 ECS / 轻量服务器

适合：不愿用 GitHub Pages、需要国内访问加速、想把博客与其他服务并列在一台 VPS 上。

### 系统要求

- Ubuntu 22.04 LTS / 24.04 LTS（其他发行版命令稍有差异）
- 1 核 1G 起步，2 核 2G 推荐（jekyll build 时内存峰值 ~600MB）
- 公网 IP + 已备案的域名（阿里云域名服务）

### 一次性安装

```bash
# 1) 系统依赖
sudo apt update
sudo apt install -y ruby-full build-essential nginx git python3-pip certbot python3-certbot-nginx

# 2) 拉源码
sudo mkdir -p /var/www
sudo chown $USER:$USER /var/www
cd /var/www
git clone https://github.com/<你的用户名>/Big-Yellow-J.github.io.git blog
cd blog

# 3) 装 gem（首次约 3-5 分钟）
gem install bundler
bundle config set --local path 'vendor/bundle'   # 装到项目内，避免污染系统
bundle install

# 4) 构建一次
bundle exec jekyll build --destination=/var/www/blog/_site
```

### Nginx 站点

`/etc/nginx/sites-available/blog.conf`：

```nginx
server {
    listen 80;
    server_name www.example.com example.com;
    root /var/www/blog/_site;
    index index.html;

    # gzip
    gzip on;
    gzip_types text/plain text/css application/javascript application/json application/xml image/svg+xml;
    gzip_min_length 1024;

    # HTML 短缓存，资源长缓存（与 Jekyll 输出的 buildAt query string 配合）
    location ~* \.(css|js|png|jpg|jpeg|webp|gif|ico|svg|woff2?)$ {
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
    location / {
        try_files $uri $uri/ $uri.html =404;
    }
}
```

启用 + 申请 HTTPS：

```bash
sudo ln -s /etc/nginx/sites-available/blog.conf /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d www.example.com -d example.com    # Let's Encrypt 自动配 SSL
```

### 自动拉新 + 重建

新文章 push GitHub 后让服务器自动同步。`/etc/systemd/system/blog-rebuild.service`：

```ini
[Unit]
Description=Pull and rebuild blog

[Service]
Type=oneshot
WorkingDirectory=/var/www/blog
ExecStart=/bin/bash -c 'git pull --ff-only && bundle exec jekyll build --destination=/var/www/blog/_site'
User=ubuntu
```

加上定时器 `/etc/systemd/system/blog-rebuild.timer`：

```ini
[Unit]
Description=Rebuild blog every 5 minutes

[Timer]
OnBootSec=2min
OnUnitActiveSec=5min

[Install]
WantedBy=timers.target
```

启用：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now blog-rebuild.timer
```

更优的做法：在 GitHub repo 设 webhook，push 时回调服务器一个内网接口触发 `blog-rebuild.service`，省掉轮询。

### 阿里云特别提醒

- **安全组**：放行 80/443，关掉默认 22 暴露（改非默认端口 + key 登录）。
- **DNS**：解析记录类型选 A（域名 → ECS 公网 IP），云解析免费够用。
- **备案**：未备案的域名无法在国内 ECS 上 80/443 开放。可以临时用 8080 测试。
- **CDN**：阿里云 CDN/DCDN 套在 Nginx 上，源站只服务 CDN 回源 IP，进一步加速。

---

## 部署方式 C：树莓派自托管

适合：玩客、家庭服务器爱好者、跑各种内网服务的派友。

### 硬件 / 系统

- 推荐 Raspberry Pi 4B 4GB 或 Pi 5 4GB+
- 系统：Raspberry Pi OS Bookworm 64-bit（**必须 64-bit**，32-bit 装 Ruby gem 易出错）
- 散热壳 + 风扇，长期运行避免降频

### 步骤

跟[阿里云](#部署方式-b-阿里云-ecs--轻量服务器)基本一致，只有几处差异：

```bash
# 1) 系统依赖（zlib / libssl 在 aarch64 上需明确装）
sudo apt update
sudo apt install -y ruby-full build-essential nginx git python3-pip \
                    zlib1g-dev libssl-dev libffi-dev libyaml-dev

# 2) 拉源码、bundle install（aarch64 编译 nokogiri 等 native gem，首次约 20-40 分钟，请耐心）
git clone https://github.com/<你的用户名>/Big-Yellow-J.github.io.git ~/blog
cd ~/blog
bundle config set --local path 'vendor/bundle'
bundle install -j4

# 3) 构建
bundle exec jekyll build --destination=/srv/blog
```

Nginx 配置同阿里云那段，`root` 改成 `/srv/blog`。

### 公网访问（选一个）

派在内网，要让公网访问到博客，三种方式：

| 方式 | 优点 | 缺点 |
| --- | --- | --- |
| **Cloudflare Tunnel**（推荐）| 免费、不开端口、自带 CDN/HTTPS | 走 CF 节点，国内速度看运气 |
| **frp** 反代到云服务器 | 速度由 VPS 决定，可控 | 要额外一台 VPS |
| **IPv6 + DDNS**（如果运营商给公网 v6）| 直连零中转 | 仅 v6 客户端可访问 |

**Cloudflare Tunnel 简版**：

```bash
# 装 cloudflared
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb -o cf.deb
sudo dpkg -i cf.deb

# 登录 + 建隧道（按提示去浏览器授权）
cloudflared tunnel login
cloudflared tunnel create blog
cloudflared tunnel route dns blog www.example.com

# 配置 ~/.cloudflared/config.yml：
# tunnel: <隧道 ID>
# credentials-file: /home/pi/.cloudflared/<隧道 ID>.json
# ingress:
#   - hostname: www.example.com
#     service: http://localhost:80
#   - service: http_status:404

sudo cloudflared service install
```

CF Dashboard 会自动签 SSL，无需在派上跑 certbot。

### 自动拉新

systemd 配置与阿里云段相同。或者更省心地用 `cron`：

```bash
crontab -e
# 每 5 分钟拉一次新内容并重建
*/5 * * * * cd ~/blog && /usr/bin/git pull --ff-only && /home/pi/.local/share/gem/ruby/3.1.0/bin/bundle exec jekyll build --destination=/srv/blog >> ~/blog-rebuild.log 2>&1
```

### 树莓派注意点

- **SD 卡寿命**：长期写日志会磨损卡。`_site/` 建议放在挂载的 SSD（USB 3.0）上，否则两年一卡。
- **内存**：jekyll build 在 4GB 派上 OK；2GB 派要加 swap（`sudo dphys-swapfile setup` 配 2G）。
- **首次构建慢**：68 篇文章首次构建 ~90s；启用 `.jekyll-cache` 后增量 ~10s。

---

## 可选 Secrets / 环境变量

部署方式 A 在 GitHub repo Settings → Secrets and variables → Actions 中添加（缺失不报错，只是跳过对应步骤）：

| Secret | 作用 | 怎么拿 |
| --- | --- | --- |
| `GOOGLE_SA_JSON` | Google Indexing API 服务账号 JSON 全文 | [Google Cloud Console](https://console.cloud.google.com) → IAM → 服务账号 → 创建 → 启用 Indexing API → 下载 JSON → 整段贴进 secret value |
| `BING_INDEXNOW_KEY` | Bing IndexNow 32 位 key | [IndexNow](https://www.bing.com/indexnow) 生成；同时把 key 写到 `Bing_indexnow.txt` 部署到站点根目录 |
| `BAIDU_PUSH_TOKEN` | 百度普通收录 push token | [百度站长平台](https://ziyuan.baidu.com) → 普通收录 → API 推送 → 复制 token |

部署方式 B / C（自建服务器）则放在 shell 环境变量里，例如 `~/.bashrc`：

```bash
export GOOGLE_SA_JSON="$(cat /etc/blog/google_api.json)"   # 或直接保留 google_api.json 文件
export BING_INDEXNOW_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
export BAIDU_PUSH_TOKEN="xxxxxxxxxxxxxxxx"
export BLOG_SUBMIT_PROXY="http://127.0.0.1:7890"          # 本地需要代理才能访问 Google 时填
```

`optim_post.py` 用的另一组（写到 `./API_KEY.env`）：

```dotenv
SMMS_API_LIST=<sm.ms API token>
LLM_URL=<doubao/deepseek 的 OpenAI 兼容 base_url>
API_KEY=<对应 API key>
```

---

## 常用脚本

| 脚本 | 用途 |
| --- | --- |
| `./blog.sh run [port]` | 启动本地 server（默认 8080） |
| `./blog.sh build` | 构建到 `dist/` |
| `./blog.sh deploy` | 构建 + `cos-upload` 到腾讯云 COS + 刷 CDN |
| `./submit_github.sh` | 跑 `optim_post.py` 后 `git add . && commit && push` |
| `python3 optim_post.py` | 扫 `_posts/` + `writing/`，下载图 → 转 webp → 上传 sm.ms → 改写 markdown 为 `<img width height>` → LLM 生成 description |
| `python3 submit_url.py` | 读 `_site/static/xml/sitemap.xml` 后推 Google + Bing + 百度（凭证从环境变量读） |
| `python3 scripts/validate_front_matter.py` | 校验所有 `_posts/*.md` 的 YAML 头必填字段 |

---

## 安装新 Jekyll 插件

1. 编辑 `Gemfile` 增加 `gem "jekyll-xxxx"`。
2. 本地 `bundle install`。
3. 在 `_config.yml` 的 `plugins:` 下追加同名条目。
4. 重启 `jekyll serve` 看效果，无误后 commit 并 push。

GitHub Pages 默认只信任[白名单插件](https://pages.github.com/versions/)；当前流水线用 `actions/upload-pages-artifact` 走自定义构建，不在白名单限制内，任何插件只要本地 `bundle install` 通过都行。

---

## License & 来源

主题派生自 [TMaize/tmaize-blog](https://github.com/TMaize/tmaize-blog)（MIT）。
博客文章版权 © [HuangJie](https://www.big-yellow-j.top/pages/about.html) 保留所有权利。
