#!/usr/bin/env bash
# 一键部署私密 life feed：拉GitHub代码 → 依赖 → 自检 → systemd自启 → 热重启 → tailscale暴露
# 用法：在树莓派 app.py 所在目录执行  bash deploy.sh
# 注意：只更新代码(app.py/templates/static)，不动 moments.db 和 uploads/（你的数据）
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
SVC=life
PORT=8000
PY="$APP_DIR/venv/bin/python"

# GitHub 源（公开仓库匿名可拉；私有仓库把下面 URL 换成 https://<token>@... 形式）
REPO="Big-Yellow-J/Big-Yellow-J.github.io"
BRANCH="master"
SUBDIR="writing/BlogLife"

echo "==> 1/6 从 GitHub 拉取最新代码（只下需要的几个文件，不下整个仓库）"
# 国内直连 GitHub 慢：默认走镜像加速，失败回退直连。镜像挂了可换 https://ghproxy.net/ 等；置空 "" 则直连
GH_MIRROR="https://ghfast.top"
RAW="https://raw.githubusercontent.com/$REPO/$BRANCH/$SUBDIR"
# 需要同步的文件清单（以后 templates/static 下新增文件，记得在这里补一项）
FILES="app.py templates/feed.html templates/deny.html static/style.css static/app.js"
dl(){  # 下载单个文件到 $APP_DIR/$1：镜像优先、失败回退直连（%/ 去尾斜杠再补，避免 URL 少斜杠）
  mkdir -p "$(dirname "$APP_DIR/$1")"
  { [ -n "$GH_MIRROR" ] && curl -fsSL --connect-timeout 15 "${GH_MIRROR%/}/${RAW}/$1" -o "$APP_DIR/$1"; } \
    || curl -fsSL --connect-timeout 15 "${RAW}/$1" -o "$APP_DIR/$1"
}
fail=0
for f in $FILES; do dl "$f" && echo "   ✓ $f" || { echo "   ✗ $f 失败"; fail=1; }; done
[ "$fail" = 0 ] && echo "   代码已更新（moments.db、uploads/ 保留不动）" \
                || echo "   !! 部分文件拉取失败（私有仓库需 token / 网络问题），用本地现有代码继续"

MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple   # 清华源，国内不卡
echo "==> 2/6 安装依赖到 venv（清华源，显示完整 pip 过程）"
python3 -m venv "$APP_DIR/venv"
PIP="$APP_DIR/venv/bin/pip"
"$PIP" install --upgrade pip -i "$MIRROR"
"$PIP" install flask pillow pillow-heif -i "$MIRROR" || {
  echo "!! 依赖安装失败；若卡在 pillow-heif，先执行：sudo apt install -y libheif1 再重跑"; exit 1; }

echo "==> 3/6 自检"
"$PY" "$APP_DIR/app.py" test

echo "==> 4/6 写 systemd 服务 + 设开机自启"
sudo tee /etc/systemd/system/$SVC.service >/dev/null <<EOF
[Unit]
Description=private life feed
After=network-online.target
Wants=network-online.target
[Service]
User=$USER
WorkingDirectory=$APP_DIR
ExecStart=$PY $APP_DIR/app.py
Restart=always
[Install]
WantedBy=multi-user.target
EOF

echo "==> 5/6 热重启服务"
sudo systemctl daemon-reload
sudo systemctl enable "$SVC"
sudo systemctl restart "$SVC"
sleep 1
if systemctl is-active --quiet "$SVC"; then
  echo "   服务 active；本地探活：$(curl -sI http://localhost:$PORT | head -1)"
else
  echo "!! 启动失败，最近日志："; journalctl -u "$SVC" -n 20 --no-pager; exit 1
fi

echo "==> 6/6 tailscale 暴露到 tailnet"
tailscale serve --bg "$PORT" || {
  echo "!! tailscale serve 失败：确认已 tailscale up，且后台已开 MagicDNS + HTTPS Certificates"; exit 1; }
sleep 1
URL=$(tailscale serve status 2>/dev/null | grep -oE 'https://[^ ]+' | head -1 || true)

echo
echo "============================================"
echo " 部署完成 ✅  代码已拉取、依赖就绪、开机自启已设、服务已热重启"
echo " tailnet 访问： ${URL:-未取到，手动 tailscale serve status 查看}"
echo " 域名访问：    https://life.big-yellow-j.top （需 cloudflared 在跑）"
echo "============================================"
