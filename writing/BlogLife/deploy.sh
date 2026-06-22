#!/usr/bin/env bash
# 一键部署私密 life feed：依赖 → 自检 → systemd 开机自启 → 热重启 → tailscale 暴露 → 打印网址
# 用法：在树莓派 app.py 所在目录执行  bash deploy.sh
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
SVC=life
PORT=8000
PY="$APP_DIR/venv/bin/python"

MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple   # 清华源，国内不卡；过程全程打印便于排查
echo "==> 1/5 安装依赖到 venv ($APP_DIR/venv) —— 走清华源，显示完整 pip 过程"
python3 -m venv "$APP_DIR/venv"
PIP="$APP_DIR/venv/bin/pip"
"$PIP" install --upgrade pip -i "$MIRROR"
"$PIP" install flask waitress pillow pillow-heif -i "$MIRROR" || {
  echo "!! 依赖安装失败；若卡在 pillow-heif，先执行：sudo apt install -y libheif1 再重跑"; exit 1; }

echo "==> 2/5 自检"
"$PY" "$APP_DIR/app.py" test

echo "==> 3/5 写 systemd 服务 + 设开机自启"
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

echo "==> 4/5 热重启服务"
sudo systemctl daemon-reload
sudo systemctl enable "$SVC"
sudo systemctl restart "$SVC"
sleep 1
if systemctl is-active --quiet "$SVC"; then
  echo "   服务 active；本地探活：$(curl -sI http://localhost:$PORT | head -1)"
else
  echo "!! 启动失败，最近日志："; journalctl -u "$SVC" -n 20 --no-pager; exit 1
fi

echo "==> 5/5 tailscale 暴露到 tailnet"
tailscale serve --bg "$PORT" || {
  echo "!! tailscale serve 失败：确认已 tailscale up，且后台已开 MagicDNS + HTTPS Certificates"; exit 1; }
sleep 1
URL=$(tailscale serve status 2>/dev/null | grep -oE 'https://[^ ]+' | head -1 || true)

echo
echo "============================================"
echo " 部署完成 ✅  开机自启已设，代码已热重启"
echo " 访问地址（仅 tailnet 设备可打开）："
echo "   ${URL:-未取到，手动运行 tailscale serve status 查看}"
echo "============================================"
