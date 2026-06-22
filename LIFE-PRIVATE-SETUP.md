# 私密「生活」相册 — 树莓派 + 博客配置

目标:
- 「生活」页**公开内容照旧可直接看**;同一页加一个「私密相册」入口,私密部分**仅 tailnet 可访问**,走邀请制。
- 朋友受邀后能**直接传图 + 写一句话**,自动成为一条朋友圈式动态,**图片存树莓派本地**。
- 朋友想进私密区 → 发邮件申请 → 你把他加进 tailnet。

设计取舍:不装 NAS、不用 Immich/Postgres。一个 ~90 行 Flask 单文件 + SQLite + 本地 `uploads/` 目录就够。
鉴权不自己写——`tailscale serve` 会把访客的**已验证邮箱**写进请求头,应用只绑 `127.0.0.1`(外部到不了,头伪造不了),按白名单放行即可。

---

## 架构一句话

```
朋友(tailnet 设备) ──HTTPS──> tailscale serve ──127.0.0.1:8000──> Flask 应用
                                  │(注入 Tailscale-User-Login 头)      │
                                  └ 证书/HTTPS 自动                     └ SQLite + 本地 uploads/
公开访客 ──> GitHub Pages「生活」页:公开内容照看 + 一个"私密相册(需邀请)"入口
```

两层门禁:不在 tailnet → 根本连不上,只看到公开博客的申请页;在 tailnet 但不在白名单 → 应用显示申请页。

---

## 一、树莓派配置

### 1. 系统 + Tailscale
```bash
# Raspberry Pi OS Lite 64-bit 即可
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
tailscale status          # 记下本机名,完整地址形如 raspberrypi.<你的tailnet>.ts.net
```
在 Tailscale 管理后台(login.tailscale.com)→ DNS → 打开 **MagicDNS** 和 **HTTPS Certificates**(serve 的 HTTPS 要用)。

### 2. 放应用
```bash
sudo apt install -y python3-pip
pip3 install flask waitress
mkdir -p ~/life && cd ~/life
# 把下面的 app.py 存到 ~/life/app.py，改第 11 行 OWNER 为你的邮箱
python3 app.py test       # 自检,打印 ok 即逻辑正常
```

`~/life/app.py`:
```python
#!/usr/bin/env python3
# 朋友圈式私密 life feed —— 仅 tailnet 可访问，身份由 Tailscale Serve 注入
# 运行: python3 app.py        (waitress, 监听 127.0.0.1:8000)
#       python3 app.py test   (自检)
import os, sqlite3, time, secrets, sys
from flask import (Flask, request, redirect, g, send_from_directory,
                   render_template_string, abort)

BASE  = os.path.dirname(os.path.abspath(__file__))
DB    = os.path.join(BASE, "moments.db")
UP    = os.path.join(BASE, "uploads")
OWNER = "you@xxx.top"                          # ← 改成你的邮箱，始终可发可看
ALLOW = os.path.join(BASE, "allowed.txt")     # 每行一个被邀请邮箱；文件不存在 = 只有你
OK_EXT = {"jpg", "jpeg", "png", "webp", "gif"}
MAX_MB = 25

os.makedirs(UP, exist_ok=True)
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024

def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB)
        g.db.execute("CREATE TABLE IF NOT EXISTS posts("
                     "id INTEGER PRIMARY KEY, author TEXT, text TEXT, img TEXT, ts INTEGER)")
    return g.db

@app.teardown_appcontext
def _close(e):
    d = g.pop("db", None)
    if d: d.close()

def viewer():
    # 只绑 127.0.0.1，外部到不了 → 这个头只可能由 tailscale serve 写入，可信
    return request.headers.get("Tailscale-User-Login", "").lower()

def allowed(email):
    if not email: return False
    if email == OWNER.lower(): return True
    try:
        with open(ALLOW, encoding="utf-8") as f:
            return email in {l.strip().lower() for l in f if l.strip()}
    except FileNotFoundError:
        return False

def ext_ok(name):
    return "." in name and name.rsplit(".", 1)[1].lower() in OK_EXT

FEED = """<!doctype html><html lang=zh><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>生活</title>
<style>
body{max-width:600px;margin:0 auto;padding:16px;background:#f5f5f5;color:#222;
 font-family:system-ui,"PingFang SC","Microsoft YaHei",sans-serif}
form,.card{background:#fff;padding:12px;border-radius:10px;margin-bottom:14px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
textarea{width:100%;border:1px solid #ddd;border-radius:6px;padding:8px;font:inherit;box-sizing:border-box}
.row{display:flex;gap:8px;align-items:center;margin-top:8px}
button{background:#3366cc;color:#fff;border:0;border-radius:6px;padding:8px 16px;cursor:pointer}
.meta{font-size:12px;color:#888;margin-bottom:6px}.text{white-space:pre-wrap;line-height:1.6}
.card img{max-width:100%;border-radius:8px;margin-top:8px;display:block}
</style><h2>🦄 生活</h2>
<form method=post action=/post enctype=multipart/form-data>
<textarea name=text rows=2 placeholder="说点什么…"></textarea>
<div class=row><input type=file name=img accept=image/*><button>发布</button></div></form>
{% for r in rows %}<div class=card><div class=meta>{{ r.author }} · {{ r.when }}</div>
{% if r.text %}<div class=text>{{ r.text }}</div>{% endif %}
{% if r.img %}<img src="/uploads/{{ r.img }}" loading=lazy>{% endif %}</div>
{% else %}<p style="color:#888">还没有内容，发第一条吧。</p>{% endfor %}
<p style="font-size:12px;color:#aaa">登录身份：{{ me }}</p></html>"""

DENY = """<!doctype html><html lang=zh><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>私密内容</title>
<style>body{max-width:480px;margin:60px auto;padding:24px;text-align:center;color:#333;
 font-family:system-ui,sans-serif}</style><h2>🔒 私密生活相册</h2>
<p>你的身份 <b>{{ me or '未识别' }}</b> 暂未获授权。</p>
<p>想看的话，发邮件到 <a href="mailto:{{ owner }}">{{ owner }}</a> 申请，我会把你加进来。</p></html>"""

@app.route("/")
def index():
    me = viewer()
    if not allowed(me):
        return render_template_string(DENY, me=me, owner=OWNER), 403
    rows = [dict(author=a, text=t, img=i,
                 when=time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)))
            for a, t, i, ts in
            db().execute("SELECT author,text,img,ts FROM posts ORDER BY id DESC")]
    return render_template_string(FEED, me=me, rows=rows)

@app.route("/post", methods=["POST"])
def post():
    me = viewer()
    if not allowed(me): abort(403)
    text = (request.form.get("text") or "").strip()
    f = request.files.get("img")
    img = ""
    if f and f.filename:
        if not ext_ok(f.filename): abort(400, "只支持图片")
        img = secrets.token_hex(8) + "." + f.filename.rsplit(".", 1)[1].lower()
        f.save(os.path.join(UP, img))     # 文件名随机生成，不信任客户端名
    if text or img:
        db().execute("INSERT INTO posts(author,text,img,ts) VALUES(?,?,?,?)",
                     (me, text, img, int(time.time())))
        db().commit()
    return redirect("/")

@app.route("/uploads/<name>")
def media(name):
    if not allowed(viewer()): abort(403)
    return send_from_directory(UP, name)

def selftest():
    assert ext_ok("a.jpg") and ext_ok("A.PNG") and not ext_ok("x.exe") and not ext_ok("noext")
    open(ALLOW, "a").close()
    assert allowed(OWNER.lower()) and not allowed("nobody@x.com") and not allowed("")
    print("ok")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        selftest()
    else:
        from waitress import serve
        serve(app, host="127.0.0.1", port=8000)
```

### 3. 开机自启 + 暴露到 tailnet
```bash
# systemd 守护应用
sudo tee /etc/systemd/system/life.service >/dev/null <<EOF
[Unit]
After=network-online.target
[Service]
User=$USER
WorkingDirectory=$HOME/life
ExecStart=/usr/bin/python3 $HOME/life/app.py
Restart=always
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl enable --now life

# 用 tailnet HTTPS 暴露(自动证书 + 注入身份头)，重启后保留
tailscale serve --bg 8000
tailscale serve status     # 看到 https://<本机>.<tailnet>.ts.net 即成功
```
访问地址就是上面那个 `.ts.net`。只有 tailnet 设备能打开。

### 4. 备份(别省)
```bash
# 每天把 数据库+图片 同步到第二块盘/另一台机，单点存储=等着丢
(crontab -l 2>/dev/null; echo "0 3 * * * rsync -a ~/life/uploads ~/life/moments.db /mnt/backup/") | crontab -
```

---

## 二、博客配置(GitHub Pages 这边)

**现有公开「生活」内容、菜单都不动。** 只在 `pages/life.html` 现有内容的**最上方**插入一个私密入口,让这页变成"公开 + 私密"两块:

```html
<div class="life-access">
  <span class="life-access-tab is-active">🦄 公开</span>
  <a class="life-access-tab" href="https://<本机>.<tailnet>.ts.net">🔒 私密相册（需邀请）</a>
  <p class="life-access-note">
    私密内容仅邀请制：发邮件到 <a href="mailto:you@xxx.top">you@xxx.top</a> 申请，
    受邀加入后用上方链接访问（未受邀点击打不开属正常）。
  </p>
</div>
<!-- ↓ 下面原有的公开时间线保持不变 -->
```

可选样式(加到 `static/css/page.css`):
```css
.life-access{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin:0 0 18px}
.life-access-tab{padding:5px 14px;border-radius:16px;border:1px solid var(--bk-card-border);
 font-size:14px;text-decoration:none;color:var(--bk-text)}
.life-access-tab.is-active{background:var(--bk-accent,#3366cc);color:#fff;border-color:transparent}
.life-access-note{flex-basis:100%;margin:0;font-size:12px;color:var(--bk-muted)}
```

把 `<本机>.<tailnet>.ts.net` 换成第一步 `tailscale serve status` 看到的地址,`you@xxx.top` 换成你的邮箱。
私密地址直接写在公开页没风险:它是不可路由的 tailnet 地址,非受邀者点了只会超时打不开。

可选:本文件在公开仓库可见(无密钥,占位符而已);介意就在 `_config.yml` 的 `exclude:` 加一行 `- LIFE-PRIVATE-SETUP.md`。

---

## 三、邀请流程

1. 朋友按申请页发邮件给你。
2. Tailscale 后台 → **Invite external users / Share**,把他拉进你的 tailnet;他装 Tailscale 登录加入。
3. (可选,想只放部分 tailnet 成员看)在树莓派 `~/life/allowed.txt` 加一行他的邮箱。不加这文件 = 只有你 OWNER 能用;加了文件 = 仅名单内 + 你。
4. 把 `https://<本机>.<tailnet>.ts.net` 发给他,他在 tailnet 设备上打开即可看 + 发。

---

```
跳过:NAS/Samba、Immich/数据库服务、自建登录(Tailscale 身份头白嫖)、Jekyll 重建管线(应用本身就是 life)。
需要时再加:朋友多到要点赞/评论/相册 → 上 Pixelfed;图多到一块盘装不下 → 再谈存储扩展。
底线:OWNER 改成你的邮箱;uploads 必须有备份;app 只绑 127.0.0.1 不要改成 0.0.0.0(改了身份头就能被伪造)。
```
