#!/usr/bin/env python3
# 朋友圈式私密 life feed —— 身份由前置代理(Tailscale Serve 或 Cloudflare Access)注入
# 运行: python3 app.py        (waitress, 监听 127.0.0.1:8000)
#       python3 app.py test   (自检)
# 结构: app.py 后端 + templates/ 模板 + static/ 样式与交互
import os, sqlite3, time, secrets, sys
from flask import (Flask, request, redirect, g, send_from_directory,
                   render_template, abort)

# HEIC(iPhone 默认格式)支持：装了 pillow-heif 才开启，上传时转成 webp
try:
    import pillow_heif
    from PIL import Image, ImageOps
    pillow_heif.register_heif_opener()
    HEIF_OK = True
except ImportError:
    HEIF_OK = False

BASE  = os.path.dirname(os.path.abspath(__file__))
DB    = os.path.join(BASE, "moments.db")
UP    = os.path.join(BASE, "uploads")
OWNER = "2802311325@qq.com"                   # ← 你的邮箱，始终可发可看
ALLOW = os.path.join(BASE, "allowed.txt")     # 每行一个被邀请邮箱；文件不存在 = 只有你
OK_EXT   = {"jpg", "jpeg", "png", "webp", "gif"}
HEIC_EXT = {"heic", "heif"}
if HEIF_OK:
    OK_EXT |= HEIC_EXT                         # 没装 pillow-heif 就直接拒 HEIC
MAX_MB = 25

os.makedirs(UP, exist_ok=True)
app = Flask(__name__)                          # 默认用同级 templates/ 与 static/
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024

def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB)
        g.db.execute("CREATE TABLE IF NOT EXISTS posts("
                     "id INTEGER PRIMARY KEY, author TEXT, text TEXT, img TEXT, ts INTEGER)")
        g.db.execute("CREATE TABLE IF NOT EXISTS comments("
                     "id INTEGER PRIMARY KEY, post_id INTEGER, author TEXT, text TEXT, ts INTEGER)")
    return g.db

@app.teardown_appcontext
def _close(e):
    d = g.pop("db", None)
    if d: d.close()

def viewer():
    # 身份由前置代理注入：Cloudflare Access 或 Tailscale Serve。
    # app 只绑 127.0.0.1，外部到不了 → 这两个头不可伪造，可信。
    return (request.headers.get("Cf-Access-Authenticated-User-Email")
            or request.headers.get("Tailscale-User-Login", "")).lower()

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

def nick(email):
    return (email or "?").split("@")[0]

def save_upload(fs):
    """存上传图片到 UP；HEIC 转 webp 并修正方向，其余原样存。返回文件名，非法返回 None。"""
    name = fs.filename or ""
    if not ext_ok(name): return None
    ext = name.rsplit(".", 1)[1].lower()
    stem = secrets.token_hex(8)
    if ext in HEIC_EXT:
        out = stem + ".webp"
        im = ImageOps.exif_transpose(Image.open(fs.stream))   # iPhone 照片常带旋转 EXIF
        im.convert("RGB").save(os.path.join(UP, out), "WEBP", quality=85)
    else:
        out = stem + "." + ext
        fs.save(os.path.join(UP, out))
    return out

@app.route("/")
def index():
    me = viewer()
    if not allowed(me):
        return render_template("deny.html", me=me, owner=OWNER), 403
    cmap = {}
    for pid, a, t, ts in db().execute(
            "SELECT post_id,author,text,ts FROM comments ORDER BY id"):
        cmap.setdefault(pid, []).append(dict(nick=nick(a), text=t, ts=ts))
    rows = [dict(id=pid, nick=nick(a), text=t, img=i, ts=ts, comments=cmap.get(pid, []))
            for pid, a, t, i, ts in
            db().execute("SELECT id,author,text,img,ts FROM posts ORDER BY id DESC")]
    return render_template("feed.html", me=me, rows=rows)

@app.route("/post", methods=["POST"])
def post():
    me = viewer()
    if not allowed(me): abort(403)
    text = (request.form.get("text") or "").strip()
    f = request.files.get("img")
    img = ""
    if f and f.filename:
        img = save_upload(f)
        if img is None: abort(400, "只支持图片")
    if text or img:
        db().execute("INSERT INTO posts(author,text,img,ts) VALUES(?,?,?,?)",
                     (me, text, img, int(time.time())))
        db().commit()
    return redirect("/")

@app.route("/comment", methods=["POST"])
def comment():
    me = viewer()
    if not allowed(me): abort(403)
    pid  = request.form.get("pid", type=int)
    text = (request.form.get("text") or "").strip()
    if pid and text:
        db().execute("INSERT INTO comments(post_id,author,text,ts) VALUES(?,?,?,?)",
                     (pid, me, text, int(time.time())))
        db().commit()
    return redirect("/")

@app.route("/uploads/<name>")
def media(name):
    if not allowed(viewer()): abort(403)
    return send_from_directory(UP, name)

def selftest():
    assert ext_ok("a.jpg") and ext_ok("A.PNG") and not ext_ok("x.exe") and not ext_ok("noext")
    assert nick("bob@x.com") == "bob"
    open(ALLOW, "a").close()
    assert allowed(OWNER.lower()) and not allowed("nobody@x.com") and not allowed("")
    if HEIF_OK:                              # 装了 heif 才测转码：内存造图 → 存盘应得 .webp
        import io
        from werkzeug.datastructures import FileStorage
        buf = io.BytesIO()
        Image.new("RGB", (8, 8), "red").save(buf, "PNG"); buf.seek(0)
        out = save_upload(FileStorage(stream=buf, filename="x.heic"))
        assert out.endswith(".webp") and os.path.exists(os.path.join(UP, out))
        os.remove(os.path.join(UP, out))
    print("ok (heif=%s)" % HEIF_OK)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        selftest()
    else:
        from waitress import serve
        serve(app, host="127.0.0.1", port=8000)
