#!/usr/bin/env python3
# 朋友圈式私密 life feed —— 身份由前置代理(Tailscale Serve 或 Cloudflare Access)注入
# 运行: python3 app.py            (Flask 内置服务器, 监听 127.0.0.1:8000)
#       LIFE_DEV=1 python3 app.py (本地调试：免鉴权放行 + 打印识别到的身份)
#       python3 app.py test       (自检)
# 结构: app.py 后端 + templates/ 模板 + static/ 样式与交互
import os, sqlite3, time, secrets, sys
from flask import (Flask, request, redirect, g, send_from_directory,
                   render_template, abort, jsonify)

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
OWNER = "2802311325@qq.com"                   # 仅用于未识别身份时的申请联系邮箱
DEV   = os.environ.get("LIFE_DEV") == "1"     # 本地调试：无身份头也放行，并打印识别到的身份
OK_EXT   = {"jpg", "jpeg", "png", "webp", "gif"}
HEIC_EXT = {"heic", "heif"}
if HEIF_OK:
    OK_EXT |= HEIC_EXT                         # 没装 pillow-heif 就直接拒 HEIC
MAX_MB = 25

os.makedirs(UP, exist_ok=True)
app = Flask(__name__)                          # 默认用同级 templates/ 与 static/
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024
if DEV:
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0   # 本地调试：static 不缓存，改完刷新即生效

def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB)
        g.db.execute("CREATE TABLE IF NOT EXISTS posts("
                     "id INTEGER PRIMARY KEY, author TEXT, text TEXT, img TEXT, ts INTEGER)")
        g.db.execute("CREATE TABLE IF NOT EXISTS comments("
                     "id INTEGER PRIMARY KEY, post_id INTEGER, author TEXT, text TEXT,"
                     " reply_to TEXT, ts INTEGER)")
        g.db.execute("CREATE TABLE IF NOT EXISTS likes("
                     "post_id INTEGER, author TEXT, ts INTEGER, PRIMARY KEY(post_id,author))")
        try:                                   # 老库迁移：comments 补 reply_to 列
            g.db.execute("ALTER TABLE comments ADD COLUMN reply_to TEXT")
        except sqlite3.OperationalError:
            pass
    return g.db

@app.teardown_appcontext
def _close(e):
    d = g.pop("db", None)
    if d: d.close()

def viewer():
    # 身份由前置代理注入：Cloudflare Access 或 Tailscale Serve。
    # app 只绑 127.0.0.1，外部到不了 → 这两个头不可伪造，可信。
    me = (request.headers.get("Cf-Access-Authenticated-User-Email")
          or request.headers.get("Tailscale-User-Login", "")).lower()
    if DEV:                                     # 本地调试：打印识别到的身份，无头也放行
        print("[life] viewer=%r  CF=%r  TS=%r" % (
            me, request.headers.get("Cf-Access-Authenticated-User-Email"),
            request.headers.get("Tailscale-User-Login")), flush=True)
        me = me or "local@dev"
    return me

def allowed(email):
    # 白名单已由前置代理(Cloudflare Access 名单 / tailnet 成员)把关；
    # 能带上已验证身份的人即放行，人人可看、可发图、可评论。
    return bool(email)

def ext_ok(name):
    return "." in name and name.rsplit(".", 1)[1].lower() in OK_EXT

def nick(email):
    return (email or "?").split("@")[0]

def save_upload(fs):
    """存单张上传图片到 UP。有 Pillow 时统一压缩转 webp(HEIC/大图都缩到最长边 2000px 省流量)，
    gif 保留动图、解析失败回退原样存。返回文件名，非法返回 None。"""
    name = fs.filename or ""
    if not ext_ok(name): return None
    ext = name.rsplit(".", 1)[1].lower()
    stem = secrets.token_hex(8)
    if HEIF_OK and ext != "gif":                  # gif 保留动图，其余统一压缩
        try:
            im = ImageOps.exif_transpose(Image.open(fs.stream))   # 修正手机旋转 EXIF
            im.thumbnail((2000, 2000))                            # 最长边 2000，等比缩放
            out = stem + ".webp"
            im.convert("RGB").save(os.path.join(UP, out), "WEBP", quality=82, method=6)
            return out
        except Exception:
            fs.stream.seek(0)                     # 坏图/解析失败 → 回退原样存，不 500
    out = stem + "." + ext
    fs.save(os.path.join(UP, out))
    return out

@app.route("/")
def index():
    me = viewer()
    if not allowed(me):
        return render_template("deny.html", me=me, owner=OWNER), 403
    d = db()
    cmap = {}                                  # 每帖评论（含回复对象）
    for pid, a, t, rt, ts in d.execute(
            "SELECT post_id,author,text,reply_to,ts FROM comments ORDER BY id"):
        cmap.setdefault(pid, []).append(dict(nick=nick(a), text=t, reply_to=rt, ts=ts))
    lcount = {pid: n for pid, n in
              d.execute("SELECT post_id,COUNT(*) FROM likes GROUP BY post_id")}
    myliked = {pid for (pid,) in
               d.execute("SELECT post_id FROM likes WHERE author=?", (me,))}
    rows = []
    for pid, a, t, i, ts in d.execute(
            "SELECT id,author,text,img,ts FROM posts ORDER BY id DESC"):
        imgs = [x for x in (i or "").split(",") if x]
        rows.append(dict(id=pid, author=a, nick=nick(a), text=t, imgs=imgs, ts=ts,
                         comments=cmap.get(pid, []),
                         likes=lcount.get(pid, 0), liked=(pid in myliked),
                         mine=(a == me or me == OWNER.lower())))
    return render_template("feed.html", me=me, rows=rows)

@app.route("/post", methods=["POST"])
def post():
    me = viewer()
    if not allowed(me): abort(403)
    text = (request.form.get("text") or "").strip()
    names = []                                  # 支持多图
    for f in request.files.getlist("img"):
        if f and f.filename:
            n = save_upload(f)
            if n is None: abort(400, "只支持图片")
            names.append(n)
    img = ",".join(names)
    if text or img:
        db().execute("INSERT INTO posts(author,text,img,ts) VALUES(?,?,?,?)",
                     (me, text, img, int(time.time())))
        db().commit()
    return redirect("/")

@app.route("/comment", methods=["POST"])
def comment():
    me = viewer()
    if not allowed(me): abort(403)
    pid = request.form.get("pid", type=int)
    text = (request.form.get("text") or "").strip()
    reply_to = (request.form.get("reply_to") or "").strip()    # 回复对象昵称，空=直接评论
    if pid and text:
        db().execute("INSERT INTO comments(post_id,author,text,reply_to,ts) VALUES(?,?,?,?,?)",
                     (pid, me, text, reply_to, int(time.time())))
        db().commit()
    return redirect("/")

@app.route("/like", methods=["POST"])
def like():
    me = viewer()
    if not allowed(me): abort(403)
    pid = request.form.get("pid", type=int)
    if not pid: abort(400)
    d = db()
    if d.execute("SELECT 1 FROM likes WHERE post_id=? AND author=?", (pid, me)).fetchone():
        d.execute("DELETE FROM likes WHERE post_id=? AND author=?", (pid, me))
        liked = False
    else:
        d.execute("INSERT INTO likes(post_id,author,ts) VALUES(?,?,?)", (pid, me, int(time.time())))
        liked = True
    d.commit()
    n = d.execute("SELECT COUNT(*) FROM likes WHERE post_id=?", (pid,)).fetchone()[0]
    return jsonify(liked=liked, count=n)

@app.route("/delete_post", methods=["POST"])
def delete_post():
    me = viewer()
    if not allowed(me): abort(403)
    pid = request.form.get("pid", type=int)
    d = db()
    row = d.execute("SELECT author,img FROM posts WHERE id=?", (pid,)).fetchone()
    if not row: abort(404)
    author, img = row
    if me != author and me != OWNER.lower():    # 只能删自己的；OWNER 可删任意
        abort(403)
    for n in (img or "").split(","):            # 删图片文件
        if n:
            try: os.remove(os.path.join(UP, n))
            except FileNotFoundError: pass
    d.execute("DELETE FROM posts WHERE id=?", (pid,))
    d.execute("DELETE FROM comments WHERE post_id=?", (pid,))
    d.execute("DELETE FROM likes WHERE post_id=?", (pid,))
    d.commit()
    return redirect("/")

@app.route("/uploads/<name>")
def media(name):
    if not allowed(viewer()): abort(403)
    return send_from_directory(UP, name)

def selftest():
    assert ext_ok("a.jpg") and ext_ok("A.PNG") and not ext_ok("x.exe") and not ext_ok("noext")
    assert nick("bob@x.com") == "bob"
    assert allowed("anyone@x.com") and not allowed("")     # 有身份即放行，空身份拒
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
        # Flask 内置服务器：私密低流量场景够用；threaded 让传图时不阻塞其他请求
        # 端口默认 8000，本地若被占可用 PORT=8001 临时换
        app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8000)), threaded=True)
