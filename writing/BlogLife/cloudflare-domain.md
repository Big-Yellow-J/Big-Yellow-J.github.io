# 用自己域名访问私密 life feed（big-yellow-j.top + Cloudflare Tunnel + Access）

目标：`https://life.big-yellow-j.top` 公网可达，但只有**白名单邮箱**能进；主站 `big-yellow-j.top`（GitHub Pages）继续正常跑。

两个事实：
- **没有公网 IP 没关系**：cloudflared 从树莓派主动出站连 Cloudflare（穿 NAT），专治没公网 IP。
- **前提是 DNS 托管在 Cloudflare**：免费版不支持只委派子域，必须把整个 `big-yellow-j.top` 的 NS 从阿里云迁到 Cloudflare（下面第 0 步，一次性）。

原理：cloudflared 出站隧道把 `life` 子域接到树莓派 `127.0.0.1:8000`；Cloudflare Access 在边缘验证邮箱，通过后注入 `Cf-Access-Authenticated-User-Email` 头给后端。app 绑 `127.0.0.1`，外部到不了 → 头不可伪造。

---

## 零、把 big-yellow-j.top 的 DNS 迁到 Cloudflare（一次性，主站不掉线）

**先记录现状**：阿里云控制台 → 云解析 DNS → `big-yellow-j.top` → 把当前所有解析记录截图/导出（A、CNAME、MX、TXT 等都要，后面在 CF 一条条对齐，别漏邮箱/验证记录）。

1. 注册 Cloudflare → **Add a site（添加站点）** 填 `big-yellow-j.top` → 选 **Free** 套餐。
2. CF 会自动扫描现有记录，但常扫不全。**对照阿里云导出逐条核对补齐**。主站 GitHub Pages 必须有这几条（apex 用 A 记录）：

   | 类型 | 名称 | 值 | 代理状态 |
   |---|---|---|---|
   | A | `@` | `185.199.108.153` | 先 DNS only(灰云) |
   | A | `@` | `185.199.109.153` | 先 DNS only |
   | A | `@` | `185.199.110.153` | 先 DNS only |
   | A | `@` | `185.199.111.153` | 先 DNS only |
   | CNAME | `www` | `big-yellow-j.github.io` | 先 DNS only |

   > 主站记录**先全设 DNS only(灰云)**，行为和现在阿里云一致，确认主站没问题后再考虑开橙云代理（开代理后 SSL 模式选 Full）。其余阿里云里的 MX/TXT 等照搬。

3. CF 会给你两个 nameserver（形如 `xxx.ns.cloudflare.com`）。到**域名注册商**后台（`.top` 域名在哪注册就在哪改；若注册商就是阿里云，则在阿里云域名管理→DNS 修改）把 NS 改成 CF 这两个。
4. 等 NS 生效（几十分钟~最长 48h），Cloudflare 后台该域名状态变 **Active**。
5. 验证主站没掉：浏览器开 `https://big-yellow-j.top` 正常即可。`CNAME` 文件（仓库根，内容 `big-yellow-j.top`）和 GitHub 端 custom domain 都不用动。

NS 生效后再做下面两步。

---

## 一、装 cloudflared 建隧道（树莓派）

```bash
# 1. 安装（树莓派 arm64）
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64 \
  -o /tmp/cloudflared && sudo install /tmp/cloudflared /usr/local/bin/cloudflared

# 2. 浏览器授权，选中 big-yellow-j.top
cloudflared tunnel login

# 3. 建隧道（记下输出的 Tunnel ID 和凭证 json 路径）
cloudflared tunnel create life

# 4. 自动写好 life 子域的 DNS（指向隧道，不用手动加记录）
cloudflared tunnel route dns life life.big-yellow-j.top
```

配置 `~/.cloudflared/config.yml`（把 Tunnel ID 换成上一步实际值）：
```yaml
tunnel: ee6173eb-9b88-4a16-ac3c-70dabc6f8d09
credentials-file: /home/huangjie/.cloudflared/ee6173eb-9b88-4a16-ac3c-70dabc6f8d09.json
ingress:
  - hostname: life.big-yellow-j.top
    service: http://127.0.0.1:8000
  - service: http_status:404
```

开机自启（service install 以 root 运行，要把配置放到 /etc 它才找得到）：
```bash
sudo mkdir -p /etc/cloudflared
sudo cp ~/.cloudflared/config.yml /etc/cloudflared/config.yml
sudo cloudflared service install
sudo systemctl enable --now cloudflared
systemctl status cloudflared --no-pager      # active 即成功
```

此时 `https://life.big-yellow-j.top` 已能打开——**但还没鉴权，先别外传，下一步加门禁。**

---

## 二、加邮箱白名单门禁（Cloudflare Zero Trust → Access）

### 2.1 第一次进 Zero Trust（只做一次）

1. Cloudflare 主控制台左侧菜单点 **「Zero Trust」**，进入独立的 Zero Trust 仪表板。
2. 首次进入要设置 **团队域名（Team domain）**：随便起名，如 `huangjie` → 得到登录门户 `huangjie.cloudflareaccess.com`（朋友登录时会看到这地址，正常）。
3. 选套餐：选 **Free（免费版）**，50 人以内够用。可能要求绑卡验证，**免费版不扣费**。

### 2.2 添加应用（自托管 → 公共DNS）

4. 左侧 **「访问 Access」→「应用程序 Applications」→「添加应用程序 Add an application」**。
5. 类型选 **「自托管 Self-hosted」**。
6. 接着会让你选**目标类型**，四个选项里选 **「公共DNS」**：

   | 选项 | 用途 | 选吗 |
   |---|---|---|
   | **公共DNS**（Public DNS） | 有公开域名的网站（走 Tunnel/CDN） | ✅ **选这个** |
   | 私有目标（Private targets） | 用 WARP 客户端访问内网私有 IP | ❌ |
   | Workers | 保护 Cloudflare Workers 应用 | ❌ |
   | 服务身份验证（Service Auth） | 机器对机器 token 鉴权（无人登录） | ❌ |

   > 原则：**哪个选项能让你填公开域名 `life.big-yellow-j.top`，就是它**，界面措辞再变也认这个。

7. 填应用信息：
   - **应用程序名称**：`life`
   - **会话持续时间 Session Duration**：`24 小时`（随意）
   - **应用程序域**：子域 `life` + 域 `big-yellow-j.top`（若是单框直接填 `life.big-yellow-j.top`），路径留空
   - 点 **下一步 Next**。

### 2.3 加白名单策略（谁能进）

8. 进入「添加策略 Add policy」：
   - **策略名称**：`friends`
   - **操作 Action**：选 **「Allow 允许」**
   - **配置规则（Include / 包括）**：
     - **选择器 Selector**：选 **「Emails 电子邮件」**
     - **值 Value**：填 `2802311325@qq.com`，再把朋友邮箱逐个加进去
   - 点 **下一步 Next**。

### 2.4 登录方式（默认即可）

9. 最后一页直接 **保存 / 添加应用程序**。
   - 默认启用 **One-time PIN（一次性 PIN）**：访问者输邮箱 → 收验证码邮件 → 输入即进，**免费、零配置**。
   - 想加 Google 登录：**「设置 Settings」→「身份验证 Authentication」** 里加（可选）。

### 中英术语对照

| 中文 | 英文 |
|---|---|
| 访问 | Access |
| 应用程序 | Applications |
| 自托管 | Self-hosted |
| 公共DNS | Public DNS |
| 策略 | Policy |
| 操作 → 允许 | Action → Allow |
| 电子邮件 | Emails |
| 一次性 PIN | One-time PIN |

效果：非白名单的人连页面都到不了，CF 边缘直接拦；通过的人，CF 把邮箱写进 `Cf-Access-Authenticated-User-Email` 头，`app.py` 据此识别身份。

---

## 三、邀请新朋友

`app.py` 已放宽为「凡通过前置鉴权、带得上身份的人都可看可发可评」，所以邀请只需**一处**：

- Cloudflare Zero Trust → **访问 Access → 应用程序 → life → 策略 → Emails**，加一行他的邮箱 → 保存。

树莓派 app 那边**不用动**（不再依赖 `allowed.txt`）。

---

## 四、和原 Tailscale 入口的关系

- 两条入口可并存：tailnet 设备走 `https://<本机>.<tailnet>.ts.net`，其他人走 `https://life.big-yellow-j.top`，`app.py` 两种身份头都认。
- 走 tailnet 入口要 `tailscale serve --bg 8000` 在树莓派跑着；只想用域名可以不开 serve，但 `life.service`（app 本身）必须一直跑。

---

## 五、常见问题排查

**访问域名提示 `That account does not have access`**
= Cloudflare Access 白名单已生效，但你登录用的邮箱不在 Allow 策略里。
- 核对：**Access → 应用程序 → life → 策略 → Emails** 里的邮箱，和你**登录时输入的邮箱**必须完全一致。
- 典型：白名单是 `2802311325@qq.com`，你却用 Gmail 登录 → 被拒。解决：用名单内邮箱登录，或把登录邮箱加进 Emails 保存。
- 改完等十几秒生效；重试前在 `huangjie.cloudflareaccess.com` 退出登录或用无痕窗口，避开旧会话缓存。

**本地 `curl -sI http://localhost:8000` 返回 403**
= 正常。直连本地没经过 Cloudflare/Tailscale，没有身份头 → 被挡。能返回 403 而不是 500，反而说明 app 和模板都正常。真正测试请用浏览器走域名。

**域名能开但页面很丑 / 报 500**
= 树莓派上 `~/BlogLife` 缺新版文件。新版 `app.py` 用 `render_template`，必须同时有 `templates/`（feed.html、deny.html）和 `static/`（style.css、app.js）。
- 检查：`ls ~/BlogLife`（要有 `app.py templates/ static/`）。
- 查错：`journalctl -u life -n 20 --no-pager`，`TemplateNotFound` 就是缺模板。

**部署/更新 app 后**：`sudo systemctl restart life` 重启生效。

---

## 六、查看树莓派部署日志

systemd 托管的两个服务都用 `journalctl` 看：

```bash
# app 本身（life 服务）
journalctl -u life -f                   # 实时跟踪（Ctrl+C 退出）
journalctl -u life -n 50 --no-pager     # 最近 50 行
journalctl -u life --since "10 min ago" # 最近 10 分钟

# cloudflared 隧道
journalctl -u cloudflared -f
journalctl -u cloudflared -n 50 --no-pager
```

服务状态：`systemctl status life --no-pager` / `systemctl status cloudflared --no-pager`。

**本地调试（免鉴权 + 打印身份）**：
```bash
LIFE_DEV=1 python3 app.py    # 浏览器开 http://localhost:8000 直接进，
                             # 控制台/日志打印每次请求识别到的身份(CF / Tailscale 头)
```
`LIFE_DEV=1` 仅本地临时用；systemd 不设这个变量，线上仍需身份头，行为不变。

---

## 安全底线

- `app.py` 必须保持绑 `127.0.0.1`，**绝不能改 0.0.0.0**——否则有人直连树莓派 8000 就能伪造身份头绕过门禁。
- 想更严：在 app 里校验 Cloudflare 签发的 `Cf-Access-Jwt-Assertion`（JWT 验签 + 校验 aud）。绑 127.0.0.1 已能防外部伪造，JWT 是加固项，按需再加。
