# 用自己域名访问私密 life feed（big-yellow-j.top + Cloudflare Tunnel + Access）

目标：`https://life.big-yellow-j.top` 公网可达，但只有**白名单邮箱**能进；主站 `big-yellow-j.top`（GitHub Pages）继续正常跑。

两个事实：
- **没有公网 IP 没关系**：cloudflared 从树莓派主动出站连 Cloudflare（穿 NAT），专治没公网 IP。
- **前提是 DNS 托管在 Cloudflare**：免费版不支持只委派子域，必须把整个 `big-yellow-j.top` 的 NS 从阿里云迁到 Cloudflare（下面第 0 步，一次性）。

原理：cloudflared 出站隧道把 `life` 子域接到树莓派 `127.0.0.1:8000`；Cloudflare Access 在边缘验证邮箱，通过后注入 `Cf-Access-Authenticated-User-Email` 头给后端。app 绑 `127.0.0.1`，外部到不了 → 头不可伪造。

---

## 零、把 big-yellow-j.top 的 DNS 迁到 Cloudflare（一次性，主站不掉线）

**先记录现状**：阿里云控制台 → 云解析 DNS → `big-yellow-j.top` → 把当前所有解析记录截图/导出（A、CNAME、MX、TXT 等都要，后面在 CF 一条条对齐，别漏邮箱/验证记录）。

1. 注册 Cloudflare → **Add a site** 填 `big-yellow-j.top` → 选 **Free** 套餐。
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

# 3. 建隧道
cloudflared tunnel create life      # 记下 Tunnel ID 和凭证 json 路径

# 4. 自动写好 life 子域的 DNS（指向隧道，不用手动加记录）
cloudflared tunnel route dns life life.big-yellow-j.top
```

配置 `~/.cloudflared/config.yml`：
```yaml
tunnel: <上面的 Tunnel ID>
credentials-file: /home/huangjie/.cloudflared/<Tunnel ID>.json
ingress:
  - hostname: life.big-yellow-j.top
    service: http://127.0.0.1:8000
  - service: http_status:404
```

开机自启：
```bash
sudo cloudflared service install
sudo systemctl enable --now cloudflared
systemctl status cloudflared --no-pager      # active 即成功
```

此时 `https://life.big-yellow-j.top` 已能打开——**但还没鉴权，先别外传，下一步加门禁。**

---

## 二、加邮箱白名单门禁（Cloudflare Zero Trust → Access）

1. Cloudflare 后台 → **Zero Trust** → **Access** → **Applications** → **Add an application** → **Self-hosted**。
2. Application domain 填 `life.big-yellow-j.top`。
3. 加 **Policy**：Action **Allow** → Include → **Emails** → 填 `2802311325@qq.com` 和受邀朋友邮箱。
4. 登录方式默认 **One-time PIN**（邮箱收验证码），免费够用。

效果：非白名单的人连页面都到不了，CF 边缘直接拦；通过的人，CF 把邮箱写进 `Cf-Access-Authenticated-User-Email` 头，`app.py` 据此识别身份。

---

## 三、邀请新朋友

每加一人改**两处**（纵深防御）：
1. Cloudflare Access 的 Policy → Emails 加他邮箱（决定能否进站）。
2. 树莓派 `~/BlogLife/allowed.txt` 加一行他邮箱（app 第二道 + 决定显示）。

---

## 四、和原 Tailscale 入口的关系

- 两条入口可并存：tailnet 设备走 `https://<本机>.<tailnet>.ts.net`，其他人走 `https://life.big-yellow-j.top`，`app.py` 两种身份头都认。
- 只想用域名：可不再 `tailscale serve`，但 `life.service`（app 本身）必须一直跑。

---

## 安全底线

- `app.py` 必须保持绑 `127.0.0.1`，**绝不能改 0.0.0.0**——否则有人直连树莓派 8000 就能伪造身份头绕过门禁。
- 想更严：在 app 里校验 Cloudflare 签发的 `Cf-Access-Jwt-Assertion`（JWT 验签 + 校验 aud）。绑 127.0.0.1 已能防外部伪造，JWT 是加固项，按需再加。
