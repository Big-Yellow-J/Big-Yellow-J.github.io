---
layout: mypost
title: Claude Code安装使用
categories: agent
address: 北京🦞
extMath: true
show_footer_image: true
tags:
- agent
- claude code
---

## Claude Code安装使用
### 安装与卸载
以win电脑为例按照官方过程直接输入命令即可，对于**桌面端安装**直接访问[链接](https://code.claude.com/docs/en/desktop)然后安装即可，对于**终端安装**参考[链接](https://code.claude.com/docs/zh-CN/terminal-guide)首先安装 [`git`](https://git-scm.com/install/windows)然后直接终端（win+r 然后输入 cmd）执行安装即可：`curl -fsSL https://claude.ai/install.cmd -o install.cmd && install.cmd && del install.cmd`或者直接：
![20260518221435773](https://files.seeusercontent.com/2026/05/18/b7wO/20260518221435773.png)
打开 PowerShell然后输入安装命令：`irm https://claude.ai/install.ps1 | iex`
**处理报错**，如果遇到如下报错：
![20260518215550368](https://files.seeusercontent.com/2026/05/18/C1fm/20260518215550368.png)
**解决措施**：（理论上）直接将VPN节点换到美国即可
**处理报错**，如果遇到报错 `Failed to fetch version from https://downloads.claude.ai/claude-code-releases/latest: ECONNREFUSED`
![20260518215850658](https://files.seeusercontent.com/2026/05/18/gF2v/20260518215850658.png)
**解决措施**：将VPN开启虚拟网卡模式或者直接将VPN开全局代理即可
最后终端中出现如下界面表示安装完成
![20260518220850516](https://files.seeusercontent.com/2026/05/18/H5eq/20260518220850516.png)
值得注意的是里面提到：`Native installation exists.....`，这是因为没有配置好系统环境变量，直接`win+r`然后输入 `sysdm.cpl`
![20260518221208684](https://files.seeusercontent.com/2026/05/18/5ylT/20260518221208684.png)
点击确认即可完成环境变量配置处理。然后终端直接输入 `claude`
**卸载过程**就比较简单直接输入：`Remove-Item -Recurse -Force "C:\Users\hjie\.local\bin"` 然后 `Remove-Item -Recurse -Force "C:\Users\hjie\.claude"`
### 配置其他API
以[DeepSeek](https://platform.deepseek.com/usage)配置过程为例，打开目录 `C:\Users\hjie\.claude` 然后
![20260518222937445](https://files.seeusercontent.com/2026/05/18/J6ng/20260518222937445.png)
```json
{
  "env": {
    "ANTHROPIC_AUTH_TOKEN": "sk-e6c7bae330f7478fb707354e03779108",
    "ANTHROPIC_BASE_URL": "https://api.deepseek.com/anthropic",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "deepseek-v4-flash",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "deepseek-v4-pro",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "deepseek-v4-pro",
    "ANTHROPIC_MODEL": "deepseek-v4-pro"
  },
  "autoUpdatesChannel": "latest",
  "theme": "dark"
}
```
在配置完毕之后，打开 Powershell然后执行 `claude`
![20260518223208225](https://files.seeusercontent.com/2026/05/18/xjG8/20260518223208225.png)
### 简单使用
![20260518224707197](https://files.seeusercontent.com/2026/05/18/rNe2/20260518224707197.png)
在终端中所有的命令可以直接输入 `/` 来进行执行比如说切换模型 `/model` 然后按不同箭头选择模型（enter 回车选择模型）