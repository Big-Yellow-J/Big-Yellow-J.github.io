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
description: Windows端Claude Code支持桌面端、终端两种安装路径，终端安装可通过CMD或PowerShell执行官方命令完成，遇网络连接报错可切换美国VPN节点、开启全局代理或虚拟网卡模式解决，安装完成后需配置系统环境变量，可搭配rtk工具优化token消耗。终端输入claude即可启动，输入/可调用操作命令，支持切换模型、安装superpowers等官方插件。内置技能分为项目级、全局级两类，可通过复制文件夹或命令行安装第三方可复用prompt，也支持自定义技能，适配自动、手动两种触发方式。
---

## Claude Code安装使用
### 安装环境准备
目前大部分的skills等都会用到python因此就需要去安装python等，因此需要去安装python，除此之外还需要权重npm等，因此：**在win电脑上更加建议直接使用wsl去搭建Claude Code**，[WSL安装方式](https://www.runoob.com/linux/windows-wsl-linux.html)，安装完毕之后其他命令就和linux安装命令相同，先去介绍基于WSL安装claude过程，执行如下命令：
> 对于配置DeepSeek：[配置DeepSeek订阅](#配置其他api)

```bash
# ========== 1. 基础系统依赖 ==========
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential python3 python3-pip python3-venv curl wget unzip git-all

# ========== 2. 安装 nvm + Node.js ==========
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.4/install.sh | bash
# 重新加载 shell 配置
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
nvm install 26
node -v
npm -v
npm config set registry https://registry.npmmirror.com

# ========== 3. 安装 Miniconda ==========
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86.sh
bash Miniconda3-latest-Linux-x86.sh -b   # -b 静默安装，跳过交互
~/miniconda3/bin/conda init bash
source ~/.bashrc

# ========== 4. 安装 Claude Code ==========
curl -fsSL https://claude.ai/install.sh | bash
# 确保 PATH 包含 ~/.local/bin（只添加一次）
grep -q 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc || \
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# ========== 5. 配置 DeepSeek API ==========
mkdir -p ~/.claude
cat > ~/.claude/settings.json << 'EOF'
{
  "env": {
    "ANTHROPIC_AUTH_TOKEN": "sk-你的DeepSeek-API-Key",
    "ANTHROPIC_BASE_URL": "https://api.deepseek.com/anthropic",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "deepseek-v4-flash",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "deepseek-v4-pro",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "deepseek-v4-pro",
    "ANTHROPIC_MODEL": "deepseek-v4-pro"
  },
  "autoUpdatesChannel": "latest",
  "theme": "dark"
}
EOF

# ========== 6. 安装 rtk ==========
curl -fsSL https://raw.githubusercontent.com/rtk-ai/rtk/refs/heads/master/install.sh | sh
source ~/.bashrc
rtk init --global
```

对于WSL后续使用直接在终端里面输入 `wsl` 然后输入 `claude` 即可
### 安装与卸载
> ⭐**在win电脑上更加建议直接使用wsl去搭建Claude Code**，[WSL安装方式](https://www.runoob.com/linux/windows-wsl-linux.html)，安装完毕之后其他命令就和linux安装命令相同

**以win电脑为例**按照官方过程直接输入命令即可，对于**桌面端安装**直接访问[链接](https://code.claude.com/docs/en/desktop)然后安装即可，对于**终端安装**参考[链接](https://code.claude.com/docs/zh-CN/terminal-guide)首先安装 [`git`](https://git-scm.com/install/windows)然后直接终端（win+r 然后输入 cmd）执行安装即可：`curl -fsSL https://claude.ai/install.cmd -o install.cmd && install.cmd && del install.cmd`或者直接：
![20260518221435773](https://files.seeusercontent.com/2026/05/19/vg0U/20260518221435773.webp)
打开 PowerShell然后输入安装命令：`irm https://claude.ai/install.ps1 | iex`
**处理报错**，如果遇到如下报错：
![20260518215550368](https://files.seeusercontent.com/2026/05/19/2xMq/20260518215550368.webp)
**解决措施**：（理论上）直接将VPN节点换到美国即可
**处理报错**，如果遇到报错 `Failed to fetch version from https://downloads.claude.ai/claude-code-releases/latest: ECONNREFUSED`
![20260518215850658](https://files.seeusercontent.com/2026/05/19/Nag5/20260518215850658.webp)
**解决措施**：将VPN开启虚拟网卡模式或者直接将VPN开全局代理即可
最后终端中出现如下界面表示安装完成
![20260518220850516](https://files.seeusercontent.com/2026/05/19/dBy8/20260518220850516.webp)
值得注意的是里面提到：`Native installation exists.....`，这是因为没有**配置好系统环境变量**，直接`win+r`然后输入 `sysdm.cpl`
![20260518221208684](https://files.seeusercontent.com/2026/05/19/F4ug/20260518221208684.webp)
点击确认即可完成环境变量配置处理。然后终端直接输入 `claude`
**卸载过程**就比较简单直接去删除对应文件即可，**为了节约token**选择直接[安装](https://github.com/rtk-ai/rtk/blob/develop/README_zh.md)`rtk`，首先[下载对应文件](https://github.com/rtk-ai/rtk/releases/download/dev-0.41.0-rc.227/rtk-x86_64-pc-windows-msvc.zip)并且解压（**记住解压的路径**），解压完成之后将对应文件添加到环境变量中（过程和上面配置claude code环境变量相同），然后 `rtk.exe init --global`（如果不行就直接输入`D:\ClaudeCode\rtk.exe init --global`，前面的路径为你解压路径）运行之后就可以开启节约token，值得注意的是如果环境变量添加了，终端输入 `rtk` 没有效果可以直接去 `C:\Users\hjie\.claude\settings.json` 里面修改配置，将里面的hook修改为
![20260519211105815](https://files.seeusercontent.com/2026/05/19/4bCo/20260519211105815.webp)
### 配置其他API
以[DeepSeek](https://platform.deepseek.com/usage)配置过程为例，打开目录 `C:\Users\hjie\.claude` 然后
![20260518222937445](https://files.seeusercontent.com/2026/05/19/bG7z/20260518222937445.webp)
```json
{
  "env": {
    "ANTHROPIC_AUTH_TOKEN": "sk-",
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
![20260518223208225](https://files.seeusercontent.com/2026/05/19/qy1O/20260518223208225.webp)

### 简单使用
![20260518224707197](https://files.seeusercontent.com/2026/05/19/fj9N/20260518224707197.webp)
在终端中所有的命令可以直接输入 `/` 来进行执行比如说切换模型 `/model` 然后按不同箭头选择模型（enter 回车选择模型），或者直接使用 `@` 去访问文件等，简单使用过程中一般而言用的比较多的几个命令：1、`/compact`：去压缩对话，一般而言模型上下文有限去对上下文进行压缩
```cmd
> /compact
# 基本压缩：将历史对话总结为关键要点
> /compact 保留认证模块的变更细节和当前测试失败信息
# 定向压缩：指定需要保留的重点内容
```
2、`clear`：完全重置，清空当前会话的全部对话历史和上下文，回到初始状态；3、`/context`查看上下文使用，显示当前上下文的占用情况和分类统计。除了基础命令之外有些时候还需要去控制claude 在执行一些 **简单安全任务**（比如说去搜索整理文件这些任务）在启动时候可以直接开启 `claude --permission-mode auto`**强制自动判断需不要询问用户**（一定要让他执行可信任务，不然可能删除一些重要东西导致出现错误，可以先去跑通一个任务之后，后续再去实现类似任务可以直接开启强制自动判断）：
![20260520215009426](https://files.seeusercontent.com/2026/05/20/Y3oc/20260520215009426.webp)
而后再去执行 `/sandbox`（建立一个 **沙盒**隔离环境避免直接操作自己电脑导致误删文件等）
![20260520225631202](https://files.seeusercontent.com/2026/05/20/2pNt/20260520225631202.webp)
除此之外要**监控各类task任务**（比如说有些python自动脚本会到建立task）进行情况可以直接安装：`npx claude-task-viewer`然后就可以看到正在进行进程：
![20260520212935955](https://files.seeusercontent.com/2026/05/20/Cil9/20260520212935955.webp)
### 插件安装
比如说安装[用量检查插件](https://github.com/jarrodwatts/claude-hud/blob/main/README.zh.md)或者安装superpowers：`/plugin install superpowers@claude-plugins-official`
### MCP
*MCP简单理解为赋予Claude Code工具调用权限，比如说想让Claude Code去收索文书等*
比如说安装`playwright`（直接用自然语言描述你想做的事，而后调用对应的浏览器操作）：`claude mcp add playwright npx @playwright/mcp@latest`如果要卸载直接`claude mcp remove playwright`，安装完毕之后
![](https://files.seeusercontent.com/2026/05/20/7Ame/20260520163227086.webp)
可以看到MCP已经启动了，如果要去使用这个MCP直接 `使用 playwright 打开浏览器访问 https://www.big-yellow-j.top/posts/2026/04/20/torch-basic-distribute-1.html 并且检查还有什么需要补充的分布式训练方式，以及内容上还有什么不足` 去让Claude Code去调用playwright的MCP。

### Skills
> 如果要去找其他的Skills直接可以访问：[https://www.skills.sh/](https://www.skills.sh/)

*skills简单理解为一个可以复用的任务prompt（比如说创建PPT）不用每次都去重新写prompt直接通过skills进行复用即可*
> 安装前简单了解一下skills有“两个目录”：**1、项目目录**（那么你的skills就只在这个项目起到效果），比如直接在你的文件夹里面进行打开就会访问项目目录比如说 `D:\ClaudeCode\.claude`；**2、根目录**（所有项目都可以用到这个skills），这个就是你的claude code安装目录比如说：`C:\Users\hjie\.claude`

**第一种、安装他人skills**[^1]有如下几种：**1、复制文件夹**直接将别人skills所有内容复制到skills目录下即可，**2、直接下载**，以获取微信文章为例直接访问[地址](https://www.skills.sh/)然后搜索`wechat-article-extractor`在得到安装命令`npx skills add https://github.com/freestylefly/wechat-article-extractor-skill --skill wechat-article-extractor`在你的终端进行安装即可，安装完过程中可能要选择是 project 还是 global 根据提示进行选择即可
> 需要安装npm，安装方式参考：[链接](https://www.cnblogs.com/liushunli/p/18663191)

安装完毕之后skills触发有两种：**1、自动进行触发**，比如在上述skills中有description字段
![20260519214202838](https://files.seeusercontent.com/2026/05/19/Yb8y/20260519214202838.webp)
当你在claude中输入 `提取微信文章：https://mp.weixin.qq.com/s/8axsDd-vY247Nd3oPZ9_zQ` 他就会自动触发这个skills进行处理
![20260519214401207](https://files.seeusercontent.com/2026/05/19/c7Nx/20260519214401207.webp)
**2、手动触发**，可以直接在 claude中输入 '/we' 会在下面触发联想，然后通过箭头选择自己需要内容直接按 `tab` 补全命令
![20260519214750902](https://files.seeusercontent.com/2026/05/19/9Wma/20260519214750902.webp)
**第二种、自定义skills**：直接去看别人skills怎么写的然后进行修改即可，或者直接让claude code写skills
#### 自动化搜索skills
一般而言在工作中可能需要去调研（比如说搜集论文、收集相关数据集等），推荐直接使用 [crawl4ai](https://github.com/unclecode/crawl4ai) 去自动搜索爬取内容，可以直接使用他的[skills](https://docs.crawl4ai.com/assets/crawl4ai-skill.zip)然后放到自己的目录下面，然后再终端中执行`claude --permission-mode auto`，然后输入自己要求：
![20260520223001394](https://files.seeusercontent.com/2026/05/20/Dts8/20260520223001394.webp)
最后输出结果（任务比较费时用来12min52s完成）
![20260520223828636](https://files.seeusercontent.com/2026/05/20/B9jx/20260520223828636.webp)
通过上面一轮对话处理下来token消耗
![20260520224434337](https://files.seeusercontent.com/2026/05/20/kRv5/20260520224434337.webp)

## 参考
[^1]: [https://www.bilibili.com/video/BV1BFouBYERu/?spm_id_from=333.337.search-card.all.click&vd_source=881c4826193cfb648b5cdd0bad9f19f0](https://www.bilibili.com/video/BV1BFouBYERu/?spm_id_from=333.337.search-card.all.click&vd_source=881c4826193cfb648b5cdd0bad9f19f0)
[^2]: [https://www.cnblogs.com/youring2/p/20065433](https://www.cnblogs.com/youring2/p/20065433)
[^3]: [https://zhuanlan.zhihu.com/p/1933624323849032925](https://zhuanlan.zhihu.com/p/1933624323849032925)