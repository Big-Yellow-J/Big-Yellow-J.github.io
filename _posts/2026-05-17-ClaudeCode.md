---
layout: mypost
title: Claude Code安装使用及Agent相关架构问题分析
categories: agent
address: 北京🦞
extMath: true
show_footer_image: true
tags:
- agent
- claude code
description: 在Windows上通过WSL安装Claude Code是推荐方式，支持桌面端和终端安装，需配置环境变量并处理VPN节点或全局代理以解决安装报错。Skills分为项目目录和根目录，可自动触发（基于description字段）或手动触发（/命令联想），安装他人Skills可通过复制文件夹或执行npx命令。自定义Skills需在SKILL.md中定义name、description、when_to_use等YAML头部字段，结合prompt和脚本实现function
  calling机制。Agent Team模式允许任务拆分给多个子Agent并行执行，子Agent各自拥有独立上下文窗口。抓包分析显示，Claude Code通过缓存命中（复用相同block）降低计算开销，上下文工程采用分阶段压缩：先廉价清理工具输出（文件预览+本地路径缓存），再手动/自动执行LLM压缩（输出`<analysis>`+`<summary>`双层结果），最后将压缩后内容替换到上下文。基础命令包括/compact压缩对话、clear重置、/context查看占用、/sandbox沙盒隔离、claude
  --permission-mode auto强制自动判断权限，以及npx claude-task-viewer监控任务。插件与MCP赋予工具调用能力，如playwright实现浏览器操作。整体Agent架构需处理上下文不受限时的缓存设计与压缩策略。
---

本文详细介绍Claude Code在windows上安装使用，对于一般非计算机行业用户或者对底层技术了解兴趣不深的可以只看：1、第一部分Claude Code简单使用；2、Claude Code进阶使用；3、Claude Code底层原理中的Skills开发。
## Claude Code简单使用
### 安装环境准备
目前大部分的skill或者说Clade Code运行任务（比如说读取文件等）都会用到python，因此就需要去安装python，除此之外如果要去安装一些skills还需要权重npm等，因此：**在win电脑上更加建议直接使用wsl去搭建Claude Code**，[WSL安装方式](https://www.runoob.com/linux/windows-wsl-linux.html)，安装完毕之后其他命令就和linux安装命令相同，先去介绍基于WSL安装claude过程，执行如下命令：
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

### 基础命令
![20260518224707197](https://files.seeusercontent.com/2026/05/19/fj9N/20260518224707197.webp)
在终端中所有的命令可以直接输入 `/` 来进行执行比如说切换模型 `/model` 然后按不同箭头选择模型（enter 回车选择模型），或者直接使用 `@` 去访问文件等，简单使用过程中一般而言用的比较多的几个命令：1、`/compact`：去压缩对话，一般而言模型上下文有限去对上下文进行压缩
```cmd
> /compact
# 基本压缩：将历史对话总结为关键要点
> /compact 保留认证模块的变更细节和当前测试失败信息
# 定向压缩：指定需要保留的重点内容
```
2、`clear`：完全重置，清空当前会话的全部对话历史和上下文，回到初始状态；3、`/context`查看上下文使用，显示当前上下文的占用情况和分类统计。除了基础命令之外有些时候还需要去控制claude 在执行一些 **简单安全任务**（比如说去搜索整理文件这些任务）在启动时候可以直接开启 `claude --permission-mode auto`**强制自动判断需不要询问用户**（一定要让他执行可信任务，不然可能删除一些重要东西导致出现错误，可以先去跑通一个任务之后，后续再去实现类似任务可以直接开启强制自动判断）或者直接在对话系统里面按键 `shift+tab`进行切换（直接切换到 `auto mode on`即可）：
![20260520215009426](https://files.seeusercontent.com/2026/05/20/Y3oc/20260520215009426.webp)
而后再去执行 `/sandbox`（建立一个 **沙盒**隔离环境避免直接操作自己电脑导致误删文件等）
![20260520225631202](https://files.seeusercontent.com/2026/05/20/2pNt/20260520225631202.webp)
除此之外要**监控各类task任务**（比如说有些python自动脚本会到建立task）进行情况可以直接安装：`npx claude-task-viewer`然后就可以看到正在进行进程：
![20260520212935955](https://files.seeusercontent.com/2026/05/20/Cil9/20260520212935955.webp)
除此之外有些时候可能需要[一次性进行多个任务](https://code.claude.com/docs/zh-CN/agent-view)（比如说去搜索A相关事情、搜索B相关事情）可以执行`claude agents`，这样就可以快速进行多组对话进行切换（或者直接在对话系统里面直接按箭头左右也可以）
### 插件安装
比如说安装[用量检查插件](https://github.com/jarrodwatts/claude-hud/blob/main/README.zh.md)或者安装superpowers：`/plugin install superpowers@claude-plugins-official`
### MCP
*MCP简单理解为赋予Claude Code工具调用权限，比如说想让Claude Code去收索文书等*
比如说安装`playwright`（直接用自然语言描述你想做的事，而后调用对应的浏览器操作）：`claude mcp add playwright npx @playwright/mcp@latest`如果要卸载直接`claude mcp remove playwright`，安装完毕之后
![](https://files.seeusercontent.com/2026/05/20/7Ame/20260520163227086.webp)
可以看到MCP已经启动了，如果要去使用这个MCP直接 `使用 playwright 打开浏览器访问 https://www.big-yellow-j.top/posts/2026/04/20/torch-basic-distribute-1.html 并且检查还有什么需要补充的分布式训练方式，以及内容上还有什么不足` 去让Claude Code去调用playwright的MCP。
### Skills使用
*skills简单理解为一个可以复用的任务prompt（比如说创建PPT）不用每次都去重新写prompt直接通过skills进行复用即可*，如果要去找其他的Skills直接可以访问：[https://www.skills.sh/](https://www.skills.sh/)或者直接去安装[find-skills](https://www.skills.sh/vercel-labs/skills/find-skills)让其帮你自动去搜索一些skills，
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
#### 搜索自动化
一般而言在工作中可能需要去调研（比如说搜集论文、收集相关数据集等），推荐直接使用 [crawl4ai](https://github.com/unclecode/crawl4ai) 去自动搜索爬取内容，可以直接使用他的[skills](https://docs.crawl4ai.com/assets/crawl4ai-skill.zip)然后放到自己的目录下面，然后再终端中执行`claude --permission-mode auto`，然后输入自己要求：
![20260520223001394](https://files.seeusercontent.com/2026/05/20/Dts8/20260520223001394.webp)
最后输出结果（任务比较费时用来12min52s完成）
![20260520223828636](https://files.seeusercontent.com/2026/05/20/B9jx/20260520223828636.webp)
通过上面一轮对话处理下来token消耗
![20260520224434337](https://files.seeusercontent.com/2026/05/20/kRv5/20260520224434337.webp)
### Agent Team协作
一般而言在使用claude code进行对话时候，都是执行一个任务，在等待结果之后再去执行一个新的任务，比如说法律一个案子需要同时去收集案子以及法条然后再去对材料进行整理，这里就需要用到**agent team概念**，简单介绍Agent team概念参考里面对于subagents和agent teams之间对比[^4]：
![20260522220431712](https://files.seeusercontent.com/2026/05/22/V9ek/20260522220431712.png)
两者之间差异点在于subagents相当于一个老师将任务分配给几位学生然后学生给老师一个反馈，agent teams相当于老师给学生分配任务，学生之间还在不断交流，那么以[长沙货拉拉案](https://news.cctv.com/2021/09/30/ARTIUszTVNT8vTB7EhBbgvgT210930.shtml)为例，以一个法律从业者出发（本人非专业人员）去构建一个agent team来解决这个案子，**首先**本地创建一个文件夹然后执行claude code
```bash
mkdir CSHuolala # 创建文件夹
mkdir .claude   # 创建文件夹
echo '{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  }
}' > /home/hjie/ClaudeCode/CShuolala/.claude/settings.json # 相当于写入文件
 claude --permission-mode auto --teammate-mode in-process  # teammate-mode 控制显示模式

# 测试的提示词是
为货拉拉案法庭辩护材料准备材料辩护词
1、任务 A：证据审查与现场环境重构（由 Evidence-Agent 认领）
2、任务 B：刑法因果关系与无罪/罪轻辩护词起草（由 Criminal-Law-Agent 认领）
3、任务 C：被害人过错分析及同行案例检索（由 Case-Search-Agent 认领）
4、任务 D：最后对辩护词进行审查（有Lawer-Agent认领）
执行顺序上首先执行任务A而后并行执行B和C，最后执行D
辩护词合适标准：1、辩护词合乎法律文书书写；2、所有收集材料都在本地文件夹保存；3、法律必需都有自己出处
```
> 值得注意的是上面任务都是基于网络进行搜索，实际情况可能需要对自己具体材料（比如说收集的材料）进行处理，可能就需要用到其他skills/mcp工具，**在规划任务时候越详细越好**（更加好控制模型） 

![](https://files.seeusercontent.com/2026/05/22/r7Ak/20260522223624142.png)
可以看到目前之后一个 `Evidence-Agent` 在执行工作，可以直接通过键盘↓去进入 `@main @Evidence-Agent`然后键盘←→箭头进行跳装查看工作如何，或者直接通过`npx claude-task-viewer`然后进入对应网页查看如下：
![](https://files.seeusercontent.com/2026/05/22/9Ttd/20260522224446512.png)
那么在`Evidence-Agent`中执行效果如下：
![](https://files.seeusercontent.com/2026/05/22/cG0g/20260522224004375.png)
很明显看到执行 *争取审查+任务重构*符合我的任务A要求，A执行完毕就会进行BC比如说得到如下：
![](https://files.seeusercontent.com/2026/05/22/Na8o/20260522224706671.png)
可以看到另外两个agent也开始执行了，**最后**所有的整理的材料以及最后的书写得到的辩护词在[百度网盘](https://pan.baidu.com/s/1wzkkb8n_HqIcmLZqeynuYA?pwd=ve8w)总共是花费了大概4元（DeepSeek-4-pro）

## Claude Code工具开发
### Skills开发
> **最简单方法直接看别人怎么写然后进行仿写即可**

以crawl4ai提供的skills为例，参考这个例子**规范文书写作skills**（**大部分内容基于AI出发，不一定符合实际法律工作内容**），首先介绍如何去定义自己的skills而后去介绍在claude code中如何去定义自己的skills，比如说在[crawl4ai](https://docs.crawl4ai.com/assets/crawl4ai-skill.zip)文件中**主要是如下几个文件**[^5]（不同skills可能差异，比如crawl4ai可能还有references）：
```bash
my-skill/
├── SKILL.md           # 主要说明（必需）
├── template.md        # Claude 要填写的模板
├── reference.md       # 详细介绍API使用文档，比如说在scripts里面会提前告诉我的模型运行scripts需要哪些参数
├── examples/
│   └── sample.md      # 显示预期格式的示例输出
└── scripts/
    └── validate.sh    # Claude 可以执行的脚本
```
首先核心内容就集中在 `SKILL.md`文件中，简单介绍一下他的 **模板**描写方式：
```markdown
---
name: api-conventions
description: API design patterns for this codebase
---

When writing API endpoints:
- Use RESTful naming conventions
- Return consistent error formats
- Include request validation
```
**首先**对于上述模板中 `---` 之间内容表示提前告诉我的模型这个 `skills`的一些基本配置，他一般有如下字段：

| 字段 | 必需 | 描述 |
|:--:|:--:|:--|
| name | 否 | Skill 的显示名称。**如果省略，使用目录名称**。仅小写字母、数字和连字符（最多 64 个字符）。 |
| description | 推荐 | Skill 的功能以及何时使用它。Claude 使用它来决定何时应用该 skill。如果省略，使用 markdown 内容的第一段。前置关键用例：组合的 description 和 when_to_use 文本在技能列表中被截断为 1,536 个字符以减少上下文使用。 |
| when_to_use | 否 | 关于 Claude 何时应该调用该 skill 的额外上下文，例如触发短语或示例请求。附加到技能列表中的 description，并计入 1,536 个字符的上限。 |
| argument-hint | 否 | 自动完成期间显示的提示，指示预期的参数。示例：[issue-number] 或 [filename] [format]。 |
| arguments | 否 | 用于 skill 内容中 $name 替换的命名位置参数。接受空格分隔的字符串或 YAML 列表。名称按顺序映射到参数位置。 |
| disable-model-invocation | 否 | **设置为 true 以防止 Claude 自动加载此 skill**。用于你想使用 /name 手动触发的工作流。也防止该 skill 被预加载到 subagents 中。默认值：false。 |
| user-invocable | 否 | 设置为 false 以从 / 菜单中隐藏。用于用户不应直接调用的背景知识。默认值：true。 |
| allowed-tools | 否 | 当此 skill 处于活动状态时，Claude 可以使用而无需请求权限的工具。接受空格分隔的字符串或 YAML 列表。 |
| model | 否 | 当此 skill 处于活动状态时要使用的模型。覆盖适用于当前轮的其余部分，不保存到设置；会话模型在你的下一个提示时恢复。接受与 /model 相同的值，或 inherit 以保持活动模型。 |
| effort | 否 | 当此 skill 处于活动状态时的工作量级别。覆盖会话工作量级别。默认值：继承自会话。选项：low、medium、high、xhigh、max；可用级别取决于模型。 |
| context | 否 | 设置为 fork 以在分叉的 subagent 上下文中运行。 |
| agent | 否 | 当设置 context: fork 时要使用的 subagent 类型。 |
| hooks | 否 | 限定于此 skill 生命周期的 hooks。有关配置格式，请参阅 Skills 和代理中的 Hooks。 |
| paths | 否 | Glob 模式，限制何时激活此 skill。接受逗号分隔的字符串或 YAML 列表。设置后，Claude 仅在处理与模式匹配的文件时自动加载该 skill。使用与路径特定规则相同的格式。 |
| shell | 否 | 用于此 skill 中 !`command` 和 \`\`\`! 块的 shell。接受 bash（默认）或 powershell。设置 powershell 在 Windows 上通过 PowerShell 运行内联 shell 命令。需要 CLAUDE_CODE_USE_POWERSHELL_TOOL=1。 |

其实对于上述字段中最核心的就是：1、description：告诉模型这个skills都是干什么的；2、name：告诉模型你的skills叫什么（如果没有他就会用上层的文件夹名称），除此之外其他的也可以去注意一下使用（具体描述看上面的表格里面描述）：disable-model-invocation，在定义完skills中yaml之后就是关于第二部分，prompt书写（这部分其实就是相当于你将你一贯的prompt直接机械的告诉大模型，让他按照你这个内容去机械的进行操作），回到最上面去实现一个**规范文书写作skills**（很粗糙具体内容可能需要具体修改），首先明确我的skills都需要哪些内容：1、定义名称（`legaldocnorm`）；2、定义一些小的脚本工具，比如说对于文档（假设为docx文件）需要让模型去打开文件就需要一些脚本；3、其他。按照上述内容开始一个skills创作（全部基于wsl上，有些命令都是linux命令可以自然的转化到win上，比如创建文件夹）。
**第一步**、去构建一个所有文件“系统”以及大致SKILL.md文件（**头部信息推荐英文**，后续内容中英都行）最后所有文件见[Github链接](https://github.com/Big-Yellow-J/Big-Yellow-J.github.io/tree/master/code/Python/skills/legaldocnorm)
**第二步**、去构建我的脚本 `script`（不会写直接让AI帮你写即可），最后的script见[Github链接](https://github.com/Big-Yellow-J/Big-Yellow-J.github.io/tree/master/code/Python/skills/legaldocnorm)
**第三步**、去构建一个reference，因为法律文书在书写上比较规划，模型可能不知道具体如何书写可以简单给一个参考让模型规范输出（规范文本可以直接用最高法院提供模板），这里只提供两种规范文本供参考：1、[广州市海珠区人民法院——民事答辩状](https://www.gzhzcourt.gov.cn/news/45007004.cshtml)；2、[广州市海珠区人民法院——民事起诉状)](https://www.gzhzcourt.gov.cn/news/45007009.cshtml)，最后所有的reference见[Github链接](https://github.com/Big-Yellow-J/Big-Yellow-J.github.io/tree/master/code/Python/skills/legaldocnorm)
**skills底层原理**：还是一个function calling，所谓 **function calling**比如说：“北京今天天气如何？”输入模型模型（大模型本身只能输出文本不能去搜索网页等功能）通过分析用户文本输出结构化信息：`{"name": "get_weather", "arguments": {"date":xxx, ....}}` 而后通过结构化信息进行工具调用（比如说调用搜索天气相关的API进行天气检索）。因此虽然claude code中skills都是文本prompt，大模型在检索到要使用的skills之后通过分析skills中内容自动解析处需要进行操作，因此claude code中skills底层就是：`Prompt+Tool Description+ Few-shot examples+ Execution`
### MCP 开发

## Claude Code抓包
### 简单网络抓包
为了分析Claude Code中每一步系统层都在发生什么就需要最Claude Code进行抓包，具体过程如下，首先对环境进行配置（**基于wsl**）：
```bash
# 基于wsl
conda create -n mitm python=3.11
conda activate mitm
pip install mitmproxy
mitmweb --listen-host 0.0.0.0 --listen-port 9870 # 8080 为代理端口 cc走这里 8081 web ui看流量

# 新建窗口
export http_proxy=http://0.0.0.0:9870
export https_proxy=http://0.0.0.0:9870
export HTTP_PROXY=http://0.0.0.0:9870
export HTTPS_PROXY=http://0.0.0.0:9870
export ALL_PROXY=http://0.0.0.0:9870
export NODE_EXTRA_CA_CERTS=~/.mitmproxy/mitmproxy-ca-cert.pem
export SSL_CERT_FILE=~/.mitmproxy/mitmproxy-ca-cert.pem
claude --permission-mode auto
```
在启动完毕之后，可以看到终端
![](https://files.seeusercontent.com/2026/05/27/4hnK/20260527214050041.png)
而后直接去Claude code随便测试：1、你好；2、`/crawl4ai 搜索一下Yolo系列论文`，**直接去终端里面提供的url地址**然后可以直接`~c 200`（因为访问  https://api.anthropic.com/api/event_logging/v2/batch **可能会**有很多失败会显示400，因此重点看一下链接成功的），比如说通过解析：
![](https://files.seeusercontent.com/2026/05/27/5dcD/20260528001036233.png)
在输入第一段对话 `你好` 模型会输入很多内容如tools/skills描述等，但是这些内容绝大部分会“命中缓存”（观察deepseek中就有这种内容）[^8]
> 所谓命中缓存意思就是说：**这部分内容不用模型去重新计算可以直接复用**，这是因为大模型生成过程是prefill+decode，prefill阶段就是对我的prompt进行编码，比如说上面输入模型是一个结构化文档，在[vllm的cache逻辑中](https://www.big-yellow-j.top/posts/2026/03/15/vllm-3.html)会将输入token进行block处理（比如每个block存储n个token）当出现block相同时候就会复用结果（必须保证如：输入1：QWERASDF和输入2：QWERASDF此时可以命中缓存，但是如果输入1改成WQERASDF就不行，因此**为了更加大的命中缓存会直接将容易命中缓存输入放到模型输入前面如tools、system等**），而decode就是对prefill后内容开始解码一个token一个token进行输出，**即使如此还是建议如果skills不用就不要开启**

而后可以看模型输出部分think+输出：
![](https://files.seeusercontent.com/2026/05/27/Iro1/20260528001620963.png)
而后在第二部分对话中 `/crawl4ai 搜索一下Yolo系列论文` 首先可以看模型终端输出：
![](https://files.seeusercontent.com/2026/05/27/y4Wa/20260528002707498.png)
模型进行如下4次对话处理：
**1、任务重写**：从模糊到精确（第 1-3 轮）
第 1 轮：输入“搜索 Yolo 论文”，模型携带了 crawl4ai 的技能定义。
关键转折（你发现的 1 & 2 点）：**crawl4ai生成了两个极其精确的搜索关键词**：`YOLO series papers object detection survey 2024 2025`（针对综述）；`YOLOv1 to YOLOv12 paper list arxiv`（针对列表）
第 2-3 轮： 框架将这两个“进阶关键词”重新喂回给模型。这时，模型不再是处理你那个模糊的原始需求，而是拿到了已经被优化过的检索指令。目的： 这就是 Query Expansion（查询扩展）。Agent 知道直接搜“Yolo 论文”结果太乱，所以先把它拆解成“搜综述”和“搜最新版本列表”两个子任务。
**2、执行与回传**（第 4 轮）
动作： 携带了这两个精确关键词的对话被送入 web_search 工具。
结果： 搜索引擎返回了你看到的那一串包含 YOLOv1 到 YOLOv26 的 Arxiv 链接和详细摘要。
总结： 最终 DeepSeek 拿到这些被精准定位后的素材，进行最后的整合输出。
> *值得注意的是*,web_search返回内容还是通过模型/工具进行总结的，比如说一个搜索得到5-6结果模型进行总结即可

### 抓包分析功能
通过上面抓包方式可以进一步去分析claude code中其中功能比如说 `/compact`，`/resume` 等，**抓包分析resume**：终端执行 `/resume` 之后而后直接输入新的内容在claude code直接加载历史对话而后补充到上下文中比如所大致结构如下：
```json
{
    "model": "deepseek-v4-pro",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            历史用户输入对话信息
          },
          {
            新的对话信息
          }
        ],
        .....
      }
```
在新的对话信息里面也会直接加载启用的skills等以及后续模型的tools也会加载（和第一次对话加载信息相同只是多了一组用户历史信息），**但是**这里简单介绍一些claude code中是如何去缓存对话信息的在路径 `~/.claude` 中文件结构如下，对于每一组对话都会存储在 history.jsonl 中每组对话结构为 `{"display":"你是什么模型","pastedContents":{},"timestamp":1780404540849,"project":"/home/huangjie/MdiriCode/Claude","sessionId":"5186f729-1c6e-4698-99e6-7ad2cd667cde"}` 最后的sessionid对应每次对话的id信息，都会在 **sessions文件夹**（_主要是记录上下文对话信息_）中记录[^9]
![](https://files.seeusercontent.com/2026/06/02/5Ekt/20260602214220864.png)
除此之外在**projects文件夹**中还会记录更加多内容，其大致结构如下：
```bash
projects/
 ├── sessionid.jsonl   # 区别于上面 session 里面只记录对话信息，此sessionid.jsonl则是记录所有 “日志记录”，比如说  {"type":"permission-mode","permissionMode":"auto","sessionId":"b69647ef-a95b-428d-a0ac-530440c7853f"}
 ├── memory/
 ├── subagents/
 ├──── xxxx.meta.jsonl # 主要记录 subagets 具体信息比如说：{"agentType":"claude-code-guide","description":"Explain Claude Code compaction","toolUseId":"call_00_0UXP3yM5PsH4VZ2ZbA1c5929"}
 ├──── xxxx.jsonl      # 子agents产生对话信息
```
**主要介绍一些subagents**，在claude code中一般是 **主agent收到信息去分发给不同子agent，而后将子agent信息接收进行总结分析等**（具体构建等[见下面分析](#sub-agents服务)）。简单总结一些子agents：1、子代理只是向Anthropic发出的独立请求，每个请求都有自己的系统消息。 2、每个子代理都有自己的上下文窗口。它不会与其他子代理或主代理共享内存或聊天记录。 3、“主”Claude（委托代理）为子代理编写提示词，该提示词最终将成为子代理收到的第一条用户消息。 子代理完成任务后，会将其发现的摘要发送回主Claude，随后主Claude会尝试整合所有报告并对其进行解读。 在不断积累上下文之后就需要考虑对内容进行压缩，在claude中**压缩过程**有两种：1、直接 `/compact`；2、输入对应文本 `/compact xxxx`对于两种压缩方式通过抓包分析发现使用的提示词都是大致相同的（更加具体的压缩过程[见后续分析](#context-engine)）具体的**压缩提示词**如下：
```json
CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.

- Do NOT use Read, Bash, Grep, Glob, Edit, Write, or ANY other tool.
- You already have all the context you need in the conversation above.
- Tool calls will be REJECTED and will waste your only turn — you will fail the task.
- Your entire response must be plain text: an <analysis> block followed by a <summary> block.

Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.
This summary should be thorough in capturing technical details, code patterns, and architectural decisions that would be essential for continuing development work without losing context.

Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts and ensure you've covered all necessary points. In your analysis process:

1. Chronologically analyze each message and section of the conversation. For each section thoroughly identify:
   - The user's explicit requests and intents
   - Your approach to addressing the user's requests
   - Key decisions, technical concepts and code patterns
   - Specific details like:
     - file names
     - full code snippets
     - function signatures
     - file edits
   - Errors that you ran into and how you fixed them
   - Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
   - Note any security-relevant instructions or constraints the user stated (e.g., sensitive files or data to avoid, operations that must not be performed, credential or secret handling rules). These MUST be preserved verbatim in the summary so they continue to apply after compaction.
2. Double-check for technical accuracy and completeness, addressing each required element thoroughly.

Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Pay special attention to the most recent messages and include full code snippets where applicable and include a summary of why this file read or edit is important.
4. Errors and fixes: List all errors that you ran into, and how you fixed them. Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
6. All user messages: List ALL user messages that are not tool results. These are critical for understanding the users' feedback and changing intent. Preserve any security-relevant instructions or constraints verbatim so they remain in effect after compaction.
7. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.
8. Current Work: Describe in detail precisely what was being worked on immediately before this summary request, paying special attention to the most recent messages from both user and assistant. Include file names and code snippets where applicable.
9. Optional Next Step: List the next step that you will take that is related to the most recent work you were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's most recent explicit requests, and the task you were working on immediately before this summary request. If your last task was concluded, then only list next steps if they are explicitly in line with the users request. Do not start on tangential requests or really old requests that were already completed without confirming with the user first.
If there is a next step, include direct quotes from the most recent conversation showing exactly what task you were working on and where you left off. This should be verbatim to ensure there's no drift in task interpretation.

Here's an example of how your output should be structured:

<example>
<analysis>
[Your thought process, ensuring all points are covered thoroughly and accurately]
</analysis>

<summary>1. Primary Request and Intent:
   [Detailed description]

2. Key Technical Concepts:
   - [Concept 1]
   - [Concept 2]
   - [...]

3. Files and Code Sections:
   - [File Name 1]
      - [Summary of why this file is important]
      - [Summary of the changes made to this file, if any]
      - [Important Code Snippet]
   - [File Name 2]
      - [Important Code Snippet]
   - [...]

4. Errors and fixes:
    - [Detailed description of error 1]:
      - [How you fixed the error]
      - [User feedback on the error if any]
    - [...]

5. Problem Solving:
   [Description of solved problems and ongoing troubleshooting]

6. All user messages: 
    - [Detailed non tool use user message]
    - [...]

7. Pending Tasks:
   - [Task 1]
   - [Task 2]
   - [...]

8. Current Work:
   [Precise description of current work]

9. Optional Next Step:
   [Optional Next step to take]

</summary>
</example>

Please provide your summary based on the conversation so far, following this structure and ensuring precision and thoroughness in your response. 

There may be additional summarization instructions provided in the included context. If so, remember to follow these instructions when creating the above summary. Examples of instructions include:
<example>
## Compact Instructions
When summarizing the conversation focus on typescript code changes and also remember the mistakes you made and how you fixed them.
</example>

<example>
# Summary instructions
When you are using compact - please focus on test output and code changes. Include file reads verbatim.
</example>


REMINDER: Do NOT call any tools. Respond with plain text only — an <analysis> block followed by a <summary> block. Tool calls will be rejected and you will fail the task.
```
## Agent架构问题
上面无论式skills涉及还是tools使用其实都会带来很多Agent底层设计问题比如说：**1、上下文工程（Context Engine）** 随着用户不断对话那么对话历史就会不断变长，如果将对话历史全部塞到对话窗口里面就会导致上下文过长问题（导致模型可能丢失/处理不好）；2、tools如果没有命中怎么办？ 等等诸如此类问题，因此下面对Agent设计过程中会遇到问题以及架构设计上内容进行介绍
### Cache机制
最上面简单介绍一下缓存机制，通过相同cache缓存命中去选择复用，不过值得注意的是虽然在对话过程中（进行抓包分析）每次用户输入都是在最上面，但是模型会将这些信息进行编排 `tools + system+ messages`这种方式，保存缓存命中。 在不同模型中缓存命中方式存在些许差异，比如说：**1、DeepSeek缓存命中机制**：；**2、Claude缓存命中机制**
https://grok.com/c/b083cb5c-cf7a-40dd-8890-6998438f3d2a?rid=7f9573b4-b0ac-44f8-baa5-015f9f9caf5e
https://zhuanlan.zhihu.com/p/17239625983
### Context Engine
所谓上下文工程指的是，随着用户之间对话那么窗口token就肯定会超出模型限制（比如DeepSeek的1M），那么就需要对历史对话进行处理如压缩等（不过值得注意的是假设模型上下文窗口是1M在claude code中不会将所有上下文都用完才会去压缩，会设定阈值进行压缩，比如说达到80%自动进行压缩），以如下代码为例：
```python
from openai import OpenAI
client = OpenAI(api_key="xxx",base_url="https://api.deepseek.com")
messages = [
    {
        "role": "system",
        "content": "你是一个专业AI助手"
    }
]

while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    messages.append({
        "role": "user",
        "content": user_input
    })
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.7
    )
    assistant_reply = response.choices[0].message.content
    messages.append({
        "role": "assistant",
        "content": assistant_reply
    })
    print("User:", user_input)
    print("\nAssistant:", assistant_reply)
```
模拟进行对话最后得到输出文本:
```markdown
[{'content': '你是一个专业AI助手', 'role': 'system'},
 {'content': '我是小红来自美国，我现在问你一些问题请不多于100个字进行输出，请简单介绍一下北京和上海', 'role': 'user'},
 {'content': '北京是中国的首都，历史悠久，拥有故宫、长城等著名景点；上海是国际大都市，以现代金融和繁华商业闻名，外滩、东方明珠是标志性地标。',
  'role': 'assistant'},
 {'content': '北京和上海两个城市哪一个更加推荐我去？', 'role': 'user'},
 {'content': '这取决于你的旅行偏好。如果你喜欢历史文化，推荐北京，可以探索故宫、长城和胡同。如果你偏爱现代都市与商业氛围，上海更合适，有外滩、陆家嘴和时尚购物区。',
  'role': 'assistant'},
 {'content': '我想去看无人机表演推荐去哪个城市？深圳怎么样？', 'role': 'user'},
 {'content': '深圳是中国无人机表演最出色的城市之一，常有大疆等企业举办大型灯光秀。若想看科技感强的表演，深圳是首选。北京和上海也有相关活动，但深圳更具创新氛围。',
  'role': 'assistant'},
 {'content': '我是谁来自哪里？', 'role': 'user'},
 {'content': '你叫小红，来自美国。', 'role': 'assistant'}]
```
下面简单介绍几种常见的上下文压缩方式[^7]:
**1、Claude Code上下文压缩方案**：比如说在上面抓包分析Claude Code的压缩过程就可以看到他所使用的压缩提示词，但是实际过程中不可能全部只通过压缩提示词进行上下文压缩，参考对Claude Code的逆向分析[^10]简单总结其压缩方案：**优先使用廉价规则清理，再到轻量摘要，最后才进行昂贵的 LLM 总结**。_廉价处理过程_（**对话持续进行持续压缩避免占用上下文**）：将部分特定工具（比如说文件读取，那么就会获取很长的上下文）输出进行压缩，直接简化将文件内容缓存本地，并且只生成简单预览以及本地路径（通过预览如果需要复用直接通过路径再去获取文件信息）。 _模型压缩过程_（**手动/自动执行压缩命令执行**）：直接通过提示词去对所有的对话历史进行压缩（具体提示词见上面抓包得到的）。最后输出`<analysis>` + `<summary>` 两层[^12]，其中`<analysis>`：允许模型在内部把对话按时间线梳理一遍，减少遗漏，`<summary>`：对外提供结构化的最终压缩结果，便于机器再加工（比如客户端替换历史、写入本地存档）。压缩完毕之后就会涉及到信息复用，claude code中直接把总结内容复用到上下文中即可比如说：
![](https://files.seeusercontent.com/2026/06/03/n0hN/20260604000325877.png)

[//]: # (**2、Manus上下文压缩方案**：[^11])
[//]: # (### Sub Agents服务)
[//]: # (在上面抓包分析过程中发现一个subagents服务，简单去了解一些其整个过程)

## 参考
[^1]: [https://www.bilibili.com/video/BV1BFouBYERu/?spm_id_from=333.337.search-card.all.click&vd_source=881c4826193cfb648b5cdd0bad9f19f0](https://www.bilibili.com/video/BV1BFouBYERu/?spm_id_from=333.337.search-card.all.click&vd_source=881c4826193cfb648b5cdd0bad9f19f0)
[^2]: [https://www.cnblogs.com/youring2/p/20065433](https://www.cnblogs.com/youring2/p/20065433)
[^3]: [https://zhuanlan.zhihu.com/p/1933624323849032925](https://zhuanlan.zhihu.com/p/1933624323849032925)
[^4]: [https://code.claude.com/docs/zh-CN/agent-teams](https://code.claude.com/docs/zh-CN/agent-teams)
[^5]: [https://code.claude.com/docs/zh-CN/skills](https://code.claude.com/docs/zh-CN/skills)
[^6]: [https://www.big-yellow-j.top/posts/2026/03/15/vllm-3.html](https://www.big-yellow-j.top/posts/2026/03/15/vllm-3.html)
[^7]: [https://arxiv.org/pdf/2507.13334](https://arxiv.org/pdf/2507.13334)
[^8]: [https://code.claude.com/docs/zh-CN/prompt-caching](https://code.claude.com/docs/zh-CN/prompt-caching)
[^9]: [https://code.claude.com/docs/zh-CN/claude-directory#ce-global-projects](https://code.claude.com/docs/zh-CN/claude-directory#ce-global-projects)
[^10]: [Claude Code 上下文压缩算法深度分析](https://github.com/win4r/cc-notebook/blob/main/Claude_Code%E4%B8%8A%E4%B8%8B%E6%96%87%E5%8E%8B%E7%BC%A9%E7%AE%97%E6%B3%95%E6%B7%B1%E5%BA%A6%E5%88%86%E6%9E%90.md)
[^11]: [AI代理的上下文工程：构建Manus的经验教训](https://manus.im/zh-cn/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
[^12]: [https://zhuanlan.zhihu.com/p/2004602569171935364](https://zhuanlan.zhihu.com/p/2004602569171935364)