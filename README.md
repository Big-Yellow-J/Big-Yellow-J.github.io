主题来自：https://github.com/TMaize/tmaize-blog
# 头文件

```yaml
---
layout: mypost
title: 深入浅出了解生成模型-3：Diffusion模型原理以及代码
categories: 生成模型
address: 武汉🏯
tags: [cv-backbone,生成模型,diffusion model]
extMath: true
show_footer_image: true
show: true
images: true
description: 日常使用比较多的生成模型比如GPT/Qwen等这些大多都是“文生文”模型（当然GPT有自己的大一统模型可以“文生图”）但是网上流行很多AI生成图像，而这些生成图像模型大多都离不开下面三种模型：1、GAN；2、VAE；3、Diffusion Model。因此本文通过介绍这三个模型作为生成模型的入门。本文主要介绍第三类Diffusion Model
---
```

# 本地化部署

## Window本地化部署
Wins上直接在wsl下进行使用：
第一步：
```bash
sudo apt install ruby-full build-essential ruby-bundler
```

第二步：

```bash
gem sources --add https://mirrors.tuna.tsinghua.edu.cn/rubygems/ --remove https://rubygems.org/
gem sources -l
gem sources --clear-all
gem sources --update
export GEM_HOME="~/.gems"
gem install bundler
bundle config mirror.https://rubygems.org https://mirrors.tuna.tsinghua.edu.cn/rubygems
bundle config list
bundle config set path "~/.gems"
```

通过下面命令启动/编译项目（进入到网页文件中）
```bash
cd Big-Yellow-J.github.io/
bundle install
bundle exec jekyll serve --watch --host=127.0.0.1 --port=8080
bundle exec jekyll build --destination=dist
```
