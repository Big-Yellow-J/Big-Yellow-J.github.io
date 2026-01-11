---
layout: mypost
title: 深入浅出了解生成模型-7：加速生成策略
categories: 生成模型
extMath: true
images: true
address: 长沙🌷
show_footer_image: true
tags:
- 生成模型
- diffusion model
- qwen image
- z-image
show: true
stickie: true
description: 
---
## 扩散模型生成加速策略

### 加速框架进行加速
https://huggingface.co/docs/diffusers/main/en/optimization/cache_dit

### 模型量化进行加速生成
https://github.com/nunchaku-ai/nunchaku


### 其他
> ModelScope: [https://www.modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo/summary](https://www.modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo/summary)
> DEMO: [https://tongyi-mai.github.io/Z-Image-blog/](https://tongyi-mai.github.io/Z-Image-blog/)
> Paper：[https://arxiv.org/pdf/2511.22699](https://arxiv.org/pdf/2511.22699)

在Z-image模型中，目前（2025.12.30）只开源了文生图的模型，对于图像编码的模型权重没有开源。因此介绍一下其基本的模型结构如下：
![image.png](https://s2.loli.net/2025/12/30/R6tTQ5pAHuMW4q9.png)
在模型的使用上Z-image选择的模型如下：1、文本编码器选择的是：lightweight Qwen3-4B；2、VAE选择的模型是：Flux VAE；3、额外的视觉编码器：SigLIP 2；从模型结构上直接通过其给出的示意图就可以快速的了解其模型结构，后续主要关注其模型训练方式、如何实现快速图像生成等内容

## TwinFlow原理
> [https://arxiv.org/pdf/2512.05150](https://arxiv.org/pdf/2512.05150)

TwinFlow一个生成的优化算法可以在通过较少的NFS下快速的生成图像
![](https://s2.loli.net/2025/12/30/DJRs8bXIzW1epPm.png)

## 测试效果
值得注意的是下面测试过程中对于Z-image-Turpo使用的是9步生图，对于TwinFlow用的是4NFs

| prompt | 测试效果（Z-image-Turpo） | 测试效果（TwinFlow-Z-image） |
| --- | --- |---|
|一位古典中国美女，全身像，年龄约25岁，拥有精致的瓜子脸、柳叶眉、丹凤眼和樱桃小嘴，皮肤如凝脂般白皙光滑，乌黑长发盘成优雅的发髻饰以金簪和珠花，她穿着华丽的红色丝绸汉服，宽袖上绣满金丝凤凰和牡丹花纹，腰间系着流苏玉佩，层层叠叠的裙摆如云雾般飘逸，散发淡淡的古风韵味；表情温柔而神秘，嘴角微微上扬形成浅浅微笑，眼眸深邃如秋水般含情脉脉，透露出内心的宁静与智慧；动作优雅自然，她一只手轻持一把精雕玉扇遮掩半边脸庞，另一只手轻轻抚摸耳畔的发丝，姿势微微侧身，展示出S形曲线身材的柔美；氛围梦幻而诗意，背景是古代江南园林的竹林小径，周围环绕粉红樱花瓣随风飘落，薄雾缭绕在青石小桥和池塘边，夕阳余晖从树隙洒下金色光斑形成戏剧性光影，水面反射出她的倩影，空气中仿佛弥漫花香和鸟鸣，整体色调温暖柔和，高动态范围；超高细节，真实摄影风格，电影级光效，精细纹理如丝绸褶皱和皮肤毛孔，8K分辨率，黄金比例复杂构图，艺术大师级作品| ![image.png](https://s2.loli.net/2025/12/30/TlZfFwgnok7JE1V.png)|![](https://s2.loli.net/2025/12/30/TpxNseVEluf6XGb.png)|
|"一张逼真的年轻东亚女性肖像，位于画面中心偏左的位置，带着浅浅的微笑直视观者。她身着以浓郁的红色和金色为主的传统中式服装。她的头发被精心盘起，饰有精致的红色和金色花卉和叶形发饰。她的眉心之间额头上绘有一个小巧、华丽的红色花卉图案。她左手持一把仿古扇子，扇面上绘有一位身着传统服饰的女性、一棵树和一只鸟的场景。她的右手向前伸出，手掌向上，托着一个悬浮的发光的霓虹黄色灯牌，上面写着“TwinFlow So Fast”，这是画面中最亮的元素。背景是模糊的夜景，带有暖色调的人工灯光，一场户外文化活动或庆典。在远处的背景中，她头部的左侧略偏，是一座高大、多层、被暖光照亮的西安大雁塔。中景可见其他模糊的建筑和灯光，暗示着一个繁华的城市或文化背景。光线是低调的，灯牌为她的脸部和手部提供了显著的照明。整体氛围神秘而迷人。人物的头部、手部和上半身完全可见，下半身被画面底部边缘截断。图像具有中等景深，主体清晰聚焦，背景柔和模糊。色彩方案温暖，以红色、金色和闪电的亮黄色为主。"| ![image.png](https://s2.loli.net/2025/12/30/532tOKREhgUdlyi.png)| ![](https://s2.loli.net/2025/12/30/FbuJfZAwzTiEvVG.png)|

