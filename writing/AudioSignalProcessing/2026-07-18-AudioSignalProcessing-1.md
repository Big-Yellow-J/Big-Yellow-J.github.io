---
layout: mypost
title: 音频型号处理————VAD基本原理以及常用模型
categories: 音频信号处理
address: 北京🎑
extMath: true
mermaid: true
special_tag: 更新中
show_footer_image: true
tags:
- 音频信号处理
- VAD
description: 
---
# VAD模型简要概述与原理
## VAD模型简要概述
VAD 的核心任务是 **判断一段音频中哪些时间段有人说话哪些时间段没有人说话** ，也就是区分语音和非语音（或静音）部分。想象一下，在一个有背景噪音的环境中，VAD 就像一个智能“守门人”，它能准确识别什么时候有人在说话，什么时候是纯粹的环境噪音或沉默。
## VAD模型工作原理
**音频数字化过程**：声音通过震动转化为电压的模拟信号，但是电脑只能处理数字信号（01等），因此在数字化信号处理过程中通过ADC（模数转换器）进行处理，其主要是如下几件事情：**1、采样**每隔固定时间测量一次声音幅度，常用指标 *Hz*表示每秒采样多少声音副本（常用的8Hz、16Hz）；**2、量化**采样后数值结果是一批连续点，但是计算机存储数据范围有限如8bit就只有 `2^8=256`因此需要将数值进行量化处理如 `0.123456V-->37`，常用指标为 *位深*如8bit、16bit等；实际代码去理解音频处理过程（*以模型训练中数据加载进行理解*），在[测试音频](https://datasets-server.huggingface.co/cached-assets/edinburghcstr/ami/--/46f28f2503e2ec48f8867a84eef356c70476beab/--/ihm/train/50/audio/audio.wav?Expires=1784359125&Signature=svrspzw2YN9rWfF4o6DYYXf96iKc7qB~zVVwZ6XpODe4leVgGVb9tNW8wooOguZVR-mPeGVnWiUhOGfkmwCZTJFYHV4q98U4hGJasBxkN1jA-wXpvUp0WrQoN173pIrjhHvwnRmPmhlqvNOhGu22HYCsOId1PJDw2-1Tr8vWvkg8d4CGtTkL1k9nax7UHqFjkv9Qml5CxW-raSAAwtV4H9vmUX0WZVkDPsYkRcfP54SpUL4XPxTGI1S2DOTDerhw-ikcEwJDV-E8r8frX5HdjyvnJqqX7Xx1wtVTkDmDwjCbMvxURtjGnfxuFZ5QsJDaguoy0qeN8hogl0A~NqOyKA__&Key-Pair-Id=KII6SEJ68IEHF)中通过绘制出波形图、Mel频谱图：
![](https://files.seeusercontent.com/2026/07/18/nfX5/20260718145407476.png)
其中在**波形图**中，X 轴 = 时间，Y 轴 = 振幅（空气振动强度），密集振荡的区域 → 有声音，平坦区域 → 静音。在**Mel频谱图**中：X轴 = 时间，Y 轴 = Mel 频率（人耳感知的频率刻度，低频分辨更精细），颜色 = 能量 (dB)。它将一维时序信号变成二维热力图，同时展示什么频率在什么时候出现。
回到VAD模型处理过程中，理想的VAD模型效果就是在上述波形图中找到静音区域然后切分出来（亦或者从Mel图中剔除掉db小的内容）
# 常用VAD模型架构

# 参考
