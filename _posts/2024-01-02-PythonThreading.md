---
layout: mypost
title: Python进阶知识：多进程/多线程/任务并行/装饰器
categories: 编程
extMath: true
images: true
address: changsha
show_footer_image: true
description: 主要介绍Python进阶知识：多线程/任务并行/装饰器以及具体代码
---

本文写作于2025.3.20，恰好作者正好在外面实习，于此同时在实际工作中遇到这些知识点，因此就进行一个简短汇总方便后续回顾。

## 简短理解一下什么是多进程/多线程/任务并行/装饰器

**多进程**：指在同一个程序中同时运行多个独立的进程。每个进程都有自己的内存空间和资源，互不干扰。常用在CPU密集型任务中。
比如说：比如你打开了多个浏览器窗口，每个窗口就是一个独立的进程，互不影响。即使一个窗口崩溃，其他窗口也不会受到影响。
**多线程**：指在同一个进程内同时运行多个线程，多个线程共享同一块内存空间。适合I/O密集型任务，线程之间的切换比进程更轻便。
比如说：你在看视频的同时，后台也在下载文件。这些操作都是通过不同的线程完成的，视频播放和下载互不干扰。
**任务并行**：指在同一时刻有多个任务并行执行，可以是多进程、多线程或异步方式的结合，目的是提高效率。
比如说：同时进行多个数据处理任务，每个任务都在独立执行，快速完成整体工作。
**装饰器**：一种特殊的函数，能在不修改原函数代码的情况下，给函数添加额外的功能。
比如说：在一个函数前后自动记录执行时间或日志，常用 @decorator_name 语法

## 在实际任务中使用多进程/多线程/任务并行/装饰器

假设有一个任务是将8000000个数字1相加，我们提前假设我们已经构建好了这样一个数组，并且我们需要记录一下代码运行需要的时间

```python
import time

start_time = time.time()
num_list = [1]*8000000
```

普通处理思路：

```
sum_num = 0
for i in num_list:
    sum_num += i
print(f"Used Time:{time.time()- start_time}")
```

需要时间：`Used Time:1.0201961994171143`，那么有一个思路我先去把num_list拆分为8份（因为数字相加并不会冲突，用分块知识解决）然后计算时间

## 值得注意的

在使用`from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor`（前者为显存，后者为进程）里面的 **ThreadPoolExecutor**和 **ProcessPoolExecutor**需要注意一个问题，后者在执行时候，比如说：

```python
with ProcessPoolExecutor(max_workers= len(current_detection_region)) as executor:
    futures = {executor.submit(process_region, i, region, frame): i for i, region in enumerate(current_detection_region)}
```

会执行任务 **process_region**那么这个时候可能会出现 **ModuleNotFoundError**问题，主要原因是 ProcessPoolExecutor **可能导致不同的进程环境之间无法共享某些依赖或模块**