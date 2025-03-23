---
layout: mypost
title: Python进阶知识：多进程/多线程/装饰器
categories: 编程
extMath: true
images: true
address: changsha
show_footer_image: true
description: 主要介绍Python进阶知识：多线程/多进程/装饰器以及具体代码
---

本文写作于2025.3.20，恰好作者正好在外面实习，于此同时在实际工作中遇到这些知识点，因此就进行一个简短汇总方便后续回顾。建议直接看第三点：**3、如何在代码中使用多进程/多线程/装饰器**

## 1、简短理解一下什么是多进程/多线程/装饰器

**多进程**：指在同一个程序中同时运行多个独立的进程。每个进程都有自己的内存空间和资源，互不干扰。常用在CPU密集型任务中。
比如说：比如你打开了多个浏览器窗口，每个窗口就是一个独立的进程，互不影响。即使一个窗口崩溃，其他窗口也不会受到影响。
**多线程**：指在同一个进程内同时运行多个线程，多个线程共享同一块内存空间。适合I/O密集型任务，线程之间的切换比进程更轻便。
比如说：你在看视频的同时，后台也在下载文件。这些操作都是通过不同的线程完成的，视频播放和下载互不干扰。
**装饰器**：一种特殊的函数，能在不修改原函数代码的情况下，给函数添加额外的功能。
比如说：在一个函数前后自动记录执行时间或日志，常用 @decorator_name 语法

## 2、在实际任务中使用多进程/多线程

假设有一个任务是将8000000个数字1相加，我们提前假设我们已经构建好了这样一个数组，并且我们需要记录一下代码运行需要的时间，普通处理思路：

```
start_time = time.time()
sum_num = 0
for i in num_list:
    sum_num += i
print(f"Used Time:{time.time()- start_time}")
```

需要时间：`Used Time:0.9650969505310059`，那么有一个思路我先去把num_list拆分为8份（因为数字相加并不会冲突，用分块知识解决）然后计算时间 `Used Time:0.07107281684875488`，但是这样数据计算是串行的（执行完第一块，然后去计算第二块），那么我们可以考虑多线程直接8块一起计算然后将最后结果汇总起来，这样得到的时间为：`Used Time:0.09244751930236816`，于此同时使用多进程计算得到结果：`Used Time:0.854262113571167`。这样就会又一个有意思现象，理论上来说多进程，多线程速度应该是都大于常规的分割法，出现这个原因是因为：

1. **GIL（全局解释器锁）影响**  
- Python 的 `ThreadPoolExecutor` 受 **GIL 限制**，多个线程并不会真正并行执行，而是交替运行，因此它比单线程快，但 **提升有限**。
- 由于 `sum` 计算是 **CPU 密集型任务**，线程池无法充分发挥 CPU 多核优势，导致性能 **不如串行分块计算**。

2. **进程间通信（IPC）开销**  
- `ProcessPoolExecutor` 会 **为每个进程创建独立的 Python 解释器**，数据需要 **在主进程和子进程之间传输**，但 `num_list` 非常大，导致 **数据传输和进程调度成本过高**，反而影响性能。

3. **任务拆分的额外开销**  
- 由于 `sum` 操作本身非常简单，计算时间短，线程池和进程池的 **管理开销**（线程/进程创建、调度、回收）可能超过计算本身的成本，导致整体运行时间反而变长。

用人话来说就是，使用多进程，就需要考虑到通信的花销，用多线程就要考虑到 **GIL**限制，换言之得到的**结论就是**：
- **多线程（ThreadPoolExecutor）** 适用于 **I/O 密集型任务**（如文件读写、网络请求），但 **CPU 计算任务受 GIL 限制**，提升有限。
- **多进程（ProcessPoolExecutor）** 适用于 **CPU 密集型任务**，但数据传输开销大，对 **短时间计算任务** 可能不适用。

## 3、如何在代码中使用多进程/多线程/装饰器

* 1、多线程使用

多线程使用方式比较简单，以下面例子为例：

```python
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers= n) as executor:
    futures = [executor.submit(sum, chunk) for chunk in chunks]  # 提交任务
    results = [future.result() for future in as_completed(futures)]  # 获取结果
```

一般来说使用过程中只需要注意如下几个操作：1、向你创建的进程中提交任务（提交的内容是：你要进行计算的函数，函数所需要的参数）；2、获取你提交任务所得到的结果（因为是多线程，因此返回得到的结果也就是不同线程的结果）
需要注意的就是下面几个内容：1、`submit` 提交你的任务；2、`as_completed` 执行你的任务
**不过需要小心的一点是**，同时向一个文件里面写入时候，比如说我通过使用LLM的api执行时候，我有一个较长的文本，先将他拆分（保证是模型的最大允许输入），然后“一次性”（假设的是线程数量恰好和分割数量一致）将其进行api访问（这样时间消耗肯定比普通的要少）将处理结果然后写入到一个文件中就需要考虑进程锁问题，因为所有任务结果都写入同一个问题就会出现问题，因此此类处理代码是：

方式一：带进程锁的处理，边处理边写入

```python
import threading

lock = threading.Lock()

def llm_api_result(...):
    '''通过llm api获取结果'''
    ...

def write_to_text(api_result, lock):
    with lock:
        with open(...) as f:
            ...

def process_and_write(chunk, lock):
    result = llm_api_result(chunk)
    write_to_text(result, lock)

def main():
    input_text, chunk_size = ..., ...
    threads = []
    for i in range(0, len(input_text), chunk_size):
        chunk = input_text[i:i + chunk_size]  # 切分 chunk
        thread = threading.Thread(target=process_and_write, args=(chunk, lock))
        threads.append(thread)  # 添加线程到列表
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()
```

方式二：不带进程锁处理，直接多线程处理所有结果，然后再去将结果写入

```python
from concurrent.futures import ThreadPoolExecutor

def llm_api_result(chunk):
    '''通过llm api获取结果'''
    ...

def write_to_text(results):
    with open(...) as f:
        ...

def main():
    input_text, chunk_size = ..., ...
    chunk_size = len(num_list) // n
    chunks = [num_list[i * chunk_size:(i + 1) * chunk_size] for i in range(n)]
    with ThreadPoolExecutor(max_workers= 2) as executor:
        futures = [executor.submit(llm_api_result, chunk) for chunk in chunks]
        results = [future.result() for future in as_completed(futures)]
    write_to_text(results)
```

## 值得注意的

在使用`from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor`（前者为线程，后者为进程）里面的 **ThreadPoolExecutor**和 **ProcessPoolExecutor**需要注意一个问题，后者在执行时候，比如说：

```python
with ProcessPoolExecutor(max_workers= len(current_detection_region)) as executor:
    futures = {executor.submit(process_region, i, region, frame): i for i, region in enumerate(current_detection_region)}
```

会执行任务 **process_region**那么这个时候可能会出现 **ModuleNotFoundError**问题，主要原因是 ProcessPoolExecutor **可能导致不同的进程环境之间无法共享某些依赖或模块**

## 参考
1、https://docs.python.org/zh-cn/3.13/library/concurrent.futures.html