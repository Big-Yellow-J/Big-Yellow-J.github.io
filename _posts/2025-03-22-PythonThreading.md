---
layout: mypost
title: Python进阶知识：多进程/多线程/装饰器
categories: 编程
extMath: true
images: true
address: changsha
show_footer_image: true
description: 主要介绍Python进阶知识：多线程/多进程/装饰器以及具体代码
tags: [python, 多进程, 多线程, 装饰器]
---

本文写作于2025.3.20，恰好作者正好在外面实习，于此同时在实际工作中遇到这些知识点，因此就进行一个简短汇总方便后续回顾，可以直接看[第三节](#3如何在代码中使用多进程多线程装饰器)

## 1、简短理解一下什么是多进程/多线程/装饰器和一些基本概念

**多进程**：指在同一个程序中同时运行多个独立的进程。每个进程都有自己的内存空间和资源，互不干扰。常用在CPU密集型任务中。
比如说：比如你打开了多个浏览器窗口，每个窗口就是一个独立的进程，互不影响。即使一个窗口崩溃，其他窗口也不会受到影响。
**多线程**：指在同一个进程内同时运行多个线程，多个线程共享同一块内存空间。适合**I/O密集型任务**（主要涉及到输入输出操作的任务。这些任务的执行时间主要花费在等待IO操作的完成上），线程之间的切换比进程更轻便。
比如说：你在看视频的同时，后台也在下载文件。这些操作都是通过不同的线程完成的，视频播放和下载互不干扰。
**装饰器**：一种特殊的函数，能在不修改原函数代码的情况下，给函数添加额外的功能。
比如说：在一个函数前后自动记录执行时间或日志，常用 @decorator_name 语法
**全局解释器锁**（GIL）：它使得任何时刻仅有一个线程在执行。即便在多核心处理器上，使用 GIL 的解释器也只允许同一时间执行一个线程。常见的使用 GIL 的解释器有CPython与Ruby MRI。

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
- Python 的 `ThreadPoolExecutor` 受 **GIL 限制**，多个线程并不会真正并行执行，而是**交替运行**，因此它比单线程快，但 **提升有限**。
- 由于 `sum` 计算是 **CPU 密集型任务**，线程池无法充分发挥 CPU 多核优势，导致性能 **不如串行分块计算**。

2. **进程间通信（IPC）开销**  
- `ProcessPoolExecutor` 会 **为每个进程创建独立的 Python 解释器**，数据需要 **在主进程和子进程之间传输**，但 `num_list` 非常大，导致 **数据传输和进程调度成本过高**，反而影响性能。

3. **任务拆分的额外开销**  
- 由于 `sum` 操作本身非常简单，计算时间短，线程池和进程池的 **管理开销**（线程/进程创建、调度、回收）可能超过计算本身的成本，导致整体运行时间反而变长。

用人话来说就是，使用多进程，就需要考虑到通信的花销，用多线程就要考虑到 **GIL**限制，换言之得到的**结论就是**：
- **多线程（ThreadPoolExecutor）** 适用于 **I/O 密集型任务**（如文件读写、网络请求），但 **CPU 计算任务受 GIL 限制**，提升有限。
- **多进程（ProcessPoolExecutor）** 适用于 **CPU 密集型任务**，但数据传输开销大，对 **短时间计算任务** 可能不适用。

## 3、如何在代码中使用多进程/多线程/装饰器

* **1、多线程使用**

多线程使用方式比较简单，以下面例子为例：

```python
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers= n) as executor:
    futures = [executor.submit(sum, chunk) for chunk in chunks]  # 提交任务
    results = [future.result() for future in as_completed(futures)]  # 获取结果
```

当软还有另外一种执行方式：

```python
import threading
thread_1 = threading.Thread(target= sum)
thread_2 = threading.Thread(target= sum)

thread_1.start()
thread_2.start()

thread_1.join()
thread_2.join()

```

第一种相对而言比较简单（自动管理线程），而第二种需要我去创建多个进程，然后对不同进程之间进行 `start()` 以及 `join()`，实际使用如果是一个长期执行任务可以用 `threading.Thread`（比如说要一致保持摄像头开启就可以直接 `threading.Thread(target=video_capture_thread, daemon=True).start()` ）而并行任务可以选择 `ThreadPoolExecutor`不用去手动创建

一般来说使用过程中只需要注意如下几个操作：1、向你创建的进程中提交任务（提交的内容是：你要进行计算的函数，函数所需要的参数）；2、获取你提交任务所得到的结果（因为是多线程，因此返回得到的结果也就是不同线程的结果）
需要注意的就是下面几个内容：1、`submit` 提交你的任务；2、`as_completed` 执行你的任务
**不过需要小心的一点是**，使用多线程，需要保证 thread-safe（线程安全），比如说同时向一个文件里面写入时候，我通过使用LLM的api执行时候，我有一个较长的文本，先将他拆分（保证是模型的最大允许输入），然后“一次性”（假设的是线程数量恰好和分割数量一致）将其进行api访问（这样时间消耗肯定比普通的要少）将处理结果然后写入到一个文件中就需要考虑进程锁问题，因为所有任务结果都写入同一个问题可能会导致 **进程冲突**，比如说：

```python
from concurrent.futures import ThreadPoolExecutor
import time

def llm_api_result(num):
    time.sleep(2)
    return f"{num}"*100

def write_to_file(num):
    content = f"Thread-{num}: " + llm_api_result(num)
    with open("./output-without-lock.txt", "a", encoding= "utf-8") as f:
        f.write(content)
        f.write("\n")

def main():
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(write_to_file, num) for num in range(10)]
        for future in futures:
            future.result()

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Used Time:", time.time()- start_time)
```

这样一来得到的结果为：

![](https://s2.loli.net/2025/03/24/MNp1SjxQmhFbEnW.png)

但是上面代码中并没有对进程加锁（`lock = threading.Lock()`），但是结果还是可以正常（`write`是一个 **原子操作** ）写入（有时候会出现遗漏掉内容），但是写入顺序是不对的。

> **原子操作**（atomic operation） 指的是 不可被中断的操作，它要么 完整执行，要么 完全不执行

将代码改为下面代码，通过使用进程锁来保护**原子操作**：

```python
import threading
lock = threading.Lock()
def write_to_file(num):
    content = f"Thread-{num}: " + llm_api_result(num)
    with lock:
        with open("output-with-lock.txt", "a", encoding="utf-8") as f:
            f.write(content)
            f.write("\n")
```

![](https://s2.loli.net/2025/03/24/LB2i35sGSK9EDXu.png)

这样一来就可以正常写入结果

* **2、多进程使用**

Python 的 multiprocessing 模块基于 fork 或 spawn 机制，可以创建多个独立进程，让它们并行执行任务，从而绕过**GIL（全局解释器锁）**，提高 CPU 密集型任务的性能（数学运算、数据处理等）。使用起来也比较简单

一个具体例子：

```python
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# 计算斐波那契数列的函数
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

def single_process():
    start_time = time.time()
    for _ in range(4):
        fibonacci(35)
    end_time = time.time()
    print(f"Single-process time: {end_time - start_time:.2f} seconds")

def multi_thread():
    start_time = time.time()
    with ThreadPoolExecutor(max_workers= 4) as executor:
        futures = [executor.submit(fibonacci, 35) for _ in range(4)]
        result = [future.result() for future in futures]
    end_time = time.time()
    print(f"Multi-thread time: {end_time - start_time:.2f} seconds")

def multi_process1():
    start_time = time.time()
    processes = []
    for _ in range(4):
        process = multiprocessing.Process(target=fibonacci, args=(35,))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()
    end_time = time.time()
    print(f"Multi-process-1 time: {end_time - start_time:.2f} seconds")

def multi_process2():
    start_time = time.time()
    with ProcessPoolExecutor(max_workers= 4) as executor:
        futures = [executor.submit(fibonacci, 35) for _ in range(4)]
        result = [future.result() for future in futures]
    end_time = time.time()
    print(f"Multi-process-2 time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    single_process()
    multi_thread()
    multi_process1()
    multi_process2()

Single-process time: 8.93 seconds
Multi-thread time: 9.89 seconds
Multi-process-1 time: 3.81 seconds
Multi-process-2 time: 3.67 seconds
```

python里面使用多进程和多线程代码上没有多大区别，只不过使用多进程需要注意的是上面代码使用必须（在window系统上）要用到下面代码，但是linux系统就没有这个问题，这是因为[两种启动进程的方式是不同的](https://docs.python.org/zh-cn/3.13/library/multiprocessing.html#multiprocessing-programming:~:text=%E6%A0%B9%E6%8D%AE%E4%B8%8D%E5%90%8C%E7%9A%84%E5%B9%B3%E5%8F%B0%EF%BC%8C%20multiprocessing%20%E6%94%AF%E6%8C%81%E4%B8%89%E7%A7%8D%E5%90%AF%E5%8A%A8%E8%BF%9B%E7%A8%8B%E7%9A%84%E6%96%B9%E6%B3%95)。

```python
if __name__ == "__main__":
```

这是因为创建子进程时，会重新导入主模块。如果不将多进程代码放在 `if __name__ == "__main__":` 块中，可能会导致递归创建子进程，甚至引发程序崩溃。更加底层的原因可以直接参考python[官方解释](https://docs.python.org/zh-cn/3.13/library/multiprocessing.html)

* **3、装饰器**

https://liaoxuefeng.com/books/python/functional/decorator/index.html

## 值得注意的

在使用`from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor`（前者为线程，后者为进程）里面的 **ThreadPoolExecutor**和 **ProcessPoolExecutor**需要注意一个问题，后者在执行时候，比如说：

```python
with ProcessPoolExecutor(max_workers= len(current_detection_region)) as executor:
    futures = {executor.submit(process_region, i, region, frame): i for i, region in enumerate(current_detection_region)}
```

会执行任务 **process_region**那么这个时候可能会出现 **ModuleNotFoundError**问题，主要原因：ProcessPoolExecutor **可能导致不同的进程环境之间无法共享某些依赖或模块**

## 参考
1、https://docs.python.org/zh-cn/3.13/library/concurrent.futures.html
2、https://docs.python.org/zh-cn/3.13/library/threading.html
3、https://zh.wikipedia.org/zh-cn/%E5%85%A8%E5%B1%80%E8%A7%A3%E9%87%8A%E5%99%A8%E9%94%81
4、https://zh.wikipedia.org/wiki/CPU%E5%AF%86%E9%9B%86%E5%9E%8B
5、https://docs.python.org/zh-cn/3.13/library/multiprocessing.html