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

multi_process1()

# multi_process2()

# if __name__ == "__main__":
#     single_process()
#     multi_thread()
#     multi_process1()
#     multi_process2()