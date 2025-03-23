import time
import math
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# 生成一个大数列表，范围 100000000 ~ 100000500
num_list = list(range(100_000_000, 100_000_500))
num_workers = 4  # 线程或进程数

def is_prime(n):
    """判断一个数是否为素数"""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    for i in range(5, int(math.sqrt(n)) + 1, 2):  # 仅检查奇数
        if n % i == 0:
            return False
    return True

def count_primes(nums):
    """计算列表中素数的个数"""
    return sum(1 for n in nums if is_prime(n))

if __name__ == "__main__":  # Windows 需要
    ### **单线程计算（基准）**
    start_time = time.time()
    prime_count = count_primes(num_list)
    print(f"Single-thread Time: {time.time() - start_time:.4f} sec, Primes: {prime_count}")

    ### **多线程计算**
    start_time = time.time()
    chunk_size = len(num_list) // num_workers
    chunks = [num_list[i * chunk_size:(i + 1) * chunk_size] for i in range(num_workers)]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(count_primes, chunks))

    print(f"ThreadPool Time: {time.time() - start_time:.4f} sec, Primes: {sum(results)}")

    ### **多进程计算**
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(count_primes, chunk) for chunk in chunks]
        results = [future.result() for future in as_completed(futures)]

    print(f"ProcessPool Time: {time.time() - start_time:.4f} sec, Primes: {sum(results)}")
