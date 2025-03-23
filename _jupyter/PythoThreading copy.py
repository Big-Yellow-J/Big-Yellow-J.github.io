import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# 初始化数据
num_list = [1] * 8000000  # 800万元素
num_workers = 8  # 进程数
chunk_size = len(num_list) // num_workers  # 每个进程处理的数据块大小

def chunk_sum(data_chunk):
    """计算数据块的总和"""
    return sum(data_chunk)

if __name__ == "__main__":  # Windows 需要这个保护
    start_time = time.time()
    chunks = [num_list[i * chunk_size:(i + 1) * chunk_size] for i in range(num_workers)]
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(chunk_sum, chunk) for chunk in chunks]
        results = [future.result() for future in as_completed(futures)]

    total_sum = sum(results)
    print(f"Execution Time:{time.time() - start_time}")