import multiprocessing
import os
import random
from concurrent.futures import ThreadPoolExecutor

import multiprocessing
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# 生产者线程
def producer(file_path, queue):
    print(f"Producer {os.getpid()} reading from file: {file_path}")
    with open(file_path, "r") as file:
        for line in file:
            queue.put(line)
            time.sleep(0.1)
    queue.put(None)  # 添加结束标志

# 消费者进程
def consumer(queue, num_producers):
    print(f"Consumer {os.getpid()} started")
    end_count = 0
    while True:
        item = queue.get()
        if item is None:  # 检测到结束标志
            end_count += 1
            if end_count == num_producers:
                break
        else:
            print(f"Consumer {os.getpid()} received: {item.strip()}")
    print(f"Consumer {os.getpid()} finished")

def main(file_paths, max_concurrent_files=4):
    num_producers = len(file_paths)
    queue = multiprocessing.Queue()

    # 创建消费者进程
    consumer_process = multiprocessing.Process(target=consumer, args=(queue, num_producers))

    # 启动消费者进程
    consumer_process.start()

    # 创建并启动生产者线程
    with ThreadPoolExecutor(max_workers=max_concurrent_files) as executor:
        futures = [executor.submit(producer, file_path, queue) for file_path in file_paths]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Failed to read file: {e}")

    # 等待消费者进程结束
    consumer_process.join()


def create_large_file(file_path, num_lines=100):
    with open(file_path, "w") as file:
        for _ in range(num_lines):
            file.write(f"{random.randint(1, 1000000)}\n")


if __name__ == "__main__":
    file1 = "large_file1.txt"
    file2 = "large_file2.txt"
    file3 = "large_file3.txt"

    # create_large_file(file1)
    # create_large_file(file2)
    # create_large_file(file3)

    file_paths = [file1, file2, file3]
    num_files_to_read_simultaneously = 2
    main(file_paths, num_files_to_read_simultaneously)
