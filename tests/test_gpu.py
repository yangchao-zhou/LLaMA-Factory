import torch
import time
from threading import Thread

def gpu_floating_point_benchmark(gpu_id, size=(4096, 4096)):
    """
    持续在指定 GPU 上运行浮点运算以达到高负载。

    :param gpu_id: GPU 的编号
    :param size: 张量的大小
    """
    device = torch.device(f"cuda:{gpu_id}")
    # print(f"GPU {gpu_id} 开始浮点运算测试...")

    # 分配随机张量
    a = torch.rand(size, device=device)
    b = torch.rand(size, device=device)

    # 持续进行矩阵乘法操作
    try:
        start_time = time.time()
        iterations = 0
        while True:
            result = torch.mm(a, b)  # 矩阵乘法
            result = result.sum()   # 防止优化器忽略计算
            iterations += 1
            if iterations % 100 == 0:
                elapsed_time = time.time() - start_time
                print(f"GPU {gpu_id} 已运行 {iterations} 次，耗时 {elapsed_time:.2f} 秒。")
                start_time = time.time()
    except KeyboardInterrupt:
        print(f"GPU {gpu_id} 测试中断，退出浮点运算。")
    except RuntimeError as e:
        print(f"GPU {gpu_id} 运行时错误：{e}")

def monitor_gpu_utilization(interval=5):
    """
    定期监控 GPU 使用情况（需安装 NVIDIA 的 nvidia-smi 工具）。

    :param interval: 监控的时间间隔，单位为秒
    """
    import subprocess

    try:
        while True:
            # print("\n--- GPU 使用情况 ---")
            subprocess.run(["nvidia-smi"], check=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("监控中断，退出。")

if __name__ == "__main__":
    # 检查 GPU 是否可用
    if not torch.cuda.is_available():
        print("GPU 不可用，请检查环境配置。")
        exit(1)

    # 获取可用 GPU 数量
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 张 GPU。")

    # 启动监控线程
    monitor_thread = Thread(target=monitor_gpu_utilization, args=(5,), daemon=True)
    monitor_thread.start()

    # 为每张 GPU 启动独立的浮点计算任务
    threads = []
    for gpu_id in range(min(num_gpus, 8)):  # 确保不超过 8 张卡
        t = Thread(target=gpu_floating_point_benchmark, args=(gpu_id,))
        t.start()
        threads.append(t)

    # 等待所有线程完成
    for t in threads:
        t.join()
