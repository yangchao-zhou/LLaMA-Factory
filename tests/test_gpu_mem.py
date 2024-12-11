import torch
import time
from threading import Thread

def gpu_full_memory_floating_point(gpu_id, size=(8192, 8192), memory_fill_factor=0.9):
    """
    在指定 GPU 上分配大矩阵以填充显存，同时进行高负载浮点运算。

    :param gpu_id: GPU 的编号
    :param size: 单次矩阵计算的大小
    :param memory_fill_factor: 填充显存的比例（0-1），默认为 90%
    """
    device = torch.device(f"cuda:{gpu_id}")
    # print(f"GPU {gpu_id} 开始运行...")

    # 分配占用显存的大矩阵
    total_memory = torch.cuda.get_device_properties(device).total_memory
    reserve_memory = int(total_memory * memory_fill_factor)

    try:
        num_elements = reserve_memory // 4  # 每个浮点数占用 4 字节
        large_tensor = torch.empty(num_elements, dtype=torch.float32, device=device)  # 填充显存

        # print(f"GPU {gpu_id} 已分配 {reserve_memory / (1024**2):.2f} MB 的显存。")

        # 分配用于浮点运算的小矩阵
        a = torch.rand(size, device=device)
        b = torch.rand(size, device=device)

        iterations = 0
        start_time = time.time()

        while True:
            # 持续矩阵乘法计算
            result = torch.mm(a, b)
            result = result.sum()
            iterations += 1

            if iterations % 100 == 0:
                elapsed_time = time.time() - start_time
                # print(f"GPU {gpu_id} 已完成 {iterations} 次矩阵计算，耗时 {elapsed_time:.2f} 秒。")
                start_time = time.time()
    except KeyboardInterrupt:
        pass
        # print(f"GPU {gpu_id} 测试中断，退出运行。")
    except RuntimeError as e:
        pass
        # print(f"GPU {gpu_id} 运行错误：{e}")

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
        pass
        # print("监控中断，退出。")

if __name__ == "__main__":
    # 检查 GPU 是否可用
    if not torch.cuda.is_available():
        # print("GPU 不可用，请检查环境配置。")
        exit(1)

    # 获取可用 GPU 数量
    num_gpus = torch.cuda.device_count()
    # print(f"检测到 {num_gpus} 张 GPU。")

    # 启动监控线程
    monitor_thread = Thread(target=monitor_gpu_utilization, args=(5,), daemon=True)
    monitor_thread.start()

    # 为每张 GPU 启动独立的浮点计算任务
    threads = []
    for gpu_id in range(min(num_gpus, 8)):  # 确保不超过 8 张卡
        t = Thread(target=gpu_full_memory_floating_point, args=(gpu_id,))
        t.start()
        threads.append(t)

    # 等待所有线程完成
    for t in threads:
        t.join()
