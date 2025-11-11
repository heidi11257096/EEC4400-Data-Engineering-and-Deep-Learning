# import tensorflow as tf

# # 打印 TensorFlow 版本
# print("TensorFlow Version:", tf.__version__)

# # 列出所有可用的物理设备（包括 CPU 和 GPU）
# gpus = tf.config.list_physical_devices('GPU')

# if gpus:
#     print(f"检测到 {len(gpus)} 个 GPU 设备:")
#     try:
#         # 打印每个 GPU 的详细信息
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#             print(" ", gpu)
#         logical_gpus = tf.config.list_logical_devices('GPU')
#         print(f"创建了 {len(logical_gpus)} 个逻辑 GPU 设备")
#         print("GPU 设置成功！")
#     except RuntimeError as e:
#         # 如果在程序启动后设置内存增长，可能会出现错误
#         print(e)
# else:
#     print("未检测到 GPU 设备。TensorFlow 将使用 CPU。")

import tensorflow as tf
import time

# 开启设备放置日志
tf.debugging.set_log_device_placement(True)

# 定义一个更大的矩阵尺寸
MATRIX_SIZE = 15000  # 从 5000 增加到 15000

print(f"测试 GPU 计算性能 (矩阵尺寸: {MATRIX_SIZE}x{MATRIX_SIZE})...")

# 在 CPU 上执行
with tf.device('/CPU:0'):
    print("CPU 计算中...")
    start_time = time.time()
    a = tf.random.normal([MATRIX_SIZE, MATRIX_SIZE])
    b = tf.random.normal([MATRIX_SIZE, MATRIX_SIZE])
    c = tf.matmul(a, b)
    c.numpy() # 确保计算完成
    cpu_time = time.time() - start_time
    print(f"CPU 耗时: {cpu_time:.4f} 秒")

# 在 GPU 上执行 (如果可用)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    with tf.device('/GPU:0'):
        print("\nGPU 计算中...")
        
        # --- 增加一个预热步骤 ---
        print("GPU 预热...")
        warmup_a = tf.random.normal([100, 100])
        warmup_b = tf.random.normal([100, 100])
        tf.matmul(warmup_a, warmup_b).numpy()
        print("预热完成。")
        # -------------------------

        start_time = time.time()
        a = tf.random.normal([MATRIX_SIZE, MATRIX_SIZE])
        b = tf.random.normal([MATRIX_SIZE, MATRIX_SIZE])
        c = tf.matmul(a, b)
        c.numpy() # 确保计算完成
        gpu_time = time.time() - start_time
        print(f"GPU 耗时: {gpu_time:.4f} 秒")
        
        if cpu_time > 0 and gpu_time > 0:
            print(f"\nGPU 比 CPU 快了约 {cpu_time / gpu_time:.2f} 倍！")
else:
    print("\n未找到 GPU，跳过 GPU 性能测试。")



