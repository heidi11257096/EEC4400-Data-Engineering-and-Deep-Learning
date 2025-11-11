```
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
# --- 设置一个美观的绘图风格 ---
# 'seaborn-v0_8-whitegrid' 是一个非常流行的风格，带有网格，看起来很清爽
plt.style.use('seaborn-v0_8-whitegrid')


def plot_eval_rwd_mean(eval_mean_list):
    """
    绘制评估阶段的平均奖励随时间变化的曲线。

    Args:
        eval_mean_list (list or np.ndarray): 包含每次评估的平均奖励的列表。
    """
    # 创建一个图形和坐标轴，设置图形大小
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # x轴坐标就是评估的次数
    x_axis = range(len(eval_mean_list))
    
    # 绘制曲线
    ax.plot(x_axis, eval_mean_list, 
            color='dodgerblue',          # 使用一种好看的蓝色
            marker='o',                  # 在每个数据点上加个圆圈标记
            linestyle='-',               # 使用实线连接
            linewidth=2,                 # 线条宽度
            markersize=5,                # 标记大小
            label='Mean Reward')         # 图例标签
            
    # 设置标题和坐标轴标签，并增加字体大小
    ax.set_title('Evaluation Reward Mean Over Time', fontsize=16)
    ax.set_xlabel('Evaluation Step', fontsize=12)
    ax.set_ylabel('Mean Reward', fontsize=12)
    
    # 显示图例
    ax.legend()
    
    # 显示图形
    plt.tight_layout()  # 调整布局以防止标签重叠
    plt.show()


def plot_eval_rwd_var(eval_var_list):
    """
    绘制评估阶段的奖励方差随时间变化的曲线。

    Args:
        eval_var_list (list or np.ndarray): 包含每次评估的奖励方差的列表。
    """
    # 创建一个图形和坐标轴
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # x轴坐标
    x_axis = range(len(eval_var_list))
    
    # 绘制曲线
    ax.plot(x_axis, eval_var_list, 
            color='forestgreen',         # 使用绿色
            marker='s',                  # 使用方形标记
            linestyle='--',              # 使用虚线
            linewidth=2,
            markersize=5,
            label='Reward Variance')
            
    # 设置标题和标签
    ax.set_title('Evaluation Reward Variance Over Time', fontsize=16)
    ax.set_xlabel('Evaluation Step', fontsize=12)
    ax.set_ylabel('Reward Variance', fontsize=12)
    
    # 显示图例
    ax.legend()
    
    # 显示图形
    plt.tight_layout()
    plt.show()


def plot_smoothed_training_rwd(train_rwd_list, window_size=20):
    """
    使用移动平均法绘制平滑后的训练奖励曲线。
    同时将原始的、未平滑的奖励作为背景，以提供对比。

    Args:
        train_rwd_list (list or np.ndarray): 包含每个训练回合总奖励的列表。
        window_size (int): 移动平均的窗口大小。
    """
    if len(train_rwd_list) < window_size:
        print(f"Warning: train_rwd_list has fewer elements ({len(train_rwd_list)}) than window_size ({window_size})." 
              " Cannot compute moving average.")
        return

    # 创建一个图形和坐标轴
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 1. 绘制原始的训练奖励，使用较浅的颜色和透明度作为背景
    ax.plot(train_rwd_list, 
            color='lightgray', 
            alpha=0.6, 
            label='Raw Reward per Episode')
            
    # 2. 计算移动平均值
    # np.convolve 是一个高效计算移动平均的方法
    # mode='valid' 确保我们只在窗口完全覆盖数据时才计算平均值
    smoothed_rewards = np.convolve(train_rwd_list, np.ones(window_size)/window_size, mode='valid')
    # np.ones(window_size)/window_size 创建权重数组（例如 [0.05, 0.05, ..., 0.05]，20个0.05）
    # np.convolve(..., mode='valid') 卷积计算移动平均
    # mode='valid' 只在窗口完全覆盖时计算，结果长度 = len(train_rwd_list) - window_size + 1
    
    # 3. 绘制平滑后的奖励曲线，使用醒目的颜色
    # 注意：平滑后的数据点数会减少，x轴需要对齐
    smoothed_x_axis = np.arange(window_size - 1, len(train_rwd_list))
    ax.plot(smoothed_x_axis, smoothed_rewards, 
            color='crimson',             # 使用深红色
            linewidth=2.5, 
            label=f'Smoothed Reward (Window Size = {window_size})')
            
    # 设置标题和标签
    ax.set_title('Smoothed Training Rewards Over Episodes', fontsize=16)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Reward', fontsize=12)
    
    # 显示图例
    ax.legend()
    
    # 显示图形
    plt.tight_layout()
    plt.show()

```