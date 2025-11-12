import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5" 
import time
import numpy as np
import random
from collections import deque

import gymnasium as gym

def evaluation(model, max_timesteps=500):
    eval_env = gym.make("CartPole-v1")
    state_size = eval_env.observation_space.shape[0] # Number of observations (CartPole)
    action_size = eval_env.action_space.n            # Number of possible actions
    eval_reward = []

    for i in range (5):
        round_reward = 0
        state, _ = eval_env.reset()
        state = np.reshape(state, [1, state_size])

        for i in range(max_timesteps):
            action = np.argmax(model.predict(state, verbose=0)[0])               #verbose=0 to 用来设置日志静默模式(其他可选值为1进度条模式，2精简日志模式)                                                                                    
            next_state, reward, terminated, truncated, _ = eval_env.step(action) #argmax返回最大值的索引
            next_state = np.reshape(next_state, [1, state_size])

            round_reward += reward
            state = next_state

            if terminated or truncated:
                eval_reward.append(round_reward)
                break

    eval_env.close()

    eval_reward_mean = np.sum(eval_reward)/len(eval_reward)
    eval_reward_var = np.var(eval_reward)

    return eval_reward_mean, eval_reward_var


def get_run_logdir(k):
    root_logdir = os.path.join(os.curdir, "eec4400_logs", k) #curdir 获取当前工作目录
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S") #strftime() 根据指定的格式，将当前时间格式化成一个字符串
    return os.path.join(root_logdir, run_id)


# Store (state, action, reward, next_state, done)
def store_experience(replay_buffer, state, action, reward, next_state, done):
        replay_buffer.append((state, action, reward, next_state, done))

# Sample (state, action, reward, next_state, done) mini-batch for training
def sample_experience(replay_buffer, batch_size):

    # Ensure we have enough samples
    assert len(replay_buffer) >= batch_size, (
        f"Not enough samples in buffer to sample {batch_size} items.")

    # Sample a mini-batch
    minibatch = random.sample(replay_buffer, batch_size)

    states, actions, rewards, next_states, dones = zip(*minibatch)  #将数据按类别分开

    states = np.array(states, dtype=np.float32).squeeze()   #.squeeze() 用于移除数组中维度为 1 的轴
    next_states = np.array(next_states, dtype=np.float32).squeeze()
    actions = np.array(actions, dtype=np.int32)
    rewards = np.array(rewards, dtype=np.float32)
    dones = np.array(dones, dtype=np.float32)

    return states, actions, rewards, next_states, dones