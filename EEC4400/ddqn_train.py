import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import time
import numpy as np
import json     # 用于保存/加载训练状态（如epsilon, episode）
import pickle   # 用于保存/加载经验回放池
from collections import deque

import gymnasium as gym

from keras.models import Sequential
from keras.layers import Dense

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam 
from utils import evaluation, get_run_logdir, store_experience, sample_experience
from model import MLP

# Use the following set of NN hyperparameters for ALL FOUR baseline policies
lr =  0.00025        #@param {type:"number"}               # learning rate
epochs =  1     #@param {type:"number"}               # epochs
episode = 500  #@param {type:"number"}               # episodes

epsilon = 1           #@param {type:"number"}     # Starting exploration rate
epsilon_min = 0.01    #@param {type:"number"}     # Exploration rate min
epsilon_decay = 0.999     #@param {type:"number"}     # Exploration rate decay

gamma = 0.99          #@param {type:"number"}     # Agent discount factor

# Use the following set of NN hyperparameters for Naive DQN, DQN and DDQN policies
ba =  32       #@param {type:"number"}               # batch_size

# Use the following set of RL hyperparameters for DQN and DDQN policies
target_update_freq = 1000 # @param {type:"number"}    # Target network update frequency

state_size = 4  # Number of observations (CartPole)
action_size = 2 # Number of possible actions

#训练代码
eval_model = MLP(state_size, action_size)
eval_model.compile(loss='mse', optimizer=Adam(learning_rate=lr))
target_model = MLP(state_size, action_size)
target_model.set_weights(eval_model.get_weights())

# For logging
train_reward_lst = []
eval_reward_mean_lst = []
eval_reward_var_lst = []

# Set up environment
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0] # Number of observations (CartPole)
action_size = env.action_space.n            # Number of possible actions

model_dir = "ddqn_baseline"  # TensorBoard log directory
#cb = keras.callbacks.TensorBoard(log_dir = get_run_logdir(model_dir), histogram_freq=1)
# 【新代码】手动创建 TensorBoard 的 summary writer
logdir = get_run_logdir(model_dir)
summary_writer = tf.summary.create_file_writer(logdir + "/train")


# Train Counter for weight syncing
train_counter = 0

# For timing training
total_training_time = 0

# Define replay buffer
replay_buffer = deque(maxlen=10000)

for ep in range(episode):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    # record start time
    start = time.time()

    for _ in range(500):

        # Interact with the environment with epsilon-greedy policy
        if np.random.rand() <= epsilon:
            action = np.random.choice(action_size)
        else:
            #pass # remove pass and use 2 lines below
            q_values = eval_model.predict(state, verbose=0)
            action = np.argmax(q_values)

        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        done = terminated or truncated

        # store experience into replay buffer
        # [WriteCode]
        store_experience(replay_buffer, state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        if done:
            break

        if len(replay_buffer) >= ba:
            train_counter += 1
            states, actions, rewards, next_states, dones = sample_experience(replay_buffer, ba)

            #下面几行是在构造训练目标标签
            # Update policy with mini-batches if replay buffer contains enough samples
            # Update policy using Double Deep Q-Learning update:
            # Q(s, a) = r + gamma * Q_target(S', argmax Q_eval(S', a))
            # [WriteCode]
            q_next_eval = eval_model.predict(next_states, verbose=0) # Shape: (batch_size, action_size)
            best_actions_next = np.argmax(q_next_eval, axis=1) # Shape: (batch_size,)
            q_next_target = target_model.predict(next_states, verbose=0)
            target_q_for_actions = q_next_target[range(ba), best_actions_next]  # Shape: (batch_size,)
            # Compute target Q-values:
            # - If done, Q-target = reward (no future reward)
            # - Otherwise, Q-target = reward + gamma * Q_target(S', argmax Q_eval(S', a))
            target_q_values = rewards + (gamma * target_q_for_actions * (1 - dones))

            # Predict current Q-values for state using eval_model
            # Use eval_model to determine best action in next_state
            # Use target_model to compute Q-value for that action
            current_q_values = eval_model.predict(states, verbose=0)
            target_for_fit = current_q_values
            
            # Update only the Q-value for the taken action
            target_for_fit[range(ba), actions] = target_q_values

            #目标标签构建好，将训练输入和目标传入fit函数中进行训练
            # Fit the model:
            # - Inputs: state
            # - Targets: updated Q-values (with action Q-value replaced by computed target)
            history = eval_model.fit(states, target_for_fit, epochs=1, verbose=0)  #这里训练一个epoch，其实就是上面那一个batch  # 不再需要 callbacks=[cb]
            # Update exploration rate
            # 【新代码】手动记录损失值
            loss = history.history['loss'][0]  # 从 history 对象中获取损失值

            # 【修改后代码】在这里记录 Loss 和 Average Q-Value
            # 计算当前批次状态的平均最大Q值
            avg_max_q = np.mean(np.max(current_q_values, axis=1))
            with summary_writer.as_default():
                tf.summary.scalar('batch_loss', data=loss, step=train_counter)
                tf.summary.scalar('average_max_q_value', data=avg_max_q, step=train_counter)

            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

            # Periodically update the target network
            if train_counter % target_update_freq == 0:
               target_model.set_weights(eval_model.get_weights())

    # record end time and log training time
    end = time.time()
    total_training_time += end - start

    # Evaluation
    # [WriteCode]
    eval_reward_mean, eval_reward_var = evaluation(eval_model, max_timesteps=500)

    # 【修改后代码】在这里记录 Evaluation Reward
    with summary_writer.as_default():
        # 我们使用 episode 数量作为 x 轴，因为它在每个 episode 结束时才计算
        tf.summary.scalar('evaluation_reward_mean', data=eval_reward_mean, step=ep)
        # 也可以记录训练时的奖励，但评估奖励更具代表性
        tf.summary.scalar('training_episode_reward', data=total_reward, step=ep)

    print(f"Episode {ep + 1}/{episode} | Ep. Total Reward: {total_reward}"
        f" | Epsilon : {epsilon:.3f}"
        f" | Eval Rwd Mean: {eval_reward_mean:.2f}"
        f" | Eval Rwd Var: {eval_reward_var:.2f}")

    # Log
    eval_reward_mean_lst.append(eval_reward_mean)
    eval_reward_var_lst.append(eval_reward_var)
    train_reward_lst.append(total_reward)

    # Early Stopping Condition to avoid overfitting
    # If the evaluation reward reaches the specified threshold, stop training early.
    # The default threshold is set to 500, but you should adjust this based on observed training performance.
    if eval_reward_mean > 500: # [Modify this threshold as needed]
        print(f"Early stopping triggered at Episode {ep + 1}.")
        break

# record end time and calculate average training time per episode
# evaluate average training time per episode
print(f"Training time: {total_training_time/episode:.4f} seconds per episode")

env.close()