# Q-Network Baseline 完整实现代码
# 将这些代码复制到对应的notebook单元格中

# ============================================
# 1. 模型定义单元格 (Cell 27)
# ============================================
"""
# Q-Network Baseline Model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


# [WriteCode] from ... import ...
# Already imported Adam above

# Define the Q-network
model = Sequential()

# [WriteCode]
# model.add(...
model.add(Dense(64, input_dim=4, activation='relu'))  # 输入层：4个状态特征，64个神经元
model.add(Dense(64, activation='relu'))                # 隐藏层：64个神经元
model.add(Dense(2, activation='linear'))               # 输出层：2个动作的Q值

# Compile the model
# [WriteCode]
model.compile(loss='mse', optimizer=Adam(learning_rate=lr))

# Print the model summary
# [WriteCode]
model.summary()
"""

# ============================================
# 2. 训练循环中的Q-learning更新 (Cell 28)
# ============================================
"""
# 在训练循环中，替换以下部分：

# 1. 修复epsilon-greedy策略选择动作的部分：
        if np.random.rand() <= epsilon:
            action = np.random.choice(action_size)
        else:
            q_values = model.predict(state, verbose=0)
            action = np.argmax(q_values[0])

# 2. 替换Q-learning更新部分：
        # Train model using Q-Learning update:  Q(s, a) = r + gamma * max Q(s', a')
        # [WriteCode]
        
        # 1. 预测当前状态的Q值
        q_values = model.predict(state, verbose=0)[0]
        
        # 2. 计算目标Q值
        if done:
            # 如果episode结束，目标Q值就是当前奖励
            target_q = reward
        else:
            # 否则，目标Q值 = 当前奖励 + gamma * 下一状态的最大Q值
            next_q_values = model.predict(next_state, verbose=0)[0]
            target_q = reward + gamma * np.max(next_q_values)
        
        # 3. 更新当前动作的Q值（保持其他动作的Q值不变）
        q_values[action] = target_q
        
        # 4. 训练模型
        model.fit(state, q_values.reshape(1, -1), epochs=epoch, verbose=0, callbacks=[cb])
"""

# ============================================
# 3. 评估函数调用 (Cell 28)
# ============================================
"""
# 在训练循环结束后，替换评估部分：

    # Evaluation
    # [WriteCode]
    eval_reward_mean, eval_reward_var = evaluation(model)
    
    print(f"Episode {ep + 1}/{episode} | Ep. Total Reward: {total_reward}"
        f" | Epsilon : {epsilon:.3f}"
        f" | Eval Rwd Mean: {eval_reward_mean:.2f}"
        f" | Eval Rwd Var: {eval_reward_var:.2f}")

    # Log
    eval_reward_mean_lst.append(eval_reward_mean)
    eval_reward_var_lst.append(eval_reward_var)
    train_reward_lst.append(total_reward)

    # Early Stopping Condition to avoid overfitting
    if eval_reward_mean > 495:  # 接近500分时停止
        print(f"Early stopping triggered at Episode {ep + 1}.")
        break
"""

# ============================================
# 4. 绘图函数调用 (Cell 29)
# ============================================
"""
# Write code to plot
# 1) Moving Averaged Training Reward, 2) Evaluation Mean, 3) Evaluation Variance
# [Write Code]

plot_smoothed_training_rwd(train_reward_lst, window_size=20)
plot_eval_rwd_mean(eval_reward_mean_lst)
plot_eval_rwd_var(eval_reward_var_lst)
"""


