# Q-Network Baseline 代码逐行解释与模型原理

## 一、整体架构

这个代码实现了 **Q-Network (Deep Q-Network 的简化版)**，用于解决 Cart-Pole 强化学习问题。它使用一个神经网络来近似 Q 函数 $Q(s, a)$，表示在状态 $s$ 下采取动作 $a$ 的预期累积奖励。

---

## 二、代码逐行解释

### 1. 初始化部分 (第1-16行)

```python
# For logging
train_reward_lst = []          # 存储每个episode的训练总奖励
eval_reward_mean_lst = []      # 存储每次评估的平均奖励
eval_reward_var_lst = []       # 存储每次评估的奖励方差
```
**作用**：创建列表来记录训练过程中的性能指标，用于后续分析和可视化。

---

```python
# Set up environment
env = gym.make("CartPole-v1")  # 创建CartPole环境
state_size = env.observation_space.shape[0]  # 状态空间维度：4 (位置、速度、角度、角速度)
action_size = env.action_space.n             # 动作空间大小：2 (左推/右推)
```
**作用**：
- `state_size = 4`：CartPole 的观察空间是 4 维向量
- `action_size = 2`：有两个离散动作（0=左推，1=右推）

---

```python
model_dir = "q_net_baseline"  # TensorBoard日志目录
cb = keras.callbacks.TensorBoard(log_dir = get_run_logdir(model_dir), histogram_freq=1)
```
**作用**：设置 TensorBoard 回调，用于可视化训练过程（损失、权重分布等）。

---

```python
# For timing training
total_training_time = 0  # 累计训练时间
```
**作用**：记录总训练时间，用于计算平均每个 episode 的训练时间。

---

### 2. 训练循环 - Episode 级别 (第19-22行)

```python
for ep in range(episode):  # 训练指定数量的episode (默认250个)
    state, _ = env.reset()  # 重置环境，返回初始状态
    state = np.reshape(state, [1, state_size])  # 将状态重塑为 (1, 4) 形状，用于神经网络输入
    total_reward = 0  # 初始化当前episode的总奖励
```

**原理**：
- 每个 episode 是智能体从初始状态到终止状态的完整交互过程
- `state = np.reshape(state, [1, state_size])` 将状态从 `(4,)` 变为 `(1, 4)`，因为 Keras 模型期望批量输入格式

---

### 3. 时间步循环 - Step 级别 (第27-76行)

#### 3.1 动作选择 - Epsilon-Greedy 策略 (第28-34行)

```python
for _ in range(500):  # 每个episode最多500步
    # Interact with the environment with epsilon-greedy policy
    if np.random.rand() <= epsilon:  # 以epsilon概率进行探索
        action = np.random.choice(action_size)  # 随机选择动作（探索）
    else:
        # remove pass and use 2 lines below
        q_values = model.predict(state, verbose=0)  # 使用当前策略预测Q值
        action = np.argmax(q_values[0])  # 选择Q值最大的动作（利用）
```

**Epsilon-Greedy 策略原理**：
- **探索 (Exploration)**：以概率 `epsilon` 随机选择动作，发现新的可能更好的策略
- **利用 (Exploitation)**：以概率 `1-epsilon` 选择当前最优动作（Q值最大）

**为什么需要探索？**
- 如果只利用当前最优策略，可能陷入局部最优
- 探索可以帮助发现更好的策略

**数学表示**：
$$a_t = \begin{cases}
\text{random action} & \text{with probability } \epsilon \\
\arg\max_a Q(s_t, a) & \text{with probability } 1-\epsilon
\end{cases}$$

---

#### 3.2 环境交互 (第36-38行)

```python
next_state, reward, terminated, truncated, _ = env.step(action)  # 执行动作，获得下一状态和奖励
next_state = np.reshape(next_state, [1, state_size])  # 重塑下一状态
done = terminated or truncated  # 检查episode是否结束
```

**作用**：
- `env.step(action)`：执行动作，返回：
  - `next_state`：下一时刻的状态
  - `reward`：即时奖励（CartPole 中每步都是 +1）
  - `terminated`：是否因为违反条件而终止（如杆子倒下）
  - `truncated`：是否因为达到最大步数而截断

---

#### 3.3 Q-Learning 更新 (第40-65行) ⭐ **核心算法**

这是整个算法的核心部分，实现了 **Q-Learning 更新规则**。

##### 步骤1：计算目标 Q 值 (第45-54行)

```python
# 1. Compute target Q-values:
# - If done, Q-target = reward (no future reward)
# - Otherwise, Q-target = reward + gamma * max(Q(next_state, a))
if done:
    # If done, Q-target = reward (no future reward)
    target_q = reward
else:
    # Otherwise, Q-target = reward + gamma * max(Q(next_state, a))
    next_q_values = model.predict(next_state, verbose=0)[0]  # 预测下一状态的Q值
    target_q = reward + gamma * np.max(next_q_values)  # 计算目标Q值
```

**Q-Learning 更新公式**：
$$Q(s_t, a_t) \leftarrow r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a')$$

**数学原理**：
- **如果 done**：`target_q = reward`
  - Episode 结束，没有未来奖励，目标 Q 值就是当前奖励
- **如果未 done**：`target_q = reward + gamma * max(Q(next_state))`
  - **当前奖励** `reward`：即时奖励
  - **未来奖励** `gamma * max(Q(next_state))`：折扣后的最大未来奖励
    - `gamma`（折扣因子，0.99）：未来奖励的折扣率
    - `max(Q(next_state))`：在下一状态选择最优动作的 Q 值

**为什么使用 max？**
- 因为 Q-Learning 是 **off-policy** 算法，学习的是最优策略（无论当前采用什么策略）
- 假设在下一状态总是选择最优动作

---

##### 步骤2：预测当前 Q 值并更新 (第57-60行)

```python
# 2. Predict current Q-values for state
# Update only the Q-value for the taken action
q_values = model.predict(state, verbose=0)[0]  # 预测当前状态所有动作的Q值
q_values[action] = target_q  # 只更新执行的那个动作的Q值
```

**作用**：
- `q_values` 是当前状态 $s_t$ 下所有动作的 Q 值，形状为 `(2,)`
- 只更新实际执行的动作 `action` 对应的 Q 值
- 其他动作的 Q 值保持不变（因为我们没有观察这些动作的结果）

**为什么只更新一个动作？**
- 我们只执行了一个动作，只获得了这个动作的奖励和下一状态
- 其他动作的 Q 值没有新的信息，所以保持不变

---

##### 步骤3：训练模型 (第62-65行)

```python
# 3. Fit the model:
# - Inputs: state
# - Targets: updated Q-values (with action Q-value replaced by computed target)
model.fit(state, q_values.reshape(1, -1), epochs=epoch, verbose=0, callbacks=[cb])
```

**作用**：
- **输入**：当前状态 `state`，形状 `(1, 4)`
- **目标**：更新后的 Q 值 `q_values`，形状 `(1, 2)`
- **训练**：使用均方误差 (MSE) 损失函数，更新神经网络参数

**损失函数**：
$$\mathcal{L} = (Q(s_t, a_t) - (r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a')))^2$$

**为什么 reshape？**
- `q_values.reshape(1, -1)` 将 `(2,)` 变为 `(1, 2)`，匹配批量输入格式

---

#### 3.4 更新探索率和状态 (第68-76行)

```python
# Update exploration rate
if epsilon > epsilon_min:
    epsilon *= epsilon_decay  # 指数衰减：epsilon越来越小，逐渐减少探索
```

**Epsilon 衰减原理**：
- 训练初期：`epsilon = 1`，大量探索，了解环境
- 训练后期：`epsilon` 逐渐减小，更多利用已学到的策略
- 最终：`epsilon` 接近 `epsilon_min`，主要以利用为主

```python
state = next_state  # 更新当前状态
total_reward += reward  # 累加奖励

if done:
    break  # 如果episode结束，跳出循环
```

---

### 4. 评估和记录 (第78-106行)

#### 4.1 时间记录 (第78-80行)

```python
# record end time and log the training time
end = time.time()
total_training_time += end - start
```

---

#### 4.2 评估当前策略 (第82-89行)

```python
# Evaluation
# [WriteCode]
eval_reward_mean, eval_reward_var = evaluation(model)

print(f"Episode {ep + 1}/{episode} | Ep. Total Reward: {total_reward}"
    f" | Epsilon : {epsilon:.3f}"
    f" | Eval Rwd Mean: {eval_reward_mean:.2f}"
    f" | Eval Rwd Var: {eval_reward_var:.2f}")
```

**evaluation 函数**：
- 运行 5 个评估 episode
- 使用当前策略（不探索，总是选择最优动作）
- 计算平均奖励和方差

**为什么需要评估？**
- 训练奖励受探索影响，不稳定
- 评估奖励反映策略的真实性能（纯利用）

---

#### 4.3 记录日志 (第96-99行)

```python
# Log
eval_reward_mean_lst.append(eval_reward_mean)  # 记录评估平均奖励
eval_reward_var_lst.append(eval_reward_var)    # 记录评估奖励方差
train_reward_lst.append(total_reward)          # 记录训练奖励
```

**作用**：用于后续绘制训练曲线，分析学习过程。

---

#### 4.4 早停机制 (第101-106行)

```python
# Early Stopping Condition to avoid overfitting
# If the evaluation reward reaches the specified threshold, stop training early.
if eval_reward_mean > 500:  # CartPole最大奖励是500
    print(f"Early stopping triggered at Episode {ep + 1}.")
    break
```

**作用**：
- 如果评估奖励达到 500（CartPole 的满分），说明策略已经学到最优
- 提前停止训练，避免浪费时间

---

### 5. 训练完成 (第108-111行)

```python
# evaluate average training time per episode
print(f"Training time: {total_training_time/(ep + 1):.4f} seconds per episode")

env.close()  # 关闭环境
```

---

## 三、Q-Learning 算法原理总结

### 3.1 核心思想

Q-Learning 是一种 **值函数方法**，学习动作值函数 $Q(s, a)$，表示在状态 $s$ 下采取动作 $a$ 的**预期累积奖励**。

### 3.2 更新公式

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)]$$

其中：
- $\alpha$：学习率（在这个实现中由 Adam 优化器控制）
- $\gamma$：折扣因子（0.99），权衡即时奖励和未来奖励
- $r_{t+1}$：即时奖励
- $\max_{a'} Q(s_{t+1}, a')$：下一状态的最大 Q 值

### 3.3 为什么使用神经网络？

**传统 Q-Learning**：
- 需要存储所有状态-动作对的 Q 值
- 状态空间连续或很大时，表太大无法存储

**Deep Q-Network (DQN)**：
- 使用神经网络近似 Q 函数：$Q(s, a; \theta) \approx Q(s, a)$
- 可以处理连续状态空间
- 通过梯度下降更新参数 $\theta$

### 3.4 与 Naive DQN 的区别

**Q-Network Baseline**：
- ✅ 不使用经验回放缓冲区
- ✅ 每步都立即更新（on-line learning）
- ❌ 可能不稳定，因为连续样本高度相关

**Naive DQN**：
- ✅ 使用经验回放缓冲区
- ✅ 随机采样批次训练，打破相关性
- ✅ 更稳定

---

## 四、关键参数说明

| 参数 | 值 | 作用 |
|------|-----|------|
| `lr` | 0.00025 | 学习率，控制参数更新步长 |
| `gamma` | 0.99 | 折扣因子，未来奖励的重要性 |
| `epsilon` | 1 → 0.01 | 探索率，从完全探索到几乎完全利用 |
| `epsilon_decay` | 0.995 | 探索率衰减系数 |
| `episode` | 250 | 训练的总 episode 数 |

---

## 五、训练过程可视化

训练完成后，绘制三个图：
1. **训练奖励曲线**（平滑移动平均）：显示训练过程中的奖励变化
2. **评估平均奖励**：显示策略的真实性能
3. **评估奖励方差**：显示策略的稳定性

---

## 六、常见问题

**Q1: 为什么 Q-Network 可能不稳定？**
- 连续的状态样本高度相关
- 目标 Q 值也在不断变化（使用当前网络计算）
- 没有经验回放打破相关性

**Q2: 为什么使用 gamma = 0.99？**
- 接近 1，重视长期奖励
- CartPole 是长期任务，需要平衡杆子尽可能久

**Q3: epsilon 衰减的作用？**
- 训练初期：探索环境，发现好的策略
- 训练后期：利用已学策略，提高性能

---

## 七、数学公式总结

### Bellman 方程
$$Q^*(s, a) = \mathbb{E}[r_{t+1} + \gamma \max_{a'} Q^*(s_{t+1}, a') | s_t = s, a_t = a]$$

### Q-Learning 更新
$$Q(s_t, a_t) \leftarrow r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a')$$

### 损失函数
$$\mathcal{L}(\theta) = \mathbb{E}[(Q(s_t, a_t; \theta) - y_t)^2]$$

其中 $y_t = r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a'; \theta)$ 是目标 Q 值。

---

## 八、代码执行流程

```
开始训练
  ↓
For each episode:
  ├─ 重置环境，初始化状态
  ├─ For each step (最多500步):
  │   ├─ Epsilon-greedy 选择动作
  │   ├─ 执行动作，获得奖励和下一状态
  │   ├─ 计算目标Q值
  │   ├─ 更新当前Q值
  │   ├─ 训练神经网络
  │   └─ 更新epsilon和状态
  ├─ 评估当前策略
  ├─ 记录性能指标
  └─ 检查早停条件
  ↓
训练完成，绘制结果
```

---

这份代码实现了基础的 Deep Q-Learning 算法，是理解强化学习的重要起点！

