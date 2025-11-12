# 修复模型错误：ValueError - Sequential model not built

## 错误原因

错误 `ValueError: Sequential model not built` 表示模型还没有被正确构建。这通常发生在：

1. **模型定义代码没有执行**
2. **模型层没有添加完整**
3. **模型没有编译**
4. **模型summary没有调用**（这会触发模型构建）

## 解决方案

### 检查模型定义单元格（Cell 27）

确保模型定义代码完整且已执行：

```python
# Q-Network Baseline Model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


# Define the Q-network
model = Sequential()

# 添加网络层
model.add(Dense(64, input_dim=4, activation='relu'))  # 输入层：4个状态特征
model.add(Dense(64, activation='relu'))                # 隐藏层
model.add(Dense(2, activation='linear'))               # 输出层：2个动作的Q值

# 编译模型
model.compile(loss='mse', optimizer=Adam(learning_rate=lr))

# 打印模型摘要（这很重要！会触发模型构建）
model.summary()
```

### 关键步骤

1. **确保 `model.summary()` 被调用**
   - 这会强制模型构建
   - 如果模型没有构建，`summary()` 会触发构建过程

2. **检查执行顺序**
   - 先运行模型定义cell（Cell 27）
   - 再运行训练cell（Cell 28）

3. **检查变量是否定义**
   - 确保 `lr` 变量已定义（在超参数设置cell中）

## 快速修复步骤

### 步骤1：检查并运行模型定义cell

```python
# 在Jupyter Notebook中，找到模型定义cell并运行
# 确保看到模型summary输出，类似：

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 64)                320       
                                                                 
 dense_1 (Dense)             (None, 64)                4160      
                                                                 
 dense_2 (Dense)             (None, 2)                 130       
                                                                 
=================================================================
Total params: 4610 (18.01 KB)
Trainable params: 4610 (18.01 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```

### 步骤2：检查超参数设置

确保超参数cell已运行：

```python
lr = 0.00025
epoch = 1
episode = 250
epsilon = 1
epsilon_min = 0.01
epsilon_decay = 0.995
gamma = 0.99
```

### 步骤3：修复动作选择代码

在训练循环中，确保动作选择代码正确：

```python
else:
    # remove pass and use 2 lines below
    q_values = model.predict(state, verbose=0)
    action = np.argmax(q_values[0])  # 注意：使用 [0] 索引
```

**注意**：`model.predict()` 返回的形状是 `(batch_size, num_actions)`，所以需要 `[0]` 来获取第一个（也是唯一的）样本的Q值。

## 完整检查清单

- [ ] 模型定义cell已运行
- [ ] 模型summary有输出（证明模型已构建）
- [ ] 超参数cell已运行
- [ ] `lr` 变量已定义
- [ ] 训练cell中的动作选择使用了 `q_values[0]`

## 如果仍然出错

如果完成上述步骤后仍然出错，尝试：

1. **重启内核**：在Jupyter中，Kernel → Restart
2. **重新运行所有cell**：按顺序从模型定义开始运行
3. **检查TensorFlow版本**：确保 `tf.__version__ >= "2.0"`

## 常见错误变体

### 错误1：`lr is not defined`
- **解决**：确保超参数设置cell已运行

### 错误2：`model.predict()` 返回形状错误
- **解决**：使用 `q_values[0]` 而不是 `q_values`

### 错误3：模型未构建
- **解决**：确保运行了 `model.summary()`

