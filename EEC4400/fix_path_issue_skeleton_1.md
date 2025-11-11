# 修复 skeleton_1 文件路径问题

## 问题分析

错误信息：`FailedPreconditionError: . is not a directory`

**根本原因**：
- `get_run_logdir` 函数使用了 `os.curdir`（返回 `'.'`）而不是实际路径
- 没有创建目录
- 返回的是相对路径，TensorBoard 无法正确识别

## 解决方案

### 步骤1：找到并修复 `get_run_logdir` 函数

找到 "Setting up Tensorboard" 部分的 cell（大约在第 25 个 cell），将函数修改为：

```python
def get_run_logdir(k):
    # 使用 os.getcwd() 获取当前工作目录的实际路径，而不是 os.curdir (返回 '.')
    root_logdir = os.path.join(os.getcwd(), "eec4400_logs", k)
    
    # 确保所有父目录都存在
    os.makedirs(root_logdir, exist_ok=True)
    
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    logdir = os.path.join(root_logdir, run_id)
    
    # 确保最终的运行目录也存在
    os.makedirs(logdir, exist_ok=True)
    
    # 转换为绝对路径并规范化
    logdir = os.path.abspath(logdir)
    
    # 验证目录确实存在
    if not os.path.isdir(logdir):
        raise ValueError(f"无法创建或访问目录: {logdir}")
    
    print(f"TensorBoard log directory: {logdir}")
    return logdir
```

### 步骤2：关键修改点

**原代码问题**：
```python
root_logdir = os.path.join(os.curdir, "eec4400_logs", k)  # ❌ os.curdir 返回 '.'
```

**修复后**：
```python
root_logdir = os.path.join(os.getcwd(), "eec4400_logs", k)  # ✅ 使用实际路径
os.makedirs(root_logdir, exist_ok=True)  # ✅ 创建目录
logdir = os.path.abspath(logdir)  # ✅ 转换为绝对路径
```

### 步骤3：执行顺序

1. **找到函数定义 cell**（"Setting up Tensorboard" 部分）
2. **替换函数代码**为上面的完整版本
3. **运行该 cell**，确认没有错误
4. **重新运行训练 cell**，应该可以正常工作

## 验证修复

运行函数测试：

```python
# 测试函数
test_logdir = get_run_logdir("test")
print(f"返回值: {test_logdir}")
print(f"是否为目录: {os.path.isdir(test_logdir)}")
print(f"是否为绝对路径: {os.path.isabs(test_logdir)}")
```

应该看到：
- 返回完整的绝对路径（如 `D:\...\eec4400_logs\test\run_...`）
- 目录确实存在
- 路径是绝对路径

## 如果仍然有问题

### 方法1：在训练 cell 中添加验证

在创建 TensorBoard callback 之前：

```python
model_dir = "q_net_baseline"

# 获取日志目录
logdir = get_run_logdir(model_dir)

# 验证 logdir
print(f"logdir 类型: {type(logdir)}")
print(f"logdir 值: {logdir}")
print(f"是否为目录: {os.path.isdir(logdir)}")

# 确保是绝对路径
logdir = os.path.abspath(logdir) if not os.path.isabs(logdir) else logdir

# 创建 TensorBoard callback
cb = keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
```

### 方法2：暂时禁用 TensorBoard（测试训练代码）

如果路径问题仍然存在，可以暂时禁用 TensorBoard 来测试训练代码：

```python
# 暂时禁用 TensorBoard
# cb = keras.callbacks.TensorBoard(log_dir = get_run_logdir(model_dir), histogram_freq=1)
cb = None

# 在 model.fit 中
if cb is not None:
    model.fit(..., callbacks=[cb])
else:
    model.fit(...)  # 不使用 callbacks
```

## 完整修复后的函数对比

### 原代码（有问题）：
```python
def get_run_logdir(k):
    root_logdir = os.path.join(os.curdir, "eec4400_logs", k)
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
```

### 修复后：
```python
def get_run_logdir(k):
    root_logdir = os.path.join(os.getcwd(), "eec4400_logs", k)
    os.makedirs(root_logdir, exist_ok=True)
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    logdir = os.path.join(root_logdir, run_id)
    os.makedirs(logdir, exist_ok=True)
    logdir = os.path.abspath(logdir)
    if not os.path.isdir(logdir):
        raise ValueError(f"无法创建或访问目录: {logdir}")
    print(f"TensorBoard log directory: {logdir}")
    return logdir
```

## 关键改进

1. ✅ 使用 `os.getcwd()` 而不是 `os.curdir`
2. ✅ 创建所有必要的目录
3. ✅ 返回绝对路径
4. ✅ 验证目录存在
5. ✅ 添加调试打印

修复后，TensorBoard 应该能够正常工作！

