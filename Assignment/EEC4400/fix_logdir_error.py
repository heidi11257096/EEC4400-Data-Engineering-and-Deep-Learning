# 修复 TensorBoard 日志目录错误的完整解决方案

# 方法1：修改 get_run_logdir 函数（推荐）
def get_run_logdir(k):
    root_logdir = os.path.join(os.curdir, "eec4400_logs", k)
    
    # 确保所有父目录都存在
    os.makedirs(root_logdir, exist_ok=True)
    
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    logdir = os.path.join(root_logdir, run_id)
    
    # 也确保最终的运行目录存在
    os.makedirs(logdir, exist_ok=True)
    
    print(f"TensorBoard log directory: {logdir}")
    return logdir

# 方法2：如果方法1不行，先手动清理（如果 eec4400_logs 是文件）
import os
import shutil

# 检查并清理（仅在必要时运行）
if os.path.exists("eec4400_logs"):
    if os.path.isfile("eec4400_logs"):
        # 如果 eec4400_logs 是文件，删除它
        os.remove("eec4400_logs")
        print("已删除 eec4400_logs 文件")
    elif os.path.isdir("eec4400_logs"):
        print("eec4400_logs 目录已存在")

# 方法3：更健壮的版本（处理所有情况）
def get_run_logdir_robust(k):
    root_logdir = os.path.join(os.curdir, "eec4400_logs", k)
    
    # 检查并处理 eec4400_logs 如果是文件的情况
    parent_dir = os.path.join(os.curdir, "eec4400_logs")
    if os.path.exists(parent_dir) and os.path.isfile(parent_dir):
        os.remove(parent_dir)
        print(f"已删除文件 {parent_dir}，将创建目录")
    
    # 确保所有父目录都存在
    try:
        os.makedirs(root_logdir, exist_ok=True)
    except OSError as e:
        print(f"创建目录失败: {e}")
        raise
    
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    logdir = os.path.join(root_logdir, run_id)
    
    # 确保最终的运行目录也存在
    os.makedirs(logdir, exist_ok=True)
    
    # 验证目录是否真的存在
    if not os.path.isdir(logdir):
        raise ValueError(f"无法创建目录: {logdir}")
    
    print(f"TensorBoard log directory: {logdir}")
    return logdir

