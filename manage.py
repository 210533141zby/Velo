#!/usr/bin/env python3
import os
import sys
import subprocess
import platform
import time
import signal
from pathlib import Path

# =============================================================================
# 配置区域
# =============================================================================

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent

# Redis 配置 (Windows)
REDIS_WIN_DIR = r"D:\Redis"
REDIS_WIN_EXE = "redis-server.exe"
REDIS_WIN_CONF = "redis.windows.conf"

# Redis 配置 (Linux/Mac)
# 通常假设已安装在系统路径或通过包管理器安装
REDIS_UNIX_CMD = "redis-server"

# 后端启动配置
UVICORN_CMD = ["uvicorn", "backend.app.main:app", "--reload", "--host", "127.0.0.1", "--port", "8000"]

# =============================================================================
# 工具函数
# =============================================================================

def print_info(msg, flush=True):
    print(f"\033[94m[INFO] {msg}\033[0m", flush=flush)

def print_success(msg, flush=True):
    print(f"\033[92m[SUCCESS] {msg}\033[0m", flush=flush)

def print_error(msg):
    print(f"\033[91m[ERROR] {msg}\033[0m", flush=True)

def print_warn(msg):
    print(f"\033[93m[WARN] {msg}\033[0m", flush=True)

# =============================================================================
# Redis 管理
# =============================================================================

def start_redis():
    system = platform.system()
    print_info(f"检测到操作系统: {system}")
    
    if system == "Windows":
        redis_path = Path(REDIS_WIN_DIR) / REDIS_WIN_EXE
        conf_path = Path(REDIS_WIN_DIR) / REDIS_WIN_CONF
        
        if not redis_path.exists():
            print_warn(f"未在 {REDIS_WIN_DIR} 找到 Redis。尝试使用系统 PATH...")
            # 尝试直接调用
            try:
                subprocess.Popen(["redis-server"], creationflags=subprocess.CREATE_NEW_CONSOLE)
                print_success("Redis 服务已尝试启动 (系统路径)")
                return
            except FileNotFoundError:
                print_error("无法启动 Redis。请确保已安装 Redis 或修改脚本中的路径。")
                print_warn("系统将降级使用内存缓存运行。")
                return

        print_info(f"正在启动 Windows Redis: {redis_path}")
        try:
            # 在新窗口启动 Redis
            cmd = f'start "Wiki Redis Server" "{redis_path}" "{conf_path}"'
            os.system(cmd)
            print_success("Redis 服务已在后台启动")
        except Exception as e:
            print_error(f"启动 Redis 失败: {e}")

    elif system in ["Linux", "Darwin"]: # Darwin is macOS
        print_info("正在启动 Unix Redis...")
        try:
            # 检查 Redis 是否已经在运行
            check = subprocess.run(["pgrep", "redis-server"], stdout=subprocess.PIPE)
            if check.returncode == 0:
                print_info("Redis 服务已在运行")
                return

            # 尝试后台启动
            subprocess.Popen([REDIS_UNIX_CMD], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print_success("Redis 服务已启动")
        except FileNotFoundError:
            print_warn("未找到 redis-server 命令。请使用 apt/yum/brew 安装 Redis。")
            print_warn("系统将降级使用内存缓存运行。")

# =============================================================================
# 数据库管理
# =============================================================================

def init_db():
    print_info("正在检查并初始化数据库...")
    # 调用 backend/init_db_script.py (如果存在) 或直接调用模块
    # 这里我们直接使用 python -m 方式运行，确保路径正确
    try:
        # 设置 PYTHONPATH 包含 backend
        env = os.environ.copy()
        python_path = sys.executable
        
        # 运行初始化脚本
        script_path = BASE_DIR / "backend" / "init_db_script.py"
        if script_path.exists():
            # 捕获输出以避免干扰主界面，除非出错
            result = subprocess.run(
                [python_path, str(script_path)], 
                env=env, 
                capture_output=True, 
                text=True
            )
            if result.returncode == 0:
                print_success("数据库初始化完成")
            else:
                print_error("数据库初始化脚本返回错误")
                print_error(result.stderr)
                print_info(result.stdout)
        else:
            print_error(f"找不到初始化脚本: {script_path}")
            
    except subprocess.CalledProcessError as e:
        print_error(f"数据库初始化失败: {e}")
    except Exception as e:
        print_error(f"执行出错: {e}")

# =============================================================================
# 主程序启动
# =============================================================================

def start_server():
    print_info("正在启动后端服务 (FastAPI)...")
    try:
        # 添加 backend 到 PYTHONPATH
        env = os.environ.copy()
        backend_path = BASE_DIR / "backend"
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{str(backend_path)}{os.pathsep}{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = str(backend_path)
            
        # 使用 sys.executable -m uvicorn 确保使用相同的 Python 环境
        # 并且将 app 视为顶级包 (因为 backend 在 PYTHONPATH 中)
        # 添加 --reload-delay 1.0 以缓解 Windows 下文件锁导致的重载卡死问题
        cmd_str = f'"{sys.executable}" -m uvicorn app.main:app --reload --reload-delay 1.0 --host 127.0.0.1 --port 8000'
        
        # 运行 Uvicorn
        # 使用 os.system 直接调用，避免 subprocess 对信号和缓冲区的过度封装，
        # 从而解决 WatchFiles 重载时导致父进程挂起的问题
        print_info(f"执行命令: {cmd_str}")
        
        # 为了让 os.system 使用正确的 PYTHONPATH，我们需要更新当前进程的环境变量
        # 或者在命令中设置(Windows下比较麻烦)，所以直接修改 os.environ
        os.environ["PYTHONPATH"] = env["PYTHONPATH"]
        
        ret = os.system(cmd_str)
        
        if ret != 0:
             # os.system 返回的是 exit status (wait result)，在 Windows 上通常是 exit code
             # 如果是被信号杀死的，可能不一样，但这里主要检测非正常退出
             # 正常 Ctrl+C 退出可能会返回 0 或 1 (取决于实现)，这里不做严格检查
             pass
    except KeyboardInterrupt:
        print_info("服务已停止")
    except Exception as e:
        print_error(f"服务启动失败: {e}")

# =============================================================================
# 命令行入口
# =============================================================================

def main():
    print("="*60)
    print("      Wiki AI 项目统一管理 (Cross-Platform)")
    print("="*60)
    
    # 1. 启动 Redis
    start_redis()
    
    # 等待 Redis 启动
    time.sleep(1)
    
    # 2. 初始化数据库
    # 注意: FastAPI 应用在启动时也会检查数据库 (app.main:lifespan)
    # 为了加快启动速度，我们在这里跳过 redundant 的检查，除非明确需要独立运行
    # init_db() 
    
    print("-" * 60)
    print("项目启动中... 按 Ctrl+C 停止")
    print(f"访问地址: http://127.0.0.1:8000")
    print("-" * 60)
    
    # 3. 启动应用
    start_server()

if __name__ == "__main__":
    main()
