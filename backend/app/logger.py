"""
=============================================================================
文件: logger.py
描述: 日志配置模块

核心功能：
1. 统一日志管理：使用 Loguru 库接管所有日志输出，替代 Python 原生的 logging。
2. 上下文追踪：通过 contextvars 自动在日志中记录 Request ID、User ID 等信息，方便排查问题。
3. 日志分流：
   - 控制台 (Console)：输出 INFO 级别以上的日志，带颜色高亮。
   - 文件 (File)：输出 DEBUG 级别以上的日志，自动按大小分割、压缩和保留。

依赖组件:
- loguru: 一个功能强大且易用的 Python 日志库。
=============================================================================
"""

import sys
from pathlib import Path
from loguru import logger
import logging
import contextvars

# =============================================================================
# 全局配置
# =============================================================================

# 1. 定义项目根目录
# 路径推导: backend/app/logger.py -> backend/app -> backend -> Wiki (项目根目录)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_PATH = PROJECT_ROOT / "logs" / "velo_app.log"

# 定义上下文变量 (Context Variables)
# 这是一个线程安全（协程安全）的全局变量，用于在同一个请求的处理流程中传递信息。
# 比如：在中间件里生成一个 request_id，在后续的所有函数调用里都能取到它，打印在日志里。
request_id_ctx = contextvars.ContextVar("request_id", default=None)
user_id_ctx = contextvars.ContextVar("user_id", default=None)
ip_address_ctx = contextvars.ContextVar("ip_address", default=None)

# 2. 移除 loguru 默认的处理器 (防止重复输出)
logger.remove()

# =============================================================================
# 日志过滤器 (Filter)
# =============================================================================

def context_filter(record):
    """
    上下文注入过滤器
    
    Logic Flow:
        每次打印日志时，自动从 contextvars 里取出当前的 request_id 等信息，
        塞进日志记录的 `extra` 字段里，这样格式化字符串时就能用了。
    """
    record["extra"]["request_id"] = request_id_ctx.get()
    record["extra"]["user_id"] = user_id_ctx.get()
    record["extra"]["ip_address"] = ip_address_ctx.get()
    return True

# =============================================================================
# 日志输出配置 (Sink)
# =============================================================================

# 3. 配置控制台输出 (Console)
# 开发人员看屏幕用的，要求清晰、高亮。
logger.add(
    sys.stderr,
    level="INFO",
    filter=context_filter,
    # 格式说明: 时间 | 级别 | 模块:函数:行号 | RequestID - 消息内容
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <magenta>{extra[request_id]}</magenta> - <level>{message}</level>",
    colorize=True,
)

# 4. 配置文件输出 (File)
# 生产环境查问题用的，要求详细、持久化。
logger.add(
    str(LOG_PATH),
    level="DEBUG", # 记录更详细的调试信息
    filter=context_filter,
    rotation="10 MB", # 文件超过 10MB 就自动切割成新文件
    retention="10 days", # 只保留最近 10 天的日志，防止占满硬盘
    compression="zip", # 切割后的旧日志自动压缩，节省空间
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {extra[request_id]} | {extra[user_id]} | {extra[ip_address]} - {message}",
    encoding="utf-8"
)

# =============================================================================
# 日志拦截器 (Interceptor)
# =============================================================================

class InterceptHandler(logging.Handler):
    """
    标准库 logging 拦截器
    
    Why:
    Uvicorn 和 FastAPI 内部使用的是 Python 自带的 `logging` 模块。
    为了统一管理，我们需要把那些日志“拦截”下来，转发给 `loguru` 来输出。
    """
    def emit(self, record):
        # 获取对应的 Loguru level
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # 找到调用者的 frame (为了在日志里显示正确的文件名和行号，而不是显示 logger.py)
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

# 导出 logger 供其他模块使用
__all__ = ["logger", "InterceptHandler", "request_id_ctx", "user_id_ctx", "ip_address_ctx"]
