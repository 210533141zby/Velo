import sys
from pathlib import Path
from loguru import logger
import logging
import contextvars

# 1. 定义项目根目录 (backend/ 的上一级，即 Wiki/)
# backend/app/logger.py -> backend/app -> backend -> Wiki
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_PATH = PROJECT_ROOT / "logs" / "velo_app.log"

# 定义上下文变量 (用于在日志中追踪请求信息)
request_id_ctx = contextvars.ContextVar("request_id", default=None)
user_id_ctx = contextvars.ContextVar("user_id", default=None)
ip_address_ctx = contextvars.ContextVar("ip_address", default=None)

# 2. 移除默认配置
logger.remove()

# 过滤器函数，将上下文变量注入到 extra 字段中
def context_filter(record):
    record["extra"]["request_id"] = request_id_ctx.get()
    record["extra"]["user_id"] = user_id_ctx.get()
    record["extra"]["ip_address"] = ip_address_ctx.get()
    return True

# 3. 配置 Console 输出 (INFO级别，高亮)
logger.add(
    sys.stderr,
    level="INFO",
    filter=context_filter,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <magenta>{extra[request_id]}</magenta> - <level>{message}</level>",
    colorize=True,
)

# 4. 配置 File 输出 (DEBUG级别，滚动/保留/压缩)
logger.add(
    str(LOG_PATH),
    level="DEBUG",
    filter=context_filter,
    rotation="10 MB",
    retention="10 days",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {extra[request_id]} | {extra[user_id]} | {extra[ip_address]} - {message}",
    encoding="utf-8"
)

# 5. 定义 InterceptHandler 用于拦截标准 logging (Uvicorn/FastAPI)
class InterceptHandler(logging.Handler):
    def emit(self, record):
        # 获取对应的 Loguru level
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # 找到调用者的 frame
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

# 导出 logger 供其他模块使用
__all__ = ["logger", "InterceptHandler", "request_id_ctx", "user_id_ctx", "ip_address_ctx"]
