import logging
import sys
import os
import json
import time
import uuid
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path

# =============================================================================
# 上下文变量 (ContextVars)
# 用于在异步请求中传递 Request ID 和 User Info
# =============================================================================
request_id_ctx = ContextVar("request_id", default=None)
user_id_ctx = ContextVar("user_id", default="Guest")
ip_address_ctx = ContextVar("ip_address", default="0.0.0.0")

# =============================================================================
# 日志配置
# =============================================================================

# 日志目录
LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

class JSONFormatter(logging.Formatter):
    """
    JSON 格式化器
    将日志记录输出为 JSON 格式，包含所有必需字段
    """
    def format(self, record):
        # 基础字段
        log_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "file": record.filename,
            "line": record.lineno,
        }

        # 上下文信息
        log_record["request_id"] = request_id_ctx.get() or "system"
        log_record["user_id"] = user_id_ctx.get()
        log_record["client_ip"] = ip_address_ctx.get()

        # 异常堆栈
        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)
            log_record["stack_trace"] = self.formatStack(record.stack_info) if record.stack_info else None

        # 额外字段 (通过 extra={} 传递)
        if hasattr(record, "extra_data"):
            log_record.update(record.extra_data)

        # 性能指标 (如果存在)
        if hasattr(record, "duration"):
            log_record["duration_ms"] = f"{record.duration:.2f}ms"

        return json.dumps(log_record, ensure_ascii=False)

class ConsoleFormatter(logging.Formatter):
    """
    控制台彩色格式化器
    开发环境下提供更直观的可读性
    """
    COLORS = {
        'DEBUG': '\033[94m',    # Blue
        'INFO': '\033[92m',     # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[41m', # Red background
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        request_id = request_id_ctx.get()
        request_str = f"[{request_id}]" if request_id else ""
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        msg = f"{color}{timestamp} | {record.levelname:8} | {request_str} {record.getMessage()}{self.RESET}"
        
        if record.exc_info:
            msg += f"\n{self.formatException(record.exc_info)}"
            
        return msg

def setup_logger(name="wiki_app"):
    """
    初始化 Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO) # 默认级别

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    # 控制台 Handler (带颜色)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ConsoleFormatter())
    logger.addHandler(console_handler)

    # 文件 Handler (JSON 格式，生产环境友好)
    file_handler = logging.FileHandler(LOG_DIR / "app.json.log", encoding="utf-8")
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)
    
    # 普通文本文件 Handler (便于快速查看)
    text_handler = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
    text_handler.setFormatter(ConsoleFormatter()) 
    logger.addHandler(text_handler)

    # 专门的用户操作日志 (user_ops.log)
    # 格式要求: 简洁明了，一行一个操作
    user_ops_handler = logging.FileHandler(LOG_DIR / "user_ops.log", encoding="utf-8")
    class UserOpsFormatter(logging.Formatter):
        def format(self, record):
            # 自定义格式：时间 | 用户 | 操作 | 资源 | 状态
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 尝试从 extra_data 中获取详细信息
            extra = getattr(record, "extra_data", {})
            user = user_id_ctx.get() or "Guest"
            event = extra.get("event", record.getMessage())
            
            # 如果不是特定的业务日志，可能没有 details
            return f"{timestamp} | {user} | {event} | {record.getMessage()}"

    user_ops_handler.setFormatter(UserOpsFormatter())
    # 只有当日志显式标记为 user_op 时才记录到这个文件？
    # 或者我们简单地让所有 INFO 级别日志都流向这里？
    # 为了避免太杂，我们可以只让 LogService 调用特定的 logger，或者在这里过滤
    # 简单起见，我们添加一个 Filter
    class UserOpFilter(logging.Filter):
        def filter(self, record):
            # 只有带有 extra_data 且 event 不为空的才记录，或者消息包含特定关键词
            return hasattr(record, "extra_data") and "event" in record.extra_data

    user_ops_handler.addFilter(UserOpFilter())
    logger.addHandler(user_ops_handler)

    return logger

# 全局 Logger 实例
logger = setup_logger()

# 配置第三方库日志级别，避免干扰
logging.getLogger("uvicorn").setLevel(logging.INFO)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING) # 减少访问日志，使用自定义中间件记录
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING) # 仅记录警告和错误