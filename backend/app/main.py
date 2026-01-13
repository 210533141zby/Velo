"""
=============================================================================
文件: main.py
描述: 应用入口文件
      配置 FastAPI 应用实例，中间件，路由注册及生命周期管理
=============================================================================
"""

import logging
import time
import uuid
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.database import engine
from app.cache import redis_manager
from app.api.api import api_router
from app.db_init import init_db
from app.logger import logger, InterceptHandler, request_id_ctx, user_id_ctx, ip_address_ctx

# =============================================================================
# 生命周期管理
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    
    启动时:
    1. 配置 Loguru 拦截标准 Logging
    2. 初始化 Redis 连接
    3. 执行数据库初始化
    
    关闭时:
    1. 关闭 Redis 连接
    2. 关闭数据库连接池
    """
    # 1. 配置日志拦截 (将 uvicorn/fastapi 日志重定向到 loguru)
    logging.getLogger().handlers = [InterceptHandler()]
    logging.getLogger("uvicorn.access").handlers = [InterceptHandler()]
    logging.getLogger("uvicorn.error").handlers = [InterceptHandler()]
    
    # 启动阶段
    logger.info("应用正在启动...", extra={"extra_data": {"event": "startup"}})

    try:
        await redis_manager.init_redis()
        
        # 运行数据库初始化和迁移检查 (自动建表)
        await init_db()
        
        print("数据库初始化完成，连接成功！", flush=True)
        logger.info("数据库连接成功", extra={"extra_data": {"event": "db_connected"}})
    except Exception as e:
        logger.error(f"启动过程中发生错误: {e}")
        # 不一定要抛出异常，取决于是否希望应用在数据库失败时仍能启动
        # raise e 
    
    yield
    
    # 关闭阶段
    logger.info("应用正在关闭...", extra={"extra_data": {"event": "shutdown"}})
    await redis_manager.close()
    
    # 关闭数据库连接池
    await engine.dispose()
    logger.info("数据库连接池已关闭", extra={"extra_data": {"event": "db_disconnected"}})

# =============================================================================
# 应用实例配置
# =============================================================================

app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=lifespan,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# =============================================================================
# 中间件配置
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    请求日志中间件
    
    功能:
    1. 生成并设置 Request ID
    2. 记录请求开始和结束
    3. 捕获性能指标和异常
    """
    start_time = time.time()
    
    # 1. 设置 Request ID
    request_id = str(uuid.uuid4())
    request_id_ctx.set(request_id)
    
    # 2. 设置其他上下文信息
    client_ip = request.client.host if request.client else "unknown"
    ip_address_ctx.set(client_ip)
    
    # 模拟用户认证 (实际项目中应从 Token 解析)
    user_id = "Admin" 
    user_id_ctx.set(user_id)
    
    url = request.url.path
    method = request.method
    
    # 排除静态资源和健康检查日志，避免刷屏
    if not url.startswith("/static") and url != "/health":
        logger.info(f"请求开始: {method} {url}", extra={
            "extra_data": {
                "event": "request_start",
                "method": method,
                "url": url,
                "user_agent": request.headers.get("user-agent"),
            }
        })

    try:
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000 # ms
        
        if not url.startswith("/static") and url != "/health":
            logger.info(f"请求结束: {response.status_code}", extra={
                "duration": process_time,
                "extra_data": {
                    "event": "request_end",
                    "status_code": response.status_code,
                    "method": method,
                    "url": url
                }
            })
            
        # 在响应头中返回 Request ID
        response.headers["X-Request-ID"] = request_id
        return response
        
    except Exception as e:
        process_time = (time.time() - start_time) * 1000
        logger.error(f"请求处理异常: {str(e)}", exc_info=True, extra={
            "duration": process_time,
            "extra_data": {
                "event": "request_error",
                "method": method,
                "url": url
            }
        })
        raise e

# =============================================================================
# 路由注册
# =============================================================================

app.include_router(api_router, prefix=settings.API_V1_STR)

# =============================================================================
# 根路由 (前端入口 - 如果需要)
# =============================================================================

@app.get("/")
async def read_root():
    """
    返回 API 信息或前端入口
    """
    return {"message": f"Welcome to {settings.PROJECT_NAME} API"}
