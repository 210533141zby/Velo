"""
=============================================================================
文件: api.py
描述: API 路由汇总

核心功能：
1. 路由注册：将各个功能模块的 Router (documents, folders, agent 等) 统一挂载到主路由上。
2. 路径前缀管理：定义统一的 URL 前缀（如 /documents, /agent）。

依赖组件:
- fastapi.APIRouter: 用于组织路由。
=============================================================================
"""

from fastapi import APIRouter
from app.api import documents, agent, folders, completion

api_router = APIRouter()

# 注册各个模块的路由
# Logic Flow:
# 这里的 prefix 会自动拼接到子模块的所有路由前面。
# 比如 documents.router 里定义了 "/", 实际访问地址就是 "/documents/"。
api_router.include_router(documents.router, prefix="/documents", tags=["documents"]) # 文档管理模块
api_router.include_router(folders.router, prefix="/folders", tags=["folders"])       # 文件夹管理模块
api_router.include_router(agent.router, prefix="/agent", tags=["agent"])             # AI 代理模块
api_router.include_router(completion.router, prefix="/code", tags=["completion"])    # 代码补全模块
