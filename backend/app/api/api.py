from fastapi import APIRouter
from app.api import documents, agent, folders, completion

api_router = APIRouter()

# 注册各个模块的路由
api_router.include_router(documents.router, prefix="/documents", tags=["documents"]) # 文档管理模块
api_router.include_router(folders.router, prefix="/folders", tags=["folders"])       # 文件夹管理模块
api_router.include_router(agent.router, prefix="/agent", tags=["agent"])             # AI 代理模块
api_router.include_router(completion.router, prefix="/code", tags=["completion"])    # 代码补全模块
