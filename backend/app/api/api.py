"""注册后端对外暴露的全部 HTTP 路由。

这个文件本身不承载业务逻辑，只负责把内容管理、补全和知识库问答
三个子路由挂到统一入口上，便于 `main.py` 在启动时一次性接入。
"""

from fastapi import APIRouter

from app.api import completion, content, rag

api_router = APIRouter()
api_router.include_router(content.documents_router)
api_router.include_router(content.folders_router)
api_router.include_router(rag.rag_router)
api_router.include_router(completion.completion_router)
