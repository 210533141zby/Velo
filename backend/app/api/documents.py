"""
=============================================================================
文件: documents.py
描述: 文档管理接口
      提供文档的创建、查询、更新等 RESTful API
      所有业务逻辑已迁移至 Service 层，本文件仅负责路由和参数解析
=============================================================================
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.database import get_db, AsyncSessionLocal
from app.schemas import DocumentCreate, DocumentResponse, DocumentSummary
from app.services.document_service import DocumentService
from app.services.log_service import LogService

router = APIRouter()

# -----------------------------------------------------------------------------
# 辅助函数: 获取服务实例
# -----------------------------------------------------------------------------

def get_document_service(db: AsyncSession = Depends(get_db)) -> DocumentService:
    """依赖注入: 获取文档服务实例"""
    return DocumentService(db)

def get_log_service(db: AsyncSession = Depends(get_db)) -> LogService:
    """依赖注入: 获取日志服务实例"""
    return LogService(db)

# -----------------------------------------------------------------------------
# 路由定义
# -----------------------------------------------------------------------------

@router.post("/", response_model=DocumentResponse)
async def create_document(
    doc: DocumentCreate, 
    background_tasks: BackgroundTasks,
    doc_service: DocumentService = Depends(get_document_service)
):
    """
    创建新文档
    
    - 调用 DocumentService 创建文档
    - 触发后台任务: 记录操作日志
    """
    new_doc = await doc_service.create_document(doc, background_tasks)
    
    # 定义后台日志记录包装器
    async def log_wrapper(doc_id: int):
        async with AsyncSessionLocal() as session:
            svc = LogService(session)
            await svc.log_operation("CREATE_DOC", "DOCUMENT", str(doc_id))
            
    background_tasks.add_task(log_wrapper, new_doc.id)
    
    return new_doc

@router.get("/", response_model=List[DocumentSummary])
async def list_documents(
    doc_service: DocumentService = Depends(get_document_service)
):
    """
    获取所有文档列表
    
    - 优先从 Redis 缓存读取
    - 返回文档摘要列表
    """
    return await doc_service.list_documents()

@router.put("/{doc_id}", response_model=DocumentResponse)
async def update_document(
    doc_id: int,
    doc_update: DocumentCreate,
    background_tasks: BackgroundTasks,
    doc_service: DocumentService = Depends(get_document_service)
):
    """
    更新文档
    """
    updated_doc = await doc_service.update_document(doc_id, doc_update, background_tasks)
    if not updated_doc:
        raise HTTPException(status_code=404, detail="未找到文档")
        
    # 记录操作日志
    async def log_wrapper(d_id: int):
        async with AsyncSessionLocal() as session:
            svc = LogService(session)
            await svc.log_operation("UPDATE_DOC", "DOCUMENT", str(d_id))
            
    background_tasks.add_task(log_wrapper, doc_id)
    
    return updated_doc

@router.get("/{doc_id}", response_model=DocumentResponse)
async def get_document(
    doc_id: int,
    doc_service: DocumentService = Depends(get_document_service)
):
    """
    获取单个文档详情
    """
    doc = await doc_service.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="未找到文档")
    return doc

@router.delete("/{doc_id}")
async def delete_document(
    doc_id: int,
    background_tasks: BackgroundTasks,
    doc_service: DocumentService = Depends(get_document_service)
):
    """
    删除文档 (软删除)
    """
    success = await doc_service.delete_document(doc_id, background_tasks)
    if not success:
        raise HTTPException(status_code=404, detail="未找到文档")
        
    # 记录操作日志
    async def log_wrapper(d_id: int):
        async with AsyncSessionLocal() as session:
            svc = LogService(session)
            await svc.log_operation("DELETE_DOC", "DOCUMENT", str(d_id))
            
    background_tasks.add_task(log_wrapper, doc_id)
    
    return {"message": "文档已删除"}