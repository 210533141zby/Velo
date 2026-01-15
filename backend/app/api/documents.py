"""
=============================================================================
文件: documents.py
描述: 文档管理接口

核心功能：
1. 文档 CRUD：提供创建、查询、更新、删除文档的 HTTP 接口。
2. 职责分离：只负责接收请求、参数校验和返回响应，具体业务逻辑委托给 DocumentService。
3. 审计日志：在执行关键操作（增删改）后，通过后台任务记录操作日志。

依赖组件:
- DocumentService: 处理文档业务逻辑。
- LogService: 处理日志记录。
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
    
    Logic Flow:
        1. **业务处理**: 调用 `doc_service.create_document` 完成数据库写入和 AI 索引触发。
        2. **日志记录**: 添加一个后台任务，记录 "CREATE_DOC" 操作。
        
    Why:
        为什么要用内部函数 `log_wrapper`？
        因为 `background_tasks` 需要一个可调用的函数。这里用闭包封装一下，
        方便在后台任务里重新获取一个新的 DB Session (AsyncSessionLocal)，
        避免和主请求共用 Session 导致“连接已关闭”的报错。
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
    
    Logic Flow:
        直接调用 Service 层方法，Service 层会自动处理缓存逻辑 (Redis)。
        返回的是摘要列表 (DocumentSummary)，不包含全文内容。
    """
    return await doc_service.get_all_documents()

@router.put("/{doc_id}", response_model=DocumentResponse)
async def update_document(
    doc_id: int,
    doc_update: DocumentCreate,
    background_tasks: BackgroundTasks,
    doc_service: DocumentService = Depends(get_document_service)
):
    """
    更新文档
    
    Logic Flow:
        1. **业务处理**: 调用 Service 更新文档内容。
        2. **404 检查**: 如果 Service 返回 None，说明文档不存在，抛出 404 异常。
        3. **日志记录**: 记录 "UPDATE_DOC" 操作。
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
    
    Logic Flow:
        1. **业务处理**: 调用 Service 执行软删除。
        2. **日志记录**: 记录 "DELETE_DOC" 操作。
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
