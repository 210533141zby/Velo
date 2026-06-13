"""内容域 HTTP 接口。

这里承接前端文档树、正文编辑和文件夹管理相关的请求，并把
“保存正文后触发索引刷新”“删除文档后同步清理向量索引”等
跨模块动作串起来。
"""

from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas import DocumentCreate, DocumentResponse, DocumentSummary, DocumentUpdate, FolderCreate, FolderResponse
from app.services.audit.log_service import record_operation_log
from app.services.content.content_service import DocumentService, FolderService
from app.services.rag.vector_index_service import delete_document_chunks, index_document_chunks

documents_router = APIRouter(prefix='/documents', tags=['documents'])
folders_router = APIRouter(prefix='/folders', tags=['folders'])


def get_document_service(db: AsyncSession = Depends(get_db)) -> DocumentService:
    """为当前请求创建文档服务实例。

    通过依赖注入把数据库会话交给服务层，避免路由函数直接操作 ORM，
    也便于后续统一测试和替换实现。
    """
    return DocumentService(db)


def get_folder_service(db: AsyncSession = Depends(get_db)) -> FolderService:
    """为当前请求创建文件夹服务实例。

    文件夹操作与文档操作共享同一数据库会话，但保持独立服务对象，
    便于接口职责清晰，也方便后续分别扩展。
    """
    return FolderService(db)


@documents_router.post('/', response_model=DocumentResponse)
async def create_document(
    doc: DocumentCreate,
    background_tasks: BackgroundTasks,
    doc_service: DocumentService = Depends(get_document_service),
):
    """创建一篇新文档，并在有正文时安排后台建索引。

    文档创建完成后不会阻塞等待向量化结束，而是把索引任务放入
    `BackgroundTasks`，保证编辑器侧响应更快。
    """
    new_doc = await doc_service.create_document(doc)
    if new_doc.content:
        background_tasks.add_task(index_document_chunks, new_doc.id, new_doc.title, new_doc.content)
    background_tasks.add_task(record_operation_log, 'CREATE_DOC', 'DOCUMENT', str(new_doc.id))
    return new_doc


@documents_router.get('/', response_model=List[DocumentSummary])
async def list_documents(doc_service: DocumentService = Depends(get_document_service)):
    """返回当前全部有效文档的摘要列表。

    列表接口只返回标题、时间和正文预览，供前端左侧列表渲染使用，
    不承担整篇正文回传职责。
    """
    return await doc_service.get_all_documents()


@documents_router.put('/{doc_id}', response_model=DocumentResponse)
async def update_document(
    doc_id: int,
    doc_update: DocumentUpdate,
    background_tasks: BackgroundTasks,
    doc_service: DocumentService = Depends(get_document_service),
):
    """更新文档内容，并根据正文变化同步维护检索索引。

    只要标题或正文发生变化，就会重新索引；如果正文被清空，
    则删除该文档对应的向量片段，避免知识库继续命中过期内容。
    """
    updated_doc = await doc_service.update_document(doc_id, doc_update)
    if not updated_doc:
        raise HTTPException(status_code=404, detail='未找到文档')
    if doc_update.title is not None or doc_update.content is not None:
        if updated_doc.content:
            background_tasks.add_task(index_document_chunks, updated_doc.id, updated_doc.title, updated_doc.content)
        else:
            background_tasks.add_task(delete_document_chunks, updated_doc.id)
    background_tasks.add_task(record_operation_log, 'UPDATE_DOC', 'DOCUMENT', str(doc_id))
    return updated_doc


@documents_router.get('/{doc_id}', response_model=DocumentResponse)
async def get_document(doc_id: int, doc_service: DocumentService = Depends(get_document_service)):
    """读取单篇文档的完整内容。"""
    doc = await doc_service.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail='未找到文档')
    return doc


@documents_router.delete('/{doc_id}')
async def delete_document(
    doc_id: int,
    background_tasks: BackgroundTasks,
    doc_service: DocumentService = Depends(get_document_service),
):
    """逻辑删除文档，并异步清理其向量索引与操作日志。"""
    success = await doc_service.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail='未找到文档')
    background_tasks.add_task(delete_document_chunks, doc_id)
    background_tasks.add_task(record_operation_log, 'DELETE_DOC', 'DOCUMENT', str(doc_id))
    return {'message': '文档已删除'}


@folders_router.post('/', response_model=FolderResponse)
async def create_folder(
    folder: FolderCreate,
    background_tasks: BackgroundTasks,
    folder_service: FolderService = Depends(get_folder_service),
):
    """创建文件夹节点，供前端文档树挂载使用。"""
    new_folder = await folder_service.create_folder(folder)
    background_tasks.add_task(record_operation_log, 'CREATE', 'FOLDER', str(new_folder.id))
    return new_folder


@folders_router.get('/all', response_model=List[FolderResponse])
async def read_all_folders(folder_service: FolderService = Depends(get_folder_service)):
    """返回全部有效文件夹，用于一次性绘制完整目录树。"""
    return await folder_service.get_all_folders()


@folders_router.get('/{folder_id}', response_model=FolderResponse)
async def read_folder(folder_id: int, folder_service: FolderService = Depends(get_folder_service)):
    """读取单个文件夹节点的元信息。"""
    folder = await folder_service.get_folder(folder_id)
    if folder is None:
        raise HTTPException(status_code=404, detail='未找到文件夹')
    return folder


@folders_router.get('/{folder_id}/contents')
async def read_folder_contents(folder_id: int, folder_service: FolderService = Depends(get_folder_service)):
    """返回指定文件夹下的直属子文件夹与文档列表。

    这里不做递归展开，只返回当前层级内容，方便前端按需懒加载。
    """
    return await folder_service.get_folder_contents(None if folder_id == 0 else folder_id)
