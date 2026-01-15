"""
=============================================================================
文件: document_service.py
描述: 文档核心业务服务
      本服务负责所有与文档生命周期相关的操作，包括 CRUD、缓存同步、
      以及触发后台任务 (如 RAG 索引构建)。

依赖关系:
- agent_service: 用于创建文档后，异步触发向量索引构建。
- redis_manager: 用于在读写操作时维护文档列表的缓存一致性。
- models.Document: 数据库模型。
=============================================================================
"""

from datetime import datetime
import json
from sqlalchemy import select, func
from fastapi import BackgroundTasks
from app.models import Document
from app.schemas import DocumentCreate
from app.cache import redis_manager
from app.logger import logger
from app.services.base import BaseService
from app.services.agent_service import AgentService

class DocumentService(BaseService):
    """
    文档服务类
    处理文档的 CRUD、缓存管理及与 Agent 服务的交互
    """
    
    async def create_document(self, doc_data: DocumentCreate, background_tasks: BackgroundTasks) -> Document:
        """
        创建新文档
        
        Input:
            doc_data (DocumentCreate): 包含 title, content, folder_id 等
            background_tasks (BackgroundTasks): FastAPI 后台任务管理器
        Output:
            Document: 创建成功的数据库对象
            
        Logic Flow:
            1. **DB Insert**: 将 Markdown 内容原样写入数据库。
            2. **Commit & Refresh**: 提交事务并刷新对象以获取生成的 ID。
            3. **Cache Invalidation**: 调用 `_invalidate_cache` 清除文档列表缓存，确保前端拉取到最新列表。
            4. **Async Indexing**: 实例化 `AgentService` 并添加后台任务 `index_document`。
               - Why Background? 索引过程涉及 OpenAI API 调用和向量库写入，耗时较长 (1-3s)，
                 不应阻塞创建接口的响应。
        """
        # Markdown 内容原样保存
        new_doc = Document(
            title=doc_data.title, 
            content=doc_data.content, 
            folder_id=doc_data.folder_id
        )
        self.db.add(new_doc)
        await self.db.commit()
        await self.db.refresh(new_doc)
        
        # 缓存失效
        await self._invalidate_cache()
        
        # RAG 索引
        # 在此处实例化 AgentService，避免类级别的循环依赖
        agent_service = AgentService(self.db)
        background_tasks.add_task(agent_service.index_document, new_doc.id, new_doc.title, new_doc.content)
        
        logger.info(f"创建文档: {new_doc.title}", extra={
            "extra_data": {
                "event": "document_created",
                "document_id": new_doc.id,
                "folder_id": new_doc.folder_id,
                "title": new_doc.title
            }
        })
        
        return new_doc

    async def get_document(self, doc_id: int) -> Document:
        """
        获取单个文档详情
        
        Input: doc_id
        Output: Document or None
        
        Logic Flow:
            1. **Direct DB Query**: 直接查询数据库，并过滤 `is_active=True` (软删除检查)。
            2. **Why No Cache?**: 单个文档详情通常用于编辑场景，需要极高的实时性。
               如果使用缓存，可能会导致用户在多端编辑时看到旧数据。
               且文档内容可能较大，缓存单个大文档对 Redis 内存压力较大。
        """
        query = select(Document).where(Document.id == doc_id, Document.is_active == True)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def update_document(self, doc_id: int, doc_update: DocumentCreate, background_tasks: BackgroundTasks) -> Document:
        """
        更新文档
        
        Logic Flow:
            1. **Check Existence**: 检查文档是否存在且未被删除。
            2. **Update Fields**: 更新标题、内容、文件夹等字段。
            3. **Update Timestamp**: 手动更新 `updated_at`。
            4. **Cache Invalidation**: 清除文档列表缓存 (因为标题或摘要可能变了)。
            5. **Re-Indexing**: 触发后台任务重新生成向量索引。
               - 注意: 这是一个全量更新索引的操作 (AgentService 侧通常会先覆盖或重新添加)。
        """
        doc = await self.get_document(doc_id)
        if not doc:
            return None
            
        doc.title = doc_update.title
        doc.content = doc_update.content # Markdown 原样保存
        doc.updated_at = datetime.utcnow()
        
        if doc_update.folder_id is not None:
             doc.folder_id = doc_update.folder_id
        
        await self.db.commit()
        await self.db.refresh(doc)
        
        # 缓存失效
        await self._invalidate_cache()
        
        # 更新索引
        agent_service = AgentService(self.db)
        background_tasks.add_task(agent_service.index_document, doc.id, doc.title, doc.content)
        
        logger.info(f"更新文档: {doc.title}", extra={
            "extra_data": {
                "event": "document_updated",
                "document_id": doc.id,
                "title": doc.title
            }
        })
        
        return doc

    async def list_documents(self) -> list[dict]:
        """
        获取文档列表
        
        Logic Flow (Read-Through Pattern):
            1. **Cache Get**: 尝试从 Redis 读取 `documents_list`。
               - 如果命中，直接返回 (极速响应)。
            2. **DB Query**: 如果缓存未命中，从数据库查询。
               - 只查询必要字段 (ID, Title, Substr(Content)作为摘要)。
               - 按 `updated_at` 倒序排列。
            3. **Cache Set**: 将查询结果序列化后存入 Redis，设置 5 分钟过期 (TTL=300)。
               - 为什么是 5 分钟? 文档列表不需要毫秒级实时，5分钟的延迟在大多数 Wiki 场景可接受。
               - 且 create/update/delete 操作会主动失效缓存，保证了即时一致性。
        """
        # 1. 尝试缓存
        try:
            cached_docs = await redis_manager.get("documents_list")
            if cached_docs:
                return json.loads(cached_docs)
        except Exception as e:
            logger.error(f"缓存读取错误: {e}", exc_info=True)

        # 2. 查询数据库
        query = select(
            Document.id,
            Document.title,
            func.substr(Document.content, 1, 200).label("content"),
            Document.created_at,
            Document.updated_at,
            Document.folder_id
        ).where(Document.is_active == True).order_by(Document.updated_at.desc())
        
        result = await self.db.execute(query)
        rows = result.all()
        
        docs_data = [
            {
                "id": row.id,
                "title": row.title,
                "content": row.content,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                "folder_id": row.folder_id
            }
            for row in rows
        ]
        
        # 3. 写入缓存 (TTL 5分钟)
        try:
            await redis_manager.set("documents_list", json.dumps(docs_data), expire=300)
        except Exception as e:
            logger.error(f"缓存写入错误: {e}", exc_info=True)
            
        return docs_data

    async def delete_document(self, doc_id: int, background_tasks: BackgroundTasks) -> bool:
        """
        删除文档 (软删除)
        
        Logic Flow:
            1. **Soft Delete**: 将 `is_active` 置为 False，保留数据以便恢复。
            2. **Cache Invalidation**: 清除列表缓存。
            3. **Index Deletion**: 触发后台任务，从向量库中物理删除该文档的索引。
               - 必须删除索引，否则 RAG 仍会检索到已删除的文档内容。
        """
        doc = await self.get_document(doc_id)
        if not doc:
            return False
            
        doc.is_active = False
        await self.db.commit()
        
        # 缓存失效
        await self._invalidate_cache()
        
        # 从向量库中删除
        agent_service = AgentService(self.db)
        background_tasks.add_task(agent_service.delete_document_index, doc_id)
        
        logger.info(f"删除文档: {doc_id}", extra={
            "extra_data": {
                "event": "document_deleted",
                "document_id": doc_id
            }
        })
        
        return True

    async def _invalidate_cache(self):
        """
        Helper: 清除文档列表缓存
        通常在写操作 (C/U/D) 后调用
        """
        try:
            await redis_manager.delete("documents_list")
        except Exception as e:
            logger.error(f"缓存清除错误: {e}", exc_info=True)
