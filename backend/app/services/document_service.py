"""
=============================================================================
文件: document_service.py
描述: 文档核心业务服务

核心功能：
1. 文档 CRUD：创建、查询、更新、删除 Markdown 文档。
2. 缓存同步：在修改文档时，自动失效或更新 Redis 缓存。
3. AI 索引触发：在创建/更新文档后，异步触发 Agent 服务进行向量索引。

依赖组件:
- agent_service: 负责将文档向量化。
- redis_manager: 负责缓存文档列表。
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
    处理文档的 CRUD、缓存管理及与 Agent 服务的交互。
    """
    
    async def create_document(self, doc_data: DocumentCreate, background_tasks: BackgroundTasks) -> Document:
        """
        创建新文档
        
        Input:
            doc_data (DocumentCreate): 包含 title, content, folder_id 等。
            background_tasks (BackgroundTasks): FastAPI 后台任务管理器。
        Output:
            Document: 创建成功的数据库对象。
            
        Logic Flow:
            1. **写入数据库 (DB Insert)**: 
               将用户提交的 Markdown 内容原样写入 `documents` 表。
            2. **清理缓存 (Cache Invalidation)**: 
               因为新增了文档，旧的文档列表缓存已经过时，所以要调用 `_invalidate_cache` 清除它。
            3. **触发 AI 索引 (Async Indexing)**: 
               为了支持 RAG 问答，文档创建后需要立刻做向量化。
               我们实例化 `AgentService`，并将 `index_document` 方法加入后台任务队列。
               
        Why:
            为什么要用 BackgroundTasks？
            索引过程涉及 OpenAI API 调用（可能慢）和向量库写入，耗时通常在 1-3 秒。
            如果在主线程里等，用户点“保存”后界面会卡住。用后台任务可以让接口在 50ms 内瞬间返回。
        """
        # 1. 创建数据库对象
        new_doc = Document(
            title=doc_data.title, 
            content=doc_data.content, 
            folder_id=doc_data.folder_id
        )
        self.db.add(new_doc)
        await self.db.commit()
        await self.db.refresh(new_doc)
        
        # 2. 缓存失效
        await self._invalidate_cache()
        
        # 3. 异步触发 RAG 索引
        # 在此处实例化 AgentService，避免类级别的循环导入问题
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
        
        Logic Flow:
            1. **直接查库**: 直接去数据库查 `id=doc_id` 且 `is_active=True` 的记录。
            
        Why:
            为什么不加缓存？
            单个文档详情通常用于“编辑”场景，需要极高的实时性。
            如果用了缓存，用户在 A 电脑改了，B 电脑可能还在显示旧内容，导致编辑冲突。
            而且文档内容可能很大，缓存单个大 Key 对 Redis 内存压力也大。
        """
        query = select(Document).where(Document.id == doc_id, Document.is_active == True)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def update_document(self, doc_id: int, doc_update: DocumentCreate, background_tasks: BackgroundTasks) -> Document:
        """
        更新文档
        
        Logic Flow:
            1. **检查存在性**: 先查一下文档在不在。
            2. **更新字段**: 遍历用户提交的字段，有值的就更新。
            3. **清理缓存**: 列表数据变了，清空列表缓存。
            4. **重建索引**: 
               如果用户改了内容 (content)，我们需要重新做向量索引。
               由于 ChromaDB 不支持增量更新单个片段，通常策略是“先删后增”或者“覆盖更新”。
               这里我们再次调用 `index_document`，它会重新处理整个文档。
        """
        db_doc = await self.get_document(doc_id)
        if not db_doc:
            return None
        
        # 更新字段
        update_data = doc_update.dict(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_doc, key, value)
            
        await self.db.commit()
        await self.db.refresh(db_doc)
        
        # 缓存失效
        await self._invalidate_cache()
        
        # 如果内容有变更，重新建立索引
        if "content" in update_data:
            agent_service = AgentService(self.db)
            # 先删旧索引 (可选，取决于 AgentService 实现，这里简化为直接覆盖/新增)
            # background_tasks.add_task(agent_service.delete_document_index, doc_id)
            background_tasks.add_task(agent_service.index_document, db_doc.id, db_doc.title, db_doc.content)
            
        return db_doc

    async def delete_document(self, doc_id: int, background_tasks: BackgroundTasks):
        """
        删除文档 (软删除)
        
        Logic Flow:
            1. **标记删除**: 不直接 delete，而是把 `is_active` 设为 False。
            2. **清理缓存**: 清空列表缓存。
            3. **删除索引**: 调用 `delete_document_index` 从向量库中彻底移除该文档的数据，
               避免 RAG 问答时搜出已删除的文档。
        """
        db_doc = await self.get_document(doc_id)
        if db_doc:
            # 软删除
            db_doc.is_active = False
            await self.db.commit()
            
            # 缓存失效
            await self._invalidate_cache()
            
            # 删除向量索引
            agent_service = AgentService(self.db)
            background_tasks.add_task(agent_service.delete_document_index, doc_id)
            
            return True
        return False

    async def get_all_documents(self):
        """
        获取文档列表 (带缓存)
        
        Logic Flow:
            1. **查缓存**: 先看 Redis 有没有 `documents_list`。如果有，直接解析 JSON 返回。
            2. **查数据库**: 缓存没命中，去数据库查所有 `is_active=True` 的文档。
               注意：为了性能，我们截取了 `content` 的前 200 个字作为预览，而不是查全文。
            3. **写缓存**: 把查到的结果序列化成 JSON 存入 Redis，设置 5 分钟过期。
            
        Why:
            为什么缓存 5 分钟？
            文档列表页是高频访问接口，但不需要毫秒级实时。
            5 分钟的延迟在 Wiki 场景完全可接受，能极大降低数据库压力。
            而且我们有主动失效机制 (`_invalidate_cache`)，一旦有增删改，缓存会立刻被清空，
            所以实际上用户几乎感觉不到延迟。
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
            # 只取前 200 个字作为摘要/预览
            func.substr(Document.content, 1, 200).label("content"),
            Document.created_at,
            Document.updated_at,
            Document.folder_id
        ).where(Document.is_active == True).order_by(Document.updated_at.desc())
        
        result = await self.db.execute(query)
        rows = result.all()
        
        # 转换为字典列表
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
        
        # 3. 写入缓存
        try:
            await redis_manager.set("documents_list", json.dumps(docs_data), ex=300)
        except Exception as e:
            logger.error(f"缓存写入错误: {e}", exc_info=True)
            
        return docs_data

    async def _invalidate_cache(self):
        """
        内部方法: 清除文档列表缓存
        """
        try:
            await redis_manager.delete("documents_list")
        except Exception as e:
            logger.error(f"缓存清理错误: {e}", exc_info=True)
