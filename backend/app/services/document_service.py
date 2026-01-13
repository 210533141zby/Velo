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
        
        流程:
        1. 写入数据库 (Markdown 内容原样保存，不转义)
        2. 清除列表缓存
        3. 添加后台任务: 建立向量索引
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
        """获取单个文档详情"""
        query = select(Document).where(Document.id == doc_id, Document.is_active == True)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def update_document(self, doc_id: int, doc_update: DocumentCreate, background_tasks: BackgroundTasks) -> Document:
        """
        更新文档
        
        流程:
        1. 检查文档是否存在
        2. 更新字段
        3. 提交事务
        4. 清除缓存
        5. 触发后台任务: 更新向量索引
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
        优先读取 Redis 缓存
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
        """清除文档列表缓存"""
        try:
            await redis_manager.delete("documents_list")
        except Exception as e:
            logger.error(f"缓存清除错误: {e}", exc_info=True)
