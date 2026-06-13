"""内容域业务服务。

这里封装文档与文件夹的数据库操作，并承担少量缓存协同逻辑，
让接口层只关心请求参数和返回结果，不直接处理持久化细节。
"""

import json

from sqlalchemy import func, select

from app.cache import redis_manager
from app.logger import logger
from app.models import Document, Folder
from app.schemas import DocumentCreate, DocumentUpdate, FolderCreate


class DocumentService:
    def __init__(self, db):
        """保存当前请求共享的数据库会话。"""
        self.db = db

    async def create_document(self, doc_data: DocumentCreate) -> Document:
        """写入一篇新文档，并刷新文档列表缓存。

        这里只负责数据库层面的新增；知识库索引刷新由路由层在保存成功后
        另行安排后台任务执行。
        """
        new_doc = Document(title=doc_data.title, content=doc_data.content, folder_id=doc_data.folder_id)
        self.db.add(new_doc)
        await self.db.commit()
        await self.db.refresh(new_doc)

        await self._invalidate_cache()
        logger.info(
            f'创建文档: {new_doc.title}',
            extra={'extra_data': {'event': 'document_created', 'document_id': new_doc.id, 'folder_id': new_doc.folder_id, 'title': new_doc.title}},
        )
        return new_doc

    async def get_document(self, doc_id: int) -> Document | None:
        """按编号读取一篇仍处于有效状态的文档。"""
        result = await self.db.execute(select(Document).where(Document.id == doc_id, Document.is_active == True))
        return result.scalar_one_or_none()

    async def update_document(self, doc_id: int, doc_update: DocumentUpdate) -> Document | None:
        """按提交字段局部更新文档，并同步清理列表缓存。"""
        db_doc = await self.get_document(doc_id)
        if not db_doc:
            return None

        update_data = doc_update.dict(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_doc, key, value)
        await self.db.commit()
        await self.db.refresh(db_doc)
        await self._invalidate_cache()
        return db_doc

    async def delete_document(self, doc_id: int) -> bool:
        """对文档执行逻辑删除，而不是直接物理移除记录。"""
        db_doc = await self.get_document(doc_id)
        if not db_doc:
            return False

        db_doc.is_active = False
        await self.db.commit()
        await self._invalidate_cache()
        return True

    async def get_all_documents(self):
        """读取全部文档摘要，并优先复用 Redis 缓存结果。

        文档列表在编辑器左侧会被频繁请求，缓存命中可以显著减少数据库读取。
        """
        try:
            cached_docs = await redis_manager.get('documents_list')
            if cached_docs:
                return json.loads(cached_docs)
        except Exception as exc:
            logger.error(f'缓存读取错误: {exc}', exc_info=True)

        result = await self.db.execute(
            select(
                Document.id,
                Document.title,
                func.substr(Document.content, 1, 200).label('content'),
                Document.created_at,
                Document.updated_at,
                Document.folder_id,
            )
            .where(Document.is_active == True)
            .order_by(Document.updated_at.desc())
        )
        rows = result.all()
        docs_data = [
            {
                'id': row.id,
                'title': row.title,
                'content': row.content,
                'created_at': row.created_at.isoformat() if row.created_at else None,
                'updated_at': row.updated_at.isoformat() if row.updated_at else None,
                'folder_id': row.folder_id,
            }
            for row in rows
        ]

        try:
            await redis_manager.set('documents_list', json.dumps(docs_data), ex=300)
        except Exception as exc:
            logger.error(f'缓存写入错误: {exc}', exc_info=True)
        return docs_data

    async def _invalidate_cache(self):
        """清除受文档变更影响的缓存键。

        目前除了文档列表，还会顺带清除基于旧正文生成的 RAG 回答缓存，
        避免知识库命中过期答案。
        """
        try:
            await redis_manager.delete('documents_list')
            await redis_manager.delete_prefix('rag:response:')
        except Exception as exc:
            logger.error(f'缓存清理错误: {exc}', exc_info=True)


class FolderService:
    def __init__(self, db):
        """保存文件夹服务使用的数据库会话。"""
        self.db = db

    async def create_folder(self, folder: FolderCreate) -> Folder:
        """创建一个新的文件夹节点。"""
        db_folder = Folder(title=folder.title, parent_id=folder.parent_id)
        self.db.add(db_folder)
        await self.db.commit()
        await self.db.refresh(db_folder)
        return db_folder

    async def get_all_folders(self) -> list[Folder]:
        """返回全部有效文件夹，供前端构造目录树。"""
        result = await self.db.execute(select(Folder).where(Folder.is_active == True))
        return result.scalars().all()

    async def get_folder(self, folder_id: int) -> Folder | None:
        """读取单个文件夹节点。"""
        result = await self.db.execute(select(Folder).where(Folder.id == folder_id))
        return result.scalar_one_or_none()

    async def get_folder_contents(self, folder_id: int | None = None) -> dict:
        """读取某一层级下的直属文件夹和文档。

        该接口服务于树形目录的逐层展开，因此只返回当前层级内容，
        不递归展开全部后代节点。
        """
        folder_query = select(Folder).where(Folder.is_active == True)
        folder_query = folder_query.where(Folder.parent_id == (None if folder_id is None else folder_id))

        doc_query = select(Document).where(Document.is_active == True)
        doc_query = doc_query.where(Document.folder_id == (None if folder_id is None else folder_id))

        folders = (await self.db.execute(folder_query)).scalars().all()
        docs = (await self.db.execute(doc_query)).scalars().all()
        return {'folders': folders, 'documents': docs}
