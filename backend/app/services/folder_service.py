"""
=============================================================================
文件: folder_service.py
描述: 文件夹管理服务

核心功能：
1. 目录结构管理：处理文件夹的增删改查。
2. 树形结构查询：获取文件夹下的子文件夹和文档。

依赖组件:
- models.Folder: 文件夹数据库模型。
=============================================================================
"""

from sqlalchemy import select
from app.models import Folder, Document
from app.schemas import FolderCreate
from app.services.base import BaseService

class FolderService(BaseService):
    """
    文件夹服务类
    """
    
    async def create_folder(self, folder: FolderCreate) -> Folder:
        """
        创建文件夹
        
        Logic Flow:
            1. 接收参数：标题、父文件夹 ID。
            2. 写入数据库。
        """
        db_folder = Folder(title=folder.title, parent_id=folder.parent_id)
        self.db.add(db_folder)
        await self.db.commit()
        await self.db.refresh(db_folder)
        return db_folder

    async def get_all_folders(self) -> list[Folder]:
        """
        获取所有文件夹
        
        Logic Flow:
            一次性拉取数据库里所有 `is_active=True` 的文件夹。
            前端拿到后，通常会在浏览器端组装成树形结构。
        """
        query = select(Folder).where(Folder.is_active == True)
        result = await self.db.execute(query)
        return result.scalars().all()

    async def get_folder(self, folder_id: int) -> Folder:
        """
        获取单个文件夹详情
        """
        query = select(Folder).where(Folder.id == folder_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_folder_contents(self, folder_id: int = None) -> dict:
        """
        获取文件夹内容 (子目录 + 文档)
        
        Input:
            folder_id: 文件夹 ID。如果为 None，表示查询根目录。
            
        Logic Flow:
            1. 查子文件夹：找所有 `parent_id` 等于当前 folder_id 的文件夹。
            2. 查文档：找所有 `folder_id` 等于当前 folder_id 的文档。
            3. 打包返回：返回一个包含 `folders` 和 `documents` 的字典。
            
        Why:
            这是一个聚合查询接口，方便前端在一个 API 请求里就把目录页需要的数据全拿完，
            减少网络请求次数。
        """
        # 1. 获取子文件夹
        folder_query = select(Folder).where(Folder.is_active == True)
        if folder_id is None:
            folder_query = folder_query.where(Folder.parent_id == None)
        else:
            folder_query = folder_query.where(Folder.parent_id == folder_id)
            
        folders_res = await self.db.execute(folder_query)
        folders = folders_res.scalars().all()
        
        # 2. 获取文档
        doc_query = select(Document).where(Document.is_active == True)
        if folder_id is None:
            doc_query = doc_query.where(Document.folder_id == None)
        else:
            doc_query = doc_query.where(Document.folder_id == folder_id)
            
        docs_res = await self.db.execute(doc_query)
        docs = docs_res.scalars().all()
        
        return {"folders": folders, "documents": docs}
