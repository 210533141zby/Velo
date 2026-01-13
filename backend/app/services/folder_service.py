from sqlalchemy import select
from app.models import Folder, Document
from app.schemas import FolderCreate
from app.services.base import BaseService

class FolderService(BaseService):
    """
    文件夹服务类
    处理文件夹层级结构的 CRUD 操作
    """
    
    async def create_folder(self, folder_data: FolderCreate) -> Folder:
        """创建新文件夹"""
        folder = Folder(title=folder_data.title, parent_id=folder_data.parent_id)
        self.db.add(folder)
        await self.db.commit()
        await self.db.refresh(folder)
        return folder

    async def get_all_folders(self) -> list[Folder]:
        """获取所有文件夹"""
        query = select(Folder).where(Folder.is_active == True)
        result = await self.db.execute(query)
        return result.scalars().all()

    async def get_folder(self, folder_id: int) -> Folder:
        """获取单个文件夹详情"""
        query = select(Folder).where(Folder.id == folder_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def get_folder_contents(self, folder_id: int = None) -> dict:
        """
        获取指定文件夹下的内容 (子文件夹 + 文档)
        folder_id 为 None 时表示根目录
        """
        # 获取子文件夹
        folder_query = select(Folder).where(Folder.is_active == True)
        if folder_id is None:
            folder_query = folder_query.where(Folder.parent_id == None)
        else:
            folder_query = folder_query.where(Folder.parent_id == folder_id)
            
        folders_res = await self.db.execute(folder_query)
        folders = folders_res.scalars().all()
        
        # 获取文档
        doc_query = select(Document).where(Document.is_active == True)
        if folder_id is None:
            doc_query = doc_query.where(Document.folder_id == None)
        else:
            doc_query = doc_query.where(Document.folder_id == folder_id)
            
        docs_res = await self.db.execute(doc_query)
        docs = docs_res.scalars().all()
        
        return {"folders": folders, "documents": docs}
