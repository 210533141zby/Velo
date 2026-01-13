"""
=============================================================================
文件: folders.py
描述: 文件夹管理接口
      提供文件夹的创建、查询、层级结构获取等 RESTful API
      所有业务逻辑已迁移至 Service 层，本文件仅负责路由和参数解析
=============================================================================
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db, AsyncSessionLocal
from app.schemas import FolderCreate, FolderResponse
from app.services.folder_service import FolderService
from app.services.log_service import LogService

router = APIRouter()

# -----------------------------------------------------------------------------
# 辅助函数: 获取服务实例
# -----------------------------------------------------------------------------

def get_folder_service(db: AsyncSession = Depends(get_db)) -> FolderService:
    """依赖注入: 获取文件夹服务实例"""
    return FolderService(db)

# -----------------------------------------------------------------------------
# 路由定义
# -----------------------------------------------------------------------------

@router.post("/", response_model=FolderResponse)
async def create_folder(
    folder: FolderCreate, 
    background_tasks: BackgroundTasks,
    folder_service: FolderService = Depends(get_folder_service)
):
    """
    创建新文件夹
    """
    new_folder = await folder_service.create_folder(folder)
    
    # 定义后台日志记录包装器
    async def log_wrapper(folder_id: int):
        async with AsyncSessionLocal() as session:
            svc = LogService(session)
            await svc.log_operation("CREATE", "FOLDER", str(folder_id))
            
    background_tasks.add_task(log_wrapper, new_folder.id)
    
    return new_folder

@router.get("/all", response_model=List[FolderResponse])
async def read_all_folders(
    folder_service: FolderService = Depends(get_folder_service)
):
    """
    获取所有文件夹
    用于构建完整的目录树结构
    """
    return await folder_service.get_all_folders()

@router.get("/{folder_id}", response_model=FolderResponse)
async def read_folder(
    folder_id: int,
    folder_service: FolderService = Depends(get_folder_service)
):
    """
    获取单个文件夹详情
    """
    folder = await folder_service.get_folder(folder_id)
    
    if folder is None:
        raise HTTPException(status_code=404, detail="未找到文件夹")
    return folder

@router.get("/{folder_id}/contents")
async def read_folder_contents(
    folder_id: int, # 注意：路由参数是 int，但如果是根目录可能需要特殊处理
    folder_service: FolderService = Depends(get_folder_service)
):
    """
    获取文件夹内容
    同时返回该目录下的子文件夹和文档
    如果 folder_id 为 0，则获取根目录内容
    """
    target_id = None if folder_id == 0 else folder_id
    return await folder_service.get_folder_contents(target_id)
