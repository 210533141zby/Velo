"""
=============================================================================
文件: folders.py
描述: 文件夹管理接口

核心功能：
1. 目录管理：提供文件夹的创建、查询接口。
2. 结构树获取：支持获取完整目录树，或获取指定目录下的内容。
3. 审计日志：在创建文件夹时，记录操作日志。

依赖组件:
- FolderService: 处理文件夹业务逻辑。
- LogService: 处理日志记录。
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
    
    Logic Flow:
        1. **业务处理**: 调用 `folder_service.create_folder` 写入数据库。
        2. **日志记录**: 后台任务记录 "CREATE FOLDER" 操作。
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
    
    Logic Flow:
        获取系统中所有文件夹的扁平列表。
        前端通常会拿到这个列表后，在浏览器端根据 `parent_id` 组装成树形结构。
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
    folder_id: int,
    folder_service: FolderService = Depends(get_folder_service)
):
    """
    获取文件夹内容
    
    Input:
        folder_id: 文件夹 ID。如果传 0，则表示获取根目录内容。
        
    Logic Flow:
        同时返回该目录下的：
        1. 子文件夹列表 (folders)
        2. 文档列表 (documents)
    """
    target_id = None if folder_id == 0 else folder_id
    return await folder_service.get_folder_contents(target_id)
