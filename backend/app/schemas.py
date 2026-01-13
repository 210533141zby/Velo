"""
=============================================================================
文件: schemas.py
描述: Pydantic 数据模式定义
      用于 API 请求参数验证和响应数据序列化
      包含文件夹、文档、聊天相关的 Schema 定义
=============================================================================
"""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

# =============================================================================
# 文件夹模式
# =============================================================================

class FolderBase(BaseModel):
    """文件夹基础模式"""
    title: str
    parent_id: Optional[int] = None

class FolderCreate(FolderBase):
    """创建文件夹请求模式"""
    pass

class FolderUpdate(FolderBase):
    """更新文件夹请求模式"""
    pass

class FolderResponse(FolderBase):
    """文件夹响应模式"""
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

# =============================================================================
# 文档模式
# =============================================================================

class DocumentBase(BaseModel):
    """文档基础模式"""
    title: str
    content: Optional[str] = None
    folder_id: Optional[int] = None

class DocumentCreate(DocumentBase):
    """创建文档请求模式"""
    pass

class DocumentUpdate(BaseModel):
    """更新文档请求模式"""
    title: Optional[str] = None
    content: Optional[str] = None
    folder_id: Optional[int] = None
    summary: Optional[str] = None
    tags: Optional[str] = None

class DocumentSummary(BaseModel):
    """文档摘要响应模式"""
    id: int
    title: str
    content: Optional[str] = None
    folder_id: Optional[int] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class DocumentResponse(DocumentBase):
    """文档详细响应模式"""
    id: int
    summary: Optional[str] = None
    tags: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    folder_id: Optional[int] = None

    class Config:
        from_attributes = True

# =============================================================================
# 聊天模式
# =============================================================================

class ChatMessage(BaseModel):
    """聊天消息模式"""
    role: str # user, assistant, system
    content: str

class ChatRequest(BaseModel):
    """聊天请求模式"""
    messages: List[ChatMessage]
    # 上下文选项
    use_rag: bool = True
    doc_id: Optional[int] = None # 针对特定文档聊天

class ChatResponse(BaseModel):
    """聊天响应模式"""
    response: str
    sources: Optional[List[Dict[str, Any]]] = None # RAG 来源引用
