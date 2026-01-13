"""
=============================================================================
文件: models.py
描述: 数据库模型定义
      使用 SQLAlchemy ORM 定义系统核心实体：文件夹、文档、日志
      所有模型继承自 Base，自动处理表名和基础字段
=============================================================================
"""

from app.database import Base
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

class Folder(Base):
    """
    文件夹模型类
    
    描述:
        支持无限层级嵌套的目录结构
        用于组织文档
    """
    __tablename__ = "folders"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False, comment="文件夹名称")
    parent_id = Column(Integer, ForeignKey("folders.id"), nullable=True, comment="父文件夹ID")
    
    is_active = Column(Boolean, default=True, comment="是否有效(软删除标记)")
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), comment="更新时间")

    # 关系定义
    children = relationship("Folder", backref="parent", remote_side=[id])
    documents = relationship("Document", back_populates="folder")


class Document(Base):
    """
    文档模型类
    
    描述:
        存储文档的核心内容、元数据及版本信息
        关联到特定的文件夹
    """
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False, comment="文档标题")
    content = Column(Text, nullable=True, comment="文档内容(Markdown)")
    
    # RAG/知识图谱元数据
    summary = Column(Text, nullable=True, comment="AI生成的摘要")
    tags = Column(String(255), nullable=True, comment="标签(逗号分隔)")
    
    # 版本控制/审计
    version = Column(Integer, default=1, comment="版本号")
    is_active = Column(Boolean, default=True, comment="是否有效(软删除标记)")
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), comment="更新时间")

    # 文件夹关联
    folder_id = Column(Integer, ForeignKey("folders.id"), nullable=True, comment="所属文件夹ID")
    folder = relationship("Folder", back_populates="documents")


class Log(Base):
    """
    系统日志模型类
    
    描述:
        记录所有用户操作和系统事件
        用于审计、故障排查和用户行为分析
    """
    __tablename__ = "system_logs"

    id = Column(Integer, primary_key=True, index=True)
    
    # 操作人 (Who)
    user_id = Column(String(50), nullable=True, comment="操作用户ID")
    
    # 操作时间 (When)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), comment="操作时间")
    
    # 操作内容 (What)
    action = Column(String(100), nullable=False, comment="动作类型") # 例如: "CREATE_DOC", "CHAT_QUERY"
    resource_type = Column(String(50), nullable=True, comment="资源类型") # 例如: "DOCUMENT"
    resource_id = Column(String(50), nullable=True, comment="资源ID")
    
    # 详情 (Details)
    details = Column(JSON, nullable=True, comment="详细信息(JSON)")
    status = Column(String(20), default="SUCCESS", comment="操作状态") # SUCCESS, ERROR
    error_message = Column(Text, nullable=True, comment="错误信息")
