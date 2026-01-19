"""
=============================================================================
文件: models.py
描述: 数据库模型定义

核心功能：
1. 数据结构映射：使用 SQLAlchemy ORM 将 Python 类映射到数据库表。
2. 关系管理：定义文件夹、文档、日志之间的关联关系（一对多、自关联）。
3. 字段定义：规定每个字段的数据类型、约束条件和默认值。

依赖组件:
- sqlalchemy: Python 中最流行的 ORM 框架。
=============================================================================
"""

from app.database import Base
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

# =============================================================================
# 文件夹模型 (Folder)
# =============================================================================

class Folder(Base):
    """
    文件夹模型类
    
    Logic Flow:
        这是一个自关联的树形结构。
        每个文件夹可以有一个父文件夹 (`parent_id`)，也可以有多个子文件夹 (`children`)。
        同时，一个文件夹下可以包含多个文档 (`documents`)。
        
    Why:
        为什么要设计成无限层级？
        为了模拟真实操作系统的文件管理体验，让用户可以自由组织知识结构，而不是只能只有一层目录。
    """
    __tablename__ = "folders"

    # 基础字段
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False, comment="文件夹名称")
    
    # 树形结构核心字段
    # 指向父文件夹的 ID，为空表示是根目录
    parent_id = Column(Integer, ForeignKey("folders.id"), nullable=True, comment="父文件夹ID")
    
    # 状态字段
    is_active = Column(Boolean, default=True, comment="是否有效(软删除标记)")
    
    # 时间戳 (自动维护)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), comment="更新时间")

    # 关系定义
    # 1. 自关联：父子文件夹关系
    # remote_side=[id] 告诉 SQLAlchemy 这是一个指向同一张表的关系
    children = relationship("Folder", backref="parent", remote_side=[id])
    
    # 2. 一对多：文件夹包含的文档
    documents = relationship("Document", back_populates="folder")


# =============================================================================
# 文档模型 (Document)
# =============================================================================

class Document(Base):
    """
    文档模型类
    
    Logic Flow:
        文档是知识库的核心单元。
        它必须属于某一个文件夹（或者根目录），存储了用户编写的 Markdown 内容。
        同时还包含了一些为了 AI 搜索（RAG）准备的元数据，比如摘要和标签。
    """
    __tablename__ = "documents"

    # 基础字段
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False, comment="文档标题")
    content = Column(Text, nullable=True, comment="文档内容(Markdown)")
    
    # RAG/知识图谱元数据
    summary = Column(Text, nullable=True, comment="AI生成的摘要")
    tags = Column(String(255), nullable=True, comment="标签(逗号分隔)")
    
    # 版本控制/审计
    version = Column(Integer, default=1, comment="版本号")
    is_active = Column(Boolean, default=True, comment="是否有效(软删除标记)")
    
    # 时间戳
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), comment="更新时间")

    # 关系定义
    # 指向所属文件夹的外键
    folder_id = Column(Integer, ForeignKey("folders.id"), nullable=True, comment="所属文件夹ID")
    folder = relationship("Folder", back_populates="documents")


# =============================================================================
# 系统日志模型 (Log)
# =============================================================================

class Log(Base):
    """
    系统日志模型类
    
    Logic Flow:
        这是一个只能追加、不能修改的流水账表。
        不管用户做了什么操作（查、增、删、改），都会在这里留下一条记录。
        
    Why:
        - 审计需求：谁在什么时候干了什么？
        - 故障排查：报错时的输入参数是什么？
    """
    __tablename__ = "system_logs"

    id = Column(Integer, primary_key=True, index=True)
    
    # 操作人 (Who)
    # 目前是简易版，存用户ID字符串。未来对接用户系统后可改为外键。
    user_id = Column(String(50), nullable=True, comment="操作用户ID")
    
    # 操作时间 (When)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), comment="操作时间")
    
    # 操作内容 (What)
    action = Column(String(100), nullable=False, comment="动作类型") # 例如: "CREATE_DOC", "CHAT_QUERY"
    resource_type = Column(String(50), nullable=True, comment="资源类型") # 例如: "DOCUMENT"
    resource_id = Column(String(50), nullable=True, comment="资源ID")
    
    # 详情 (Details)
    # 使用 JSON 类型存储灵活的结构化数据，比如请求参数、返回结果摘要
    details = Column(JSON, nullable=True, comment="详细信息(JSON)")
    
    # 结果
    status = Column(String(20), default="SUCCESS", comment="操作状态") # SUCCESS, ERROR
    error_message = Column(Text, nullable=True, comment="错误信息")
