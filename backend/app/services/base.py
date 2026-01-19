"""
=============================================================================
文件: base.py
描述: 服务层基类

核心功能：
1. 统一依赖注入：所有 Service 类都继承自此类，确保自动获得数据库会话 (db session)。
2. 代码复用：未来可以在这里添加通用的数据库操作方法（如通用的 find_by_id, delete 等）。

依赖组件:
- sqlalchemy: 异步数据库会话管理。
=============================================================================
"""

from sqlalchemy.ext.asyncio import AsyncSession

class BaseService:
    """
    基础服务类
    
    Logic Flow:
        这是一个抽象基类（虽然没强制用 ABC）。
        它规定了所有 Service 初始化时必须接收一个 `AsyncSession`。
        
    Why:
        为什么要搞个基类？
        为了统一管理数据库连接。这样在 Controller (API) 层实例化 Service 时，
        只需要把 `db` 传进去，Service 内部就能直接用了，不用到处写 `self.db = db`。
    """
    def __init__(self, db: AsyncSession):
        self.db = db
