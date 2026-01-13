from sqlalchemy.ext.asyncio import AsyncSession

class BaseService:
    """
    基础服务类
    提供数据库会话的通用封装
    """
    def __init__(self, db: AsyncSession):
        self.db = db
