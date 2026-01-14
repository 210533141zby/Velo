from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.models import Document
from app.services.base import BaseService
from app.cache import redis_manager

class StatsService(BaseService):
    """
    统计服务类
    提供仪表盘所需的数据聚合功能
    """
    def __init__(self, db: AsyncSession):
        super().__init__(db)

    async def get_dashboard_stats(self):
        """
        获取仪表盘统计数据
        """
        # 1. 统计文档总数 (仅统计未删除的)
        stmt_count = select(func.count(Document.id)).where(Document.is_active == True)
        result_count = await self.db.execute(stmt_count)
        total_documents = result_count.scalar() or 0

        # 2. 统计总字数 (所有文档内容的长度之和)
        # 注意: 如果内容为空，length返回NULL，sum忽略NULL
        stmt_words = select(func.sum(func.length(Document.content))).where(Document.is_active == True)
        result_words = await self.db.execute(stmt_words)
        total_words = result_words.scalar() or 0

        # 3. Redis 状态
        cache_status = "online" if redis_manager.use_redis else "offline (memory fallback)"
        
        return {
            "total_documents": total_documents,
            "total_words": total_words,
            "cache_status": cache_status
        }
