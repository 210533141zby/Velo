from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.services.stats_service import StatsService

router = APIRouter()

def get_stats_service(db: AsyncSession = Depends(get_db)) -> StatsService:
    return StatsService(db)

@router.get("/")
async def get_dashboard_stats(
    stats_service: StatsService = Depends(get_stats_service)
):
    """
    获取系统统计数据 (仪表盘)
    """
    return await stats_service.get_dashboard_stats()
