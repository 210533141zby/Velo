from sqlalchemy.ext.asyncio import AsyncSession
from app.models import Log
from app.logger import logger
from app.services.base import BaseService

class LogService(BaseService):
    """
    日志服务类
    负责记录系统操作日志，通常在后台任务中异步执行
    """
    async def log_operation(self, action: str, resource_type: str, resource_id: str, user_id: str = "system", details: dict = None):
        """
        记录操作日志
        
        参数:
            action: 操作动作 (如 CREATE, DELETE)
            resource_type: 资源类型 (如 DOCUMENT, FOLDER)
            resource_id: 资源ID
            user_id: 操作用户ID
            details: 额外的详细信息字典
        """
        try:
            log = Log(
                user_id=user_id,
                action=action,
                resource_type=resource_type,
                resource_id=str(resource_id),
                details=details
            )
            self.db.add(log)
            await self.db.commit()
        except Exception as e:
            logger.error(f"日志记录失败: {e}", exc_info=True, extra={
                "extra_data": {
                    "event": "log_operation_failed",
                    "resource_type": resource_type,
                    "resource_id": resource_id
                }
            })
