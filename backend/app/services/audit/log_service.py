"""操作日志服务。

这个模块负责把文档创建、修改、删除等关键动作落成结构化日志，
便于后续排查问题、演示系统行为链路，或在答辩时说明系统具备基本审计能力。
"""

from __future__ import annotations

from app.database import AsyncSessionLocal
from app.logger import logger
from app.models import Log


class LogService:
    def __init__(self, db):
        """保存当前日志写入所使用的数据库会话。"""
        self.db = db

    async def log_operation(
        self,
        action: str,
        resource_type: str,
        resource_id: str,
        user_id: str = 'system',
        details: dict | None = None,
    ):
        """把一次业务动作写入日志表。

        记录内容尽量保持结构化，后续可以按资源类型、资源编号和动作类别
        做过滤，而不是只依赖模糊文本检索。
        """
        try:
            log = Log(
                user_id=user_id,
                action=action,
                resource_type=resource_type,
                resource_id=str(resource_id),
                details=details,
            )
            self.db.add(log)
            await self.db.commit()
        except Exception as exc:
            logger.error(
                f'日志记录失败: {exc}',
                exc_info=True,
                extra={'extra_data': {'event': 'log_operation_failed', 'resource_type': resource_type, 'resource_id': resource_id}},
            )


async def record_operation_log(
    action: str,
    resource_type: str,
    resource_id: str,
    user_id: str = 'system',
    details: dict | None = None,
):
    """在独立数据库会话中写入一条操作日志。

    路由层或后台任务可以直接调用这个辅助函数，无需自己管理日志写入会话。
    """
    async with AsyncSessionLocal() as session:
        await LogService(session).log_operation(action, resource_type, resource_id, user_id=user_id, details=details)
