"""
=============================================================================
文件: log_service.py
描述: 操作日志服务

核心功能：
1. 审计记录：将用户的关键操作（增删改查）写入数据库的日志表。
2. 故障留痕：记录操作失败时的上下文信息，方便后续排查。

依赖组件:
- models.Log: 日志数据库模型。
=============================================================================
"""

from sqlalchemy.ext.asyncio import AsyncSession
from app.models import Log
from app.logger import logger
from app.services.base import BaseService

class LogService(BaseService):
    """
    日志服务类
    
    Logic Flow:
        专门负责往 `system_logs` 表里插数据。
        通常不在 API 响应的主线程里直接调，而是扔到后台任务 (BackgroundTasks) 里异步执行。
    """
    
    async def log_operation(self, action: str, resource_type: str, resource_id: str, user_id: str = "system", details: dict = None):
        """
        记录操作日志
        
        Input:
            action: 做了什么？(如 "CREATE", "DELETE")
            resource_type: 对谁做的？(如 "DOCUMENT", "FOLDER")
            resource_id: 资源的 ID
            user_id: 谁做的？
            details: 还有什么细节？(字典格式)
            
        Logic Flow:
            1. 构建日志对象：把传入的参数填入 Log 模型。
            2. 写入数据库：执行 insert 操作。
            3. 异常捕获：
               如果写日志本身报错了（比如数据库断连），我们捕获异常并打印到控制台日志，
               **绝对不能**因为记录日志失败，而导致原来的业务操作（比如创建文档）也回滚失败。
               
        Why:
            为什么要 try-except？
            日志是辅助功能，不能喧宾夺主。主业务成功了就是成功了，日志没记上顶多是审计缺失，
            不能因此报错给用户。
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
            # 写数据库失败时，至少在文件日志里留个痕
            logger.error(f"日志记录失败: {e}", exc_info=True, extra={
                "extra_data": {
                    "event": "log_operation_failed",
                    "resource_type": resource_type,
                    "resource_id": resource_id
                }
            })
