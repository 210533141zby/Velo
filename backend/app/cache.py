import redis.asyncio as redis
from app.core.config import settings
import time
import logging
from typing import Optional, Any, Tuple

# 设置日志记录器
logger = logging.getLogger(__name__)

class CacheManager:
    """
    企业级混合策略缓存管理器
    
    策略:
    1. 尝试连接 Redis (生产/Linux 环境标准).
    2. 如果 Redis 不可用或超时 (本地开发/Windows 常见),
       优雅降级到内存缓存.
    3. 确保无阻塞操作影响应用启动或请求延迟.
    """
    
    def __init__(self):
        self.redis: Optional[redis.Redis] = None
        self.use_redis = False
        # 内存缓存降级方案: {key: (value, expire_at)}
        self._memory_cache: dict[str, Tuple[Any, float]] = {}

    async def init_redis(self):
        """
        初始化 Redis 连接，设置严格超时以防挂起
        """
        try:
            # 强制使用 IP 避免 localhost 解析延迟
            host = "127.0.0.1" if settings.REDIS_HOST == "localhost" else settings.REDIS_HOST
            
            # 创建客户端
            client = redis.from_url(
                f"redis://{host}:{settings.REDIS_PORT}",
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=0.5, # 极速失败 (0.5s)
                socket_timeout=0.5,         # 极速失败 (0.5s)
                retry_on_timeout=False
            )
            
            # 验证连接
            await client.ping()
            
            self.redis = client
            self.use_redis = True
            logger.info("✅ Redis 缓存: 连接成功")
            print("Redis 已连接 (企业模式)")
            
        except (redis.ConnectionError, redis.TimeoutError, Exception) as e:
            self.use_redis = False
            # 确保 client 被关闭
            if 'client' in locals() and client:
                 await client.aclose()
            self.redis = None
            logger.warning(f"⚠️ Redis 不可用: {str(e)}. 切换到内存缓存")
            print(f"Redis 连接失败: {e}. 使用内存缓存 (降级模式)")

    async def close(self):
        """
        关闭连接
        """
        if self.redis:
            await self.redis.close()
        self._memory_cache.clear()

    async def get(self, key: str) -> Optional[Any]:
        """
        从缓存获取值 (Redis -> 内存)
        """
        # 1. 尝试 Redis
        if self.use_redis and self.redis:
            try:
                return await self.redis.get(key)
            except Exception as e:
                logger.error(f"Redis 读取错误: {e}. 降级到内存")
                # 遇到瞬时错误不永久切换模式，仅跳过此次读取
                pass

        # 2. 降级到内存
        if key in self._memory_cache:
            value, expire_at = self._memory_cache[key]
            # 检查过期
            if expire_at and time.time() > expire_at:
                del self._memory_cache[key]
                return None
            return value
        
        return None

    async def set(self, key: str, value: Any, ex: int = None):
        """
        设置缓存值 (Redis -> 内存)
        """
        # 1. 尝试 Redis
        if self.use_redis and self.redis:
            try:
                await self.redis.set(key, value, ex=ex)
                return
            except Exception as e:
                logger.error(f"Redis 写入错误: {e}. 降级到内存")
        
        # 2. 写入内存 (作为备份或主存)
        # 注意: 如果 Redis 活跃，我们可能为了节省 RAM 跳过内存写入，
        # 但为了在不稳定环境下的混合一致性，写入内存更安全，
        # 尽管在多实例下可能导致脏数据。
        # 对于此单实例应用，这没问题。
        expire_at = (time.time() + ex) if ex else float('inf')
        self._memory_cache[key] = (value, expire_at)

    async def delete(self, key: str):
        """
        从缓存删除值
        """
        if self.use_redis and self.redis:
            try:
                await self.redis.delete(key)
            except Exception:
                pass
        
        if key in self._memory_cache:
            del self._memory_cache[key]

# 单例实例
redis_manager = CacheManager()
