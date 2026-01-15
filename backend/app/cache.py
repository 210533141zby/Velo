"""
=============================================================================
文件: cache.py
描述: 缓存管理器
      实现了一个混合策略 (Hybrid Strategy) 的缓存系统，旨在提供高可用性和容错性。

核心特性:
- **Redis 优先**: 优先尝试连接 Redis 服务 (生产环境标准)。
- **内存降级 (Memory Fallback)**: 当 Redis 不可用 (如本地开发未启动 Redis，或网络故障) 时，
  自动平滑切换到进程内内存缓存 (`_memory_cache`)，确保业务不中断。
- **非阻塞初始化**: 连接过程设置了极短的超时时间，防止应用启动被挂起。

使用场景:
- RAG 问答结果缓存 (TTL 1小时)
- 文档列表缓存 (TTL 5分钟)
=============================================================================
"""

import redis.asyncio as redis
from app.core.config import settings
import time
import logging
from typing import Optional, Any, Tuple

# 设置日志记录器
logger = logging.getLogger(__name__)

class CacheManager:
    """
    混合策略缓存管理器
    
    Attributes:
        redis (redis.Redis): Redis 客户端实例 (如果连接成功)。
        use_redis (bool): 当前是否使用 Redis 模式。
        _memory_cache (dict): 内存缓存存储结构 {key: (value, expire_at_timestamp)}。
    """
    
    def __init__(self):
        self.redis: Optional[redis.Redis] = None
        self.use_redis = False
        # 内存缓存降级方案: {key: (value, expire_at)}
        self._memory_cache: dict[str, Tuple[Any, float]] = {}

    async def init_redis(self):
        """
        初始化 Redis 连接
        
        Logic Flow:
        1. 读取配置中的 Host 和 Port。
        2. 尝试建立连接，并发送 PING 命令验证。
        3. **Fast Fail**: 设置 socket_timeout=0.5s。如果 Redis 没反应，迅速抛出异常，
           而不是让整个应用启动卡住几十秒。
        4. 如果成功 -> 设置 use_redis=True。
        5. 如果失败 -> 捕获异常，打印警告，设置 use_redis=False (后续将使用内存缓存)。
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
            print("Redis 已连接 ")
            
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
        关闭连接并清理资源
        """
        if self.redis:
            await self.redis.close()
        self._memory_cache.clear()

    async def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值
        
        Logic Flow:
        1. **Check Redis**: 如果启用了 Redis，尝试 `await redis.get(key)`。
           - 如果 Redis 突然断连 (抛出异常)，记录错误但不崩溃，继续尝试内存缓存 (虽然此时内存可能没数据，但保证了代码健壮性)。
        2. **Check Memory**: 如果 Redis 不可用或未命中，检查 `_memory_cache`。
           - 检查 `expire_at` 时间戳，如果过期则删除并返回 None。
           - 否则返回 value。
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
        设置缓存值
        
        Logic Flow:
        1. **Write Redis**: 优先写入 Redis。如果失败，记录日志并降级。
        2. **Write Memory**: 
           - 为了保证在混合环境下的数据可用性 (例如 Redis 刚挂掉)，
             我们选择**始终写入内存** (或者在 Redis 失败时写入)。
           - 当前实现：如果 Redis 写入失败，或者根本没用 Redis，则写入内存。
           - 内存过期时间计算: `now() + ex`。
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
        删除缓存值
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
