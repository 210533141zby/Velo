"""
=============================================================================
文件: database.py
描述: 数据库连接配置

核心功能：
1. 建立连接：创建异步数据库引擎 (Engine)，负责与数据库服务器通信。
2. 会话管理：创建会话工厂 (SessionLocal)，用于生成数据库操作会话。
3. 模型基类：定义 ORM 模型的基类 (Base)。

依赖组件:
- SQLAlchemy (AsyncIO): Python 最流行的 ORM 框架，这里使用其异步版本。
=============================================================================
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

# =============================================================================
# 数据库引擎 (Engine)
# =============================================================================

# 创建异步引擎
# 它是 SQLAlchemy 的核心，负责维护连接池和执行 SQL 语句。
engine = create_async_engine(
    settings.SQLALCHEMY_DATABASE_URI, 
    echo=True,       # 开启 SQL 日志，会在控制台打印执行的 SQL 语句 (调试用)
    future=True,     # 使用 SQLAlchemy 2.0 的新特性
    pool_pre_ping=True # 每次从池子里拿连接前先 ping 一下，防止拿到断开的连接报错
)

# =============================================================================
# 会话工厂 (Session Factory)
# =============================================================================

# 异步会话工厂
# 每次需要操作数据库时，就找它要一个 session (会话)
AsyncSessionLocal = sessionmaker(
    bind=engine, 
    class_=AsyncSession, 
    expire_on_commit=False, # 提交后不立即可期对象，防止异步操作中出现属性访问错误
    autoflush=False         # 关闭自动刷新，由开发者手动控制 flush 时机
)

# ORM 模型基类
# 我们定义的 User, Document 等模型都要继承这个类
Base = declarative_base()

async def get_db():
    """
    获取数据库会话 (Dependency)
    
    Logic Flow:
        1. 创建：从工厂里拿一个新的会话。
        2. 提供：通过 yield 把会话借给调用者 (比如 API 处理函数)。
        3. 清理：等调用者用完了，自动关闭会话，把连接还回连接池。
        
    Why:
        这是 FastAPI 中标准的“依赖注入”写法。
        它保证了每个请求都有独立的数据库会话，而且用完一定会关，不会泄露连接。
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            # 无论业务逻辑是否报错，最后都要关闭会话
            await session.close()
