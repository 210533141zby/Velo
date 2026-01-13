from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

# 创建异步引擎
engine = create_async_engine(
    settings.SQLALCHEMY_DATABASE_URI, 
    echo=True,
    future=True,
    pool_pre_ping=True
)

# 异步会话工厂
AsyncSessionLocal = sessionmaker(
    bind=engine, 
    class_=AsyncSession, 
    expire_on_commit=False,
    autoflush=False
)

Base = declarative_base()

async def get_db():
    """
    获取数据库会话依赖
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
