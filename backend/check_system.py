import asyncio
import os
import sys

# 添加 backend 目录到 python path
sys.path.append(os.path.dirname(__file__))

from sqlalchemy import text
from app.database import AsyncSessionLocal
from app.core.config import settings
import redis.asyncio as redis

async def check_db():
    print("="*50)
    print("正在检查数据库连接...")
    print(f"数据库 URI: {settings.SQLALCHEMY_DATABASE_URI}")
    
    db_path = "./wiki.db"
    if os.path.exists(db_path):
        print(f"✅ SQLite 数据库文件存在: {os.path.abspath(db_path)}")
        size = os.path.getsize(db_path)
        print(f"   文件大小: {size/1024:.2f} KB")
    else:
        print("❌ SQLite 数据库文件不存在 (可能尚未初始化)")

    try:
        async with AsyncSessionLocal() as session:
            # 检查 Document 表
            result = await session.execute(text("SELECT count(*) FROM documents"))
            doc_count = result.scalar()
            print(f"✅ 成功连接数据库. Document 表记录数: {doc_count}")
            
            # 检查 Folder 表
            result = await session.execute(text("SELECT count(*) FROM folders"))
            folder_count = result.scalar()
            print(f"   Folder 表记录数: {folder_count}")

            # 简单的列出前5个文档标题
            result = await session.execute(text("SELECT title FROM documents LIMIT 5"))
            titles = result.scalars().all()
            if titles:
                print("   最新文档示例:")
                for t in titles:
                    print(f"    - {t}")
            
    except Exception as e:
        print(f"❌ 数据库连接或查询失败: {e}")

async def check_redis():
    print("\n" + "="*50)
    print("正在检查 Redis 连接...")
    print(f"Redis 目标: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
    
    # 强制使用 IP 避免 localhost 解析延迟
    host = "127.0.0.1" if settings.REDIS_HOST == "localhost" else settings.REDIS_HOST
    
    r = None
    try:
        r = redis.from_url(
            f"redis://{host}:{settings.REDIS_PORT}", 
            encoding="utf-8", 
            decode_responses=True,
            socket_connect_timeout=1, # 极速超时
            socket_timeout=1,
            retry_on_timeout=False    # 禁止重试
        )
        await r.ping()
        print("✅ Redis 连接成功!")
        
        # 简单测试读写
        await r.set("test_key", "wiki_local_test", ex=10)
        val = await r.get("test_key")
        if val == "wiki_local_test":
            print("✅ Redis 读写测试通过")
        
    except Exception as e:
        print(f"❌ Redis 连接失败: {e}")
        print("   (不用担心，系统会自动降级使用内存缓存)")
    finally:
        if r:
            await r.aclose()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(check_db())
        loop.run_until_complete(check_redis())
    finally:
        loop.close()
