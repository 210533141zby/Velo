"""
=============================================================================
文件: check_system.py
描述: 系统健康检查脚本

核心功能：
1. 数据库检查：连接数据库，检查核心表 (documents, folders) 是否存在及记录数。
2. Redis 检查：连接 Redis，执行 Ping 和简单的读写测试。
3. 环境诊断：打印数据库 URI 和 Redis URL，方便排查配置问题。

使用场景：
- 部署后验证系统各组件是否正常连通。
- 开发环境排查数据库或缓存连接问题。

依赖组件:
- sqlalchemy: 执行 SQL 查询。
- redis: 连接 Redis 服务。
- app.database: 获取数据库会话。
- app.core.config: 获取配置信息。
=============================================================================
"""

import asyncio
import os
import sys
from pathlib import Path

# 添加 backend 目录到 python path
# backend/scripts/check_system.py -> backend/scripts -> backend
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text
from app.database import AsyncSessionLocal
from app.core.config import settings
import redis.asyncio as redis

async def check_db():
    """
    检查数据库连接状态
    
    Logic Flow:
        1. **路径检查**: 如果是 SQLite，先检查物理文件是否存在。
        2. **连接测试**: 尝试建立数据库会话 (Session)。
        3. **数据查询**: 
           - 统计 `documents` 表行数。
           - 统计 `folders` 表行数。
           - 读取前 5 个文档标题，验证数据可读性。
        4. **异常捕获**: 如果连接失败，打印错误信息。
    """
    print("="*50)
    print("正在检查数据库连接...")
    print(f"数据库 URI: {settings.SQLALCHEMY_DATABASE_URI}")
    
    # 修正数据库路径检查逻辑 (假设是 SQLite 且在 backend 目录下)
    # 但实际生产环境可能是 Postgres，这里仅作 SQLite 参考
    if "sqlite" in settings.SQLALCHEMY_DATABASE_URI:
        db_path = settings.SQLALCHEMY_DATABASE_URI.replace("sqlite+aiosqlite:///", "")
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
    """
    检查 Redis 连接状态
    
    Logic Flow:
        1. **连接建立**: 使用配置中的 Host 和 Port 创建 Redis 客户端。
        2. **Ping 测试**: 发送 PING 命令，确认服务存活。
        3. **读写测试**: 写入一个临时 Key，再读取出来比对，确保功能正常。
        4. **资源清理**: 关闭连接。
    """
    print("\n" + "="*50)
    print("正在检查 Redis 连接...")
    redis_url = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}"
    print(f"Redis URL: {redis_url}")
    
    try:
        r = redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
        await r.ping()
        print("✅ 成功连接 Redis")
        
        # 简单写入读取测试
        await r.set("test_key", "test_value", ex=10)
        val = await r.get("test_key")
        if val == "test_value":
             print("✅ Redis 读写测试通过")
        else:
             print("❌ Redis 读写测试失败")
             
        await r.aclose()
        
    except Exception as e:
        print(f"❌ Redis 连接失败: {e}")

async def main():
    await check_db()
    await check_redis()
    print("\n" + "="*50)

if __name__ == "__main__":
    # Windows 下 asyncio 策略调整
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
