"""
=============================================================================
文件: db_init.py
描述: 数据库初始化脚本

核心功能：
1. 等待数据库就绪：在应用启动时，循环检查数据库是否连得上。
2. 自动建表：如果连接成功，自动创建所有定义好的数据表。

Why:
Docker 容器启动时，数据库可能还没准备好。如果后端直接连，会报错退出。
所以需要一个“重试等待”机制，确保数据库真的启动了再继续。
=============================================================================
"""

import logging
import asyncio
from sqlalchemy import text
from app.database import engine, Base
# 导入模型是为了让 SQLAlchemy 知道有哪些表需要创建
# 如果不导入，Base.metadata 里就是空的，建表操作什么也不会做
from app.models import Document, Folder, Log

logger = logging.getLogger(__name__)

async def init_db():
    """
    初始化数据库
    
    Logic Flow:
        第一步：等待连接 (Wait for Connection)。
            尝试连接数据库，如果失败就歇 1 秒再试，最多试 60 次 (1分钟)。
        第二步：创建表 (Create Tables)。
            连接成功后，调用 `Base.metadata.create_all` 自动根据代码里的模型创建数据库表。
            
    Why:
        - 为什么要循环重试？
          防止因为数据库启动慢一点点，导致整个后端服务直接崩掉。
        - 为什么要 create_all？
          为了方便部署。不用手动去数据库里敲 SQL 建表，代码跑起来表就有了。
    """
    logger.info("正在初始化数据库...")
    
    # 第一步：等待数据库连接就绪
    retries = 0
    max_retries = 60
    
    while retries < max_retries:
        try:
            # 尝试建立一个连接并执行简单的查询
            async with engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            logger.info("数据库连接建立成功！")
            break
        except Exception as e:
            retries += 1
            logger.warning(f"等待数据库就绪 ({retries}/{max_retries})... 错误: {e}")
            await asyncio.sleep(1) # 歇一秒再试
            
    if retries >= max_retries:
        raise Exception("连接数据库超时 (60秒)，请检查数据库服务是否正常启动。")

    # 第二步：自动创建表结构
    try:
        async with engine.begin() as conn:
            # run_sync 是因为 create_all 是同步方法，需要在异步环境中运行
            await conn.run_sync(Base.metadata.create_all)
            logger.info("数据库表结构已创建/更新。")
            
            # 这里以后可以加一些初始数据的插入逻辑
            # 比如创建一个默认的管理员账号
            
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        raise e
