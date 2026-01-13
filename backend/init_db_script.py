import asyncio
import os
import sys

# 添加 backend 目录到 python path
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

from app.db_init import init_db

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    print("正在初始化数据库...")
    asyncio.run(init_db())
    print("数据库初始化完成。")
