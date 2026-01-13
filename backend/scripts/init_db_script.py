import asyncio
import os
import sys
from pathlib import Path

# 添加 backend 目录到 python path
# backend/scripts/init_db_script.py -> backend/scripts -> backend
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.db_init import init_db

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    print("正在初始化数据库...")
    asyncio.run(init_db())
    print("数据库初始化完成。")
