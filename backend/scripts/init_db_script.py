"""
=============================================================================
文件: init_db_script.py
描述: 数据库初始化入口脚本

核心功能：
1. 独立运行：作为一个独立的 CLI 脚本运行，不依赖 Web 服务启动。
2. 环境配置：自动设置 sys.path 以便能导入 app 模块。
3. 初始化调用：调用 `app.db_init.init_db` 执行实际的建表和种子数据填充。

使用场景：
- 项目初次部署时初始化数据库。
- CI/CD 流程中重置数据库。
=============================================================================
"""

import asyncio
import os
import sys
from pathlib import Path

# 添加 backend 目录到 python path
# backend/scripts/init_db_script.py -> backend/scripts -> backend
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.db_init import init_db

if __name__ == "__main__":
    # Windows 下 asyncio 策略调整
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    print("正在初始化数据库...")
    asyncio.run(init_db())
    print("数据库初始化完成。")
