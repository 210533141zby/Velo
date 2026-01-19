"""
=============================================================================
文件: config.py
描述: 全局配置管理

核心功能：
1. 环境变量加载：从 .env 文件或系统环境变量中读取配置。
2. 统一管理：集中管理数据库、Redis、AI 模型等所有关键参数。

依赖组件:
- pydantic-settings: 用于自动读取环境变量并进行类型检查。
=============================================================================
"""

import os
from typing import List, Optional
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

class Settings(BaseSettings):
    """
    系统配置类
    
    Why:
    使用 Pydantic 的 BaseSettings 可以自动从环境变量中读取配置，并且自带类型检查。
    如果 .env 写错了（比如端口写成了字符串），程序启动时就会报错提醒，防止带着隐患运行。
    """
    PROJECT_NAME: str = "Wiki AI"
    API_V1_STR: str = "/api/v1"
    
    # =========================================================================
    # 数据库配置 (Database)
    # =========================================================================
    POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER", "localhost")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "wiki_db")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")
    
    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        """
        获取数据库连接 URI (动态计算)
        
        Logic Flow:
            判断是否指定了 PostgreSQL 服务器地址。
            如果是 'sqlite' 或者为空，就自动降级使用本地的 SQLite 文件数据库。
            否则，拼装标准的 PostgreSQL 连接字符串。
            
        Why:
            这样写是为了方便开发。本地跑的时候不用装复杂的 PostgreSQL，直接用 SQLite 就能跑起来。
            上线的时候配上环境变量，就能无缝切换到生产级数据库。
        """
        if self.POSTGRES_SERVER == "sqlite" or not self.POSTGRES_SERVER:
            return "sqlite+aiosqlite:///./wiki.db"
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    # =========================================================================
    # Redis 缓存配置
    # =========================================================================
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    
    # =========================================================================
    # 安全与跨域配置 (CORS)
    # =========================================================================
    # 允许哪些网站访问我们的 API。["*"] 表示允许所有人，开发阶段为了方便这样设。
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    # =========================================================================
    # AI 模型配置
    # =========================================================================
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "sk-placeholder")
    OPENAI_API_BASE: Optional[str] = os.getenv("OPENAI_API_BASE", None)
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # vLLM 配置 (如果自建模型服务的话会用到)
    VLLM_API_URL: str = os.getenv("VLLM_API_URL", "http://localhost:8001/v1")
    
    class Config:
        # 指定环境变量文件名为 .env
        env_file = ".env"
        # 忽略 .env 中多余的字段，不要报错
        extra = "ignore"

settings = Settings()
