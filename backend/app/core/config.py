import os
from typing import List, Optional
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

class Settings(BaseSettings):
    """
    系统配置类
    """
    PROJECT_NAME: str = "Wiki AI"
    API_V1_STR: str = "/api/v1"
    
    # 数据库配置
    POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER", "localhost")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "wiki_db")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")
    
    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        """
        获取数据库连接 URI
        """
        # 如果配置了 sqlite 协议或者没有配置 POSTGRES_SERVER，则使用 sqlite
        if self.POSTGRES_SERVER == "sqlite" or not self.POSTGRES_SERVER:
            return "sqlite+aiosqlite:///./wiki.db"
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    # Redis 配置
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    
    # CORS 跨域配置
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    # AI 配置
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "sk-placeholder")
    OPENAI_API_BASE: Optional[str] = os.getenv("OPENAI_API_BASE", None)
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # vLLM 配置
    VLLM_API_URL: str = os.getenv("VLLM_API_URL", "http://localhost:8001/v1")
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()