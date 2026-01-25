# 使用轻量级 Python 基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SQLITE_DB_PATH=/app/data/wiki.db \
    CHROMA_DB_PATH=/app/data/chroma_db \
    UPLOAD_DIR=/app/data/uploads

# --- [优化1] 强制更换 APT 国内源，解决下载慢的问题 ---
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --- [优化2] 先安装依赖，保证缓存生效 ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 创建必要的目录
RUN mkdir -p /app/data /app/seed_data

# --- [关键修复] ---
# 1. 先把所有文件（包括可能存在的 backend/data）复制进镜像
COPY . .

# 2. 然后用 RUN 指令进行“尝试性备份”
# 这里可以使用 Shell 语法：如果 data/wiki.db 存在，就拷到 seed_data；如果不存在，就忽略 (|| true)
RUN if [ -f "/app/data/wiki.db" ]; then cp /app/data/wiki.db /app/seed_data/; fi && \
    if [ -d "/app/data/chroma_db" ]; then cp -r /app/data/chroma_db /app/seed_data/; fi

# 赋予脚本执行权限
RUN chmod +x entrypoint.sh

# 暴露端口
EXPOSE 8000

# 启动脚本
ENTRYPOINT ["/app/entrypoint.sh"]