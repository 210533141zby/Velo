# 使用轻量级 Python 基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # 设置数据存储路径环境变量，确保代码读取正确位置
    # 注意：你需要确保你的代码 (config.py) 能读取这些环境变量，或者修改代码适配 /app/data
    SQLITE_DB_PATH=/app/data/wiki.db \
    CHROMA_DB_PATH=/app/data/chroma_db \
    UPLOAD_DIR=/app/data/uploads

# 安装系统依赖 (编译某些 Python 包可能需要)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .
# 使用清华源加速安装
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 创建数据目录和种子数据目录
RUN mkdir -p /app/data /app/seed_data

# === 关键步骤：植入初始数据 ===
# 假设构建时，上下文包含 wiki.db, chroma_db, uploads
# 如果本地文件不存在，Docker build 会失败，请确保目录下有这些文件或文件夹
# 使用 COPY --from=context 的替代方案是直接 COPY，前提是 context 正确
# 如果这些文件在 backend/ 目录下：
ONBUILD COPY wiki.db /app/seed_data/wiki.db
ONBUILD COPY chroma_db /app/seed_data/chroma_db
ONBUILD COPY uploads /app/seed_data/uploads

# 复制应用代码
COPY . .

# 赋予脚本执行权限
RUN chmod +x entrypoint.sh

# 暴露端口
EXPOSE 8000

# 设置启动脚本
ENTRYPOINT ["/app/entrypoint.sh"]
