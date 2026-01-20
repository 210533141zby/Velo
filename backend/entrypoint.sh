#!/bin/bash
set -e

# 定义数据目录
DATA_DIR="/app/data"
SEED_DIR="/app/seed_data"

# 检查数据库文件是否存在
if [ ! -f "$DATA_DIR/wiki.db" ]; then
    echo "First time startup: Initializing data from seed..."
    
    # 复制 SQLite 数据库
    if [ -f "$SEED_DIR/wiki.db" ]; then
        echo "Copying wiki.db..."
        cp "$SEED_DIR/wiki.db" "$DATA_DIR/wiki.db"
    fi
    
    # 复制向量数据库目录
    if [ -d "$SEED_DIR/chroma_db" ]; then
        echo "Copying chroma_db..."
        cp -r "$SEED_DIR/chroma_db" "$DATA_DIR/chroma_db"
    fi
    
    # 复制上传文件目录
    if [ -d "$SEED_DIR/uploads" ]; then
        echo "Copying uploads..."
        cp -r "$SEED_DIR/uploads" "$DATA_DIR/uploads"
    fi
    
    echo "Data initialization complete."
else
    echo "Data directory already initialized. Skipping seed copy."
fi

# 确保数据目录权限正确
chmod -R 755 "$DATA_DIR"

# 启动应用
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
