#!/bin/bash

# =============================================================================
# Build & Push Script for Metis
# 描述: 自动构建前后端镜像并推送到 Docker Hub
# =============================================================================

# 设置 Docker Hub 用户名 (请修改此处)
DOCKER_USERNAME="swze"

# 颜色定义
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}>>> 1. 准备构建环境...${NC}"

# 导出最新的 requirements (确保依赖是最新的)
# 注意：这里假设 backend/requirements.txt 已经是最新的，或者手动运行导出命令

echo -e "${GREEN}>>> 2. 登录 Docker Hub...${NC}"
docker login

echo -e "${GREEN}>>> 3. 构建并推送 Backend 镜像...${NC}"
cd backend
# 注意：这里会把当前目录下的 wiki.db, chroma_db 等打包进去
docker build -t $DOCKER_USERNAME/metis-backend:latest .
docker push $DOCKER_USERNAME/metis-backend:latest
cd ..

echo -e "${GREEN}>>> 4. 构建并推送 Frontend 镜像...${NC}"
cd frontend
docker build -t $DOCKER_USERNAME/metis-frontend:latest .
docker push $DOCKER_USERNAME/metis-frontend:latest
cd ..

echo -e "${GREEN}>>> 5. 完成！${NC}"
echo -e "现在你可以在任何服务器上使用 docker-compose.prod.yml 启动项目了。"
