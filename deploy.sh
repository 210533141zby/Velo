#!/bin/bash
# 如果是在 Windows 下编辑保存的，请先在服务器执行: sed -i 's/\r$//' deploy.sh

# =============================================================================
# 自动部署脚本 (Manual / Non-Docker)
# 适用系统: Ubuntu 20.04 / 22.04 LTS
# =============================================================================

# 遇到错误立即停止
set -e

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# 获取当前脚本所在目录的父级 (项目根目录)
PROJECT_ROOT=$(pwd)
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

echo -e "${GREEN}>>> 开始部署 Wiki 项目...${NC}"

# 1. 系统依赖安装
echo -e "${GREEN}>>> 1. 安装系统依赖...${NC}"
sudo apt-get update
# 增加 dos2unix 以防万一
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev build-essential libpq-dev redis-server nginx git curl dos2unix

# 安装 Node.js
if ! command -v node &> /dev/null; then
    echo "安装 Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
    sudo apt-get install -y nodejs
else
    echo "Node.js 已安装: $(node -v)"
fi

# 2. 配置后端
echo -e "${GREEN}>>> 2. 配置后端服务...${NC}"
cd $BACKEND_DIR

# 创建虚拟环境
if [ ! -d "venv" ]; then
    echo "创建 Python 虚拟环境..."
    python3.11 -m venv venv
fi

# 激活虚拟环境并安装依赖
source venv/bin/activate
echo "安装 Python 依赖 (使用清华源加速)..."
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 创建 .env 文件
if [ ! -f ".env" ]; then
    echo "创建后端 .env 配置文件..."
    cp ../.env.example .env
    sed -i 's/# POSTGRES_SERVER=db/POSTGRES_SERVER=sqlite/' .env
fi

# 3. 配置前端
echo -e "${GREEN}>>> 3. 构建前端资源...${NC}"
cd $FRONTEND_DIR

# 切换 npm 淘宝镜像
echo "设置 npm 淘宝镜像..."
npm config set registry https://registry.npmmirror.com

echo "安装 npm 依赖..."
npm install

echo "编译前端代码..."
npm run build

# 4. 配置 Systemd 服务
echo -e "${GREEN}>>> 4. 配置 Systemd 服务...${NC}"
SERVICE_FILE="/etc/systemd/system/wiki-backend.service"

sudo bash -c "cat > $SERVICE_FILE" <<EOF
[Unit]
Description=Wiki Backend Service
After=network.target redis-server.service

[Service]
User=$USER
Group=$USER
WorkingDirectory=$BACKEND_DIR
Environment="PATH=$BACKEND_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$BACKEND_DIR/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable wiki-backend
sudo systemctl restart wiki-backend

# 5. 配置 Nginx
echo -e "${GREEN}>>> 5. 配置 Nginx 反向代理...${NC}"
NGINX_CONF="/etc/nginx/sites-available/wiki"

sudo bash -c "cat > $NGINX_CONF" <<EOF
server {
    listen 80;
    server_name _;

    location / {
        root $FRONTEND_DIR/dist;
        index index.html;
        try_files \$uri \$uri/ /index.html;
    }

    location /api {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

if [ -L "/etc/nginx/sites-enabled/default" ]; then
    sudo rm /etc/nginx/sites-enabled/default
fi
if [ ! -L "/etc/nginx/sites-enabled/wiki" ]; then
    sudo ln -s /etc/nginx/sites-available/wiki /etc/nginx/sites-enabled/
fi

sudo nginx -t && sudo systemctl restart nginx

echo -e "${GREEN}====================================================${NC}"
echo -e "${GREEN}部署完成！${NC}"
echo -e "请务必执行: vim ${BACKEND_DIR}/.env 填入 OPENAI_API_KEY，然后重启服务！"
echo -e "${GREEN}====================================================${NC}"
