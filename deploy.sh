#!/bin/bash

# =============================================================================
# 自动部署脚本 (Manual / Non-Docker)
# 适用系统: Ubuntu 20.04 / 22.04 LTS
# 描述: 自动安装依赖、构建前端、配置后端服务和 Nginx 反向代理。
# =============================================================================

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
echo -e "${GREEN}>>> 1. 安装系统依赖 (Python, Redis, Nginx, Node.js)...${NC}"
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev build-essential libpq-dev redis-server nginx git curl

# 安装 Node.js (使用 NodeSource 仓库安装最新 LTS)
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
echo "安装 Python 依赖..."
pip install --upgrade pip
pip install -r requirements.txt

# 创建 .env 文件 (如果不存在)
if [ ! -f ".env" ]; then
    echo "创建后端 .env 配置文件..."
    cp ../.env.example .env
    # 强制使用 SQLite 以简化部署
    sed -i 's/# POSTGRES_SERVER=db/POSTGRES_SERVER=sqlite/' .env
    echo -e "${RED}请注意: 已自动创建 backend/.env，请务必稍后编辑填入 OPENAI_API_KEY！${NC}"
fi

# 3. 配置前端
echo -e "${GREEN}>>> 3. 构建前端资源...${NC}"
cd $FRONTEND_DIR
echo "安装 npm 依赖..."
npm install
echo "编译前端代码..."
npm run build

# 4. 配置 Systemd 服务 (让后端开机自启)
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
    server_name _;  # 匹配所有域名/IP

    # 前端静态文件
    location / {
        root $FRONTEND_DIR/dist;
        index index.html;
        try_files \$uri \$uri/ /index.html;
    }

    # 后端 API 转发
    location /api {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# 启用配置
if [ -L "/etc/nginx/sites-enabled/default" ]; then
    sudo rm /etc/nginx/sites-enabled/default
fi
if [ ! -L "/etc/nginx/sites-enabled/wiki" ]; then
    sudo ln -s /etc/nginx/sites-available/wiki /etc/nginx/sites-enabled/
fi

sudo nginx -t && sudo systemctl restart nginx

echo -e "${GREEN}====================================================${NC}"
echo -e "${GREEN}部署完成！${NC}"
echo -e "请执行以下步骤完成最后配置："
echo -e "1. 编辑 ${BACKEND_DIR}/.env 文件，填入您的 OPENAI_API_KEY。"
echo -e "2. 重启后端服务: sudo systemctl restart wiki-backend"
echo -e "3. 访问服务器 IP 即可使用。"
echo -e "${GREEN}====================================================${NC}"
