# Velo 项目 Docker 部署最佳实践指南 (国内版)

> **⚠️ 核心警告**：
> 1. **严禁使用 Snap**：绝对不要运行 `snap install docker`。
> 2. **国内环境优化**：本文档针对中国大陆网络环境优化，集成阿里云镜像源与并发下载配置。

## 1. 极速重置与安装 (One-Step Script)

如果服务器环境混乱（例如安装了 Snap 版 Docker），请直接复制以下命令块执行。它会自动清理旧环境并使用阿里云镜像源安装最新 Docker。

```bash
# === 1. 清理旧环境 (防止冲突) ===
sudo snap remove docker
sudo apt-get purge -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin docker.io
sudo rm -rf /var/lib/docker /var/lib/containerd

# === 2. 使用官方脚本 + 阿里云镜像一键安装 ===
# 这是最快、最标准的安装方式
curl -fsSL https://get.docker.com | bash -s docker --mirror Aliyun

# === 3. 启动 Docker 并设置开机自启 ===
sudo systemctl enable --now docker
```

## 2. 配置 NVIDIA Runtime 与 性能优化

这一步至关重要，它不仅启用 GPU 支持，还开启了多线程下载加速镜像拉取。

```bash
# === 1. 安装 NVIDIA Container Toolkit ===
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# === 2. 生成高性能配置文件 (关键步骤) ===
# 这里我们直接写入包含优化参数的配置
sudo tee /etc/docker/daemon.json <<EOF
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia",
    "max-concurrent-downloads": 10,
    "registry-mirrors": [
        "https://docker.m.daocloud.io",
        "https://dockerproxy.com"
    ]
}
EOF
# 注意：如果有阿里云专属加速地址，请替换 registry-mirrors 中的内容

# === 3. 重启生效 ===
sudo systemctl restart docker
```

**验证安装**：
```bash
sudo docker run --rm --gpus all ubuntu:22.04 nvidia-smi
```

## 3. 关于模型文件 (Models)

**注意：本自动化部署不包含模型文件。**

开发者请注意：
1.  **自行下载模型**：你需要自行下载所需的 LLM 模型文件。
2.  **放置位置**：请将下载好的模型文件夹放置在项目根目录下的 **`models/`** 文件夹中。

例如，如果你使用的是 Qwen 模型，目录结构应该类似：
```
/root/Velo/models/Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4/...
```

只要模型文件存在于 `models/` 目录中，Docker 启动时会自动挂载并识别。

---

**总结**：使用 **一键安装脚本** 准备好 Docker 环境后，只需将模型放入 `models` 目录，即可直接启动项目。
