你只需要关注两个核心文件： /etc/docker/daemon.json 和 docker-compose.yml 。
 1. 宿主机 Docker 配置 (一次性配置)
文件位置 : /etc/docker/daemon.json 关键点 :

- 必须显式注册 nvidia 运行时。
- 不要 随意添加 debug: true 或 iptables: false ，除非你非常清楚自己在做什么（刚才的网络链丢失就是因为配置冲突导致的）。
- 修改后必须重启 Docker： sudo systemctl restart docker 。
推荐的标准配置 :

```
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/
            nvidia-container-runtime
            ",
            "runtimeArgs": []
        }
    }
}
``` 2. 项目配置 (docker-compose.yml)
关键修改 : 针对宿主机库路径不匹配的问题，我们采用了“特权模式 + 环境变量”的强穿透方案。

下次部署时，请确保 vllm 服务包含以下 3 个关键配置 :

1. privileged: true : 开启特权模式，绕过 Runtime 的钩子限制。
2. environment : 手动声明 GPU 可见性。
3. deploy : 保留标准的资源声明，以防万一环境恢复正常。
参考片段 :

```
  vllm:
    # ... 其他配置 ...
    privileged: true  # <--- 关键点 
    1: 必开
    environment:
      - 
      NVIDIA_VISIBLE_DEVICES=all    
        # <--- 关键点 2: 必填
      - 
      NVIDIA_DRIVER_CAPABILITIES=all
        # <--- 关键点 3: 必填
      - VLLM_NCCL_SO_PATH=/usr/lib/
      x86_64-linux-gnu/libnccl.so.2
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```
### 🚨 常见报错与自救手册
如果下次你遇到类似报错，可以按以下流程快速自救：

现象/报错 原因 快速修复命令 KeyError: 'ContainerConfig' Docker 缓存状态与 Compose 不一致 docker-compose down --remove-orphans 然后重试 Chain 'DOCKER-ISOLATION-STAGE-2' not exist iptables 规则链丢失或损坏 sudo iptables -F (清空规则) 后 sudo systemctl restart docker load library failed: libnvidia-ml.so.1 宿主机驱动库路径与容器不匹配 确保开启 privileged: true ；若仍不行，手动挂载 /usr/lib/x86_64-linux-gnu vLLM 启动卡住无日志 正在下载大模型权重 耐心等待，或进入容器查看： docker logs -f metis_vllm

### 💡 给未来的建议
1. 保持环境纯净 : 尽量不要手动修改 /etc/nvidia-container-runtime/config.toml ，使用默认配置通常最稳。
2. 特权模式是“银弹” : 在开发环境或内网服务器中，如果遇到顽固的 GPU 挂载问题，开启 privileged: true 通常能瞬间解决问题，但生产环境请谨慎使用。
3. 验证脚本 : 保留好我为你写的 verify_gpu.py ，下次部署完环境先跑一遍脚本，能省去很多排查时间。
祝你的项目运行顺利！如果有新问题，随时找我。