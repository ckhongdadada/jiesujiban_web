# Docker 部署指南

## 目录结构

```
项目根目录/
├── Dockerfile                    # 主应用镜像定义
├── docker-compose.yml            # 多服务编排配置
├── .dockerignore                 # 构建排除文件
├── docker-deploy.sh              # Linux/Mac 部署脚本
├── docker-deploy.bat             # Windows 部署脚本
├── docker/
│   ├── .env.example              # 环境变量模板
│   ├── prometheus/
│   │   └── prometheus.yml        # Prometheus 配置
│   └── grafana/
│       └── provisioning/
│           └── datasources/
│               └── datasources.yml
├── models/                       # 模型文件目录（需挂载）
├── data/                         # 数据目录（需挂载）
└── logs/                         # 日志目录（需挂载）
```

## 快速开始

### 1. 前置条件

- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA Container Toolkit（GPU 支持）
- 至少 16GB 可用内存
- 至少 50GB 磁盘空间

### 2. 安装 Docker

**Windows**:
```powershell
# 下载并安装 Docker Desktop
# https://www.docker.com/products/docker-desktop

# 安装 NVIDIA Container Toolkit（GPU 支持）
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

**Linux (Ubuntu)**:
```bash
# 安装 Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# 安装 Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 安装 NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 3. 准备模型文件

```bash
# 创建模型目录
mkdir -p models

# 将模型文件放入 models 目录
# 目录结构应为:
# models/
# ├── local_roberta_model/      # 分类基础模型
# ├── final_model_fgm/          # 分类微调权重
# ├── qwen_models/
# │   └── Qwen/
# │       └── Qwen2.5-1.5B-Instruct/  # 生成模型
# └── qwen_reply_model/         # 生成微调权重
```

### 4. 配置环境变量

```bash
# 复制环境变量模板
cp docker/.env.example .env

# 编辑配置（可选）
# Windows: notepad .env
# Linux/Mac: nano .env
```

### 5. 启动服务

**Windows**:
```powershell
.\docker-deploy.bat up
```

**Linux/Mac**:
```bash
chmod +x docker-deploy.sh
./docker-deploy.sh up
```

### 6. 验证部署

```bash
# 检查服务状态
docker-compose ps

# 检查健康状态
curl http://localhost:5000/api/health

# 查看日志
docker-compose logs -f app
```

## 服务说明

### 核心服务

| 服务 | 端口 | 说明 |
|------|------|------|
| app | 5000 | 主应用服务 |
| redis | 6379 | 缓存服务 |
| neo4j | 7474/7687 | 图数据库 |

### 可选服务（监控）

| 服务 | 端口 | 说明 |
|------|------|------|
| prometheus | 9090 | 监控数据收集 |
| grafana | 3000 | 可视化面板 |

启动监控服务：
```bash
docker-compose --profile monitoring up -d
```

## 常用命令

```bash
# 启动所有服务
docker-compose up -d

# 停止所有服务
docker-compose down

# 重启服务
docker-compose restart

# 查看日志
docker-compose logs -f app

# 查看服务状态
docker-compose ps

# 进入容器
docker-compose exec app bash

# 重新构建镜像
docker-compose build --no-cache

# 清理所有数据（危险操作）
docker-compose down -v
```

## 配置说明

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| APP_PORT | 5000 | 应用端口 |
| NEO4J_PASSWORD | password123 | Neo4j 密码 |
| LOG_LEVEL | INFO | 日志级别 |
| GRAFANA_USER | admin | Grafana 用户名 |
| GRAFANA_PASSWORD | admin | Grafana 密码 |

### 资源限制

编辑 `docker-compose.yml` 调整资源限制：

```yaml
services:
  app:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 16G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## 数据持久化

以下目录通过 Docker Volume 持久化：

| Volume | 说明 |
|--------|------|
| redis-data | Redis 数据 |
| neo4j-data | Neo4j 数据 |
| neo4j-logs | Neo4j 日志 |
| prometheus-data | Prometheus 数据 |
| grafana-data | Grafana 配置 |

## 健康检查

所有服务都配置了健康检查：

```bash
# 查看健康状态
docker-compose ps

# 手动健康检查
curl http://localhost:5000/api/health
curl http://localhost:7474
redis-cli ping
```

## 故障排查

### 容器无法启动

```bash
# 查看详细日志
docker-compose logs app

# 检查资源使用
docker stats

# 检查 GPU 可用性
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi
```

### GPU 不可用

```bash
# 检查 NVIDIA Docker 支持
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi

# 如果失败，重新配置 NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 内存不足

```bash
# 检查内存使用
docker stats

# 增加 Docker 内存限制（Docker Desktop）
# Settings -> Resources -> Memory
```

### 网络问题

```bash
# 检查网络
docker network ls
docker network inspect complaint-network

# 重建网络
docker-compose down
docker-compose up -d
```

## 生产部署建议

### 1. 使用 HTTPS

```yaml
# docker-compose.yml 添加 nginx 反向代理
services:
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
```

### 2. 配置备份

```bash
# 备份 Neo4j 数据
docker-compose exec neo4j neo4j-admin database dump neo4j --to-path=/backups

# 备份 Redis 数据
docker-compose exec redis redis-cli BGSAVE
```

### 3. 日志管理

```yaml
# 限制日志大小
services:
  app:
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"
```

### 4. 安全加固

```yaml
# 以非 root 用户运行
services:
  app:
    user: "1000:1000"
    read_only: true
    tmpfs:
      - /tmp
```

## Kubernetes 部署

如需 Kubernetes 部署，可使用以下命令生成配置：

```bash
# 使用 kompose 转换
kompose convert -f docker-compose.yml

# 或手动创建 Kubernetes 配置
# 参考 k8s/ 目录
```

## 更新与回滚

```bash
# 拉取最新代码
git pull

# 重新构建并启动
docker-compose build --no-cache
docker-compose up -d

# 回滚到上一版本
docker-compose down
docker tag complaint-system:previous complaint-system:latest
docker-compose up -d
```
