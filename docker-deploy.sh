#!/bin/bash
# 接诉即办系统 Docker 部署脚本

set -e

echo "=============================================="
echo "  接诉即办智能服务系统 - Docker 部署"
echo "=============================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 检查 Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}错误: Docker 未安装${NC}"
    echo "请访问 https://docs.docker.com/get-docker/ 安装 Docker"
    exit 1
fi

if ! docker compose version &> /dev/null; then
    echo -e "${RED}错误: Docker Compose 插件未安装${NC}"
    echo "请访问 https://docs.docker.com/compose/install/ 安装 Docker Compose"
    exit 1
fi

echo -e "${GREEN}✓ Docker 环境检查通过${NC}"

# 检查 NVIDIA Docker
if docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA Docker 支持已启用${NC}"
else
    echo -e "${YELLOW}⚠ NVIDIA Docker 支持未启用，将使用 CPU 模式${NC}"
    echo "如需 GPU 加速，请安装 nvidia-container-toolkit"
fi

# 检查环境文件
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}未找到 .env 文件，使用默认配置${NC}"
    cp docker/.env.example .env
fi

# 检查模型目录
if [ ! -d "checkpoints" ]; then
    echo -e "${YELLOW}未找到 checkpoints 目录，创建中...${NC}"
    mkdir -p checkpoints
fi

# 检查数据目录
mkdir -p data/runtime logs

# 解析命令
case "${1:-up}" in
    up)
        echo ""
        echo "启动服务..."
        docker compose -p jsjb up -d
        
        echo ""
        echo -e "${GREEN}=============================================="
        echo "  服务启动成功！"
        echo "==============================================${NC}"
        echo ""
        echo "访问地址:"
        echo "  - API 服务:    http://localhost:${APP_PORT:-5000}"
        echo "  - Neo4j 控制台: http://localhost:${NEO4J_HTTP_PORT:-7474}"
        echo ""
        echo "查看日志: docker compose -p jsjb logs -f app"
        echo "停止服务: ./docker-deploy.sh down"
        ;;
    
    down)
        echo "停止服务..."
        docker compose -p jsjb down
        echo -e "${GREEN}✓ 服务已停止${NC}"
        ;;
    
    restart)
        echo "重启服务..."
        docker compose -p jsjb restart
        echo -e "${GREEN}✓ 服务已重启${NC}"
        ;;
    
    logs)
        docker compose -p jsjb logs -f ${2:-app}
        ;;
    
    build)
        echo "构建镜像..."
        docker compose -p jsjb build --no-cache
        echo -e "${GREEN}✓ 镜像构建完成${NC}"
        ;;
    
    ps)
        docker compose -p jsjb ps
        ;;
    
    *)
        echo "用法: $0 {up|down|restart|logs|build|ps}"
        echo ""
        echo "命令说明:"
        echo "  up      - 启动所有服务"
        echo "  down    - 停止所有服务"
        echo "  restart - 重启所有服务"
        echo "  logs    - 查看日志 (可指定服务名)"
        echo "  build   - 重新构建镜像"
        echo "  ps      - 查看服务状态"
        exit 1
        ;;
esac
