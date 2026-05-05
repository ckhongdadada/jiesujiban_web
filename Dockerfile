# 接诉即办智能服务系统 Dockerfile
# 多阶段构建：基础镜像 -> 依赖安装 -> 应用部署

# ==================== 基础镜像 ====================
FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

WORKDIR /app

# ==================== 依赖构建 ====================
FROM base AS builder

COPY requirements.txt requirements_optional_rag_ner.txt ./

RUN pip install --upgrade pip setuptools wheel \
    && pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118 \
    && pip install -r requirements.txt \
    && pip install -r requirements_optional_rag_ner.txt

# ==================== 最终镜像 ====================
FROM base AS runtime

COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/ ./src/
COPY configs/ ./configs/
COPY templates/ ./templates/
COPY app.py .
COPY pytest.ini .

RUN mkdir -p /app/data/runtime /app/logs /app/models

ENV PYTHONPATH=/app
ENV PYTHONPATH=/app/src:$PYTHONPATH

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

CMD ["python", "app.py"]
