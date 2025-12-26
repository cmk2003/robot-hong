# ========================================
# 小虹 - 情感陪伴机器人 Docker 镜像
# 多用户版本
# ========================================

FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # 应用默认配置
    ENV=production \
    GRADIO_SERVER_PORT=7860 \
    DATABASE_PATH=/app/data/emotional_bot.db \
    LOG_PATH=/app/logs \
    LOG_LEVEL=INFO

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制源代码
COPY src/ ./src/

# 创建数据目录和日志目录
RUN mkdir -p /app/data /app/logs

# 暴露端口
EXPOSE 7860

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" || exit 1

# 启动命令
CMD ["python", "src/main.py"]
