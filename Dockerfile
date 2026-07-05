FROM python:3.11-slim

WORKDIR /app

# 系统依赖
# - libmupdf-dev: pymupdf PDF 解析
# - libgomp1: faiss-cpu 运行时
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖（分层缓存：依赖层不常变）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 可选：预先下载 bge-m3 模型（避免首次启动时的网络延迟，约 2.2GB）
# 在国内网络环境下构建较慢，默认跳过；需要时取消注释
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

# 复制代码
COPY . .

# 创建运行时数据目录
RUN mkdir -p data/rule_documents data/rule_index data/env_variables

EXPOSE 6066

# 生产模式启动
CMD ["python", "-c", "from app import run; run()"]
