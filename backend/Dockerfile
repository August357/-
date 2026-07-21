FROM python:3.11-slim

WORKDIR /app

# 系统依赖（pypdf/docx 等无需编译；保持镜像精简）
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY backend ./backend
COPY core ./core
COPY rebuild_db_v3.py download_models.py ./

# 数据目录（模型/文档/向量库通过 volume 挂载）
RUN mkdir -p data/docs models vector_db logs

EXPOSE 8000

# JWT_SECRET 由 docker-compose 环境注入
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
