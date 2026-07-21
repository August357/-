# RAGFlow

基于 RAG 的中文智能问答系统，支持 PDF/Word/TXT 文档检索与 ChatGLM3-6B 本地问答。

## 功能特性

- 文档加载与分块（PDF、Word、TXT），结构化文档按章节/标题边界优先切分
- 特殊格式识别：表格抽取为 Markdown、代码块加围栏保留，切分时整段原子保护
  （PDF 用 pdfplumber、DOCX 用 python-docx 按正文顺序解析、TXT 识别围栏/缩进代码与管道表格）
- FAISS 向量检索 + BM25 混合检索（归一化加权融合）+ BGE 重排序
- ChatGLM3-6B 4-bit 本地推理
- 多轮对话：按 session 取最近 3 轮历史，指代/省略问题先改写再检索
- SSE 真流式输出（`/api/ask/stream`），保留非流式接口降级
- 向量库后台异步重建，轮询真实进度
- JWT 认证（注册/登录）、文件上传校验、路径穿越防护
- Vue3 前端：对话、知识库、Chunk 可视化、配置、系统监控、PDF 导出

## 项目结构

```
RAGFlow/
├── backend/          # FastAPI 后端（main.py 为唯一入口，含 Dockerfile）
├── core/             # RAG 核心（qa_chain / vector_store / llm / prompt / config）
├── frontend/         # Vue3 前端（含 Dockerfile 与 nginx.conf）
├── tests/            # pytest 测试
├── data/docs/        # 本地文档目录（不上传）
├── models/           # 本地模型（需自行下载，见 models/README.md）
├── vector_db/        # 向量库（运行后生成）
├── download_models.py
├── rebuild_db_v3.py  # 重建向量库（统一调用 core.vector_store.build_vector_db）
├── docker-compose.yml
└── requirements.txt
```

## 环境要求

- Python 3.10+
- Node.js 18+
- NVIDIA GPU（推荐 8GB+ 显存，用于 4-bit ChatGLM3-6B）

## 快速开始

### 1. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，必须设置 JWT_SECRET（随机长字符串），否则后端拒绝启动
```

也可以直接用环境变量：`export JWT_SECRET='<随机长字符串>'`。
全部可配置项见 `core/config.py` 与 `.env.example`（检索阈值、融合权重、上传限额、CORS、日志等）。

### 3. 下载模型

```bash
python download_models.py
```

模型清单（详见 [models/README.md](models/README.md)）：

| 用途 | 模型 | 本地路径 |
|------|------|----------|
| 大语言模型 | THUDM/chatglm3-6b | `models/chatglm3-6b/` |
| 文本嵌入 | BAAI/bge-large-zh-v1.5 | `models/bge-large-zh-v1.5/` |
| 重排序 | BAAI/bge-reranker-base | `models/bge-reranker-base/` |

> Embedding 模型已升级为 bge-large-zh-v1.5，更换后**必须重建向量库**。

### 4. 构建向量库

将文档放入 `data/docs/` 后执行：

```bash
python rebuild_db_v3.py
```

也可在前端「配置」页提交后台构建任务并查看实时进度。

### 5. 启动服务

```bash
# 后端
uvicorn backend.main:app --host 127.0.0.1 --port 8000

# 前端（新终端）
cd frontend
npm install
npm run dev
```

浏览器访问 `http://localhost:5173`。
模型缺失时后端仍可启动，`/api/health` 返回 `llm_ready=false`。

## Docker 部署

```bash
# JWT_SECRET 通过环境变量或根目录 .env 注入
export JWT_SECRET='<随机长字符串>'
docker compose up --build
```

- 后端：`http://localhost:8000`
- 前端（nginx）：`http://localhost:8080`
- `models/`、`data/`、`vector_db/`、`logs/` 以 volume 挂载，模型不进入镜像
- GPU 部署请取消 `docker-compose.yml` 中的 nvidia 设备注释

## 测试

```bash
pytest tests/ -v
```

- `tests/test_units.py`：纯逻辑单测，不依赖模型文件
- `tests/test_qa_chain.py`：集成测试，使用真实 Embedding/Reranker（CPU）+ mock LLM；
  模型未下载时自动 skip

## 说明

- **模型文件**、**知识库文档**、**向量库**、**node_modules** 已加入 `.gitignore`，不会上传到 GitHub
- 克隆仓库后需按上述步骤下载模型、放入文档并重建向量库
- 日志默认输出到控制台与 `logs/app.log`（`LOG_LEVEL` / `LOG_FILE` 可配）
