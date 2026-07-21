# 模型目录

本目录用于存放本地模型，**不会**提交到 Git 仓库。克隆项目后请自行下载以下模型：

| 用途 | HuggingFace 仓库 | 本地路径 |
|------|------------------|----------|
| 大语言模型 | [THUDM/chatglm3-6b](https://huggingface.co/THUDM/chatglm3-6b) | `models/chatglm3-6b/` |
| 文本嵌入 | [BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5) | `models/bge-large-zh-v1.5/` |
| 重排序 | [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) | `models/bge-reranker-base/` |

> 注意：Embedding 模型已从 bge-small-zh-v1.5 升级为 bge-large-zh-v1.5。
> 更换模型后旧向量库不兼容，下载完成后**必须重建向量库**（见下文第 2 步）。

## 下载方式

### 方式一：使用项目脚本（推荐）

```bash
pip install huggingface_hub
python download_models.py
```

国内用户脚本已默认使用 `hf-mirror.com` 镜像。

### 方式二：huggingface-cli

```bash
huggingface-cli download THUDM/chatglm3-6b --local-dir models/chatglm3-6b
huggingface-cli download BAAI/bge-large-zh-v1.5 --local-dir models/bge-large-zh-v1.5
huggingface-cli download BAAI/bge-reranker-base --local-dir models/bge-reranker-base
```

## 下载完成后

1. 将文档放入 `data/docs/`
2. 运行 `python rebuild_db_v3.py` 构建向量库
3. 启动后端：`uvicorn backend.main:app --host 127.0.0.1 --port 8000`
4. 启动前端：`cd frontend && npm install && npm run dev`
