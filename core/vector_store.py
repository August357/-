import os
import re
import shutil
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader
)
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


# ============================================
# 项目根目录
# ============================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 文档目录
DOCS_DIR = os.path.join(BASE_DIR, "data", "docs")

# 向量库存储路径
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db")

# Chunk信息保存路径
CHUNK_INFO_PATH = os.path.join(
    BASE_DIR,
    "vector_db",
    "chunk_info.txt"
)

# Embedding模型配置
# 注意：更换模型后必须重建向量库（向量维度/分布不同，旧 index.faiss 不兼容）
EMBEDDING_MODEL_NAME = "bge-large-zh-v1.5"
EMBEDDING_MODEL_PATH = os.path.join(BASE_DIR, "models", EMBEDDING_MODEL_NAME)

# ============================================
# 设备配置
# ============================================

# Embedding 使用 CPU，为 ChatGLM3-6B 保留 GPU 显存
DEVICE = "cpu"

# ============================================
# 关闭 HuggingFace 一些警告
# ============================================

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"

# ============================================
# 懒加载 Embedding（须先于 LLM 加载，否则 Windows 上 4-bit 量化会崩溃）
# ============================================

_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        # 本地模型缺失时立即报错，避免 sentence-transformers 静默联网下载导致请求挂起
        if not os.path.exists(EMBEDDING_MODEL_PATH):
            raise FileNotFoundError(
                f"Embedding模型目录不存在: {EMBEDDING_MODEL_PATH}。"
                "请先运行 python download_models.py 下载模型。"
            )
        logger.info("正在加载Embedding模型...")
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_PATH,
            model_kwargs={"device": DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding模型加载成功！")
    return _embedding_model


class _EmbeddingProxy:
    """兼容旧代码中对 embedding_model 的直接引用"""

    def embed_documents(self, texts):
        return get_embedding_model().embed_documents(texts)

    def embed_query(self, text):
        return get_embedding_model().embed_query(text)


embedding_model = _EmbeddingProxy()

# ============================================
# 文本切分器创建函数（动态配置）
# ============================================

def create_splitter(chunk_size=500, chunk_overlap=100, split_mode="zh"):
    """
    根据配置动态创建文本切分器

    Args:
        chunk_size: Chunk大小
        chunk_overlap: Chunk重叠大小
        split_mode: 切分策略（zh/paragraph/sentence）

    Returns:
        RecursiveCharacterTextSplitter: 文本切分器
    """
    # 根据切分模式选择分隔符
    if split_mode == "paragraph":
        separators = ["\n\n", "\n"]
    elif split_mode == "sentence":
        separators = ["。", "！", "？", "；"]
    else:  # zh - 中文智能切分
        separators = ["\n\n", "\n", "。", "！", "？", "；", "，"]

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )


# 默认切分器（兼容旧代码）
text_splitter = create_splitter()


# ============================================
# 结构化文档按章节/标题边界切分
# ============================================

# 章节/标题行：「实验一：」「实验3 」「第三章」「一、」等
HEADING_PATTERN = re.compile(
    r"^(实验\s*[0-9一二三四五六七八九十]+\s*[：:、\s．.]|"
    r"第\s*[0-9一二三四五六七八九十]+\s*[章节篇]\s*[：:、\s．.]?|"
    r"[一二三四五六七八九十]+、)",
    re.MULTILINE,
)


def split_by_headings(documents: list) -> list:
    """
    对结构化文档优先按章节/标题边界切分，标题归入 metadata['section']。
    无标题（或仅一处标题）的文档原样保留，由 RecursiveCharacterTextSplitter 兜底。
    返回的段落仍会再经 Recursive 切分器约束 chunk_size。
    """
    from langchain_core.documents import Document

    result = []
    for doc in documents:
        text = doc.page_content
        matches = list(HEADING_PATTERN.finditer(text))
        if len(matches) < 2:
            result.append(doc)
            continue

        # 第一个标题之前的前言部分
        if matches[0].start() > 0:
            preface = text[:matches[0].start()].strip()
            if preface:
                result.append(Document(page_content=preface, metadata=dict(doc.metadata)))

        for i, m in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section = text[m.start():end].strip()
            if not section:
                continue
            meta = dict(doc.metadata)
            meta["section"] = m.group(0).strip("：:、．. \t\n")
            result.append(Document(page_content=section, metadata=meta))

    return result


# ============================================
# 当前配置（保存到文件供查询）
# ============================================

CONFIG_PATH = os.path.join(BASE_DIR, "vector_db", "config.json")

def save_config(chunk_size, chunk_overlap, split_mode, top_k):
    """保存当前配置到文件"""
    import json
    config = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "split_mode": split_mode,
        "top_k": top_k
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

def load_config():
    """加载当前配置"""
    import json
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "chunk_size": 500,
        "chunk_overlap": 100,
        "split_mode": "zh",
        "top_k": 5
    }


# ============================================
# 文档加载函数
# ============================================

def load_documents(doc_dir):
    documents = []
    for filename in os.listdir(doc_dir):
        file_path = os.path.join(doc_dir, filename)
        try:
            if filename.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
            elif filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            else:
                continue
            docs = loader.load()

            if filename.endswith(".docx"):
                content_len = len(docs[0].page_content) if docs else 0
                logger.debug("%s 内容长度: %d", filename, content_len)

            for doc in docs:
                doc.metadata["source"] = filename
            documents.extend(docs)
            logger.info("[OK] 已加载: %s", filename)
        except Exception as e:
            logger.error("[FAIL] 加载失败: %s (%s)", filename, e)
    return documents

# ============================================
# 构建向量库函数（可被外部调用）
# ============================================

def build_vector_db(chunk_size=500, chunk_overlap=100, split_mode="zh", top_k=5, progress_callback=None):
    """
    构建向量数据库（支持动态配置）

    Args:
        chunk_size: Chunk大小（100-2000）
        chunk_overlap: Chunk重叠（0-500）
        split_mode: 切分策略（zh/paragraph/sentence）
        top_k: 检索TopK数量（1-10）
        progress_callback: 可选回调 fn(progress:int, stage:str)，
            在 加载/切分/向量化/保存 各阶段上报真实进度（0-100）

    Returns:
        bool: 是否成功
    """
    def report(progress, stage):
        logger.info("[构建进度 %d%%] %s", progress, stage)
        if progress_callback:
            try:
                progress_callback(progress, stage)
            except Exception:
                pass

    # 保存配置
    report(2, "保存配置")
    save_config(chunk_size, chunk_overlap, split_mode, top_k)

    logger.info(
        "知识库构建配置: chunk_size=%d, chunk_overlap=%d, split_mode=%s, top_k=%d",
        chunk_size, chunk_overlap, split_mode, top_k,
    )

    # 检查文档目录
    if not os.path.exists(DOCS_DIR):
        logger.error("文档目录不存在: %s", DOCS_DIR)
        return False

    # 读取文档
    report(5, "正在读取文档...")
    documents = load_documents(DOCS_DIR)

    if not documents:
        logger.error("未找到任何文档")
        return False

    report(25, f"已加载 {len(documents)} 个文档")

    for doc in documents:
        logger.debug("文档 metadata: %s", doc.metadata)

    # 文本切分：结构化文档先按章节/标题分段，再由 Recursive 切分器约束长度（兜底）
    report(30, "正在切分文本...")

    structured_docs = split_by_headings(documents)
    if len(structured_docs) != len(documents):
        logger.info("结构化切分：%d 个文档 -> %d 个章节段落", len(documents), len(structured_docs))

    splitter = create_splitter(chunk_size, chunk_overlap, split_mode)
    split_docs = splitter.split_documents(structured_docs)

    report(45, f"切分完成，共 {len(split_docs)} 个Chunk")

    # 为每个 chunk 添加唯一标识
    for i, doc in enumerate(split_docs):
        doc.metadata["chunk_id"] = i + 1  # 添加chunk_id
        # 如果没有page信息，默认设为1
        if "page" not in doc.metadata:
            doc.metadata["page"] = 1

    logger.info("总Chunk数量：%d", len(split_docs))

    # 删除旧向量库
    if os.path.exists(VECTOR_DB_PATH):
        shutil.rmtree(VECTOR_DB_PATH)

    # 创建FAISS向量库（最耗时阶段）
    report(55, "正在向量化并创建FAISS向量库...")
    vector_store = FAISS.from_documents(
        split_docs,
        get_embedding_model()
    )

    report(85, "向量化完成，正在保存向量库...")

    # 创建保存目录
    if not os.path.exists(VECTOR_DB_PATH):
        os.makedirs(VECTOR_DB_PATH)

    # 保存向量数据库
    vector_store.save_local(VECTOR_DB_PATH)

    # 须在向量库目录创建后再写入，避免被上面的 rmtree 删除
    save_chunk_info_file(split_docs)

    report(100, "向量数据库构建完成")
    logger.info("向量数据库构建完成！位置: %s", VECTOR_DB_PATH)

    return True

def save_chunk_info_file(docs):
    """将 Chunk 列表写入 chunk_info.txt（供可视化页面使用）"""
    os.makedirs(os.path.dirname(CHUNK_INFO_PATH), exist_ok=True)
    with open(CHUNK_INFO_PATH, "w", encoding="utf-8") as f:
        for i, doc in enumerate(docs):
            chunk_id = doc.metadata.get("chunk_id", i + 1)
            page = doc.metadata.get("page", "未知")
            f.write(f"Chunk {chunk_id}\n")
            f.write(f"来源:{doc.metadata.get('source', '未知')}\n")
            f.write(f"页码:P{page}\n")
            f.write(f"长度:{len(doc.page_content)}\n")
            f.write(doc.page_content)
            f.write("\n")
            f.write("=" * 100)
            f.write("\n")
    logger.info("Chunk信息已保存到: %s", CHUNK_INFO_PATH)


def ensure_chunk_info_file():
    """若 chunk_info.txt 缺失，从现有 FAISS 向量库导出"""
    if os.path.exists(CHUNK_INFO_PATH):
        return True
    faiss_path = os.path.join(VECTOR_DB_PATH, "index.faiss")
    if not os.path.exists(faiss_path):
        return False
    vector_store = load_vector_store()
    if not vector_store:
        return False
    docs = list(vector_store.docstore._dict.values())
    docs.sort(key=lambda d: d.metadata.get("chunk_id", 0))
    save_chunk_info_file(docs)
    return True


def parse_chunk_info_file():
    """解析 chunk_info.txt 为结构化列表"""
    ensure_chunk_info_file()
    if not os.path.exists(CHUNK_INFO_PATH):
        return []

    with open(CHUNK_INFO_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = []
    for block in text.split("=" * 100):
        if not block.strip():
            continue
        lines = block.strip().split("\n")
        if len(lines) < 4:
            continue

        chunk_id = 0
        source = ""
        page = 1
        length = 0
        content_lines = []

        for i, line in enumerate(lines):
            if line.startswith("Chunk "):
                chunk_id = int(line.split()[1])
            elif line.startswith("来源:"):
                source = line.split(":", 1)[1].strip()
            elif line.startswith("页码:P"):
                page = int(line.split("P")[1])
            elif line.startswith("长度:"):
                length = int(line.split(":")[1].strip())
            elif i >= 4:
                content_lines.append(line)

        if chunk_id > 0:
            chunks.append({
                "chunk_id": chunk_id,
                "source": source,
                "page": page,
                "length": length,
                "content": "\n".join(content_lines).strip(),
            })

    return chunks


def list_chunks_grouped_by_source():
    """按文档分组返回 Chunk 列表"""
    chunks = parse_chunk_info_file()
    groups = {}
    order = []

    for chunk in chunks:
        source = chunk["source"]
        if source not in groups:
            groups[source] = []
            order.append(source)
        groups[source].append(chunk)

    documents = []
    for source in order:
        doc_chunks = groups[source]
        documents.append({
            "source": source,
            "chunk_count": len(doc_chunks),
            "total_length": sum(c["length"] for c in doc_chunks),
            "chunks": doc_chunks,
        })

    return {
        "documents": documents,
        "total": len(chunks),
        "document_count": len(documents),
    }


def read_chunk_info_content():
    """读取 Chunk 可视化文本（纯文本，兼容旧接口）"""
    ensure_chunk_info_file()
    if not os.path.exists(CHUNK_INFO_PATH):
        return ""
    with open(CHUNK_INFO_PATH, "r", encoding="utf-8") as f:
        return f.read()

# ============================================
# 查看知识库文件列表
# ============================================

def list_knowledge_files():
    if not os.path.exists(DOCS_DIR):
        return []
    files = os.listdir(DOCS_DIR)
    return sorted(files)

# ============================================
# 删除知识库文件
# ============================================

def delete_file(filename):
    try:
        path = os.path.join(DOCS_DIR, filename)
        if not os.path.exists(path):
            return "文件不存在"
        os.remove(path)
        build_vector_db()
        return f"{filename} 删除成功"
    except Exception as e:
        logger.error("删除文件失败: %s (%s)", filename, e)
        return str(e)

# ============================================
# 预览文档内容
# ============================================

def preview_file(filename):
    path = os.path.join(DOCS_DIR, filename)

    if not os.path.exists(path):
        return "文件不存在"

    try:
        if filename.endswith(".txt"):
            docs = TextLoader(path, encoding="utf-8").load()
        elif filename.endswith(".pdf"):
            docs = PyPDFLoader(path).load()
        elif filename.endswith(".docx"):
            docs = Docx2txtLoader(path).load()
        else:
            return "不支持的文件格式"

        if not docs:
            return "文件内容为空"

        content = docs[0].page_content
        if len(content) > 5000:
            content = content[:5000] + "\n\n--- 内容已截断 ---"

        return content
    except Exception as e:
        logger.error("预览文件失败: %s (%s)", filename, e)
        return f"读取失败: {str(e)}"

# ============================================
# 增量更新向量库
# ============================================

def load_vector_store():
    if not os.path.exists(VECTOR_DB_PATH):
        return None
    return FAISS.load_local(
        VECTOR_DB_PATH,
        get_embedding_model(),
        allow_dangerous_deserialization=True
    )

def add_documents_to_db(file_paths):
    """向向量库添加新文档（增量更新）"""
    try:
        # 加载现有向量库或创建新库
        vector_store = load_vector_store()

        # 加载新文档
        new_docs = []
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            try:
                if filename.endswith(".txt"):
                    loader = TextLoader(file_path, encoding="utf-8")
                elif filename.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                elif filename.endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                else:
                    continue

                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = filename
                new_docs.extend(docs)
                logger.info("[OK] 已加载: %s", filename)
            except Exception as e:
                logger.error("[FAIL] 加载失败: %s (%s)", filename, e)

        if not new_docs:
            return "没有可添加的文档"

        # 切分文本
        split_docs = text_splitter.split_documents(new_docs)

        # 添加到向量库
        if vector_store:
            vector_store.add_documents(split_docs)
            logger.info("已向向量库添加 %d 个文本块", len(split_docs))
        else:
            vector_store = FAISS.from_documents(split_docs, get_embedding_model())
            logger.info("创建新向量库，包含 %d 个文本块", len(split_docs))

        # 保存向量库
        if not os.path.exists(VECTOR_DB_PATH):
            os.makedirs(VECTOR_DB_PATH)
        vector_store.save_local(VECTOR_DB_PATH)

        return f"成功添加 {len(new_docs)} 个文档，{len(split_docs)} 个文本块"

    except Exception as e:
        logger.error("增量更新失败: %s", e)
        return f"增量更新失败: {str(e)}"

# ============================================
# 直接运行时执行构建
# ============================================

if __name__ == "__main__":
    from core.logging_config import setup_logging
    setup_logging()
    build_vector_db()
