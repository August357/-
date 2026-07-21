"""
RAG问答链核心模块
整合检索、重排序、Prompt构建和LLM调用
"""

import os
import gc
import json
import re
import logging
import threading

from langchain_community.vectorstores import FAISS

from core.config import get_settings
from core.llm import chat_model, stream_chat_model
from core.prompt import prompt_builder
from core.vector_store import get_embedding_model, VECTOR_DB_PATH

logger = logging.getLogger(__name__)

# ============================================
# 项目根目录
# ============================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================
# 关闭HF联网警告
# ============================================

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"

# ============================================
# Reranker模型（懒加载）
# ============================================

_reranker = None

def load_reranker():
    """懒加载BGE-Reranker模型（CPU 运行，避免与 ChatGLM 争用 GPU 显存）"""
    global _reranker
    if _reranker is not None:
        return _reranker

    try:
        from FlagEmbedding import FlagReranker
        logger.info("正在加载BGE-Reranker...")

        local_model = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "models", "bge-reranker-base")
        )

        if not os.path.exists(local_model):
            # 禁止静默联网下载：模型缺失时跳过重排序（走检索顺序兜底）
            logger.warning(
                "本地Reranker模型不存在: %s，跳过重排序。请先运行 python download_models.py",
                local_model,
            )
            return None

        logger.info("使用本地Reranker: %s", local_model)
        _reranker = FlagReranker(
            local_model,
            use_fp16=False,
            devices="cpu",
            batch_size=16,
        )
        logger.info("BGE-Reranker加载成功！（CPU）")
        return _reranker
    except ImportError:
        logger.warning("FlagEmbedding未安装，跳过Reranker")
        return None
    except Exception as e:
        logger.error("加载Reranker失败: %s", e)
        return None


def _reset_reranker():
    """释放 Reranker 实例，避免异常后单例处于不可用状态"""
    global _reranker
    if _reranker is None:
        return
    try:
        if hasattr(_reranker, "model"):
            del _reranker.model
        if hasattr(_reranker, "tokenizer"):
            del _reranker.tokenizer
    except Exception:
        pass
    _reranker = None
    gc.collect()

# ============================================
# 动态加载向量库（按 index.faiss 的 mtime 缓存，未变化时复用内存实例）
# ============================================

_vector_store_cache = None
_vector_store_mtime = None
_vector_store_lock = threading.Lock()


def _index_mtime() -> float:
    index_path = os.path.join(VECTOR_DB_PATH, "index.faiss")
    if not os.path.exists(index_path):
        return -1
    return os.path.getmtime(index_path)


def load_vector_store():
    """加载FAISS向量库；index.faiss 未变化时直接复用缓存实例"""
    global _vector_store_cache, _vector_store_mtime
    mtime = _index_mtime()
    with _vector_store_lock:
        if _vector_store_cache is not None and mtime == _vector_store_mtime:
            return _vector_store_cache
        store = FAISS.load_local(
            VECTOR_DB_PATH,
            get_embedding_model(),
            allow_dangerous_deserialization=True
        )
        _vector_store_cache = store
        _vector_store_mtime = mtime
        return store


def reset_vector_store_cache():
    """重建向量库后调用，强制下次访问时重新加载"""
    global _vector_store_cache, _vector_store_mtime
    with _vector_store_lock:
        _vector_store_cache = None
        _vector_store_mtime = None

# ============================================
# BM25 检索通道（与向量检索加权融合）
# ============================================

_bm25_cache = None       # (BM25Okapi, docs)
_bm25_cache_mtime = None
_bm25_lock = threading.Lock()


def _tokenize(text: str) -> list:
    """中文分词：优先 jieba，缺失时退化为单字切分"""
    try:
        import jieba
        return [t for t in jieba.lcut(text) if t.strip()]
    except ImportError:
        return [c for c in text if not c.isspace()]


def _get_bm25(vector_store):
    """从 FAISS docstore 构建 BM25 索引，按 index.faiss mtime 缓存复用"""
    global _bm25_cache, _bm25_cache_mtime
    mtime = _index_mtime()
    with _bm25_lock:
        if _bm25_cache is not None and mtime == _bm25_cache_mtime:
            return _bm25_cache
        from rank_bm25 import BM25Okapi
        docs = list(vector_store.docstore._dict.values())
        corpus = [_tokenize(d.page_content) for d in docs]
        bm25 = BM25Okapi(corpus)
        _bm25_cache = (bm25, docs)
        _bm25_cache_mtime = mtime
        logger.info("BM25 索引已构建（%d 个Chunk）", len(docs))
        return _bm25_cache


def reset_bm25_cache():
    """重建向量库后调用，强制下次检索时重建 BM25 索引"""
    global _bm25_cache, _bm25_cache_mtime
    with _bm25_lock:
        _bm25_cache = None
        _bm25_cache_mtime = None


def _doc_key(doc) -> str:
    return f"{doc.metadata.get('source', '')}#{doc.metadata.get('chunk_id', id(doc))}"


def _minmax_normalize(scores: list) -> list:
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def _fuse_rankings(vector_results: list, bm25_results: list, top_k: int) -> list:
    """
    按归一化分数加权融合两个通道的检索结果。
    vector_results / bm25_results: [(doc, normalized_score)]，未命中通道的得分按 0 计。
    """
    settings = get_settings()
    fused = {}
    for doc, score in vector_results:
        key = _doc_key(doc)
        fused.setdefault(key, {"doc": doc, "score": 0.0})
        fused[key]["score"] += settings.RETRIEVAL_VECTOR_WEIGHT * score
    for doc, score in bm25_results:
        key = _doc_key(doc)
        fused.setdefault(key, {"doc": doc, "score": 0.0})
        fused[key]["score"] += settings.RETRIEVAL_BM25_WEIGHT * score

    ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in ranked[:top_k]]


# ============================================
# 文档检索（向量 + BM25 混合，支持动态TopK）
# ============================================

def retrieve(query: str, top_k: int = None) -> list:
    """
    混合检索：向量相似度 + BM25，归一化后按权重融合排序。

    Args:
        query: 用户查询
        top_k: 返回数量（可选，默认从配置文件读取）

    Returns:
        list: 检索到的文档列表
    """
    settings = get_settings()
    vector_store = load_vector_store()

    docs_with_scores = vector_store.similarity_search_with_score(
        query,
        k=settings.RETRIEVAL_CANDIDATE_K
    )

    if top_k is None:
        config_path = os.path.join(BASE_DIR, "vector_db", "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = {"top_k": 5}
        top_k = config.get("top_k", 5)

    # BGE 归一化向量下 L2 距离越小越相似；0.6 过严会漏掉相关文档
    max_distance = settings.MAX_L2_DISTANCE
    filtered = [(doc, score) for doc, score in docs_with_scores if score < max_distance]
    if not filtered:
        filtered = docs_with_scores[:top_k]

    # 向量通道：L2 距离转相似度并归一化
    vec_docs = [doc for doc, _ in filtered]
    vec_sims = _minmax_normalize([max_distance - score for _, score in filtered])
    vector_results = list(zip(vec_docs, vec_sims))

    # BM25 通道：失败时退化为纯向量检索
    bm25_results = []
    try:
        bm25, corpus_docs = _get_bm25(vector_store)
        bm25_scores = bm25.get_scores(_tokenize(query))
        top_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:settings.BM25_TOP_N]
        top_idx = [i for i in top_idx if bm25_scores[i] > 0]
        if top_idx:
            norm = _minmax_normalize([bm25_scores[i] for i in top_idx])
            bm25_results = [(corpus_docs[i], s) for i, s in zip(top_idx, norm)]
    except ImportError:
        logger.warning("rank_bm25 未安装，退化为纯向量检索")
    except Exception as e:
        logger.error("BM25 检索失败，退化为纯向量检索: %s", e)

    if not bm25_results:
        return vec_docs[:top_k]

    fused_docs = _fuse_rankings(vector_results, bm25_results, top_k=len(vector_results) + len(bm25_results))

    # 编号精确匹配优先：问题含实验编号时，含相同编号的 Chunk 提到前面
    # （解决向量分数接近时 BM25 0.3 权重无法把精确命中顶到 top1 的问题）
    query_ids = extract_experiment_ids(query)
    if query_ids:
        fused_docs.sort(
            key=lambda d: 0 if (query_ids & extract_experiment_ids(d.page_content)) else 1
        )

    docs = fused_docs[:top_k]
    logger.info(
        "混合检索：向量 %d 条 + BM25 %d 条 -> 融合后 %d 条",
        len(vector_results), len(bm25_results), len(docs),
    )
    return docs

# ============================================
# Reranker重排序
# ============================================

_CN_DIGIT = {
    "零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


def _parse_cn_number(token: str):
    token = token.strip()
    if not token:
        return None
    if token.isdigit():
        return int(token)
    if len(token) == 1 and token in _CN_DIGIT:
        return _CN_DIGIT[token]
    if token.startswith("十"):
        rest = token[1:]
        return 10 if not rest else 10 + _CN_DIGIT.get(rest, 0)
    if "十" in token:
        left, _, right = token.partition("十")
        tens = _CN_DIGIT.get(left, 1) if left else 1
        ones = _CN_DIGIT.get(right, 0) if right else 0
        return tens * 10 + ones
    return None


def extract_experiment_ids(text: str) -> set:
    """从文本中提取实验编号，如 实验四 / 实验5"""
    ids = set()
    for match in re.finditer(r"实验\s*([0-9一二三四五六七八九十]+)", text):
        num = _parse_cn_number(match.group(1))
        if num is not None:
            ids.add(num)
    return ids


def experiment_query_mismatch(query: str, docs: list) -> bool:
    """
    问题包含具体实验编号，但检索片段中没有相同编号 → 视为知识库无答案。
    例如问「实验四」，检索到「实验二/实验五」时应拒绝引用。
    """
    query_ids = extract_experiment_ids(query)
    if not query_ids:
        return False

    for doc in docs:
        blob = f"{doc.metadata.get('source', '')}\n{doc.page_content}"
        if query_ids & extract_experiment_ids(blob):
            return False
    return True


KB_MISS_ANSWER = "抱歉，知识库中未找到与您问题相关的资料，无法根据知识库内容回答该问题。"

CHAT_KEYWORDS = {
    "你好", "您好", "谢谢", "再见", "你是谁", "介绍一下你",
}

GENERAL_QUERY_PATTERNS = [
    r"(?i)python",
    r"冒泡排序|快速排序|二分查找|排序算法|递归|动态规划",
    r"写.*(?:代码|程序|脚本)|完成.*(?:代码|程序)|实现.*(?:代码|程序|算法)",
    r"编程题|算法题|代码题",
    r"今天天气|讲个笑话|帮我翻译",
]


def is_chitchat_query(query: str) -> bool:
    return query.strip() in CHAT_KEYWORDS


def is_obvious_general_query(query: str) -> bool:
    """明显与本地知识库无关的通用问题（编程、闲聊等），跳过检索"""
    q = query.strip()
    if is_chitchat_query(q):
        return True
    if extract_experiment_ids(q):
        return False
    return any(re.search(p, q) for p in GENERAL_QUERY_PATTERNS)


# ============================================
# 多轮对话：查询改写（指代/省略补全）
# ============================================

# 含指代或省略特征的问题才需要改写，避免对所有问题多调一次 LLM
REWRITE_HINT_PATTERNS = [
    r"(它|他|她|这个|那个|该|此|上述|前面|前者|后者)",
    r"(步骤|方法|流程|原因|结果|结论|时间|地点|目的|材料|原理|内容)呢[？?]?$",
    r"^(还有|然后|那么|那|继续|再)",
]


def needs_rewrite(query: str) -> bool:
    """粗判问题是否含指代/省略，需要结合历史改写"""
    q = query.strip()
    if any(re.search(p, q) for p in REWRITE_HINT_PATTERNS):
        return True
    # 极短追问（如「步骤呢」「为什么」）大概率依赖上文
    if len(q) <= 6:
        return True
    return False


def rewrite_query(query: str, history: list) -> str:
    """
    结合最近对话历史，把含指代/省略的问题改写为可独立检索的完整问题。
    改写失败或结果异常时回退为原问题。
    """
    if not history:
        return query

    history_text = "\n".join(
        f"用户：{h.get('question', '')}\n助手：{h.get('answer', '')[:200]}"
        for h in history[-3:]
    )
    prompt = f"""以下是一段对话历史和用户的最新问题。如果最新问题包含指代（如"它""这个"）或省略了主语，请结合历史将其改写为语义完整、可独立理解的问题；如果问题本身已经完整，请原样返回该问题。只输出改写后的问题本身，不要输出任何解释或回答。

对话历史：
{history_text}

最新问题：{query}
改写后的问题："""

    try:
        rewritten = chat_model(prompt).strip()
        # 去掉可能的引号包裹
        rewritten = rewritten.strip("「」\"'")
        if not rewritten or len(rewritten) > max(len(query) * 4, 200):
            logger.warning("查询改写结果异常，回退原问题: %s", rewritten[:80])
            return query
        if rewritten != query:
            logger.info("【查询改写】「%s」->「%s」", query, rewritten)
        return rewritten
    except Exception as e:
        logger.error("查询改写失败，使用原问题: %s", e)
        return query


def _kb_miss_response() -> dict:
    return {
        "answer": KB_MISS_ANSWER,
        "sources": [],
        "context_length": 0,
        "retrieved_docs": 0,
    }


def _general_llm_response(query: str, history=None) -> dict:
    logger.info("【通用模式】问题与知识库无关，直接调用 LLM（不引用来源）")
    response = _call_llm_direct(query, history=history)
    return {
        "answer": response,
        "sources": [],
        "context_length": 0,
        "retrieved_docs": 0,
    }


def rerank(query: str, docs: list, top_n: int = 3) -> tuple:
    """
    使用BGE-Reranker对检索结果重排序，并过滤低相关度文档

    Args:
        query: 用户查询
        docs: 检索到的文档列表
        top_n: 返回前N个

    Returns:
        tuple: (重排序后的文档列表, 最佳 rerank 分数或 None)
    """
    settings = get_settings()

    if not docs:
        return [], None

    valid_docs = [
        doc for doc in docs
        if getattr(doc, "page_content", None) and str(doc.page_content).strip()
    ]
    if not valid_docs:
        logger.warning("Reranker 跳过：检索结果均为空文本")
        return [], None

    reranker = load_reranker()
    if not reranker:
        return valid_docs[:top_n], None

    pairs = [(query, str(doc.page_content).strip()) for doc in valid_docs]
    try:
        scores = reranker.compute_score(pairs, batch_size=min(16, len(pairs)))
    except Exception as e:
        logger.error("Reranker 计算失败，回退为检索顺序: %s", e)
        _reset_reranker()
        return valid_docs[:top_n], None

    if not isinstance(scores, list):
        scores = [scores]

    if len(scores) != len(valid_docs):
        logger.warning("Reranker 分数数量异常，回退为检索顺序")
        return valid_docs[:top_n], None

    scored_docs = sorted(zip(valid_docs, scores), key=lambda x: x[1], reverse=True)
    best_score = scored_docs[0][1]
    logger.info("Reranker 最佳分数: %.3f (阈值: %.2f)", best_score, settings.RERANK_MIN_SCORE)

    if best_score < settings.RERANK_MIN_SCORE:
        logger.info("【相关性过滤】判定与知识库无关，不引用任何文档")
        return [], best_score

    min_score = max(best_score - 2.0, settings.RERANK_MIN_SCORE)
    filtered = [doc for doc, score in scored_docs if score >= min_score]
    filtered = filtered[:top_n]

    if filtered and experiment_query_mismatch(query, filtered):
        logger.info("【实验编号不匹配】问题与检索内容不一致，判定为知识库无答案")
        return [], best_score

    return filtered, best_score


def _call_llm_direct(query: str, history=None) -> str:
    """不基于知识库，直接调用 LLM"""
    _reset_reranker()
    try:
        return chat_model(query, history=history)
    except Exception as e:
        logger.exception("LLM 调用失败: %s", e)
        return (
            "大语言模型暂时不可用（可能因显存不足或模型未加载）。"
            "请确认后端终端仍在运行，并重启 uvicorn 后重试。"
        )


def _stream_llm_direct(query: str, history=None):
    """流式调用 LLM；流式失败时回退为一次性非流式输出"""
    _reset_reranker()
    try:
        yield from stream_chat_model(query, history=history)
    except Exception as e:
        logger.error("LLM 流式调用失败，回退非流式: %s", e)
        yield _call_llm_direct(query, history=history)

# ============================================
# 检索管线（ask / ask_stream 共用）
# ============================================

def _prepare_context(query: str, history=None, use_reranker: bool = True) -> dict:
    """
    执行 闲聊/通用判断 -> 查询改写 -> 检索 -> 重排序 -> 构建Prompt。

    Returns:
        dict: {
            mode: 'rag' | 'general' | 'kb_miss',
            prompt: RAG prompt（mode='rag' 时）,
            sources, context_length, retrieved_docs,
            effective_query: 改写后的检索用问题,
        }
    """
    settings = get_settings()
    logger.info("===== 问题: %s =====", query)

    if is_chitchat_query(query):
        logger.info("【闲聊模式】直接调用LLM")
        return {"mode": "general", "sources": [], "context_length": 0,
                "retrieved_docs": 0, "effective_query": query}

    if is_obvious_general_query(query):
        logger.info("【通用模式】识别为编程/通用问题，跳过知识库检索")
        return {"mode": "general", "sources": [], "context_length": 0,
                "retrieved_docs": 0, "effective_query": query}

    # 0. 多轮对话：含指代/省略时先改写为独立问题再检索
    effective_query = query
    if history and needs_rewrite(query):
        logger.info("【步骤0】检测到指代/省略，结合历史改写问题...")
        effective_query = rewrite_query(query, history)

    # 1. 检索文档
    logger.info("【步骤1】检索文档...")
    raw_docs = retrieve(effective_query)
    logger.info("检索到 %d 个文档", len(raw_docs))

    # 2. Reranker重排序
    rerank_best_score = None
    if use_reranker:
        logger.info("【步骤2】Reranker重排序...")
        docs, rerank_best_score = rerank(effective_query, raw_docs)
        logger.info("重排序后保留 %d 个文档", len(docs))
    else:
        docs = raw_docs

    # 3. 无相关文档：区分「知识库缺答案」与「通用问题」
    if not docs:
        if experiment_query_mismatch(effective_query, raw_docs):
            logger.info("【知识库无匹配】实验编号不匹配，返回标准拒答")
            return {"mode": "kb_miss", "sources": [], "context_length": 0,
                    "retrieved_docs": 0, "effective_query": effective_query}
        if rerank_best_score is not None and rerank_best_score < settings.GENERAL_LLM_MAX_SCORE:
            return {"mode": "general", "sources": [], "context_length": 0,
                    "retrieved_docs": 0, "effective_query": effective_query}
        logger.info("【知识库无匹配】与知识库主题相近但无答案，返回标准拒答")
        return {"mode": "kb_miss", "sources": [], "context_length": 0,
                "retrieved_docs": 0, "effective_query": effective_query}

    # 4. 输出检索结果（调试）
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "未知来源")
        chunk_id = doc.metadata.get("chunk_id", i + 1)
        page = doc.metadata.get("page", "未知")
        logger.debug(
            "【文档 %d】%s (P%s Chunk%s) 内容预览：%s...",
            i + 1, source, page, chunk_id, doc.page_content[:150],
        )

    # 5. 构建Prompt（使用统一的PromptBuilder）
    logger.info("【步骤4】构建Prompt...")
    prompt_result = prompt_builder.build_rag_prompt(effective_query, docs)
    logger.info("上下文长度: %d 字符", prompt_result["context_length"])

    return {
        "mode": "rag",
        "prompt": prompt_result["prompt"],
        "sources": prompt_result["sources"],
        "context_length": prompt_result["context_length"],
        "retrieved_docs": len(docs),
        "effective_query": effective_query,
    }


# ============================================
# 主问答函数
# ============================================

def ask(query: str, use_reranker: bool = True, history=None) -> dict:
    """
    主问答函数

    Args:
        query: 用户问题
        use_reranker: 是否使用Reranker
        history: 多轮对话历史（[{question, answer}]，最近几轮）

    Returns:
        dict: {answer, sources, context_length, retrieved_docs}
    """
    ctx = _prepare_context(query, history=history, use_reranker=use_reranker)

    if ctx["mode"] == "kb_miss":
        return _kb_miss_response()

    if ctx["mode"] == "general":
        return _general_llm_response(query, history=history)

    # 5. 调用LLM（释放 Reranker 占用的显存）
    logger.info("【步骤5】调用LLM...")
    response = _call_llm_direct(ctx["prompt"], history=history)

    logger.info("【步骤6】回答完成！")

    return {
        "answer": response,
        "sources": ctx["sources"],
        "context_length": ctx["context_length"],
        "retrieved_docs": ctx["retrieved_docs"]
    }


def ask_stream(query: str, history=None, use_reranker: bool = True):
    """
    流式问答生成器：检索/重排序完成后逐 token 输出。

    Yields:
        dict: {"type": "token", "text": 增量文本}
              或结束事件 {"type": "end", "sources", "context_length", "retrieved_docs"}
    """
    ctx = _prepare_context(query, history=history, use_reranker=use_reranker)

    if ctx["mode"] == "kb_miss":
        yield {"type": "token", "text": KB_MISS_ANSWER}
        yield {"type": "end", "sources": [], "context_length": 0, "retrieved_docs": 0}
        return

    if ctx["mode"] == "general":
        for delta in _stream_llm_direct(query, history=history):
            yield {"type": "token", "text": delta}
        yield {"type": "end", "sources": [], "context_length": 0, "retrieved_docs": 0}
        return

    logger.info("【步骤5】流式调用LLM...")
    for delta in _stream_llm_direct(ctx["prompt"], history=history):
        yield {"type": "token", "text": delta}

    logger.info("【步骤6】回答完成！")
    yield {
        "type": "end",
        "sources": ctx["sources"],
        "context_length": ctx["context_length"],
        "retrieved_docs": ctx["retrieved_docs"],
    }


# ============================================
# 测试函数
# ============================================

def test_qa():
    """测试问答功能"""
    result = ask("什么是人工智能？")
    logger.info("最终回答：%s", result["answer"])


if __name__ == "__main__":
    from core.logging_config import setup_logging
    setup_logging()
    test_qa()
