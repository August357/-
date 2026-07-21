"""
RAG问答链核心模块
整合检索、重排序、Prompt构建和LLM调用
"""

import os
import gc
import json
import re
import threading

from langchain_community.vectorstores import FAISS

from core.llm import chat_model, stream_chat_model
from core.prompt import prompt_builder
from core.vector_store import get_embedding_model, VECTOR_DB_PATH

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
        print("正在加载BGE-Reranker...")
        
        import os as _os
        local_model = _os.path.join(_os.path.dirname(__file__), "..", "models", "bge-reranker-base")
        local_model = _os.path.abspath(local_model)
        
        if _os.path.exists(local_model):
            model_path = local_model
            print(f"使用本地Reranker: {model_path}")
        else:
            # 禁止静默联网下载：模型缺失时跳过重排序（走检索顺序兜底）
            print(f"本地Reranker模型不存在: {local_model}，跳过重排序。请先运行 python download_models.py")
            return None
        
        _reranker = FlagReranker(
            model_path,
            use_fp16=False,
            devices="cpu",
            batch_size=16,
        )
        print("BGE-Reranker加载成功！（CPU）")
        return _reranker
    except ImportError:
        print("FlagEmbedding未安装，跳过Reranker")
        return None
    except Exception as e:
        print(f"加载Reranker失败: {e}")
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
# 文档检索（使用MMR，支持动态TopK）
# ============================================

def retrieve(query: str, top_k: int = None) -> list:
    """
    使用相似度检索并过滤低匹配文档
    
    Args:
        query: 用户查询
        top_k: 返回数量（可选，默认从配置文件读取）
        
    Returns:
        list: 检索到的文档列表
    """
    vector_store = load_vector_store()

    docs_with_scores = vector_store.similarity_search_with_score(
        query,
        k=10
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
    max_distance = 1.2
    docs = [doc for doc, score in docs_with_scores if score < max_distance]

    if not docs:
        docs = [doc for doc, _ in docs_with_scores[:top_k]]

    return docs[:top_k]

# ============================================
# Reranker重排序
# ============================================

# BGE-Reranker 分数低于此阈值视为与知识库无关（cross-encoder logits）
# 实测：高度相关约 4~6，主题相近但编号不符约 1~2，完全无关常为负数
RERANK_MIN_SCORE = 2.5
# 低于此分数且非「实验编号不匹配」→ 走通用 LLM（闲聊/编程等）
GENERAL_LLM_MAX_SCORE = 1.0

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
            print(f"查询改写结果异常，回退原问题: {rewritten[:80]}")
            return query
        if rewritten != query:
            print(f"【查询改写】「{query}」->「{rewritten}」")
        return rewritten
    except Exception as e:
        print(f"查询改写失败，使用原问题: {e}")
        return query


def _kb_miss_response() -> dict:
    return {
        "answer": KB_MISS_ANSWER,
        "sources": [],
        "context_length": 0,
        "retrieved_docs": 0,
    }


def _general_llm_response(query: str, history=None) -> dict:
    print("\n【通用模式】问题与知识库无关，直接调用 LLM（不引用来源）")
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
    if not docs:
        return [], None

    valid_docs = [
        doc for doc in docs
        if getattr(doc, "page_content", None) and str(doc.page_content).strip()
    ]
    if not valid_docs:
        print("Reranker 跳过：检索结果均为空文本")
        return [], None

    reranker = load_reranker()
    if not reranker:
        return valid_docs[:top_n], None

    pairs = [(query, str(doc.page_content).strip()) for doc in valid_docs]
    try:
        scores = reranker.compute_score(pairs, batch_size=min(16, len(pairs)))
    except Exception as e:
        print(f"Reranker 计算失败，回退为检索顺序: {e}")
        _reset_reranker()
        return valid_docs[:top_n], None

    if not isinstance(scores, list):
        scores = [scores]

    if len(scores) != len(valid_docs):
        print("Reranker 分数数量异常，回退为检索顺序")
        return valid_docs[:top_n], None

    scored_docs = sorted(zip(valid_docs, scores), key=lambda x: x[1], reverse=True)
    best_score = scored_docs[0][1]
    print(f"Reranker 最佳分数: {best_score:.3f} (阈值: {RERANK_MIN_SCORE})")

    if best_score < RERANK_MIN_SCORE:
        print("【相关性过滤】判定与知识库无关，不引用任何文档")
        return [], best_score

    min_score = max(best_score - 2.0, RERANK_MIN_SCORE)
    filtered = [doc for doc, score in scored_docs if score >= min_score]
    filtered = filtered[:top_n]

    if filtered and experiment_query_mismatch(query, filtered):
        print("【实验编号不匹配】问题与检索内容不一致，判定为知识库无答案")
        return [], best_score

    return filtered, best_score


def _call_llm_direct(query: str, history=None) -> str:
    """不基于知识库，直接调用 LLM"""
    _reset_reranker()
    try:
        return chat_model(query, history=history)
    except Exception as e:
        print(f"LLM 调用失败: {e}")
        import traceback
        traceback.print_exc()
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
        print(f"LLM 流式调用失败，回退非流式: {e}")
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
    print(f"\n===== 问题: {query} =====")

    if is_chitchat_query(query):
        print("\n【闲聊模式】直接调用LLM")
        return {"mode": "general", "sources": [], "context_length": 0,
                "retrieved_docs": 0, "effective_query": query}

    if is_obvious_general_query(query):
        print("\n【通用模式】识别为编程/通用问题，跳过知识库检索")
        return {"mode": "general", "sources": [], "context_length": 0,
                "retrieved_docs": 0, "effective_query": query}

    # 0. 多轮对话：含指代/省略时先改写为独立问题再检索
    effective_query = query
    if history and needs_rewrite(query):
        print("\n【步骤0】检测到指代/省略，结合历史改写问题...")
        effective_query = rewrite_query(query, history)

    # 1. 检索文档
    print("\n【步骤1】检索文档...")
    raw_docs = retrieve(effective_query)
    print(f"检索到 {len(raw_docs)} 个文档")

    # 2. Reranker重排序
    rerank_best_score = None
    if use_reranker:
        print("\n【步骤2】Reranker重排序...")
        docs, rerank_best_score = rerank(effective_query, raw_docs)
        print(f"重排序后保留 {len(docs)} 个文档")
    else:
        docs = raw_docs

    # 3. 无相关文档：区分「知识库缺答案」与「通用问题」
    if not docs:
        if experiment_query_mismatch(effective_query, raw_docs):
            print("\n【知识库无匹配】实验编号不匹配，返回标准拒答")
            return {"mode": "kb_miss", "sources": [], "context_length": 0,
                    "retrieved_docs": 0, "effective_query": effective_query}
        if rerank_best_score is not None and rerank_best_score < GENERAL_LLM_MAX_SCORE:
            return {"mode": "general", "sources": [], "context_length": 0,
                    "retrieved_docs": 0, "effective_query": effective_query}
        print("\n【知识库无匹配】与知识库主题相近但无答案，返回标准拒答")
        return {"mode": "kb_miss", "sources": [], "context_length": 0,
                "retrieved_docs": 0, "effective_query": effective_query}

    # 4. 输出检索结果（调试）
    print("\n【步骤3】检索结果详情：")
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "未知来源")
        chunk_id = doc.metadata.get("chunk_id", i + 1)
        page = doc.metadata.get("page", "未知")
        print(f"\n【文档 {i+1}】{source} (P{page} Chunk{chunk_id})")
        print(f"内容预览：{doc.page_content[:150]}...")
        print("-" * 60)

    # 5. 构建Prompt（使用统一的PromptBuilder）
    print("\n【步骤4】构建Prompt...")
    prompt_result = prompt_builder.build_rag_prompt(effective_query, docs)
    print(f"上下文长度: {prompt_result['context_length']} 字符")

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
    print("\n【步骤5】调用LLM...")
    response = _call_llm_direct(ctx["prompt"], history=history)

    print("\n【步骤6】回答完成！")

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

    print("\n【步骤5】流式调用LLM...")
    for delta in _stream_llm_direct(ctx["prompt"], history=history):
        yield {"type": "token", "text": delta}

    print("\n【步骤6】回答完成！")
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
    print("\n" + "="*50)
    print("最终回答：")
    print(result["answer"])


if __name__ == "__main__":
    test_qa()
