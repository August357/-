import gc
import os
import logging
import threading

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "true"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from core.config import get_settings

logger = logging.getLogger(__name__)

_IMPORT_ERROR = None
try:
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    import torch
except ImportError as e:
    # 依赖缺失时不终止进程（服务仍需提供 /api/health 等接口），
    # 在 load_model() 时统一抛出带提示的异常
    _IMPORT_ERROR = e
    AutoModel = AutoTokenizer = BitsAndBytesConfig = None
    torch = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "models", "chatglm3-6b")

_model = None
_tokenizer = None
_lock = threading.Lock()


def load_model():
    """懒加载模型，只在第一次调用时加载"""
    global _model, _tokenizer

    if _model is not None:
        return _model, _tokenizer

    with _lock:
        if _model is not None:
            return _model, _tokenizer

        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                f"LLM 依赖导入失败: {_IMPORT_ERROR}。"
                "请运行: pip install transformers torch bitsandbytes accelerate"
            ) from _IMPORT_ERROR

        logger.info("首次加载模型...")

        if not os.path.exists(LOCAL_MODEL_PATH):
            raise FileNotFoundError(f"模型目录不存在: {LOCAL_MODEL_PATH}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.set_per_process_memory_fraction(0.92)
            except Exception:
                pass

        logger.info("正在加载Tokenizer...")
        try:
            _tokenizer = AutoTokenizer.from_pretrained(
                LOCAL_MODEL_PATH,
                trust_remote_code=True
            )
        except Exception as e:
            raise RuntimeError(f"加载Tokenizer失败: {e}") from e

        logger.info("正在加载模型(4-bit)...")
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            _model = AutoModel.from_pretrained(
                LOCAL_MODEL_PATH,
                trust_remote_code=True,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True,
            ).eval()
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {e}") from e

        logger.info("模型加载完成")
        if hasattr(_model, "hf_device_map"):
            logger.info("device_map: %s", _model.hf_device_map)

    return _model, _tokenizer


def _normalize_history(history) -> list:
    """
    将 [{'question':..., 'answer':...}] 或 ChatGLM 格式的 history
    统一为 ChatGLM3 的 [{'role': 'user'/'assistant', 'content': ...}] 列表，
    只保留最近 HISTORY_ROUNDS 轮并截断过长内容，避免撑爆显存。
    """
    if not history:
        return []

    settings = get_settings()

    messages = []
    for item in history:
        if isinstance(item, dict) and "role" in item:
            messages.append({"role": item["role"], "content": str(item.get("content", ""))})
        elif isinstance(item, dict):
            messages.append({"role": "user", "content": str(item.get("question", ""))})
            messages.append({"role": "assistant", "content": str(item.get("answer", ""))})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            messages.append({"role": "user", "content": str(item[0])})
            messages.append({"role": "assistant", "content": str(item[1])})

    messages = messages[-settings.HISTORY_ROUNDS * 2:]
    return [
        {"role": m["role"], "content": m["content"][:settings.MAX_HISTORY_CHARS]}
        for m in messages
        if m["content"].strip()
    ]


def chat_model(query: str, max_new_tokens: int = None, history=None) -> str:
    """
    线程安全、限长推理，防止 max_length=8192 撑爆 8GB 显存导致进程被系统杀死。
    history 为多轮对话上下文（ChatGLM3 格式或 [{question, answer}]）。
    """
    settings = get_settings()
    if max_new_tokens is None:
        max_new_tokens = settings.MAX_NEW_TOKENS

    text = (query or "").strip()
    if not text:
        return "请输入有效问题。"

    if len(text) > settings.MAX_INPUT_CHARS:
        text = text[:settings.MAX_INPUT_CHARS] + "\n...(输入过长已截断)"

    chat_history = _normalize_history(history)

    # 注意：load_model 内部会获取 _lock，必须在持锁前调用，避免同线程重入死锁
    model, tokenizer = load_model()

    with _lock:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        inputs = tokenizer.build_chat_input(text, history=chat_history, role="user")
        input_len = int(inputs["input_ids"].shape[1])
        max_length = min(input_len + max_new_tokens, settings.MAX_TOTAL_TOKENS)
        if max_length <= input_len:
            max_length = min(input_len + 128, settings.MAX_TOTAL_TOKENS)

        logger.info(
            "LLM 推理: input_tokens≈%d, max_length=%d, history轮数=%d",
            input_len, max_length, len(chat_history) // 2,
        )

        response, _ = model.chat(
            tokenizer,
            text,
            history=chat_history,
            max_length=max_length,
            do_sample=False,
        )
        return response


def stream_chat_model(query: str, history=None, max_new_tokens: int = None):
    """
    流式推理：逐 token 增量 yield 新生成的文本。
    基于 ChatGLM3 的 model.stream_chat（其输出为累积文本，此处转为增量）。
    """
    settings = get_settings()
    if max_new_tokens is None:
        max_new_tokens = settings.MAX_NEW_TOKENS

    text = (query or "").strip()
    if not text:
        yield "请输入有效问题。"
        return

    if len(text) > settings.MAX_INPUT_CHARS:
        text = text[:settings.MAX_INPUT_CHARS] + "\n...(输入过长已截断)"

    chat_history = _normalize_history(history)

    # 注意：load_model 内部会获取 _lock，必须在持锁前调用，避免同线程重入死锁
    model, tokenizer = load_model()

    with _lock:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        inputs = tokenizer.build_chat_input(text, history=chat_history, role="user")
        input_len = int(inputs["input_ids"].shape[1])
        max_length = min(input_len + max_new_tokens, settings.MAX_TOTAL_TOKENS)
        if max_length <= input_len:
            max_length = min(input_len + 128, settings.MAX_TOTAL_TOKENS)

        logger.info(
            "LLM 流式推理: input_tokens≈%d, max_length=%d, history轮数=%d",
            input_len, max_length, len(chat_history) // 2,
        )

        past = ""
        for response, _ in model.stream_chat(
            tokenizer,
            text,
            history=chat_history,
            max_length=max_length,
            do_sample=False,
        ):
            delta = response[len(past):]
            past = response
            if delta:
                yield delta


def test_model():
    answer = chat_model("你好")
    logger.info("回答: %s", answer)


if __name__ == "__main__":
    from core.logging_config import setup_logging
    setup_logging()
    test_model()
