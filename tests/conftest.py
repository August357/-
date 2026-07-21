"""
pytest 共享夹具：用样例实验文档构建独立的临时向量库。

- Embedding / Reranker 使用真实本地模型（CPU），缺失时整个模块 skip
- LLM（chat_model）一律 mock，避免依赖 GPU / ChatGLM3
"""

import os
import sys

import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

os.environ.setdefault("JWT_SECRET", "pytest-secret")

from core.vector_store import EMBEDDING_MODEL_PATH  # noqa: E402

RERANKER_PATH = os.path.join(BASE_DIR, "models", "bge-reranker-base")

requires_embedding = pytest.mark.skipif(
    not os.path.exists(EMBEDDING_MODEL_PATH),
    reason=f"Embedding模型未下载: {EMBEDDING_MODEL_PATH}",
)
requires_reranker = pytest.mark.skipif(
    not os.path.exists(RERANKER_PATH),
    reason=f"Reranker模型未下载: {RERANKER_PATH}",
)

SAMPLE_DOC = """实验一：观察植物细胞的质壁分离
实验材料：紫色洋葱鳞片叶、蔗糖溶液、显微镜。
实验步骤：第一步制作临时装片，第二步滴加蔗糖溶液，第三步观察细胞变化。
实验结论：成熟植物细胞在高浓度溶液中会发生质壁分离。

实验二：测定种子发芽率
实验材料：绿豆种子、培养皿、滤纸。
实验步骤：第一步挑选100粒种子，第二步置于湿润滤纸上，第三步统计发芽数量。
实验结论：发芽率等于发芽种子数除以种子总数。

实验三：探究光照对植物生长的影响
实验材料：两盆长势相同的幼苗、遮光罩。
实验步骤：第一步将幼苗分为两组，第二步一组遮光一组照光，第三步测量株高。
实验结论：光照充足的幼苗生长更快。
"""

FAKE_ANSWER = "这是基于知识库的测试回答。"


@pytest.fixture(scope="session")
def kb(tmp_path_factory):
    """构建临时知识库并把 qa_chain/vector_store 指向它"""
    import core.vector_store as vs
    import core.qa_chain as qc

    tmp = tmp_path_factory.mktemp("kb")
    docs_dir = tmp / "docs"
    docs_dir.mkdir()
    (docs_dir / "experiments.txt").write_text(SAMPLE_DOC, encoding="utf-8")
    db_dir = tmp / "vector_db"

    # 备份并切换路径常量
    saved = {
        "vs.DOCS_DIR": vs.DOCS_DIR,
        "vs.VECTOR_DB_PATH": vs.VECTOR_DB_PATH,
        "vs.CHUNK_INFO_PATH": vs.CHUNK_INFO_PATH,
        "vs.CONFIG_PATH": vs.CONFIG_PATH,
        "qc.VECTOR_DB_PATH": qc.VECTOR_DB_PATH,
    }
    vs.DOCS_DIR = str(docs_dir)
    vs.VECTOR_DB_PATH = str(db_dir)
    vs.CHUNK_INFO_PATH = str(db_dir / "chunk_info.txt")
    vs.CONFIG_PATH = str(db_dir / "config.json")
    qc.VECTOR_DB_PATH = str(db_dir)

    ok = vs.build_vector_db(chunk_size=200, chunk_overlap=50, top_k=3)
    assert ok, "临时向量库构建失败"

    yield {"docs_dir": str(docs_dir), "db_dir": str(db_dir)}

    # 还原
    vs.DOCS_DIR = saved["vs.DOCS_DIR"]
    vs.VECTOR_DB_PATH = saved["vs.VECTOR_DB_PATH"]
    vs.CHUNK_INFO_PATH = saved["vs.CHUNK_INFO_PATH"]
    vs.CONFIG_PATH = saved["vs.CONFIG_PATH"]
    qc.VECTOR_DB_PATH = saved["qc.VECTOR_DB_PATH"]


@pytest.fixture()
def fake_llm(monkeypatch):
    """mock LLM，返回固定回答；同时屏蔽流式接口"""
    import core.qa_chain as qc

    calls = []

    def fake_chat(query, max_new_tokens=None, history=None):
        calls.append({"query": query, "history": history})
        return FAKE_ANSWER

    def fake_stream(query, history=None, max_new_tokens=None):
        yield FAKE_ANSWER

    monkeypatch.setattr(qc, "chat_model", fake_chat)
    monkeypatch.setattr(qc, "stream_chat_model", fake_stream)
    return calls
