"""
问答链路集成测试（真实 Embedding/Reranker + mock LLM）

覆盖：
1. 知识库内问题命中并返回 sources
2. 知识库外问题返回标准拒答
3. 实验编号不匹配时拒答
4. 闲聊问题不检索直接回答
"""

import pytest

from tests.conftest import (
    FAKE_ANSWER,
    requires_embedding,
    requires_reranker,
)

pytestmark = [requires_embedding, requires_reranker]


class TestKbHit:
    def test_kb_question_returns_sources(self, kb, fake_llm):
        """知识库内问题：命中检索、返回答案与来源引用"""
        from core.qa_chain import ask

        result = ask("实验一的实验材料是什么？")

        assert result["answer"] == FAKE_ANSWER
        assert result["retrieved_docs"] > 0
        assert len(result["sources"]) > 0
        assert any(s["source"] == "experiments.txt" for s in result["sources"])
        # 确认真的是 RAG 路径：LLM 收到的 prompt 应包含上下文
        assert "上下文" in fake_llm[-1]["query"]


class TestOutOfKb:
    def test_unrelated_topic_refused_or_general(self, kb, fake_llm):
        """知识库外问题：要么标准拒答，要么走通用模式，总之不得引用 sources"""
        from core.qa_chain import ask, KB_MISS_ANSWER

        result = ask("学校食堂今天中午吃什么？")

        assert result["sources"] == []
        # 两条合法路径：标准拒答，或通用 LLM 直接回答（不引用知识库）
        assert result["answer"] in {KB_MISS_ANSWER, FAKE_ANSWER}

    def test_near_domain_miss_refused(self, kb, fake_llm):
        """与知识库主题相近但无对应内容：返回标准拒答"""
        from core.qa_chain import ask, KB_MISS_ANSWER

        # 知识库没有编号为四的实验
        result = ask("实验四的实验结论是什么？")

        assert result["answer"] == KB_MISS_ANSWER
        assert result["sources"] == []


class TestExperimentIdMismatch:
    def test_mismatch_refused(self, kb, fake_llm):
        """问「实验九」但知识库只有实验一/二/三：拒答"""
        from core.qa_chain import ask, KB_MISS_ANSWER

        result = ask("实验九的操作步骤是什么？")

        assert result["answer"] == KB_MISS_ANSWER
        assert result["sources"] == []

    def test_mismatch_unit(self):
        """experiment_query_mismatch 单元逻辑"""
        from langchain_core.documents import Document
        from core.qa_chain import experiment_query_mismatch

        docs = [Document(page_content="实验二：测定种子发芽率...", metadata={"source": "t"})]
        assert experiment_query_mismatch("实验四的步骤", docs) is True
        assert experiment_query_mismatch("实验二的步骤", docs) is False
        assert experiment_query_mismatch("实验材料有哪些", docs) is False


class TestChitchat:
    def test_chitchat_skips_retrieval(self, kb, fake_llm, monkeypatch):
        """闲聊问题：不触发检索，直接调用 LLM 回答"""
        import core.qa_chain as qc

        def boom(*args, **kwargs):
            raise AssertionError("闲聊不应触发检索")

        monkeypatch.setattr(qc, "retrieve", boom)

        result = qc.ask("你好")

        assert result["answer"] == FAKE_ANSWER
        assert result["sources"] == []
        assert result["retrieved_docs"] == 0
