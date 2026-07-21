"""纯逻辑单元测试：不依赖任何模型文件"""

import pytest

from core.qa_chain import (
    _fuse_rankings,
    _minmax_normalize,
    extract_experiment_ids,
    is_chitchat_query,
    is_obvious_general_query,
    needs_rewrite,
)
from core.vector_store import split_by_headings
from core.llm import _normalize_history
from langchain_core.documents import Document


class TestExperimentIds:
    @pytest.mark.parametrize("text,expected", [
        ("实验四的步骤", {4}),
        ("实验5", {5}),
        ("实验十二", {12}),
        ("问的是实验一和实验三", {1, 3}),
        ("没有编号", set()),
    ])
    def test_extract(self, text, expected):
        assert extract_experiment_ids(text) == expected


class TestQueryRouting:
    def test_chitchat(self):
        assert is_chitchat_query("你好")
        assert not is_chitchat_query("实验一是什么")

    def test_general(self):
        assert is_obvious_general_query("用python写一个冒泡排序")
        assert not is_obvious_general_query("实验四的步骤是什么")

    def test_needs_rewrite(self):
        assert needs_rewrite("它的步骤是什么？")
        assert needs_rewrite("这个实验需要哪些材料")
        assert needs_rewrite("步骤呢")
        assert not needs_rewrite("实验四的完整流程是什么？")


class TestFusion:
    def test_minmax(self):
        assert _minmax_normalize([1.0, 3.0]) == [0.0, 1.0]
        assert _minmax_normalize([2.0, 2.0]) == [1.0, 1.0]

    def test_fusion_weights(self):
        d2 = Document(page_content="实验二", metadata={"source": "t", "chunk_id": 2})
        d4 = Document(page_content="实验四", metadata={"source": "t", "chunk_id": 4})
        # 向量分数接近时，BM25 精确命中可反超
        vec = [(d2, 0.55), (d4, 0.45)]
        bm25 = [(d4, 1.0)]
        out = _fuse_rankings(vec, bm25, 2)
        assert out[0] is d4

    def test_fusion_vector_dominant(self):
        d2 = Document(page_content="实验二", metadata={"source": "t", "chunk_id": 2})
        d4 = Document(page_content="实验四", metadata={"source": "t", "chunk_id": 4})
        vec = [(d2, 1.0), (d4, 0.1)]
        bm25 = [(d4, 1.0)]
        out = _fuse_rankings(vec, bm25, 2)
        assert out[0] is d2  # 0.7*1.0 > 0.7*0.1 + 0.3*1.0


class TestStructuredSplit:
    def test_heading_split(self):
        doc = Document(
            page_content="前言\n实验一：A\n内容一\n实验二：B\n内容二\n实验三：C\n内容三",
            metadata={"source": "t.txt"},
        )
        parts = split_by_headings([doc])
        assert len(parts) == 4
        assert parts[1].metadata["section"].startswith("实验一")
        assert "内容二" in parts[2].page_content

    def test_no_heading_passthrough(self):
        doc = Document(page_content="没有标题的普通文本", metadata={})
        assert split_by_headings([doc]) == [doc]


class TestHistoryNormalize:
    def test_qa_pairs(self):
        h = _normalize_history([{"question": "q1", "answer": "a1"}])
        assert h == [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ]

    def test_round_limit(self):
        history = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(10)]
        h = _normalize_history(history)
        assert len(h) == 6  # 默认只保留 3 轮
