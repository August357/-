"""结构化文档解析与原子切分测试（不依赖模型文件）"""

import os

import pytest

from core.doc_parser import parse_txt, parse_docx, parse_pdf, _markdown_table
from core.vector_store import split_docs_smart, create_splitter
from langchain_core.documents import Document


TXT_SAMPLE = """实验一：示例实验
这是普通说明文字。

```python
def hello():
    print("围栏代码")
```

| 名称 | 数量 | 单位 |
| --- | --- | --- |
| 烧杯 | 2 | 个 |
| 玻璃棒 | 1 | 根 |

表格后的说明。

    def legacy():
        x = 1
        y = 2
        return x + y

结尾段落。
"""


class TestParseTxt:
    def test_blocks(self, tmp_path):
        p = tmp_path / "sample.txt"
        p.write_text(TXT_SAMPLE, encoding="utf-8")
        blocks = parse_txt(str(p))

        types = [b["type"] for b in blocks]
        assert "table" in types
        assert types.count("code") == 2  # 围栏代码 + 缩进代码

        table = next(b for b in blocks if b["type"] == "table")
        assert "| 名称 | 数量 | 单位 |" in table["content"]
        assert "烧杯" in table["content"]

        fenced = next(b for b in blocks if b["type"] == "code" and "```python" in b["content"])
        assert 'print("围栏代码")' in fenced["content"]

        indented = next(b for b in blocks if b["type"] == "code" and "def legacy" in b["content"])
        assert indented["content"].startswith("```")

        # 普通文字仍在，且表格/代码没有混入 text 块
        text_all = "\n".join(b["content"] for b in blocks if b["type"] == "text")
        assert "普通说明文字" in text_all
        assert "烧杯" not in text_all

    def test_plain_text_only(self, tmp_path):
        p = tmp_path / "plain.txt"
        p.write_text("第一段。\n\n第二段。\n", encoding="utf-8")
        blocks = parse_txt(str(p))
        assert [b["type"] for b in blocks] == ["text", "text"]


class TestParseDocx:
    def test_table_and_code(self, tmp_path):
        docx = pytest.importorskip("docx")

        document = docx.Document()
        document.add_paragraph("实验一：DOCX 结构测试")
        document.add_paragraph("这是说明段落。")

        table = document.add_table(rows=3, cols=2)
        table.cell(0, 0).text = "材料"
        table.cell(0, 1).text = "用量"
        table.cell(1, 0).text = "蔗糖溶液"
        table.cell(1, 1).text = "30ml"
        table.cell(2, 0).text = "蒸馏水"
        table.cell(2, 1).text = "50ml"

        document.add_paragraph("代码如下：")
        code_para = document.add_paragraph()
        run = code_para.add_run("def add(a, b):")
        run.font.name = "Consolas"
        code_para2 = document.add_paragraph()
        run2 = code_para2.add_run("    return a + b")
        run2.font.name = "Consolas"

        p = tmp_path / "sample.docx"
        document.save(str(p))

        blocks = parse_docx(str(p))
        types = [b["type"] for b in blocks]
        assert "table" in types
        assert "code" in types

        table_block = next(b for b in blocks if b["type"] == "table")
        assert "| 材料 | 用量 |" in table_block["content"]
        assert "蔗糖溶液" in table_block["content"]

        code_block = next(b for b in blocks if b["type"] == "code")
        assert "def add(a, b):" in code_block["content"]
        assert "return a + b" in code_block["content"]
        assert code_block["content"].startswith("```")

        text_all = "\n".join(b["content"] for b in blocks if b["type"] == "text")
        assert "说明段落" in text_all


class TestParsePdf:
    def test_table(self, tmp_path):
        pytest.importorskip("pdfplumber")
        reportlab = pytest.importorskip("reportlab")

        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
        from reportlab.lib import colors

        p = str(tmp_path / "sample.pdf")
        doc = SimpleDocTemplate(p, pagesize=A4)
        data = [["Item", "Qty"], ["Beaker", "2"], ["Rod", "1"]]
        tbl = Table(data)
        tbl.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]))
        doc.build([Paragraph("Experiment One Report"), tbl, Paragraph("Tail text after table.")])

        blocks = parse_pdf(p)
        types = [b["type"] for b in blocks]
        assert "table" in types

        table_block = next(b for b in blocks if b["type"] == "table")
        assert "Beaker" in table_block["content"]
        assert "| Item | Qty |" in table_block["content"]

        text_all = "\n".join(b["content"] for b in blocks if b["type"] == "text")
        assert "Experiment One Report" in text_all
        assert "Tail text after table." in text_all
        # 表格内容不应在 text 块中重复出现
        assert "Beaker" not in text_all


class TestSmartSplit:
    def test_table_code_atomic(self):
        splitter = create_splitter(chunk_size=100, chunk_overlap=20)
        table_doc = Document(
            page_content="| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |",
            metadata={"block_type": "table", "source": "t"},
        )
        code_doc = Document(
            page_content="```\ndef f():\n    pass\n```",
            metadata={"block_type": "code", "source": "t"},
        )
        text_doc = Document(
            page_content="这是一段很长的说明文字。" * 30,
            metadata={"block_type": "text", "source": "t"},
        )
        out = split_docs_smart([table_doc, code_doc, text_doc], splitter, chunk_size=100)

        tables = [d for d in out if d.metadata.get("block_type") == "table"]
        codes = [d for d in out if d.metadata.get("block_type") == "code"]
        assert len(tables) == 1 and tables[0].page_content == table_doc.page_content
        assert len(codes) == 1 and codes[0].page_content == code_doc.page_content
        # 长文本被正常切小
        texts = [d for d in out if d.metadata.get("block_type") == "text"]
        assert len(texts) > 1

    def test_oversized_table_fallback(self):
        splitter = create_splitter(chunk_size=50, chunk_overlap=10)
        big_table = Document(
            page_content="| A |\n| --- |\n" + "| x |\n" * 100,
            metadata={"block_type": "table", "source": "t"},
        )
        out = split_docs_smart([big_table], splitter, chunk_size=50)
        assert len(out) > 1  # 超过 1.5×chunk_size 的表格仍兜底切分


class TestMarkdownTable:
    def test_basic(self):
        md = _markdown_table([["a", "b"], ["1", "2"]])
        assert md.splitlines()[0] == "| a | b |"
        assert "| 1 | 2 |" in md

    def test_none_cells(self):
        md = _markdown_table([["a", None], ["1", "2"]])
        assert "| a |  |" in md
