"""
结构化文档解析器

按格式识别文档中的特殊结构（表格、代码块），输出带类型的内容块：
    Block = {"type": "text" | "table" | "code", "content": str, "page": int}

- TXT：识别 ``` 围栏代码、4空格/Tab 缩进代码、Markdown 管道表格
- DOCX：python-docx 按正文顺序遍历段落与表格；表格转 Markdown；
        等宽字体（Consolas/Courier 等）或代码样式的连续段落识别为代码块
- PDF：pdfplumber 抽取表格转 Markdown；表外文本按版面（layout）抽取保留缩进

下游切分（vector_store.split_docs_smart）会把 table/code 块作为原子单元保留。
"""

import logging
import os
import re

logger = logging.getLogger(__name__)

# 等宽/代码字体（DOCX run 字体名命中即视为代码）
MONOSPACE_FONTS = {
    "consolas", "courier", "courier new", "menlo", "monaco",
    "dejavu sans mono", "source code pro", "fira code",
    "jetbrains mono", "cascadia code", "lucida console",
    "simsun-extb", "等线", "dengxian mono",
}

# 代码段落样式名关键词（DOCX style）
CODE_STYLE_KEYWORDS = ("code", "代码", "macro", "html code", "htmlcode")


def _markdown_table(rows: list) -> str:
    """二维数组 -> Markdown 表格文本；第一行作表头"""
    rows = [[("" if c is None else str(c)).replace("\n", " ").strip() for c in row] for row in rows]
    rows = [row for row in rows if any(cell for cell in row)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]

    def fmt(row):
        return "| " + " | ".join(row) + " |"

    lines = [fmt(rows[0]), "|" + " --- |" * width]
    lines += [fmt(r) for r in rows[1:]]
    return "\n".join(lines)


# ============================================================
# TXT
# ============================================================

_FENCE_RE = re.compile(r"^```")
_MD_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")
_INDENT_LINE = re.compile(r"^(?: {4}|\t)")


def parse_txt(path: str) -> list:
    """TXT：识别围栏代码、Markdown 表格、缩进代码块，其余按空行分段"""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    blocks = []
    text_buf = []
    i = 0
    n = len(lines)

    def flush_text():
        nonlocal text_buf
        # 按空行拆为若干 text 块
        paragraph = []
        for line in text_buf:
            if line.strip():
                paragraph.append(line)
            elif paragraph:
                blocks.append({"type": "text", "content": "\n".join(paragraph), "page": 1})
                paragraph = []
        if paragraph:
            blocks.append({"type": "text", "content": "\n".join(paragraph), "page": 1})
        text_buf = []

    while i < n:
        line = lines[i]

        # 围栏代码块 ``` ... ```
        if _FENCE_RE.match(line.strip()):
            flush_text()
            code_lines = [line]
            i += 1
            while i < n:
                code_lines.append(lines[i])
                if _FENCE_RE.match(lines[i].strip()) and len(code_lines) > 1:
                    i += 1
                    break
                i += 1
            blocks.append({"type": "code", "content": "\n".join(code_lines), "page": 1})
            continue

        # Markdown 管道表格（连续 ≥2 行 |...|）
        if _MD_TABLE_LINE.match(line):
            j = i
            table_lines = []
            while j < n and _MD_TABLE_LINE.match(lines[j]):
                table_lines.append(lines[j].strip())
                j += 1
            if len(table_lines) >= 2:
                flush_text()
                rows = [[c.strip() for c in tl.strip("|").split("|")] for tl in table_lines]
                # 去掉 Markdown 分隔行 |---|---|
                rows = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c or "---") for c in r)]
                blocks.append({"type": "table", "content": _markdown_table(rows), "page": 1})
                i = j
                continue

        # 缩进代码块（连续 ≥3 行 4空格/Tab 缩进）
        if _INDENT_LINE.match(line):
            j = i
            code_lines = []
            while j < n and (_INDENT_LINE.match(lines[j]) or not lines[j].strip()):
                code_lines.append(lines[j])
                j += 1
            stripped = [l for l in code_lines if l.strip()]
            if len(stripped) >= 3:
                flush_text()
                blocks.append({
                    "type": "code",
                    "content": "```\n" + "\n".join(code_lines).strip("\n") + "\n```",
                    "page": 1,
                })
                i = j
                continue

        text_buf.append(line)
        i += 1

    flush_text()
    return blocks


# ============================================================
# DOCX
# ============================================================

def _iter_block_items(document):
    """按正文顺序产出 Paragraph / Table"""
    from docx.document import Document as _Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P

    if not isinstance(document, _Document):
        raise ValueError("仅支持 docx 主文档")
    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


def _is_code_paragraph(para) -> bool:
    """等宽字体或代码样式 -> 代码段落"""
    style_name = (para.style.name or "").lower() if para.style is not None else ""
    if any(k in style_name for k in CODE_STYLE_KEYWORDS):
        return True
    runs = [r for r in para.runs if r.text.strip()]
    if not runs:
        return False
    mono = 0
    for r in runs:
        font = (r.font.name or "").lower()
        east = ""
        try:
            east = (r._element.rPr.rFonts.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}eastAsia") or "").lower()
        except Exception:
            pass
        if any(m in font or m in east for m in MONOSPACE_FONTS):
            mono += 1
    return mono >= max(1, len(runs) * 0.8)


def parse_docx(path: str) -> list:
    """DOCX：按正文顺序解析，表格转 Markdown，代码段落合并为围栏代码块"""
    import docx

    document = docx.Document(path)
    blocks = []
    text_buf = []
    code_buf = []

    def flush_text():
        nonlocal text_buf
        content = "\n".join(text_buf).strip()
        if content:
            blocks.append({"type": "text", "content": content, "page": 1})
        text_buf = []

    def flush_code():
        nonlocal code_buf
        content = "\n".join(code_buf).strip("\n")
        if content.strip():
            blocks.append({"type": "code", "content": f"```\n{content}\n```", "page": 1})
        code_buf = []

    for item in _iter_block_items(document):
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        if isinstance(item, Table):
            flush_text()
            flush_code()
            rows = [[cell.text for cell in row.cells] for row in item.rows]
            md = _markdown_table(rows)
            if md:
                blocks.append({"type": "table", "content": md, "page": 1})
        elif isinstance(item, Paragraph):
            if not item.text.strip():
                flush_text()
                flush_code()
                continue
            if _is_code_paragraph(item):
                flush_text()
                code_buf.append(item.text)
            else:
                flush_code()
                text_buf.append(item.text)

    flush_text()
    flush_code()
    return blocks


# ============================================================
# PDF
# ============================================================

def _obj_in_bboxes(obj, bboxes, tolerance=2) -> bool:
    """判断 pdfplumber 页面对象是否落在任一表格 bbox 内"""
    x = (obj["x0"] + obj["x1"]) / 2
    top = (obj["top"] + obj["bottom"]) / 2
    for x0, t, x1, b in bboxes:
        if x0 - tolerance <= x <= x1 + tolerance and t - tolerance <= top <= b + tolerance:
            return True
    return False


def parse_pdf(path: str) -> list:
    """PDF：pdfplumber 抽表格转 Markdown；表外文本按版面抽取（保留缩进/对齐）"""
    import pdfplumber

    blocks = []
    with pdfplumber.open(path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            try:
                tables = page.find_tables()
            except Exception as e:
                logger.warning("PDF 表格检测失败（第%d页）: %s", page_no, e)
                tables = []

            bboxes = [t.bbox for t in tables]
            for t in tables:
                try:
                    md = _markdown_table(t.extract())
                except Exception as e:
                    logger.warning("PDF 表格抽取失败（第%d页）: %s", page_no, e)
                    continue
                if md:
                    blocks.append({"type": "table", "content": md, "page": page_no})

            try:
                if bboxes:
                    text_page = page.filter(lambda obj: not _obj_in_bboxes(obj, bboxes))
                else:
                    text_page = page
                text = text_page.extract_text(layout=True) or ""
            except Exception as e:
                logger.warning("PDF 文本抽取失败（第%d页）: %s", page_no, e)
                text = ""

            text = text.strip()
            if text:
                blocks.append({"type": "text", "content": text, "page": page_no})

    return blocks


# ============================================================
# 统一入口
# ============================================================

def parse_file(path: str) -> list:
    """按扩展名解析文档为结构化 Block 列表；不支持的格式返回空列表"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        return parse_txt(path)
    if ext == ".docx":
        return parse_docx(path)
    if ext == ".pdf":
        return parse_pdf(path)
    return []


def blocks_to_documents(blocks: list, source: str) -> list:
    """Block 列表 -> langchain Documents，metadata 带 block_type/page/source"""
    from langchain_core.documents import Document

    documents = []
    for block in blocks:
        content = (block.get("content") or "").strip()
        if not content:
            continue
        documents.append(Document(
            page_content=content,
            metadata={
                "source": source,
                "page": block.get("page", 1),
                "block_type": block.get("type", "text"),
            },
        ))
    return documents
