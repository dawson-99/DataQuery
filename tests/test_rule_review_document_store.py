"""
规则审查系统文档存储模块单元测试

覆盖 src/rule_review/document_store.py 的 PDF 解析、chunk 切分、
FAISS 索引、持久化与 OCR 降级逻辑。
"""

from __future__ import annotations

import io
from pathlib import Path

import fitz
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from src.rule_review.document_store import (
    Chunk,
    DocumentStore,
    OCRProcessor,
    PageContent,
    TextBlock,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class NullOCRProcessor(OCRProcessor):
    """始终不可用的 OCR，用于测试降级路径。"""

    @property
    def is_available(self) -> bool:
        return False

    def process_page(self, image, page_number: int = 0) -> PageContent:
        raise RuntimeError("不可用")


class MockOCRProcessor(OCRProcessor):
    """测试用 OCR，按页码返回 deterministic 文本。"""

    def __init__(self, pages: dict[int, str] | None = None) -> None:
        self.pages = pages or {}

    @property
    def is_available(self) -> bool:
        return True

    def process_page(self, image, page_number: int = 0) -> PageContent:
        text = self.pages.get(page_number, "")
        return PageContent(
            text_blocks=[
                TextBlock(text=text, bbox=(0, 0, 1, 1), confidence=0.99)
            ],
            table_blocks=[],
            avg_confidence=0.99,
        )


def _make_text_pdf(path: Path) -> None:
    """使用 pymupdf 生成包含标题、正文和表格的文本 PDF（内置 CJK 字体）。"""
    doc = fitz.open()
    page = doc.new_page()
    w, h = page.rect.width, page.rect.height

    # fitz 坐标原点在左上角，y 向下递增，因此标题应使用较小的 y 值
    page.insert_text((72, 72), "第一章 总则", fontname="china-ss", fontsize=16)
    page.insert_text((72, 100), "第1条 适用范围", fontname="china-ss", fontsize=14)
    page.insert_text(
        (72, 130),
        "本规则适用于省间电力现货交易，各市场主体应严格遵守。",
        fontname="china-ss",
        fontsize=12,
    )

    # 长文本，用于测试 chunk 切分（逐行绘制，确保可被提取）
    sentence = "市场主体应遵守交易规则。"
    y = 160
    for _ in range(200):
        page.insert_text((72, y), sentence, fontname="china-ss", fontsize=12)
        y += 16
        if y > h - 72:
            page = doc.new_page()
            y = 72

    # 第二页：表格
    page = doc.new_page()
    page.insert_text((72, 72), "第二章 价格规则", fontname="china-ss", fontsize=16)
    page.insert_text((72, 100), "第2条 价格上限", fontname="china-ss", fontsize=14)
    page.insert_text(
        (72, 130),
        "下表为各省日前现货出清电价上限：",
        fontname="china-ss",
        fontsize=12,
    )

    rows = [
        ["省份", "电价上限(元/MWh)"],
        ["冀北", "760"],
        ["山西", "780"],
        ["四川主网", "800"],
    ]
    y_top = 180
    row_height = 24
    col_widths = [120, 160]
    table_width = sum(col_widths)

    # 单元格文字
    for i, row in enumerate(rows):
        y = y_top + i * row_height
        for j, cell in enumerate(row):
            x = 72 + sum(col_widths[:j])
            page.insert_text(
                (x + 5, y + 16), cell, fontname="china-ss", fontsize=12
            )

    # 横线
    for i in range(len(rows) + 1):
        y = y_top + i * row_height
        page.draw_line((72, y), (72 + table_width, y))

    # 竖线
    for j in range(len(col_widths) + 1):
        x = 72 + sum(col_widths[:j])
        page.draw_line((x, y_top), (x, y_top + len(rows) * row_height))

    doc.save(str(path))
    doc.close()


def _make_scanned_pdf(path: Path) -> None:
    """生成一页只含图片、无文本层的扫描 PDF。"""
    img = Image.new("RGB", (600, 100), color="white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, 30), "第3条 扫描页示例", fill="black", font=font)

    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")

    doc = fitz.open()
    page = doc.new_page()
    page.insert_image(page.rect, stream=img_bytes.getvalue())
    doc.save(str(path))
    doc.close()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_embed():
    """固定 8 维归一化随机 embedding。"""
    dim = 8
    rng = np.random.default_rng(42)

    def _fn(texts: list[str]) -> np.ndarray:
        vecs = rng.normal(size=(len(texts), dim)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1.0, norms)

    return _fn


@pytest.fixture
def store(tmp_path: Path, mock_embed):
    """使用临时目录和 mock embedding 的 DocumentStore。"""
    return DocumentStore(
        documents_dir=tmp_path / "docs",
        index_dir=tmp_path / "index",
        embedding_fn=mock_embed,
        embedding_dim=8,
        chunk_size=600,
        chunk_overlap=150,
    )


@pytest.fixture
def text_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "rule.pdf"
    _make_text_pdf(path)
    return path


@pytest.fixture
def scanned_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "scanned.pdf"
    _make_scanned_pdf(path)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ingest_text_pdf_creates_chunks(store: DocumentStore, text_pdf: Path):
    resp = store.ingest(text_pdf, filename="rule.pdf")
    assert resp.file_name == "rule.pdf"
    assert resp.page_count >= 2
    assert resp.chunk_count > 0

    docs = store.list_documents()
    assert len(docs) == 1
    assert docs[0].chunk_count == resp.chunk_count

    chunks = list(store._chunks.values())
    assert all(c.embedding is not None for c in chunks)
    assert all(c.embedding.shape == (8,) for c in chunks)


def test_table_fused_with_heading(store: DocumentStore, text_pdf: Path):
    store.ingest(text_pdf)
    table_chunks = [c for c in store._chunks.values() if "| 省份 |" in c.text]
    assert len(table_chunks) >= 1

    table_chunk = table_chunks[0]
    assert "第2条 价格上限" in table_chunk.text
    assert "下表为各省日前现货出清电价上限" in table_chunk.text
    assert any(
        t.get("caption", "").startswith("下表")
        for t in table_chunk.tables
    )


def test_section_hierarchy_detected(store: DocumentStore, text_pdf: Path):
    store.ingest(text_pdf)
    chunks = list(store._chunks.values())
    hierarchy_chunks = [c for c in chunks if c.section_hierarchy]
    assert len(hierarchy_chunks) > 0

    # 找到同时包含章和条的 chunk
    full = [
        c
        for c in chunks
        if any("第一章" in s for s in c.section_hierarchy)
        and any("第1条" in s for s in c.section_hierarchy)
    ]
    assert len(full) > 0


def test_chunk_size_and_overlap(store: DocumentStore, text_pdf: Path):
    # 使用更小的 chunk_size 强制切分
    store.chunk_size = 120
    store.chunk_overlap = 30
    store.ingest(text_pdf)

    text_chunks = [c for c in store._chunks.values() if "市场主体" in c.text]
    assert len(text_chunks) > 1

    for c in text_chunks:
        # 允许章节标题带来少量超出
        assert len(c.text) <= store.chunk_size + 50


def test_scanned_pdf_with_mock_ocr(store: DocumentStore, scanned_pdf: Path):
    store.ocr_engine = MockOCRProcessor({1: "第3条 扫描页示例"})
    resp = store.ingest(scanned_pdf)
    assert resp.chunk_count >= 1

    chunks = list(store._chunks.values())
    assert all(c.is_scanned for c in chunks)
    assert any("第3条" in c.text for c in chunks)


def test_scanned_pdf_without_ocr_degrades(tmp_path: Path, scanned_pdf: Path, mock_embed):
    store = DocumentStore(
        documents_dir=tmp_path / "docs",
        index_dir=tmp_path / "index",
        embedding_fn=mock_embed,
        embedding_dim=8,
        ocr_engine=NullOCRProcessor(),
    )
    resp = store.ingest(scanned_pdf)
    assert resp.chunk_count == 0


def test_vector_search_returns_top_k(store: DocumentStore, text_pdf: Path):
    store.ingest(text_pdf)
    chunks = list(store._chunks.values())
    assert len(chunks) > 0

    target = chunks[0]
    results = store.vector_search(target.embedding, top_k=1)
    assert len(results) == 1
    assert results[0].chunk.chunk_id == target.chunk_id
    assert results[0].score > 0.99


def test_delete_document_removes_chunks_and_index(store: DocumentStore, text_pdf: Path):
    resp1 = store.ingest(text_pdf, filename="rule1.pdf")
    # 复制一份内容作为第二个文档
    resp2 = store.ingest(text_pdf, filename="rule2.pdf")

    assert len(store.list_documents()) == 2

    deleted = store.delete(resp1.doc_id)
    assert deleted is True

    assert len(store.list_documents()) == 1
    assert store.get_chunk(list(store._chunks.values())[0].chunk_id) is not None
    # 被删除文档的 chunk 不应再被检索到
    for chunk in store._chunks.values():
        assert chunk.doc_id != resp1.doc_id

    # 用剩余 chunk 的 embedding 搜索，只能返回剩余文档
    remaining_chunk = list(store._chunks.values())[0]
    results = store.vector_search(remaining_chunk.embedding, top_k=10)
    assert all(r.chunk.doc_id != resp1.doc_id for r in results)


def test_persistence_save_and_load(tmp_path: Path, text_pdf: Path, mock_embed):
    index_dir = tmp_path / "index"
    docs_dir = tmp_path / "docs"

    store1 = DocumentStore(
        documents_dir=docs_dir,
        index_dir=index_dir,
        embedding_fn=mock_embed,
        embedding_dim=8,
    )
    resp = store1.ingest(text_pdf)

    store2 = DocumentStore(
        documents_dir=docs_dir,
        index_dir=index_dir,
        embedding_fn=mock_embed,
        embedding_dim=8,
    )
    assert len(store2.list_documents()) == 1
    assert store2.list_documents()[0].doc_id == resp.doc_id
    assert len(store2._chunks) == resp.chunk_count

    # 验证检索能力在 reload 后仍可用
    chunk = list(store2._chunks.values())[0]
    results = store2.vector_search(chunk.embedding, top_k=1)
    assert len(results) == 1


def test_inject_embedding_no_model_download(store: DocumentStore):
    """使用 mock embedding 时不应加载 sentence-transformers。"""
    from src.rule_review.document_store import BGEM3Embedding

    assert not isinstance(store.embedding_fn, BGEM3Embedding)
    assert store.embedding_fn is not None
    assert store._embedding_dim == 8


def test_chunk_to_dict_roundtrip():
    chunk = Chunk(
        chunk_id="c1",
        doc_id="d1",
        text="hello",
        embedding=np.array([0.1, 0.2], dtype=np.float32),
    )
    data = chunk.to_dict()
    restored = Chunk.from_dict(data)
    assert restored.chunk_id == chunk.chunk_id
    assert restored.text == chunk.text
    np.testing.assert_array_equal(restored.embedding, chunk.embedding)


def test_rows_to_markdown():
    rows = [["省份", "上限"], ["冀北", "760"]]
    md = DocumentStore._rows_to_markdown(rows)
    assert "| 省份 | 上限 |" in md
    assert "| 冀北 | 760 |" in md
    assert "---" in md


def test_table_truncation(store: DocumentStore):
    rows = [
        ["省份", "上限"],
        ["冀北", "760"],
        ["山西", "780"],
        ["四川主网", "800"],
    ]
    truncated = store._truncate_table_chunk("第2条", "表：", rows)
    assert "第2条" in truncated
    assert "表：" in truncated
    assert "..." in truncated or len(truncated) <= store.chunk_size
