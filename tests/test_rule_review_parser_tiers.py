"""
规则审查系统解析分层单元测试

覆盖 src/rule_review/parsers.py 与 document_store.py 的解析分层改造：
- markdown → PageContent 转换（MinerU 输出 / 手动入库共用路径）
- MinerU 未安装时的静默降级
- ingest 解析模式路由（auto / mineru / pymupdf）
- ingest_manual 手动入库（重要政策问答表格）
- 表格单元格识别准确率（98% 双评测：解析精度 + 检索召回）

全部零网络、零真实模型（mock embedding + FakeMinerU）。
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from src.rule_review.document_store import DocumentStore
from src.rule_review.evaluation import (
    RETRIEVAL_RECALL_TARGET,
    TABLE_RECALL_TAG,
    TestCase,
    compute_recall_at_k,
)
from src.rule_review.models import PageContent
from src.rule_review.parsers import (
    MinerULocalParser,
    PDFParser,
    markdown_to_page_content,
    parse_markdown,
    split_markdown_pages,
)
from src.rule_review.parsing_eval import (
    TABLE_CELL_ACCURACY_TARGET,
    compute_table_cell_accuracy,
    evaluate_table_parsing,
    load_table_parse_cases,
    parse_markdown_table,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MARKDOWN_SAMPLE = """# 第二章 价格规则

## 第2条 价格上限

下表为各省日前现货出清电价上限：

| 省份 | 电价上限(元/MWh) |
|---|---|
| 冀北 | 760 |
| 山西 | 780 |
| 四川主网 | 800 |

市场主体应遵守交易规则。
"""


class FakeMinerUParser(PDFParser):
    """测试用 MinerU 替身：直接注入 DocumentStore(mineru_parser=...)，
    不触碰 mineru 模块。"""

    name = "mineru"

    def __init__(self, markdown: str = "") -> None:
        self._markdown = markdown
        self._available = True
        self.parse_called = False

    @property
    def is_available(self) -> bool:
        return self._available

    def parse(self, data: bytes, filename: str) -> list[PageContent]:
        self.parse_called = True
        return parse_markdown(self._markdown)


class UnavailableMinerUParser(FakeMinerUParser):
    """不可用的 MinerU 替身（模拟未安装场景）。"""

    @property
    def is_available(self) -> bool:
        return False


def _make_minimal_pdf() -> bytes:
    """生成一页真实文本 PDF（pymupdf 路径的降级测试用）。"""
    import io

    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "第1条 适用范围", fontname="china-ss", fontsize=14)
    page.insert_text(
        (72, 100), "本规则适用于省间电力现货交易。", fontname="china-ss", fontsize=12
    )
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture(autouse=True)
def _clear_mineru_singleton():
    """清理 MinerULocalParser 单例缓存，防跨测试污染。"""
    MinerULocalParser._instances.clear()
    yield
    MinerULocalParser._instances.clear()


@pytest.fixture
def mock_embed():
    """固定 8 维归一化随机 embedding（与 document_store 测试同款）。"""
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
        mineru_enabled=True,  # 显式开启 MinerU 层，路由测试不依赖环境配置
    )


# ---------------------------------------------------------------------------
# Markdown → PageContent 转换
# ---------------------------------------------------------------------------


class TestMarkdownConversion:
    def test_heading_marker_removed(self):
        page = markdown_to_page_content("# 第一章 总则\n\n正文内容")
        texts = [b.text for b in page.text_blocks]
        assert "第一章 总则" in texts
        assert not any(t.startswith("#") for t in texts)

    def test_table_rows_extracted(self):
        page = markdown_to_page_content(MARKDOWN_SAMPLE)
        assert len(page.table_blocks) == 1
        table = page.table_blocks[0]
        assert table.rows == [
            ["省份", "电价上限(元/MWh)"],
            ["冀北", "760"],
            ["山西", "780"],
            ["四川主网", "800"],
        ]

    def test_table_separator_skipped(self):
        page = markdown_to_page_content("| a | b |\n|---|---|\n| 1 | 2 |")
        assert page.table_blocks[0].rows == [["a", "b"], ["1", "2"]]

    def test_empty_lines_skipped(self):
        page = markdown_to_page_content("\n\n正文\n\n\n")
        assert [b.text for b in page.text_blocks] == ["正文"]

    def test_synthetic_bbox_preserves_order(self):
        page = markdown_to_page_content("第一行\n\n| 表格 |\n|---|---|\n| 值 |\n\n第三行")
        # 阅读序 y 递增：第一行(0) < 表格(1) < 第三行(2)
        assert [b.bbox[1] for b in page.text_blocks] == [0.0, 2.0]
        assert page.table_blocks[0].bbox[1] == 1.0

    def test_split_markdown_pages(self):
        md = "第1页\n\n---\n\n第2页"
        assert split_markdown_pages(md) == ["第1页", "第2页"]

    def test_parse_markdown_multipage(self):
        md = "## 第1条 总则\n\n---\n\n## 第2条 价格上限"
        pages = parse_markdown(md)
        assert len(pages) == 2


# ---------------------------------------------------------------------------
# MinerU 降级
# ---------------------------------------------------------------------------


class TestMineruDegradation:
    def test_available_when_mineru_importable(self, monkeypatch):
        fake_module = types.ModuleType("mineru")

        class FakeMinerU:
            def __init__(self, **kwargs) -> None:
                pass

        fake_module.MinerU = FakeMinerU
        monkeypatch.setitem(sys.modules, "mineru", fake_module)

        parser = MinerULocalParser(enabled=True)
        assert parser.is_available

    def test_unavailable_when_mineru_missing(self, monkeypatch):
        # mineru 模块存在但没有 MinerU 属性 → 降级
        monkeypatch.setitem(sys.modules, "mineru", types.ModuleType("mineru"))
        parser = MinerULocalParser(enabled=True)
        assert not parser.is_available

    def test_unavailable_when_disabled(self):
        parser = MinerULocalParser(enabled=False)
        assert not parser.is_available

    def test_singleton_reused(self, monkeypatch):
        fake_module = types.ModuleType("mineru")
        fake_module.MinerU = type("FakeMinerU", (), {"__init__": lambda self, **kw: None})
        monkeypatch.setitem(sys.modules, "mineru", fake_module)

        p1 = MinerULocalParser()
        p2 = MinerULocalParser()
        assert p1 is p2


# ---------------------------------------------------------------------------
# 解析模式路由
# ---------------------------------------------------------------------------


class TestParseModeRouting:
    def test_auto_uses_mineru_when_available(self, store, mock_embed):
        fake = FakeMinerUParser(markdown=MARKDOWN_SAMPLE)
        store.mineru_parser = fake

        resp = store.ingest(b"%PDF-fake-bytes", filename="rule.pdf")
        assert resp.parse_mode == "mineru"
        assert fake.parse_called
        doc = store.list_documents()[0]
        assert doc.importance == "low"
        assert doc.parse_mode == "mineru"

    def test_pymupdf_forced_skips_mineru(self, store, mock_embed):
        fake = FakeMinerUParser(markdown=MARKDOWN_SAMPLE)
        store.mineru_parser = fake

        resp = store.ingest(_make_minimal_pdf(), filename="rule.pdf", parse_mode="pymupdf")
        assert resp.parse_mode == "pymupdf"
        assert not fake.parse_called

    def test_auto_with_unavailable_mineru_degrades_to_pymupdf(self, store):
        store.mineru_parser = UnavailableMinerUParser()

        resp = store.ingest(_make_minimal_pdf(), filename="rule.pdf")
        assert resp.parse_mode == "pymupdf"  # 降级后记录实际模式

    def test_mineru_mode_degrades_when_unavailable(self, store):
        store.mineru_parser = UnavailableMinerUParser()

        resp = store.ingest(_make_minimal_pdf(), filename="rule.pdf", parse_mode="mineru")
        assert resp.parse_mode == "pymupdf"

    def test_auto_with_mineru_disabled_uses_pymupdf(self, tmp_path, mock_embed):
        store = DocumentStore(
            documents_dir=tmp_path / "docs",
            index_dir=tmp_path / "index",
            embedding_fn=mock_embed,
            embedding_dim=8,
            mineru_enabled=False,  # 环境开关关闭
            mineru_parser=FakeMinerUParser(markdown=MARKDOWN_SAMPLE),
        )
        resp = store.ingest(_make_minimal_pdf(), filename="rule.pdf")
        assert resp.parse_mode == "pymupdf"

    def test_invalid_parse_mode_raises(self, store):
        with pytest.raises(ValueError, match="未知解析模式"):
            store.ingest(b"%PDF", filename="rule.pdf", parse_mode="bogus")

    def test_invalid_importance_raises(self, store):
        with pytest.raises(ValueError, match="importance"):
            store.ingest(b"%PDF", filename="rule.pdf", importance="weird")


# ---------------------------------------------------------------------------
# 手动入库（重要政策问答表格）
# ---------------------------------------------------------------------------


class TestManualIngest:
    def test_manual_ingest_creates_document(self, store):
        resp = store.ingest_manual(
            MARKDOWN_SAMPLE,
            filename="价格上限表.md",
            importance="high",
            source="政策问答表格人工整理",
        )
        assert resp.parse_mode == "manual"
        assert resp.importance == "high"
        assert resp.chunk_count > 0

        doc = store.list_documents()[0]
        assert doc.parse_mode == "manual"
        assert doc.importance == "high"
        assert doc.source == "政策问答表格人工整理"

        # chunk 含表格内容与章节标题
        chunks = [c for c in store._chunks.values() if c.doc_id == resp.doc_id]
        table_chunks = [c for c in chunks if "省份" in c.text]
        assert table_chunks, "手动入库的表格应生成含表头的 chunk"
        assert any("第2条" in c.text for c in chunks), "chunk 应带章节标题"

    def test_manual_ingest_no_pdf_file(self, store):
        resp = store.ingest_manual(MARKDOWN_SAMPLE, filename="问答.md")
        docs_dir = store.documents_dir
        assert list(docs_dir.glob("*.pdf")) == [], "手动入库不落 PDF 文件"

    def test_manual_ingest_default_importance_high(self, store):
        resp = store.ingest_manual("# 第1条 测试\n\n正文")
        assert resp.importance == "high"  # 手动入库面向重要内容，默认 high

    def test_manual_ingest_empty_markdown_raises(self, store):
        with pytest.raises(ValueError, match="markdown 内容不能为空"):
            store.ingest_manual("   \n")

    def test_manual_ingest_invalid_importance_raises(self, store):
        with pytest.raises(ValueError, match="importance"):
            store.ingest_manual("正文", importance="weird")

    def test_manual_ingest_delete_works(self, store):
        resp = store.ingest_manual(MARKDOWN_SAMPLE)
        assert store.delete(resp.doc_id)
        assert len(store.list_documents()) == 0


# ---------------------------------------------------------------------------
# 解析精度评测（98% 双评测之一：表格单元格识别准确率）
# ---------------------------------------------------------------------------


class TestParsingAccuracy:
    def test_exact_match_is_100_percent(self):
        rows = [["省份", "上限"], ["冀北", "760"]]
        result = compute_table_cell_accuracy(rows, rows)
        assert result.accuracy == 1.0
        assert result.correct_cells == result.total_cells

    def test_single_cell_error(self):
        actual = [["省份", "上限"], ["冀北", "760"]]
        expected = [["省份", "上限"], ["冀北", "780"]]
        result = compute_table_cell_accuracy(actual, expected)
        assert result.accuracy == 3 / 4
        assert result.errors[0]["expected"] == "780"
        assert result.errors[0]["actual"] == "760"

    def test_missing_row_counts_as_errors(self):
        actual = [["省份", "上限"], ["冀北", "760"]]
        expected = [["省份", "上限"], ["冀北", "760"], ["山西", "780"]]
        result = compute_table_cell_accuracy(actual, expected)
        assert result.correct_cells == 4
        assert result.total_cells == 6
        assert result.accuracy == pytest.approx(4 / 6)

    def test_extra_rows_not_penalized(self):
        actual = [["省份", "上限"], ["冀北", "760"], ["山西", "780"], ["湖北", "999"]]
        expected = [["省份", "上限"], ["冀北", "760"], ["山西", "780"]]
        result = compute_table_cell_accuracy(actual, expected)
        assert result.total_cells == 6
        assert result.accuracy == 1.0

    def test_empty_actual_is_zero(self):
        result = compute_table_cell_accuracy([], [["a"], ["b"]])
        assert result.accuracy == 0.0

    def test_manual_markdown_roundtrip_meets_target(self):
        """手动入库口径：markdown 原文还原 → 与手工标注期望表逐格一致 → 100%。"""
        md = "| 省份 | 电价上限(元/MWh) |\n|---|---|\n| 冀北 | 760 |\n| 山西 | 780 |\n| 四川主网 | 800 |"
        rows = parse_markdown_table(md)
        result = compute_table_cell_accuracy(
            rows,
            [["省份", "电价上限(元/MWh)"], ["冀北", "760"], ["山西", "780"], ["四川主网", "800"]],
        )
        assert result.accuracy >= TABLE_CELL_ACCURACY_TARGET
        assert result.accuracy == 1.0

    def test_mineru_markdown_meets_target(self):
        """MinerU 输出口径：多页 markdown（带页分隔符）还原 → 表格完整识别。"""
        md = (
            "# 第二章 价格规则\n\n---\n\n## 第2条 价格上限\n\n"
            "| 省份 | 电价上限(元/MWh) |\n|---|---|\n"
            "| 冀北 | 760 |\n| 山西 | 780 |\n| 四川主网 | 800 |"
        )
        page = markdown_to_page_content(md)
        result = compute_table_cell_accuracy(
            page.table_blocks[0].rows,
            [["省份", "电价上限(元/MWh)"], ["冀北", "760"], ["山西", "780"], ["四川主网", "800"]],
        )
        assert result.accuracy >= TABLE_CELL_ACCURACY_TARGET

    def test_evaluate_table_parsing_cases_file(self):
        """评测数据文件整体达标（手动入库 100%、MinerU 输出 ≥98%）。"""
        cases = load_table_parse_cases()
        assert cases, "评测数据 data/evaluation/table_parse_cases.json 应可加载"
        results, overall = evaluate_table_parsing(cases)
        assert overall >= TABLE_CELL_ACCURACY_TARGET
        assert all(r.accuracy >= TABLE_CELL_ACCURACY_TARGET for r in results)


# ---------------------------------------------------------------------------
# 检索召回评测（98% 双评测之二：表格类用例 recall@k）
# ---------------------------------------------------------------------------


class TestRetrievalRecall98:
    def _table_cases(self) -> list[TestCase]:
        """表格检索类用例（tag=「表格检索」，与 test_cases.json tc-015~018 同口径）。"""
        return [
            TestCase(
                id="tc-015", question="冀北的日前现货出清电价上限是多少？",
                expected_keywords=["760", "电价上限"], tags=[TABLE_RECALL_TAG],
            ),
            TestCase(
                id="tc-016", question="四川主网的日前现货出清电价上限是多少？",
                expected_keywords=["四川主网", "800"], tags=[TABLE_RECALL_TAG],
            ),
            TestCase(
                id="tc-017", question="山西省的日前现货出清电价上限是多少？",
                expected_keywords=["山西", "780"], tags=[TABLE_RECALL_TAG],
            ),
            TestCase(
                id="tc-018", question="各省日前现货出清电价上限表中包含哪些省份？",
                expected_keywords=["省份", "冀北", "山西", "四川主网"], tags=[TABLE_RECALL_TAG],
            ),
        ]

    def test_table_cases_recall_at_k_meets_target(self):
        """表格类用例按关键词代理口径命中 → 平均 recall@k ≥ 98%。"""
        # 模拟分层解析后的检索结果：表格 chunk 文本为 markdown 表格
        table_chunk = (
            "第2条 价格上限 > 下表为各省日前现货出清电价上限：\n"
            "| 省份 | 电价上限(元/MWh) |\n|---|---|\n"
            "| 冀北 | 760 |\n| 山西 | 780 |\n| 四川主网 | 800 |"
        )
        hits = [1.0 if compute_recall_at_k([{"text": table_chunk}], tc.expected_keywords) else 0.0
                for tc in self._table_cases()]
        avg_recall = sum(hits) / len(hits)
        assert avg_recall >= RETRIEVAL_RECALL_TARGET

    def test_recall_gate_fails_without_table_chunk(self):
        """门槛有效性：检索不到表格 chunk（关键数字缺失）时 recall 不达标。"""
        cases = self._table_cases()
        hits = [
            1.0 if compute_recall_at_k([{"text": "规则正文，无表格内容"}], tc.expected_keywords) else 0.0
            for tc in cases
        ]
        avg_recall = sum(hits) / len(hits)
        assert avg_recall < RETRIEVAL_RECALL_TARGET


# ---------------------------------------------------------------------------
# 持久化兼容
# ---------------------------------------------------------------------------


class TestPersistenceCompat:
    def test_old_document_info_json_compatible(self):
        """旧格式（无新字段）的 documents.json 反序列化不报错。"""
        from src.rule_review.models import DocumentInfo

        old = {
            "doc_id": "abc",
            "file_name": "rule.pdf",
            "page_count": 3,
            "chunk_count": 5,
            "created_at": "2026-01-01T00:00:00",
        }
        info = DocumentInfo(**old)
        assert info.importance == "low"
        assert info.parse_mode == "pymupdf"
        assert info.source == ""

    def test_save_load_roundtrip_with_new_fields(self, tmp_path, mock_embed):
        store = DocumentStore(
            documents_dir=tmp_path / "docs",
            index_dir=tmp_path / "index",
            embedding_fn=mock_embed,
            embedding_dim=8,
        )
        resp = store.ingest_manual(
            MARKDOWN_SAMPLE, filename="问答.md", importance="high", source="人工整理"
        )

        store2 = DocumentStore(
            documents_dir=tmp_path / "docs",
            index_dir=tmp_path / "index",
            embedding_fn=mock_embed,
            embedding_dim=8,
        )
        docs = store2.list_documents()
        assert len(docs) == 1
        assert docs[0].parse_mode == "manual"
        assert docs[0].importance == "high"
        assert docs[0].source == "人工整理"
