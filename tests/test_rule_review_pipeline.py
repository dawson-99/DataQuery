"""
规则审查系统编排器单元测试

覆盖 src/rule_review/pipeline.py 的：
- 澄清判断逻辑
- 多文档拆分逻辑
- 完整 pipeline 流式/非流式执行
- 空检索兜底路径
- LLM not_found 路径
- 多文档合并去重
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rule_review.document_store import Chunk, DocumentInfo, DocumentStore
from src.rule_review.pipeline import (
    RuleReviewPipeline,
    SSEProgress,
    _has_comparison_intent,
    _has_entity_info,
    _has_time_info,
    check_clarification_needed,
    get_default_pipeline,
    split_if_multi_document,
)
from src.rule_review.retriever import HybridRetriever, HybridSearchResult, RetrieveResult
from src.rule_review.schemas import (
    ClarificationResponse,
    LLMOutput,
    RuleReviewRequest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(chunk_id: str, text: str, page: int = 1, section: str = "") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc_1",
        text=text,
        page=page,
        section=section,
    )


def _make_hybrid_result(
    chunk_id: str, text: str, score: float = 0.9, page: int = 1
) -> HybridSearchResult:
    return HybridSearchResult(
        chunk=_make_chunk(chunk_id, text, page=page),
        score=score,
        bm25_score=0.8,
        vector_score=0.85,
    )


def _make_mock_document_store(doc_names: list[str] | None = None) -> MagicMock:
    mock = MagicMock(spec=DocumentStore)
    docs = []
    for name in (doc_names or ["测试规则.pdf"]):
        docs.append(
            DocumentInfo(
                doc_id=f"doc_{name}",
                file_name=name,
                page_count=10,
                chunk_count=5,
                created_at="2026-01-01T00:00:00",
            )
        )
    mock.list_documents.return_value = docs
    mock._chunks = {}
    return mock


# ---------------------------------------------------------------------------
# 澄清判断测试
# ---------------------------------------------------------------------------


class TestHasTimeInfo:
    def test_with_year_month_day(self):
        assert _has_time_info("2025年3月15日冀北的日前现货电价")

    def test_with_yesterday(self):
        assert _has_time_info("昨天的电价是否超过上限")

    def test_with_this_month(self):
        assert _has_time_info("本月冀北电价情况")

    def test_without_time(self):
        assert not _has_time_info("冀北电价是否超过上限")

    def test_with_holiday(self):
        assert _has_time_info("春节期间的交易是否符合规则")


class TestHasEntityInfo:
    def test_with_province_name(self):
        assert _has_entity_info("冀北的电价上限")

    def test_with_region_suffix(self):
        assert _has_entity_info("四川主网的价格规则")

    def test_without_entity(self):
        assert not _has_entity_info("电价是否超过上限")


class TestHasComparisonIntent:
    def test_with_shi_fou(self):
        assert _has_comparison_intent("是否超过价格上限")

    def test_with_fu_he(self):
        assert _has_comparison_intent("是否符合规则")

    def test_with_wei_fan(self):
        assert _has_comparison_intent("有没有违反价格上限")

    def test_without_intent(self):
        assert not _has_comparison_intent("电价和电量数据")


class TestCheckClarificationNeeded:
    def test_full_query_no_clarification(self):
        result = check_clarification_needed("2025年3月15日冀北的日前现货电价800元/MWh是否符合上限")
        assert result.needs_clarification is False

    def test_missing_time(self):
        result = check_clarification_needed("冀北的日前现货电价是否符合上限")
        assert result.needs_clarification is True
        assert any("时间" in m for m in result.missing)

    def test_missing_entity(self):
        result = check_clarification_needed("2025年3月15日电价800元/MWh是否符合上限")
        assert result.needs_clarification is True
        assert any("主体" in m for m in result.missing)

    def test_missing_comparison(self):
        result = check_clarification_needed("2025年3月15日冀北的日前现货电价")
        assert result.needs_clarification is True

    def test_all_missing(self):
        result = check_clarification_needed("测试")
        assert result.needs_clarification is True
        assert len(result.missing) >= 3

    def test_suggestions_included(self):
        result = check_clarification_needed("冀北的日前电价")
        assert result.needs_clarification is True
        assert len(result.suggestions) >= 1
        assert "请补充" in result.suggestions[0]


# ---------------------------------------------------------------------------
# 多文档拆分测试
# ---------------------------------------------------------------------------


class TestSplitIfMultiDocument:
    def test_single_document_returns_one_item(self):
        store = _make_mock_document_store(["规则.pdf"])
        result = split_if_multi_document("冀北的日前电价是否符合规则", store)
        assert len(result) == 1
        assert result[0]["doc_name"] is None

    def test_two_documents_splits(self):
        store = _make_mock_document_store(["交易规则.pdf", "监管办法.pdf"])
        result = split_if_multi_document(
            "根据《交易规则》和《监管办法》，冀北电价是否符合规定", store
        )
        assert len(result) == 2

    def test_doc_name_not_in_store(self):
        store = _make_mock_document_store(["规则A.pdf"])
        result = split_if_multi_document(
            "根据《不存在的文档》，冀北电价是否符合规定", store
        )
        # 书名号中的文档名不匹配任何已入库文档 → 单文档
        assert len(result) == 1
        assert result[0]["doc_name"] is None

    def test_no_book_title_marks(self):
        store = _make_mock_document_store(["交易规则.pdf", "监管办法.pdf"])
        result = split_if_multi_document("冀北电价是否符合规定", store)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Pipeline 流式执行测试
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_components():
    """创建 mock 版本的 Pipeline 组件。"""
    rewriter = MagicMock()
    rewriter.rewrite.return_value = "2025-03-15 冀北 日前现货出清电价 800元/MWh 是否符合 价格上限"

    doc_store = _make_mock_document_store(["测试规则.pdf"])

    # Mock retriever
    retriever = MagicMock(spec=HybridRetriever)
    mock_result = RetrieveResult(
        results=[
            _make_hybrid_result("c1", "省间日前现货出清电价上限为760元/MWh。", page=2),
            _make_hybrid_result("c2", "违反价格上限将被处罚。", page=4, score=0.85),
        ],
        not_found=False,
        bm25_hits=2,
        vector_hits=2,
        fused_hits=2,
    )

    async def _mock_retrieve(*args, **kwargs):
        return mock_result

    retriever.retrieve_with_fallback = _mock_retrieve

    # Mock generator
    generator = MagicMock()
    llm_output = LLMOutput(
        decision="不符合",
        reason="实际电价800元/MWh超过上限760元/MWh",
        evidence=[
            {
                "source": "测试规则.pdf",
                "section": "第2条 价格上限",
                "page": 2,
                "text": "省间日前现货出清电价上限为760元/MWh。",
            }
        ],
        confidence=0.95,
    )

    async def _mock_generate(*args, **kwargs):
        return llm_output

    generator.generate = _mock_generate

    return rewriter, doc_store, retriever, generator, llm_output


class TestRuleReviewPipelineStream:
    @pytest.mark.asyncio
    async def test_execute_stream_full_flow(self, mock_components):
        rewriter, doc_store, retriever, generator, llm_output = mock_components
        pipeline = RuleReviewPipeline(
            rewriter=rewriter,
            document_store=doc_store,
            retriever=retriever,
            generator=generator,
        )

        request = RuleReviewRequest(
            question="2025年3月15日冀北的日前现货出清电价800元/MWh是否符合价格上限",
            stream=True,
        )

        events = []
        async for sse_line in pipeline.execute_stream(request):
            events.append(sse_line)

        assert len(events) > 0

        # 检查各阶段标签
        full_text = "".join(events)
        assert "查询预处理中" in full_text or "改写" in full_text
        assert "检索相关知识中" in full_text

        # 最后一条应该是 done
        assert "done" in events[-1]

        # 应有内容输出
        content_events = [e for e in events if "不符合" in e and "event: message" in e]
        assert len(content_events) >= 1

    @pytest.mark.asyncio
    async def test_execute_stream_clarification(self, mock_components):
        """问题不明确时应返回澄清追问并提前结束。"""
        rewriter, doc_store, retriever, generator, _ = mock_components
        # 修改 rewriter 返回一个缺乏时间信息的 query
        rewriter.rewrite.return_value = "冀北电价是否符合上限"

        pipeline = RuleReviewPipeline(
            rewriter=rewriter,
            document_store=doc_store,
            retriever=retriever,
            generator=generator,
        )

        request = RuleReviewRequest(question="冀北电价是否符合上限", stream=True)

        events = []
        async for sse_line in pipeline.execute_stream(request):
            events.append(sse_line)

        # 应该有澄清事件
        full_text = "".join(events)
        assert "clarification" in full_text or "needs_clarification" in full_text

    @pytest.mark.asyncio
    async def test_execute_stream_empty_retrieval(self, mock_components):
        """检索无结果时应返回 not_found。"""
        rewriter, doc_store, retriever, generator, _ = mock_components

        empty_result = RetrieveResult(not_found=True)

        async def _mock_empty(*args, **kwargs):
            return empty_result

        retriever.retrieve_with_fallback = _mock_empty

        pipeline = RuleReviewPipeline(
            rewriter=rewriter,
            document_store=doc_store,
            retriever=retriever,
            generator=generator,
        )

        request = RuleReviewRequest(question="测试问题", stream=True)

        events = []
        async for sse_line in pipeline.execute_stream(request):
            events.append(sse_line)

        full_text = "".join(events)
        assert "not_found" in full_text or "无法判断" in full_text

    @pytest.mark.asyncio
    async def test_execute_stream_llm_not_found(self, mock_components):
        """LLM 返回 not_found 时不走后续步骤（Judge 等）。"""
        rewriter, doc_store, retriever, generator, _ = mock_components

        not_found_output = LLMOutput(
            decision="无法判断",
            reason="文档中未找到相关规则",
            evidence=[],
            confidence=0.0,
            not_found=True,
        )

        async def _mock_not_found(*args, **kwargs):
            return not_found_output

        generator.generate = _mock_not_found

        pipeline = RuleReviewPipeline(
            rewriter=rewriter,
            document_store=doc_store,
            retriever=retriever,
            generator=generator,
        )

        request = RuleReviewRequest(question="测试问题", stream=True)

        events = []
        async for sse_line in pipeline.execute_stream(request):
            events.append(sse_line)

        full_text = "".join(events)
        assert "not_found" in full_text or "无法判断" in full_text

    @pytest.mark.asyncio
    async def test_execute_stream_generator_failure(self, mock_components):
        """LLM 生成返回 None 时应有错误处理。"""
        rewriter, doc_store, retriever, generator, _ = mock_components

        async def _mock_none(*args, **kwargs):
            return None

        generator.generate = _mock_none

        pipeline = RuleReviewPipeline(
            rewriter=rewriter,
            document_store=doc_store,
            retriever=retriever,
            generator=generator,
        )

        request = RuleReviewRequest(question="测试问题", stream=True)

        events = []
        async for sse_line in pipeline.execute_stream(request):
            events.append(sse_line)

        full_text = "".join(events)
        assert "无法判断" in full_text or "error" in full_text


# ---------------------------------------------------------------------------
# Pipeline 非流式执行测试
# ---------------------------------------------------------------------------


class TestRuleReviewPipelineExecute:
    @pytest.mark.asyncio
    async def test_execute_full_flow(self, mock_components):
        rewriter, doc_store, retriever, generator, llm_output = mock_components
        pipeline = RuleReviewPipeline(
            rewriter=rewriter,
            document_store=doc_store,
            retriever=retriever,
            generator=generator,
        )

        request = RuleReviewRequest(question="测试问题", stream=False)
        result = await pipeline.execute(request)

        assert "result" in result
        assert result["result"]["decision"] == "不符合"
        assert len(result["stages"]) >= 4

    @pytest.mark.asyncio
    async def test_execute_clarification(self, mock_components):
        rewriter, doc_store, retriever, generator, _ = mock_components
        rewriter.rewrite.return_value = "电价是否符合上限"

        pipeline = RuleReviewPipeline(
            rewriter=rewriter,
            document_store=doc_store,
            retriever=retriever,
            generator=generator,
        )

        request = RuleReviewRequest(question="电价是否符合上限", stream=False)
        result = await pipeline.execute(request)

        assert "clarification" in result
        assert result["clarification"]["needs_clarification"] is True

    @pytest.mark.asyncio
    async def test_execute_empty_retrieval(self, mock_components):
        rewriter, doc_store, retriever, generator, _ = mock_components

        async def _mock_empty(*args, **kwargs):
            return RetrieveResult(not_found=True)

        retriever.retrieve_with_fallback = _mock_empty

        pipeline = RuleReviewPipeline(
            rewriter=rewriter,
            document_store=doc_store,
            retriever=retriever,
            generator=generator,
        )

        request = RuleReviewRequest(question="测试问题", stream=False)
        result = await pipeline.execute(request)

        assert result["result"]["decision"] == "无法判断"
        assert result["result"]["not_found"] is True


# ---------------------------------------------------------------------------
# 辅助方法测试
# ---------------------------------------------------------------------------


class TestChunksToDictList:
    def test_converts_hybrid_results(self):
        results = [
            _make_hybrid_result("c1", "规则内容A", page=2),
            _make_hybrid_result("c2", "规则内容B", page=4),
        ]
        dicts = RuleReviewPipeline._chunks_to_dict_list(results)
        assert len(dicts) == 2
        assert dicts[0]["text"] == "规则内容A"
        assert dicts[0]["page"] == 2
        assert dicts[1]["chunk_id"] == "c2"


class TestMergeRetrieveResults:
    def test_merges_and_dedup(self):
        r1 = _make_hybrid_result("c1", "内容A")
        r2 = _make_hybrid_result("c2", "内容B")
        r3 = _make_hybrid_result("c1", "内容A")  # 重复

        rr1 = RetrieveResult(results=[r1, r2], bm25_hits=2, vector_hits=2, fused_hits=2)
        rr2 = RetrieveResult(results=[r3], bm25_hits=1, vector_hits=1, fused_hits=1)

        merged = RuleReviewPipeline._merge_retrieve_results([rr1, rr2], top_k=10)
        assert len(merged.results) == 2  # 去重后
        assert merged.bm25_hits == 3
        assert merged.vector_hits == 3

    def test_merge_sorts_by_score(self):
        r1 = _make_hybrid_result("c1", "低分", score=0.5)
        r2 = _make_hybrid_result("c2", "高分", score=0.95)

        rr1 = RetrieveResult(results=[r1, r2], bm25_hits=2, vector_hits=2, fused_hits=2)

        merged = RuleReviewPipeline._merge_retrieve_results([rr1], top_k=10)
        assert merged.results[0].chunk.chunk_id == "c2"  # 高分在前


# ---------------------------------------------------------------------------
# SSE 格式化测试
# ---------------------------------------------------------------------------


class TestSSEFormatting:
    def test_sse_label(self):
        result = RuleReviewPipeline._sse_label("测试中...", "test")
        assert "data: " in result
        assert "测试中" in result
        assert "messageLabel" in result
        assert "test" in result

    def test_sse_content(self):
        result = RuleReviewPipeline._sse_content('{"ok":true}', "content")
        assert "event: message" in result
        # JSON 在 SSE 输出中被序列化为 JSON 字符串，因此是转义后的
        assert "ok" in result
        assert "true" in result

    def test_sse_done(self):
        result = RuleReviewPipeline._sse_done("q-123")
        assert "event: done" in result
        assert "done" in result
        assert "q-123" in result


# ---------------------------------------------------------------------------
# 便利方法测试
# ---------------------------------------------------------------------------


class TestConvenienceMethods:
    @pytest.mark.asyncio
    async def test_run_rewrite_and_clarify(self, mock_components):
        rewriter, doc_store, retriever, generator, _ = mock_components
        pipeline = RuleReviewPipeline(
            rewriter=rewriter,
            document_store=doc_store,
            retriever=retriever,
            generator=generator,
        )

        rewritten, clarification = await pipeline.run_rewrite_and_clarify(
            "2025年3月15日冀北的日前现货电价800是否符合上限"
        )
        assert isinstance(rewritten, str)
        assert isinstance(clarification, ClarificationResponse)


# ---------------------------------------------------------------------------
# 默认工厂测试
# ---------------------------------------------------------------------------


class TestDefaultPipeline:
    def test_get_default_pipeline(self):
        p1 = get_default_pipeline()
        p2 = get_default_pipeline()
        assert p1 is p2
        assert isinstance(p1, RuleReviewPipeline)


# ---------------------------------------------------------------------------
# multi-document 流式测试
# ---------------------------------------------------------------------------


class TestMultiDocStream:
    @pytest.mark.asyncio
    async def test_execute_stream_multi_doc(self, mock_components):
        """多文档查询应走合并检索路径。"""
        rewriter, doc_store, retriever, generator, llm_output = mock_components
        # 改写结果必须包含时间、实体、比较意图以及书名号以通过澄清+触发多文档拆分
        rewriter.rewrite.return_value = (
            "2025年3月15日 根据《交易规则》和《监管办法》 冀北的日前现货出清电价 "
            "800元/MWh 是否符合 价格上限"
        )

        # 修改 doc_store 返回多文档
        doc_store.list_documents.return_value = [
            DocumentInfo(
                doc_id="doc_a", file_name="交易规则.pdf",
                page_count=10, chunk_count=5, created_at="2026-01-01T00:00:00",
            ),
            DocumentInfo(
                doc_id="doc_b", file_name="监管办法.pdf",
                page_count=8, chunk_count=3, created_at="2026-01-01T00:00:00",
            ),
        ]

        pipeline = RuleReviewPipeline(
            rewriter=rewriter,
            document_store=doc_store,
            retriever=retriever,
            generator=generator,
        )

        request = RuleReviewRequest(
            question="根据《交易规则》和《监管办法》冀北电价是否符合规定",
            stream=True,
        )

        events = []
        async for sse_line in pipeline.execute_stream(request):
            events.append(sse_line)

        full_text = "".join(events)
        # 应包含多文档提示
        assert "多文档" in full_text or "split" in full_text
