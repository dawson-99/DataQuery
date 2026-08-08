"""
规则审查系统可观测性单元测试

覆盖 src/rule_review/observability.py、pipeline 延迟接线、审计字段补齐、
router 延迟端点与 logging_setup 的 JSON 结构化日志。
"""

from __future__ import annotations

import json
import logging
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.rule_review.audit import AuditStore
from src.rule_review.document_store import Chunk, DocumentInfo, DocumentStore
from src.rule_review.observability import LatencyStats
from src.rule_review.pipeline import RuleReviewPipeline
from src.rule_review.retriever import HybridRetriever, HybridSearchResult, RetrieveResult
from src.rule_review.schemas import LLMOutput, RuleReviewRequest


# ---------------------------------------------------------------------------
# LatencyStats 纯函数
# ---------------------------------------------------------------------------


class TestLatencyStats:
    def test_percentile_odd_samples(self):
        stats = LatencyStats()
        for ms in [10, 20, 30, 40, 50]:
            stats.record("retrieval", ms)
        assert stats.percentile("retrieval", 50) == 30.0
        # 线性插值: idx=(5-1)*0.95=3.8 → 40*0.2+50*0.8=48（与 numpy.percentile 一致）
        assert stats.percentile("retrieval", 95) == 48.0

    def test_percentile_even_samples_linear_interpolation(self):
        stats = LatencyStats()
        for ms in [10, 20, 30, 40]:
            stats.record("total", ms)
        # idx = (4-1)*0.5 = 1.5 → 20*0.5 + 30*0.5 = 25
        assert stats.percentile("total", 50) == 25.0

    def test_percentile_empty_returns_zero(self):
        stats = LatencyStats()
        assert stats.percentile("retrieval", 95) == 0.0
        assert stats.avg("retrieval") == 0.0
        assert stats.count("retrieval") == 0

    def test_avg(self):
        stats = LatencyStats()
        for ms in [10, 20, 30]:
            stats.record("generation", ms)
        assert stats.avg("generation") == 20.0

    def test_summary_structure(self):
        stats = LatencyStats()
        stats.record("retrieval", 100)
        summary = stats.summary()
        assert "retrieval" in summary
        assert summary["retrieval"]["count"] == 1
        assert summary["retrieval"]["p50_ms"] == 100.0
        assert summary["retrieval"]["p95_ms"] == 100.0
        assert summary["retrieval"]["avg_ms"] == 100.0

    def test_to_from_dict_roundtrip(self):
        stats = LatencyStats()
        stats.record("retrieval", 10.5)
        stats2 = LatencyStats()
        stats2.from_dict(stats.to_dict())
        assert stats2.count("retrieval") == 1
        assert stats2.percentile("retrieval", 50) == 10.5

    def test_save_load_roundtrip(self, tmp_path):
        path = tmp_path / "latency.json"
        stats = LatencyStats(str(path))
        stats.record("retrieval", 12.5)
        assert stats.save() == str(path)
        stats2 = LatencyStats(str(path))
        assert stats2.load() is True
        assert stats2.count("retrieval") == 1

    def test_load_missing_file(self):
        stats = LatencyStats("/nonexistent_dir/latency.json")
        assert stats.load() is False

    def test_concurrent_record_thread_safe(self):
        stats = LatencyStats()
        threads = [
            threading.Thread(target=stats.record, args=("retrieval", 10.0))
            for _ in range(50)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert stats.count("retrieval") == 50


# ---------------------------------------------------------------------------
# pipeline 延迟统计接线
# ---------------------------------------------------------------------------


def _make_mock_components():
    """构造 mock 组件（参考 test_rule_review_pipeline.py 的写法）。"""
    rewriter = MagicMock()
    rewriter.rewrite.return_value = "2025-03-15 冀北 日前现货出清电价 800元/MWh 是否符合 价格上限"

    doc_store = MagicMock(spec=DocumentStore)
    doc_store.list_documents.return_value = [
        DocumentInfo(
            doc_id="doc_1", file_name="测试规则.pdf", page_count=10,
            chunk_count=5, created_at="2026-01-01T00:00:00",
        )
    ]

    retriever = MagicMock(spec=HybridRetriever)
    retriever.retrieve_with_fallback.return_value = RetrieveResult(
        results=[
            HybridSearchResult(
                chunk=Chunk(
                    chunk_id="c1", doc_id="doc_1",
                    text="省间日前现货出清电价上限为760元/MWh。", page=2,
                ),
                score=0.9,
                bm25_score=0.8,
                vector_score=0.85,
            )
        ],
        not_found=False,
        bm25_hits=1,
        vector_hits=1,
        fused_hits=1,
    )

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
    generator.generate = AsyncMock(return_value=llm_output)

    return rewriter, doc_store, retriever, generator


class TestPipelineLatencyRecording:
    @pytest.mark.asyncio
    async def test_execute_records_all_stages(self):
        rewriter, doc_store, retriever, generator = _make_mock_components()
        stats = LatencyStats()
        pipeline = RuleReviewPipeline(
            rewriter=rewriter,
            document_store=doc_store,
            retriever=retriever,
            generator=generator,
            stats=stats,
        )
        req = RuleReviewRequest(question="2025年3月15日冀北电价是否符合上限", stream=False)
        await pipeline.execute(req)

        assert stats.count("retrieval") == 1
        assert stats.count("generation") == 1
        assert stats.count("judge") == 1  # judge=None 也记录 0ms 样本
        assert stats.count("total") == 1

    @pytest.mark.asyncio
    async def test_execute_stream_records_all_stages(self):
        rewriter, doc_store, retriever, generator = _make_mock_components()
        stats = LatencyStats()
        pipeline = RuleReviewPipeline(
            rewriter=rewriter,
            document_store=doc_store,
            retriever=retriever,
            generator=generator,
            stats=stats,
        )
        req = RuleReviewRequest(question="2025年3月15日冀北电价是否符合上限", stream=True)
        async for _ in pipeline.execute_stream(req):
            pass

        assert stats.count("retrieval") == 1
        assert stats.count("generation") == 1
        assert stats.count("total") == 1


class TestAuditFieldsPopulated:
    @pytest.mark.asyncio
    async def test_audit_model_token_and_retrieval_fields(self):
        rewriter, doc_store, retriever, generator = _make_mock_components()
        audit_store = MagicMock(spec=AuditStore)
        pipeline = RuleReviewPipeline(
            rewriter=rewriter,
            document_store=doc_store,
            retriever=retriever,
            generator=generator,
            audit_store=audit_store,
        )
        req = RuleReviewRequest(question="2025年3月15日冀北电价是否符合上限", stream=False)
        await pipeline.execute(req)

        audit_store.save.assert_called_once()
        record = audit_store.save.call_args.args[0]

        # 模型名（默认 qwen3-max）与 token 容错（mock 无 usage_metadata → 0）
        assert record.llm_generation.model == "qwen3-max"
        assert record.llm_generation.tok_input == 0
        assert record.llm_generation.tok_output == 0
        # 检索通道计数与耗时如实记录
        assert record.retrieval.sparse_k == 0  # SparseRetriever 未接线
        assert record.retrieval.bm25_k == 1
        assert record.retrieval.retrieval_latency_ms >= 0
        assert record.llm_generation.latency_ms >= 0


# ---------------------------------------------------------------------------
# router 延迟端点
# ---------------------------------------------------------------------------


class TestLatencyEndpoint:
    @pytest.mark.asyncio
    async def test_endpoint_returns_summary(self, monkeypatch):
        stats = LatencyStats()
        stats.record("retrieval", 100)
        monkeypatch.setattr(
            "src.rule_review.observability.get_default_stats", lambda: stats
        )
        from src.rule_review.router import get_latency_stats

        resp = await get_latency_stats()
        body = json.loads(resp.body)
        assert body["status"] == "success"
        assert body["data"]["retrieval"]["count"] == 1
        assert body["data"]["retrieval"]["p95_ms"] == 100.0

    @pytest.mark.asyncio
    async def test_endpoint_empty_stats(self, monkeypatch):
        monkeypatch.setattr(
            "src.rule_review.observability.get_default_stats",
            lambda: LatencyStats(),
        )
        from src.rule_review.router import get_latency_stats

        resp = await get_latency_stats()
        body = json.loads(resp.body)
        assert body["status"] == "success"
        assert body["data"] == {}


# ---------------------------------------------------------------------------
# 结构化日志
# ---------------------------------------------------------------------------


class TestJsonFormatter:
    def _make_record(self, msg: str = "完成"):
        return logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )

    def test_format_parses_with_trace_and_extra(self):
        from src.utils.logging_setup import JsonFormatter

        record = self._make_record()
        record.trace_id = "t1"
        record.stage = "total"
        record.latency_ms = 123.4

        parsed = json.loads(JsonFormatter().format(record))
        assert parsed["trace_id"] == "t1"
        assert parsed["stage"] == "total"
        assert parsed["latency_ms"] == 123.4
        assert parsed["msg"] == "完成"
        assert parsed["level"] == "INFO"

    def test_format_without_extra_ok(self):
        from src.utils.logging_setup import JsonFormatter

        record = self._make_record("x")
        parsed = json.loads(JsonFormatter().format(record))
        assert "stage" not in parsed
        assert parsed["trace_id"] == "-"
