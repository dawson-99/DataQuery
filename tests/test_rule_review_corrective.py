"""
规则审查 Corrective-RAG 回环单元测试

覆盖 src/rule_review/pipeline.py 的：
- 触发条件判断（_should_run_corrective）
- 二次检索 query 构造（_build_corrective_query）
- 回环全链路（二次检索 → 合并去重 → 重新生成 → 二次校验）
- 终止矩阵（次轮空检索 / 生成失败 / not_found / Judge 跳过）
- 审计字段与 SSE 阶段标签
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import settings
from src.rule_review.audit import AuditStore
from src.rule_review.document_store import Chunk
from src.rule_review.pipeline import (
    RuleReviewPipeline,
    _build_corrective_audit,
    _is_query_subsumed,
)
from src.rule_review.retriever import HybridRetriever, HybridSearchResult, RetrieveResult
from src.rule_review.schemas import LLMOutput, RuleReviewRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hybrid_result(
    chunk_id: str, text: str, score: float = 0.9
) -> HybridSearchResult:
    return HybridSearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id="doc_1",
            text=text,
            page=1,
            section=f"第{chunk_id[1:]}条 测试章节",
        ),
        score=score,
        bm25_score=0.8,
        vector_score=0.85,
    )


def _make_round1_judged(clean: bool = False) -> dict:
    """构造首轮 Judge 结果 dict。clean=True 时无幻觉/遗漏（不触发回环）。"""
    judged = {
        "decision": "不符合",
        "reason": "实际电价800元/MWh超过上限760元/MWh",
        "evidence": [
            {
                "source": "测试规则.pdf",
                "section": "第2条 价格上限",
                "page": 2,
                "text": "省间日前现货出清电价上限为760元/MWh。",
            }
        ],
        "confidence": 0.9,
        "judge_verified": False,
        "judge_corrections": [],
        "judge_hallucinated": [],
        "judge_missing_rules": [],
    }
    if not clean:
        judged["judge_hallucinated"] = [
            {"index": 0, "reason": "证据文本未在规则原文中找到对应内容"}
        ]
        judged["judge_missing_rules"] = [
            {"rule": "第5条 限价申报规则", "source": "测试规则.pdf"}
        ]
    return judged


def _make_round2_judged() -> dict:
    return {
        "decision": "不符合",
        "reason": "修正后：电价800元/MWh超过上限760元/MWh，且违反限价申报规则",
        "evidence": [
            {
                "source": "测试规则.pdf",
                "section": "第2条 价格上限",
                "page": 2,
                "text": "省间日前现货出清电价上限为760元/MWh。",
            },
            {
                "source": "测试规则.pdf",
                "section": "第5条 限价申报规则",
                "page": 5,
                "text": "申报电价不得高于限价申报标准。",
            },
        ],
        "confidence": 0.97,
        "judge_verified": True,
        "judge_corrections": [],
        "judge_hallucinated": [],
        "judge_missing_rules": [],
    }


def _make_pipeline(
    retriever_results: list[RetrieveResult],
    generator_outputs: list[LLMOutput],
    judge: MagicMock | None = None,
    audit_store: MagicMock | None = None,
):
    """构造带按调用次数返回结果的 retriever/generator 的 pipeline。

    调用顺序：首轮检索、二次检索各取一个结果；首轮生成、二次生成各取一个输出。
    counters 记录实际调用次数，供断言使用。

    Returns:
        (pipeline, retriever, generator, counters: {"retrieve": int, "generate": int})
    """
    rewriter = MagicMock()
    rewriter.rewrite.return_value = "2025-03-15 冀北 日前现货出清电价 800元/MWh 是否符合 价格上限"

    retriever = MagicMock(spec=HybridRetriever)
    counters = {"retrieve": 0, "generate": 0}

    def _mock_retrieve(*args, **kwargs):
        counters["retrieve"] += 1
        idx = min(counters["retrieve"] - 1, len(retriever_results) - 1)
        return retriever_results[idx]

    retriever.retrieve_with_fallback = _mock_retrieve

    generator = MagicMock()

    async def _mock_generate(*args, **kwargs):
        idx = min(counters["generate"], len(generator_outputs) - 1)
        counters["generate"] += 1
        return generator_outputs[idx]

    generator.generate = AsyncMock(side_effect=_mock_generate)

    pipeline = RuleReviewPipeline(
        rewriter=rewriter,
        document_store=MagicMock(),
        retriever=retriever,
        generator=generator,
        judge=judge if judge is not None else MagicMock(),
        audit_store=audit_store,
    )
    return pipeline, retriever, generator, counters


def _make_request(top_k: int = 10, stream: bool = False) -> RuleReviewRequest:
    return RuleReviewRequest(
        question="2025年3月15日冀北的日前现货出清电价800元/MWh是否符合价格上限",
        stream=stream,
        top_k=top_k,
    )


# ---------------------------------------------------------------------------
# 触发条件判断
# ---------------------------------------------------------------------------


class TestShouldRunCorrective:
    def test_triggered_by_missing_rules(self, monkeypatch):
        monkeypatch.setattr(settings, "RULE_REVIEW_CORRECTIVE_ENABLED", True)
        judged = _make_round1_judged()
        judged["judge_hallucinated"] = []
        assert RuleReviewPipeline._should_run_corrective(judged) is True

    def test_triggered_by_hallucinated(self, monkeypatch):
        monkeypatch.setattr(settings, "RULE_REVIEW_CORRECTIVE_ENABLED", True)
        judged = _make_round1_judged()
        judged["judge_missing_rules"] = []
        assert RuleReviewPipeline._should_run_corrective(judged) is True

    def test_triggered_by_both(self, monkeypatch):
        monkeypatch.setattr(settings, "RULE_REVIEW_CORRECTIVE_ENABLED", True)
        assert RuleReviewPipeline._should_run_corrective(_make_round1_judged()) is True

    def test_not_triggered_when_clean(self, monkeypatch):
        monkeypatch.setattr(settings, "RULE_REVIEW_CORRECTIVE_ENABLED", True)
        assert RuleReviewPipeline._should_run_corrective(_make_round1_judged(clean=True)) is False

    def test_not_triggered_when_judge_skipped(self, monkeypatch):
        monkeypatch.setattr(settings, "RULE_REVIEW_CORRECTIVE_ENABLED", True)
        judged = _make_round1_judged()
        judged["judge_skipped"] = True
        judged["judge_skipped_reason"] = "超时"
        assert RuleReviewPipeline._should_run_corrective(judged) is False

    def test_not_triggered_when_disabled(self, monkeypatch):
        monkeypatch.setattr(settings, "RULE_REVIEW_CORRECTIVE_ENABLED", False)
        assert RuleReviewPipeline._should_run_corrective(_make_round1_judged()) is False


# ---------------------------------------------------------------------------
# 二次检索 query 构造
# ---------------------------------------------------------------------------


class TestBuildCorrectiveQuery:
    def test_appends_missing_rules_into_query(self):
        judged = _make_round1_judged()
        llm_out = LLMOutput(decision="不符合", reason="r", confidence=0.9)
        query = RuleReviewPipeline._build_corrective_query("原问题", judged, llm_out)
        assert "第5条 限价申报规则" in query
        assert query.startswith("原问题")

    def test_excludes_hallucinated_evidence_text(self):
        """幻觉证据正文不进入 query（证据本身是错的，不重查）。"""
        judged = _make_round1_judged()
        judged["judge_missing_rules"] = []
        judged["judge_hallucinated"] = [
            {"index": 0, "reason": "证据文本未在规则原文中找到对应内容"}
        ]
        llm_out = LLMOutput(
            decision="不符合",
            reason="r",
            confidence=0.9,
            evidence=[
                {
                    "source": "测试规则.pdf",
                    "section": "第2条 价格上限",
                    "page": 2,
                    "text": "省间日前现货出清电价上限为760元/MWh。",
                }
            ],
        )
        query = RuleReviewPipeline._build_corrective_query("原问题", judged, llm_out)
        # 使用幻觉证据的 section 标题做关键词，但不含证据正文
        assert "第2条 价格上限" in query
        assert "省间日前现货出清电价上限为760元/MWh" not in query

    def test_dedup_overlapped_query(self):
        """与 rewritten_query 互相包含的补充跳过。"""
        judged = {
            "judge_hallucinated": [],
            "judge_missing_rules": [{"rule": "2025-03-15", "source": "x"}],
        }
        query = RuleReviewPipeline._build_corrective_query("2025-03-15 冀北电价", judged, None)
        assert query == "2025-03-15 冀北电价"

    def test_no_valid_parts_returns_rewritten_query(self):
        judged = {
            "judge_hallucinated": [{"index": 99, "reason": " "}],
            "judge_missing_rules": [],
        }
        query = RuleReviewPipeline._build_corrective_query("原问题", judged, None)
        assert query == "原问题"

    def test_truncates_and_limits_parts(self):
        long_rule = "第X条 " + "很长的规则文本" * 20
        judged = {
            "judge_hallucinated": [],
            "judge_missing_rules": [
                {"rule": long_rule, "source": "x"},
                {"rule": "条款二", "source": "x"},
                {"rule": "条款三", "source": "x"},
                {"rule": "条款四", "source": "x"},
            ],
        }
        query = RuleReviewPipeline._build_corrective_query("原问题", judged, None)
        # 最多 3 条补充
        assert query.count("条款") >= 2
        assert len(query) <= 200

    def test_hallucinated_fallback_to_reason_when_index_invalid(self):
        judged = {
            "judge_hallucinated": [{"index": 99, "reason": "价格上限条款未找到"}],
            "judge_missing_rules": [],
        }
        query = RuleReviewPipeline._build_corrective_query("原问题", judged, None)
        assert "价格上限条款未找到" in query


class TestIsQuerySubsumed:
    def test_subsumed_by_query(self):
        assert _is_query_subsumed("2025-03-15", "2025-03-15 冀北电价", []) is True

    def test_subsumed_by_part(self):
        assert _is_query_subsumed("第5条", "原问题", ["第5条 限价申报规则"]) is True

    def test_empty_candidate(self):
        assert _is_query_subsumed("", "原问题", []) is True

    def test_independent_kept(self):
        assert _is_query_subsumed("第5条 限价申报规则", "原问题", []) is False


# ---------------------------------------------------------------------------
# 回环全链路（非流式 execute）
# ---------------------------------------------------------------------------


class TestCorrectiveLoopExecute:
    @pytest.mark.asyncio
    async def test_execute_triggers_full_loop(self, monkeypatch):
        """触发时：检索 2 次、生成 2 次、Judge 2 次，最终输出第二轮结果。"""
        monkeypatch.setattr(settings, "RULE_REVIEW_CORRECTIVE_ENABLED", True)
        first_rr = RetrieveResult(results=[_make_hybrid_result("c1", "规则A")], bm25_hits=1, vector_hits=1)
        second_rr = RetrieveResult(results=[_make_hybrid_result("c2", "规则B")], bm25_hits=1, vector_hits=1)
        llm1 = LLMOutput(decision="不符合", reason="首轮", confidence=0.9)
        llm2 = LLMOutput(decision="不符合", reason="第二轮修正", confidence=0.95)

        pipeline, _, _, counters = _make_pipeline(
            [first_rr, second_rr], [llm1, llm2]
        )

        with patch(
            "src.rule_review.judge.verify_with_fallback",
            side_effect=[_make_round1_judged(), _make_round2_judged()],
        ) as mock_verify:
            result = await pipeline.execute(_make_request())

        assert mock_verify.call_count == 2
        assert counters["retrieve"] == 2
        assert counters["generate"] == 2
        assert result["result"]["decision"] == "不符合"
        assert result["result"]["judge_verified"] is True
        # 第二阶段标签
        assert any(s.get("stage") == "re_retrieval" for s in result["stages"])

    @pytest.mark.asyncio
    async def test_execute_merge_dedup(self, monkeypatch):
        """首轮 c1/c2 + 次轮 c2/c3 → 第二轮生成收到 {c1,c2,c3} 无重复。"""
        monkeypatch.setattr(settings, "RULE_REVIEW_CORRECTIVE_ENABLED", True)
        first_rr = RetrieveResult(
            results=[
                _make_hybrid_result("c1", "规则A", score=0.9),
                _make_hybrid_result("c2", "规则B", score=0.8),
            ],
            bm25_hits=2, vector_hits=2,
        )
        second_rr = RetrieveResult(
            results=[
                _make_hybrid_result("c2", "规则B", score=0.95),
                _make_hybrid_result("c3", "规则C", score=0.7),
            ],
            bm25_hits=2, vector_hits=2,
        )
        llm1 = LLMOutput(decision="不符合", reason="首轮", confidence=0.9)
        llm2 = LLMOutput(decision="不符合", reason="第二轮", confidence=0.95)

        pipeline, _, generator, _ = _make_pipeline([first_rr, second_rr], [llm1, llm2])
        received_chunks = {}

        async def _capture_generate(*args, **kwargs):
            received_chunks["chunks"] = [c.get("chunk_id") for c in kwargs.get("context_chunks", [])]
            return llm2

        generator.generate = _capture_generate

        with patch(
            "src.rule_review.judge.verify_with_fallback",
            side_effect=[_make_round1_judged(), _make_round2_judged()],
        ):
            await pipeline.execute(_make_request())

        assert set(received_chunks["chunks"]) == {"c1", "c2", "c3"}
        assert len(received_chunks["chunks"]) == 3

    @pytest.mark.asyncio
    async def test_execute_not_triggered_when_clean(self, monkeypatch):
        """Judge 无幻觉/遗漏时不触发：检索与生成各 1 次。"""
        monkeypatch.setattr(settings, "RULE_REVIEW_CORRECTIVE_ENABLED", True)
        first_rr = RetrieveResult(results=[_make_hybrid_result("c1", "规则A")], bm25_hits=1, vector_hits=1)
        llm1 = LLMOutput(decision="不符合", reason="首轮", confidence=0.9)

        pipeline, _, _, counters = _make_pipeline([first_rr], [llm1])

        with patch(
            "src.rule_review.judge.verify_with_fallback",
            return_value=_make_round1_judged(clean=True),
        ) as mock_verify:
            await pipeline.execute(_make_request())

        assert mock_verify.call_count == 1
        assert counters["retrieve"] == 1
        assert counters["generate"] == 1

    @pytest.mark.asyncio
    async def test_execute_not_triggered_when_disabled(self, monkeypatch):
        """开关关闭时不触发。"""
        monkeypatch.setattr(settings, "RULE_REVIEW_CORRECTIVE_ENABLED", False)
        first_rr = RetrieveResult(results=[_make_hybrid_result("c1", "规则A")], bm25_hits=1, vector_hits=1)
        llm1 = LLMOutput(decision="不符合", reason="首轮", confidence=0.9)

        pipeline, _, _, counters = _make_pipeline([first_rr], [llm1])

        with patch(
            "src.rule_review.judge.verify_with_fallback",
            return_value=_make_round1_judged(),
        ) as mock_verify:
            await pipeline.execute(_make_request())

        assert mock_verify.call_count == 1
        assert counters["retrieve"] == 1
        assert counters["generate"] == 1

    @pytest.mark.asyncio
    async def test_second_retrieve_empty_keeps_round1(self, monkeypatch):
        """次轮检索为空 → 保留首轮结果，不再生成/校验。"""
        monkeypatch.setattr(settings, "RULE_REVIEW_CORRECTIVE_ENABLED", True)
        first_rr = RetrieveResult(results=[_make_hybrid_result("c1", "规则A")], bm25_hits=1, vector_hits=1)
        second_rr = RetrieveResult(not_found=True)
        llm1 = LLMOutput(decision="不符合", reason="首轮", confidence=0.9)

        pipeline, _, _, counters = _make_pipeline([first_rr, second_rr], [llm1])

        with patch(
            "src.rule_review.judge.verify_with_fallback",
            return_value=_make_round1_judged(),
        ) as mock_verify:
            result = await pipeline.execute(_make_request())

        assert mock_verify.call_count == 1  # 只有首轮校验
        assert counters["generate"] == 1
        assert result["result"]["judge_verified"] is False
        re_stage = next(s for s in result["stages"] if s.get("stage") == "re_retrieval")
        assert re_stage["terminated_reason"] == "second_retrieve_empty"

    @pytest.mark.asyncio
    async def test_second_generate_none_falls_back_round1(self, monkeypatch):
        """二次生成失败（None）→ 降级输出首轮结果。"""
        monkeypatch.setattr(settings, "RULE_REVIEW_CORRECTIVE_ENABLED", True)
        first_rr = RetrieveResult(results=[_make_hybrid_result("c1", "规则A")], bm25_hits=1, vector_hits=1)
        second_rr = RetrieveResult(results=[_make_hybrid_result("c3", "规则C")], bm25_hits=1, vector_hits=1)
        llm1 = LLMOutput(decision="不符合", reason="首轮", confidence=0.9)

        # 首轮正常生成，第二轮（回环内）生成失败返回 None
        pipeline, _, _, _ = _make_pipeline([first_rr, second_rr], [llm1, None])

        with patch(
            "src.rule_review.judge.verify_with_fallback",
            return_value=_make_round1_judged(),
        ) as mock_verify:
            result = await pipeline.execute(_make_request())

        assert mock_verify.call_count == 1
        assert result["result"]["judge_verified"] is False
        re_stage = next(s for s in result["stages"] if s.get("stage") == "re_retrieval")
        assert re_stage["terminated_reason"] == "second_generate_none"

    @pytest.mark.asyncio
    async def test_second_not_found_terminates(self, monkeypatch):
        """二次生成判定无相关规则 → 输出 not_found，不再校验/三循环。"""
        monkeypatch.setattr(settings, "RULE_REVIEW_CORRECTIVE_ENABLED", True)
        first_rr = RetrieveResult(results=[_make_hybrid_result("c1", "规则A")], bm25_hits=1, vector_hits=1)
        second_rr = RetrieveResult(results=[_make_hybrid_result("c3", "规则C")], bm25_hits=1, vector_hits=1)
        llm1 = LLMOutput(decision="不符合", reason="首轮", confidence=0.9)
        llm2 = LLMOutput(decision="无法判断", reason="无相关规则", confidence=0.0, not_found=True)

        pipeline, _, _, _ = _make_pipeline([first_rr, second_rr], [llm1, llm2])

        with patch(
            "src.rule_review.judge.verify_with_fallback",
            return_value=_make_round1_judged(),
        ) as mock_verify:
            result = await pipeline.execute(_make_request())

        assert mock_verify.call_count == 1  # 未做第二轮校验
        assert result["result"]["not_found"] is True
        assert result["result"]["judge_skipped"] is True
        re_stage = next(s for s in result["stages"] if s.get("stage") == "re_retrieval")
        assert re_stage["terminated_reason"] == "second_not_found"

    @pytest.mark.asyncio
    async def test_second_judge_skipped_terminates(self, monkeypatch):
        """二次校验跳过 → 输出其跳过结果，不再三循环。"""
        monkeypatch.setattr(settings, "RULE_REVIEW_CORRECTIVE_ENABLED", True)
        first_rr = RetrieveResult(results=[_make_hybrid_result("c1", "规则A")], bm25_hits=1, vector_hits=1)
        second_rr = RetrieveResult(results=[_make_hybrid_result("c3", "规则C")], bm25_hits=1, vector_hits=1)
        llm1 = LLMOutput(decision="不符合", reason="首轮", confidence=0.9)
        llm2 = LLMOutput(decision="不符合", reason="第二轮", confidence=0.95)

        pipeline, _, _, counters = _make_pipeline([first_rr, second_rr], [llm1, llm2])

        skipped_round2 = _make_round2_judged()
        skipped_round2["judge_skipped"] = True
        skipped_round2["judge_skipped_reason"] = "超时"

        with patch(
            "src.rule_review.judge.verify_with_fallback",
            side_effect=[_make_round1_judged(), skipped_round2],
        ) as mock_verify:
            result = await pipeline.execute(_make_request())

        assert mock_verify.call_count == 2
        assert counters["generate"] == 2
        assert result["result"]["judge_skipped"] is True
        re_stage = next(s for s in result["stages"] if s.get("stage") == "re_retrieval")
        assert re_stage["terminated_reason"] == "second_judge_skipped"

    @pytest.mark.asyncio
    async def test_audit_records_corrective_fields(self, monkeypatch):
        """审计记录携带 corrective 详情与检索轮数。"""
        monkeypatch.setattr(settings, "RULE_REVIEW_CORRECTIVE_ENABLED", True)
        first_rr = RetrieveResult(results=[_make_hybrid_result("c1", "规则A")], bm25_hits=1, vector_hits=1)
        second_rr = RetrieveResult(results=[_make_hybrid_result("c2", "规则B")], bm25_hits=1, vector_hits=1)
        llm1 = LLMOutput(decision="不符合", reason="首轮", confidence=0.9)
        llm2 = LLMOutput(decision="不符合", reason="第二轮", confidence=0.95)

        audit_store = MagicMock(spec=AuditStore)
        pipeline, _, _, _ = _make_pipeline(
            [first_rr, second_rr], [llm1, llm2], audit_store=audit_store
        )

        with patch(
            "src.rule_review.judge.verify_with_fallback",
            side_effect=[_make_round1_judged(), _make_round2_judged()],
        ):
            await pipeline.execute(_make_request())

        assert audit_store.save.call_count == 1
        record = audit_store.save.call_args[0][0]
        assert record.retrieval.retrieval_rounds == 2
        assert record.retrieval.corrective_expanded is True
        assert record.llm_generation.rounds == 2
        assert record.judge_verification.rounds == 2
        assert record.corrective is not None
        assert record.corrective["triggered"] is True
        assert record.corrective["reason"] == "both"
        assert record.corrective["corrective_query"]  # 非空
        assert record.corrective["round2_verified"] is True
        assert record.corrective["terminated_reason"] == ""


class TestBuildCorrectiveAudit:
    def test_none_when_not_triggered(self):
        assert _build_corrective_audit(None) is None

    def test_none_when_triggered_false(self):
        from src.rule_review.pipeline import CorrectiveLoopResult

        assert _build_corrective_audit(CorrectiveLoopResult()) is None

    def test_audit_dict_content(self):
        from src.rule_review.pipeline import CorrectiveLoopResult

        loop = CorrectiveLoopResult(
            triggered=True,
            trigger_reason="missing_rules",
            corrective_query="补充查询",
            second_top_k=20,
            merged_chunks=[{"chunk_id": "c1", "text": "x"}],
            terminated_reason="",
        )
        loop.second_judged = {"judge_verified": True, "judge_skipped": False}
        loop.second_llm_output = LLMOutput(decision="符合", reason="r", confidence=0.9, tok_input=10, tok_output=20)

        audit = _build_corrective_audit(loop)
        assert audit["triggered"] is True
        assert audit["reason"] == "missing_rules"
        assert audit["merged_chunk_ids"] == ["c1"]
        assert audit["round2_tok_input"] == 10
        assert audit["round2_tok_output"] == 20


# ---------------------------------------------------------------------------
# 回环 SSE 流式标签
# ---------------------------------------------------------------------------


class TestCorrectiveLoopStream:
    @pytest.mark.asyncio
    async def test_execute_stream_corrective_labels(self, monkeypatch):
        """流式输出包含 re_retrieval / re_generation / re_judge 阶段标签。"""
        monkeypatch.setattr(settings, "RULE_REVIEW_CORRECTIVE_ENABLED", True)
        first_rr = RetrieveResult(results=[_make_hybrid_result("c1", "规则A")], bm25_hits=1, vector_hits=1)
        second_rr = RetrieveResult(results=[_make_hybrid_result("c2", "规则B")], bm25_hits=1, vector_hits=1)
        llm1 = LLMOutput(decision="不符合", reason="首轮", confidence=0.9)
        llm2 = LLMOutput(decision="不符合", reason="第二轮", confidence=0.95)

        pipeline, _, _, _ = _make_pipeline([first_rr, second_rr], [llm1, llm2])

        with patch(
            "src.rule_review.judge.verify_with_fallback",
            side_effect=[_make_round1_judged(), _make_round2_judged()],
        ):
            events = []
            async for sse_line in pipeline.execute_stream(_make_request(stream=True)):
                events.append(sse_line)

        full_text = "".join(events)
        assert "补充检索中" in full_text
        assert "重新推理中" in full_text
        assert "二次校验中" in full_text
        assert "re_retrieval" in full_text
        assert "done" in events[-1]

    @pytest.mark.asyncio
    async def test_execute_stream_second_retrieve_empty_no_labels(self, monkeypatch):
        """次轮检索为空时不出 re_generation/re_judge 标签。"""
        monkeypatch.setattr(settings, "RULE_REVIEW_CORRECTIVE_ENABLED", True)
        first_rr = RetrieveResult(results=[_make_hybrid_result("c1", "规则A")], bm25_hits=1, vector_hits=1)
        second_rr = RetrieveResult(not_found=True)
        llm1 = LLMOutput(decision="不符合", reason="首轮", confidence=0.9)

        pipeline, _, _, _ = _make_pipeline([first_rr, second_rr], [llm1])

        with patch(
            "src.rule_review.judge.verify_with_fallback",
            return_value=_make_round1_judged(),
        ):
            events = []
            async for sse_line in pipeline.execute_stream(_make_request(stream=True)):
                events.append(sse_line)

        full_text = "".join(events)
        assert "补充检索中" in full_text
        assert "重新推理中" not in full_text
        assert "done" in events[-1]
