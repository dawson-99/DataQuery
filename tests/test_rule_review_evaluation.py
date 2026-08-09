"""
规则审查系统评估体系单元测试

覆盖 src/rule_review/evaluation.py 的：
- recall@k / MRR 检索层指标计算
- 幻觉检测空检索 bug 修复（核心回归）
- EvalRunner 批量评估（mock pipeline）
- TestCaseManager 测试集管理
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from src.rule_review.evaluation import (
    EvalReport,
    EvalRunner,
    TestCase,
    TestCaseManager,
    compute_mrr,
    compute_recall_at_k,
    detect_hallucination,
)


# ---------------------------------------------------------------------------
# 检索层指标计算
# ---------------------------------------------------------------------------


class TestComputeRetrievalMetrics:
    def _chunks(self, texts: list[str]) -> list[dict]:
        return [{"chunk_id": f"c{i}", "text": t} for i, t in enumerate(texts)]

    def test_recall_hit_in_top_k(self):
        chunks = self._chunks(["价格上限为760元", "无关文本"])
        assert compute_recall_at_k(chunks, ["760"]) == 1.0

    def test_recall_miss(self):
        chunks = self._chunks(["无关文本A", "无关文本B"])
        assert compute_recall_at_k(chunks, ["760"]) == 0.0

    def test_recall_k_truncation(self):
        # 命中在 k 之外 → 0
        chunks = self._chunks(["无关A", "无关B", "价格上限为760元"])
        assert compute_recall_at_k(chunks, ["760"], k=2) == 0.0
        assert compute_recall_at_k(chunks, ["760"], k=3) == 1.0

    def test_recall_empty_inputs(self):
        assert compute_recall_at_k([], ["760"]) == 0.0
        assert compute_recall_at_k(self._chunks(["760元"]), []) == 0.0

    def test_recall_any_keyword_hits(self):
        chunks = self._chunks(["申报上限120%", "无关"])
        assert compute_recall_at_k(chunks, ["760", "120%"]) == 1.0

    def test_mrr_rank1(self):
        chunks = self._chunks(["价格上限为760元", "无关"])
        assert compute_mrr(chunks, ["760"]) == 1.0

    def test_mrr_rank2(self):
        chunks = self._chunks(["无关A", "价格上限为760元"])
        assert compute_mrr(chunks, ["760"]) == pytest.approx(0.5)

    def test_mrr_miss(self):
        chunks = self._chunks(["无关A", "无关B"])
        assert compute_mrr(chunks, ["760"]) == 0.0

    def test_mrr_k_truncation(self):
        chunks = self._chunks(["无关A", "价格上限为760元"])
        assert compute_mrr(chunks, ["760"], k=1) == 0.0

    def test_mrr_empty_inputs(self):
        assert compute_mrr([], ["760"]) == 0.0
        assert compute_mrr(self._chunks(["760元"]), []) == 0.0


# ---------------------------------------------------------------------------
# 幻觉检测修复回归（核心）
# ---------------------------------------------------------------------------


class TestHallucinationDetectionFix:
    def test_evidence_found_in_retrieved_not_hallucination(self):
        has_hallu, hallu_texts = detect_hallucination(
            ["省间日前现货出清电价上限为760元/MWh。"],
            ["省间日前现货出清电价上限为760元/MWh。", "其他条款"],
        )
        assert has_hallu is False
        assert hallu_texts == []

    def test_evidence_not_in_retrieved_is_hallucination(self):
        has_hallu, hallu_texts = detect_hallucination(
            ["完全虚构的条款内容不存在于任何文档"],
            ["检索到的真实条款内容"],
        )
        assert has_hallu is True
        assert len(hallu_texts) == 1

    def test_empty_retrieved_all_flagged_as_hallucination(self):
        # 修复前：空检索集 → 所有 evidence 误判为幻觉
        has_hallu, hallu_texts = detect_hallucination(
            ["真实条款内容"], []
        )
        assert has_hallu is True
        assert len(hallu_texts) == 1


class _FakePipelineWithStages:
    """返回固定 result/stages 的假 pipeline。"""

    def __init__(self, result: dict):
        self._result = result

    async def execute(self, req):
        return self._result


def _make_stage_result(
    decision: str,
    reason: str,
    evidence_texts: list[str],
    retrieved_chunks: list[dict] | None,
    expected_decision: str = "",
) -> dict:
    return {
        "query_id": "q_test",
        "result": {
            "decision": decision,
            "reason": reason,
            "evidence": [{"source": "测试规则.pdf", "text": t} for t in evidence_texts],
            "confidence": 0.9,
        },
        "stages": [
            {
                "stage": "retrieval",
                "retrieved_chunks": retrieved_chunks if retrieved_chunks is not None else [],
            }
        ],
    }


class TestEvalRunnerWithMockPipeline:
    @pytest.mark.asyncio
    async def test_run_with_retrieved_chunks(self):
        """带检索结果时:幻觉检测基于真实检索文本,recall@k/MRR 正常计算。"""
        cases = [
            TestCase(
                id="tc-a",
                question="2025年3月15日冀北电价800元/MWh是否符合上限",
                expected_decision="不符合",
                expected_keywords=["760"],
                difficulty="easy",
            ),
            TestCase(
                id="tc-b",
                question="某申报是否合规",
                expected_decision="符合",
                expected_keywords=["不存在关键词"],
                difficulty="hard",
            ),
        ]

        # tc-a: 命中检索、无幻觉;tc-b: 未命中检索、evidence 为虚构 → 幻觉
        results = {
            "2025年3月15日冀北电价800元/MWh是否符合上限": _make_stage_result(
                decision="不符合",
                reason="实际电价800元/MWh超过上限760元/MWh",
                evidence_texts=["省间日前现货出清电价上限为760元/MWh。"],
                retrieved_chunks=[
                    {"chunk_id": "c1", "source": "测试规则.pdf", "text": "省间日前现货出清电价上限为760元/MWh。"},
                ],
            ),
            "某申报是否合规": _make_stage_result(
                decision="不符合",
                reason="推断",
                evidence_texts=["完全虚构的处罚结论甲"],
                retrieved_chunks=[
                    {"chunk_id": "c2", "source": "测试规则.pdf", "text": "检索到的真实条款"},
                ],
            ),
        }

        async def _fake_execute(req):
            return results[req.question]

        pipeline = _FakePipelineWithStages.__new__(_FakePipelineWithStages)
        pipeline.execute = _fake_execute

        runner = EvalRunner(pipeline)
        report: EvalReport = await runner.run(cases, top_k=10)

        assert report.total_cases == 2
        assert report.success_cases == 2
        assert report.error_cases == 0
        # tc-a 决策匹配, tc-b 不匹配 → 准确率 0.5
        assert report.decision_accuracy == 0.5
        # tc-a recall@k=1, tc-b=0 → 平均 0.5;MRR 同理
        assert report.avg_recall_at_k == 0.5
        assert report.avg_mrr == 0.5
        # 幻觉率: tc-b 1 条幻觉 → 0.5
        assert report.hallucination_rate == 0.5

        # 按难度分组
        assert report.by_difficulty["easy"]["count"] == 1
        assert report.by_difficulty["hard"]["count"] == 1
        assert report.by_difficulty["easy"]["decision_accuracy"] == 1.0

        # 逐条详情
        m_a = next(m for m in report.details if m.case_id == "tc-a")
        assert m_a.recall_at_k == 1.0
        assert m_a.mrr == 1.0
        assert m_a.has_hallucination is False
        assert m_a.hallucination_check_skipped is False

    @pytest.mark.asyncio
    async def test_run_without_retrieved_chunks_skips_hallucination(self):
        """旧版 stages 无 retrieved_chunks(空列表):不误判为幻觉,标记跳过(核心回归)。"""
        cases = [
            TestCase(
                id="tc-legacy",
                question="旧版返回结构",
                expected_decision="不符合",
                expected_keywords=["760"],
                difficulty="easy",
            ),
        ]

        result = _make_stage_result(
            decision="不符合",
            reason="实际电价800元/MWh超过上限760元/MWh",
            evidence_texts=["省间日前现货出清电价上限为760元/MWh。"],
            retrieved_chunks=None,
        )

        async def _fake_execute(req):
            return result

        pipeline = _FakePipelineWithStages.__new__(_FakePipelineWithStages)
        pipeline.execute = _fake_execute

        runner = EvalRunner(pipeline)
        report: EvalReport = await runner.run(cases, top_k=10)

        # 修复前:空检索集会把 evidence 误判为幻觉;修复后:跳过检测且不误报
        m = report.details[0]
        assert m.has_hallucination is False
        assert m.hallucination_check_skipped is True
        assert report.hallucination_rate == 0.0

    @pytest.mark.asyncio
    async def test_run_without_pipeline_raises(self):
        runner = EvalRunner(None)
        with pytest.raises(RuntimeError):
            await runner.run([], top_k=10)

    @pytest.mark.asyncio
    async def test_run_prefers_re_retrieval_chunks_for_hallucination(self):
        """Corrective 回环：幻觉检测应基于末次（合并后）检索结果而非首轮。"""
        cases = [
            TestCase(
                id="tc-corrective",
                question="回环场景",
                expected_decision="不符合",
                expected_keywords=["760"],
                difficulty="easy",
            ),
        ]

        result = {
            "query_id": "q_corrective",
            "result": {
                "decision": "不符合",
                "reason": "修正后：电价800元/MWh超过上限760元/MWh",
                "evidence": [
                    {"source": "测试规则.pdf", "text": "省间日前现货出清电价上限为760元/MWh。"},
                    {"source": "测试规则.pdf", "text": "申报电价不得高于限价申报标准。"},
                ],
                "confidence": 0.95,
            },
            "stages": [
                # 首轮检索：证据缺失（旧检索）
                {
                    "stage": "retrieval",
                    "retrieved_chunks": [
                        {"chunk_id": "c1", "source": "测试规则.pdf", "text": "不相关内容"},
                    ],
                },
                # 回环二次检索：合并后证据完整
                {
                    "stage": "re_retrieval",
                    "triggered_by": "missing_rules",
                    "result_count": 2,
                    "retrieved_chunks": [
                        {"chunk_id": "c1", "source": "测试规则.pdf", "text": "不相关内容"},
                        {"chunk_id": "c2", "source": "测试规则.pdf", "text": "省间日前现货出清电价上限为760元/MWh。"},
                        {"chunk_id": "c3", "source": "测试规则.pdf", "text": "申报电价不得高于限价申报标准。"},
                    ],
                },
            ],
        }

        async def _fake_execute(req):
            return result

        pipeline = _FakePipelineWithStages.__new__(_FakePipelineWithStages)
        pipeline.execute = _fake_execute

        runner = EvalRunner(pipeline)
        report: EvalReport = await runner.run(cases, top_k=10)

        m = report.details[0]
        # 修复前：只取首轮 retrieval → 证据全被误判为幻觉；
        # 修复后：取末次 re_retrieval（合并后）→ 无幻觉
        assert m.has_hallucination is False
        assert m.hallucination_check_skipped is False
        # recall@k 基于合并后 chunks
        assert m.recall_at_k == 1.0

    @pytest.mark.asyncio
    async def test_run_prefers_re_retrieval_full_chunks_for_ragas(self):
        """Corrective 回环：RAGAS 上下文应取末次（合并后）完整文本。"""
        cases = [
            TestCase(
                id="tc-ragas-corrective",
                question="回环场景",
                expected_decision="不符合",
                expected_keywords=["760"],
                difficulty="easy",
            ),
        ]

        result = {
            "query_id": "q_ragas",
            "result": {
                "decision": "不符合",
                "reason": "电价800元/MWh超过上限",
                "evidence": [
                    {"source": "测试规则.pdf", "text": "省间日前现货出清电价上限为760元/MWh。"},
                ],
                "confidence": 0.9,
            },
            "stages": [
                {
                    "stage": "retrieval",
                    "retrieved_chunks_full": [
                        {"chunk_id": "c1", "source": "x", "text": "旧内容"},
                    ],
                },
                {
                    "stage": "re_retrieval",
                    "triggered_by": "hallucinated",
                    "result_count": 2,
                    "retrieved_chunks_full": [
                        {"chunk_id": "c1", "source": "x", "text": "旧内容"},
                        {"chunk_id": "c2", "source": "测试规则.pdf", "text": "省间日前现货出清电价上限为760元/MWh。"},
                    ],
                },
            ],
        }

        async def _fake_execute(req):
            return result

        pipeline = _FakePipelineWithStages.__new__(_FakePipelineWithStages)
        pipeline.execute = _fake_execute

        from src.rule_review.llm_judge_metrics import RagasMetrics

        captured = {}

        async def _fake_ragas(question, answer_text, context_chunks, reference_answer="", model=None):
            captured["context_chunks"] = context_chunks
            return RagasMetrics()

        runner = EvalRunner(pipeline)
        with patch(
            "src.rule_review.evaluation.compute_ragas_metrics",
            side_effect=_fake_ragas,
        ):
            report: EvalReport = await runner.run(cases, top_k=10)

        m = report.details[0]
        assert m.has_hallucination is False
        # RAGAS 收到的是合并后（末次 re_retrieval）的完整 chunks，包含二次检索新增的 c2
        assert captured["context_chunks"] is not None
        chunk_ids = [c.get("chunk_id") for c in captured["context_chunks"]]
        assert "c2" in chunk_ids


# ---------------------------------------------------------------------------
# 测试集管理
# ---------------------------------------------------------------------------


class TestTestCaseManager:
    def test_save_load_roundtrip(self, tmp_path):
        path = tmp_path / "cases.json"
        manager = TestCaseManager(str(path))
        cases = [
            TestCase(
                id="tc-1",
                question="测试问题",
                expected_decision="符合",
                expected_keywords=["760"],
                documents=["规则.pdf"],
                tags=["价格上限"],
                difficulty="hard",
                expected_chunk_ids=["chunk-abc"],
            )
        ]
        manager.save(cases)

        loaded = TestCaseManager(str(path)).load()
        assert len(loaded) == 1
        assert loaded[0].id == "tc-1"
        assert loaded[0].expected_chunk_ids == ["chunk-abc"]
        assert loaded[0].difficulty == "hard"

    def test_load_missing_file_returns_empty(self, tmp_path):
        manager = TestCaseManager(str(tmp_path / "not_exists.json"))
        assert manager.load() == []

    def test_load_skips_invalid_case(self, tmp_path):
        path = tmp_path / "cases.json"
        path.write_text(
            json.dumps(
                [
                    {"id": "valid", "question": "有效用例"},
                    {"id": "invalid", "expected_decision": "缺 question 字段"},
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        cases = TestCaseManager(str(path)).load()
        assert len(cases) == 1
        assert cases[0].id == "valid"

    def test_stats(self, tmp_path):
        path = tmp_path / "cases.json"
        manager = TestCaseManager(str(path))
        manager.save(
            [
                TestCase(id="a", question="q1", expected_decision="符合", tags=["t1"], difficulty="easy"),
                TestCase(id="b", question="q2", expected_decision="不符合", tags=["t2"], difficulty="hard"),
            ]
        )
        stats = TestCaseManager(str(path)).stats()
        assert stats["total_cases"] == 2
        assert stats["by_difficulty"] == {"easy": 1, "hard": 1}
        assert stats["by_tag"] == {"t1": 1, "t2": 1}
