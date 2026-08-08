"""
规则审查系统 RAGAS 风格评测指标单元测试

覆盖 src/rule_review/llm_judge_metrics.py 的：
- 四指标计算（Faithfulness / Answer Relevancy / Context Precision / Context Recall）
- 编排器 compute_ragas_metrics 的降级与错误隔离
- EvalRunner 集成（字段回填、报告聚合、save/load 往返）
- TestCase reference_answer 字段往返
- CLI 入口 main()

测试全程使用 MockJudgeModel 响应队列，不触达真实 LLM。
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage

from src.rule_review.evaluation import (
    EvalReport,
    EvalRunner,
    TestCase,
    TestCaseManager,
)
from src.rule_review.llm_judge_metrics import (
    RagasMetrics,
    compute_answer_relevancy,
    compute_context_precision,
    compute_context_recall,
    compute_faithfulness,
    compute_ragas_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockJudgeModel:
    """响应队列 mock：按调用顺序返回预置 JSON 响应。"""

    temperature = 0.0

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.call_count = 0

    async def ainvoke(self, messages, **kwargs):
        self.call_count += 1
        if not self._responses:
            raise RuntimeError("mock 响应队列已空")
        return AIMessage(content=self._responses.pop(0))


def _chunks(texts: list[str]) -> list[dict]:
    return [
        {"chunk_id": f"c{i}", "source": f"规则{i}.pdf", "section": f"第{i}章", "text": t}
        for i, t in enumerate(texts)
    ]


def _metrics_response(scores: list[str]) -> MockJudgeModel:
    return MockJudgeModel(scores)


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------


class TestComputeFaithfulness:
    def test_all_claims_supported(self):
        model = _metrics_response([
            '{"claims": [{"claim": "c1", "supported": true, "reason": "r"}, '
            '{"claim": "c2", "supported": true, "reason": "r"}]}'
        ])
        score, details = self._run(model)
        assert score == 1.0
        assert details["supported_claims"] == 2

    def test_all_claims_unsupported(self):
        model = _metrics_response([
            '{"claims": [{"claim": "c1", "supported": false, "reason": "r"}]}'
        ])
        score, _ = self._run(model)
        assert score == 0.0

    def test_partial_support(self):
        model = _metrics_response([
            '{"claims": [{"claim": "a", "supported": true, "reason": "r"}, '
            '{"claim": "b", "supported": true, "reason": "r"}, '
            '{"claim": "c", "supported": false, "reason": "r"}]}'
        ])
        score, details = self._run(model)
        assert score == pytest.approx(2 / 3)
        assert details["total_claims"] == 3

    def test_no_claims_returns_none(self):
        model = _metrics_response(['{"claims": []}'])
        score, details = self._run(model)
        assert score is None
        assert "未抽取到任何 claim" in details["reason"]

    def test_empty_context_returns_none(self):
        score, details = self._run(None, context=[])
        assert score is None
        assert "上下文" in details["skipped_reason"]

    def test_model_exception_returns_none(self):
        model = MockJudgeModel([])  # 空队列 → ainvoke 抛异常
        score, details = self._run(model)
        assert score is None
        assert details["error"]

    def test_non_json_output_returns_none(self):
        model = _metrics_response(["这不是 JSON"])
        score, details = self._run(model)
        assert score is None
        assert details["error"]

    def _run(self, model, context=None):
        import asyncio

        async def run():
            return await compute_faithfulness(
                "Q", "答",
                context if context is not None else _chunks(["规则文本"]),
                model=model,
            )
        return asyncio.run(run())


# ---------------------------------------------------------------------------
# Answer Relevancy
# ---------------------------------------------------------------------------


class TestComputeAnswerRelevancy:
    def test_normal_score(self):
        model = _metrics_response(['{"score": 0.8, "reason": "切题"}'])
        score, details = self._run(model)
        assert score == 0.8
        assert details["reason"] == "切题"

    def test_score_above_1_clamped(self):
        model = _metrics_response(['{"score": 1.5, "reason": "r"}'])
        score, _ = self._run(model)
        assert score == 1.0

    def test_score_below_0_clamped(self):
        model = _metrics_response(['{"score": -0.1, "reason": "r"}'])
        score, _ = self._run(model)
        assert score == 0.0

    def test_unparseable_score_returns_none(self):
        model = _metrics_response(['{"score": "高", "reason": "r"}'])
        score, details = self._run(model)
        assert score is None
        assert details["error"]

    def test_empty_answer_returns_none(self):
        score, details = self._run(None, answer="")
        assert score is None
        assert "回答为空" in details["skipped_reason"]

    def test_parse_failure_returns_none(self):
        model = _metrics_response(["无法解析的输出"])
        score, details = self._run(model)
        assert score is None

    def _run(self, model, answer="答"):
        import asyncio

        async def run():
            return await compute_answer_relevancy("Q", answer, model=model)
        return asyncio.run(run())


# ---------------------------------------------------------------------------
# Context Precision
# ---------------------------------------------------------------------------


class TestComputeContextPrecision:
    def test_relevant_first_rank(self):
        # [rel, nonrel] → P@1=1, P@2=0.5 → (1*1 + 0.5*0) / 1 = 1.0
        model = _metrics_response([
            '{"verdicts": [{"index": 0, "relevant": true, "reason": "r"}, '
            '{"index": 1, "relevant": false, "reason": "r"}]}'
        ])
        score, details = self._run(model)
        assert score == 1.0
        assert details["relevant_count"] == 1

    def test_relevant_second_rank(self):
        # [nonrel, rel] → P@2=0.5 → 0.5 / 1 = 0.5
        model = _metrics_response([
            '{"verdicts": [{"index": 0, "relevant": false, "reason": "r"}, '
            '{"index": 1, "relevant": true, "reason": "r"}]}'
        ])
        score, _ = self._run(model)
        assert score == pytest.approx(0.5)

    def test_all_relevant(self):
        model = _metrics_response([
            '{"verdicts": [{"index": 0, "relevant": true, "reason": "r"}, '
            '{"index": 1, "relevant": true, "reason": "r"}]}'
        ])
        score, _ = self._run(model)
        assert score == 1.0

    def test_none_relevant(self):
        model = _metrics_response([
            '{"verdicts": [{"index": 0, "relevant": false, "reason": "r"}]}'
        ])
        score, _ = self._run(model)
        assert score == 0.0

    def test_empty_chunks_returns_zero(self):
        score, _ = self._run(None, chunks=[])
        assert score == 0.0

    def test_missing_reference_returns_none(self):
        score, details = self._run(None, reference="")
        assert score is None
        assert "reference_answer" in details["skipped_reason"]

    def _run(self, model, chunks=None, reference="参考要点"):
        import asyncio

        async def run():
            return await compute_context_precision(
                "Q", reference,
                chunks if chunks is not None else _chunks(["c1", "c2"]),
                model=model,
            )
        return asyncio.run(run())


# ---------------------------------------------------------------------------
# Context Recall
# ---------------------------------------------------------------------------


class TestComputeContextRecall:
    def test_all_sentences_supported(self):
        model = _metrics_response([
            '{"sentences": [{"sentence": "s1", "supported": true, "reason": "r"}, '
            '{"sentence": "s2", "supported": true, "reason": "r"}]}'
        ])
        score, details = self._run(model)
        assert score == 1.0
        assert details["supported_sentences"] == 2

    def test_half_supported(self):
        model = _metrics_response([
            '{"sentences": [{"sentence": "s1", "supported": true, "reason": "r"}, '
            '{"sentence": "s2", "supported": false, "reason": "r"}]}'
        ])
        score, _ = self._run(model)
        assert score == pytest.approx(0.5)

    def test_none_supported(self):
        model = _metrics_response([
            '{"sentences": [{"sentence": "s1", "supported": false, "reason": "r"}]}'
        ])
        score, _ = self._run(model)
        assert score == 0.0

    def test_empty_reference_returns_none(self):
        score, details = self._run(None, reference="")
        assert score is None
        assert "reference_answer" in details["skipped_reason"]

    def test_empty_chunks_returns_zero(self):
        score, _ = self._run(None, chunks=[])
        assert score == 0.0

    def test_judgments_aligned_by_order(self):
        # 模型少给一条判定 → 按顺序对齐,缺的按不支持计
        model = _metrics_response([
            '{"sentences": [{"sentence": "s1", "supported": true, "reason": "r"}]}'
        ])
        score, details = self._run(model, reference="句一。句二。")
        assert score == pytest.approx(0.5)

    def _run(self, model, chunks=None, reference="参考句。第二句。"):
        import asyncio

        async def run():
            return await compute_context_recall(
                "Q", reference,
                chunks if chunks is not None else _chunks(["规则文本"]),
                model=model,
            )
        return asyncio.run(run())


# ---------------------------------------------------------------------------
# 编排器 compute_ragas_metrics
# ---------------------------------------------------------------------------


class TestComputeRagasMetricsOrchestrator:
    FULL_RESPONSES = [
        '{"claims": [{"claim": "c1", "supported": true, "reason": "r"}]}',
        '{"score": 0.8, "reason": "ok"}',
        '{"verdicts": [{"index": 0, "relevant": true, "reason": "r"}, '
        '{"index": 1, "relevant": false, "reason": "r"}]}',
        '{"sentences": [{"sentence": "s1", "supported": true, "reason": "r"}]}',
    ]

    def test_full_flow(self):
        model = MockJudgeModel(self.FULL_RESPONSES)
        ragas = self._run(model, chunks=_chunks(["c1", "c2"]))
        assert ragas.faithfulness == 1.0
        assert ragas.answer_relevancy == 0.8
        assert ragas.context_precision == 1.0
        assert ragas.context_recall == 1.0
        assert model.call_count == 4
        assert "judge_latency_ms" in ragas.details

    def test_missing_reference_skips_two_metrics(self):
        model = MockJudgeModel(self.FULL_RESPONSES[:2])
        ragas = self._run(model, reference="", chunks=_chunks(["c1"]))
        assert ragas.faithfulness == 1.0
        assert ragas.answer_relevancy == 0.8
        assert ragas.context_precision is None
        assert ragas.context_recall is None
        assert model.call_count == 2  # 未发 precision/recall 调用

    def test_empty_answer_skips_all(self):
        ragas = self._run(MockJudgeModel([]), answer="")
        assert ragas.skipped is True
        assert "回答为空" in ragas.skip_reason
        assert ragas.faithfulness is None
        assert ragas.answer_relevancy is None

    def test_one_metric_failure_isolated(self):
        # relevancy 输出不可解析 → 仅该项为 None,其余正常
        responses = list(self.FULL_RESPONSES)
        responses[1] = "无法解析"
        model = MockJudgeModel(responses)
        ragas = self._run(model, chunks=_chunks(["c1", "c2"]))
        assert ragas.faithfulness == 1.0
        assert ragas.answer_relevancy is None  # 输出不可解析 → 静默降级
        assert ragas.context_precision == 1.0
        assert ragas.context_recall == 1.0
        assert ragas.errors == []  # 解析失败不算异常,不污染 errors

    def test_empty_context_only_relevancy_computed(self):
        model = MockJudgeModel([
            '{"score": 0.8, "reason": "ok"}',
        ])
        ragas = self._run(model, chunks=[])
        assert ragas.faithfulness is None
        assert ragas.answer_relevancy == 0.8
        assert ragas.context_precision == 0.0
        assert ragas.context_recall == 0.0

    def _run(self, model, chunks=None, reference="参考答案。", answer="答"):
        import asyncio

        async def run():
            return await compute_ragas_metrics(
                "Q", answer,
                chunks if chunks is not None else _chunks(["c1"]),
                reference_answer=reference,
                model=model,
            )
        return asyncio.run(run())


# ---------------------------------------------------------------------------
# EvalRunner 集成
# ---------------------------------------------------------------------------


class _FakePipelineWithStages:
    """返回固定 result/stages 的假 pipeline（含 retrieved_chunks_full）。"""

    def __init__(self, result: dict):
        self._result = result

    async def execute(self, req):
        return self._result


def _make_stage_result(
    decision: str,
    reason: str,
    evidence_texts: list[str],
    retrieved_chunks_full: list[dict] | None,
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
                "retrieved_chunks_full": retrieved_chunks_full if retrieved_chunks_full is not None else [],
            }
        ],
    }


class TestEvalRunnerRagasIntegration:
    def test_ragas_fields_backfilled(self):
        """1 条用例 4 响应 → EvalMetrics 字段与报告均值正确。"""
        case = TestCase(
            id="tc-ragas",
            question="2025年3月15日冀北电价800元/MWh是否符合上限",
            expected_decision="不符合",
            reference_answer="出清电价上限为760元/MWh，800元/MWh超过上限。",
        )
        result = _make_stage_result(
            decision="不符合",
            reason="实际电价800元/MWh超过上限760元/MWh",
            evidence_texts=["省间日前现货出清电价上限为760元/MWh。"],
            retrieved_chunks_full=[
                {"chunk_id": "c1", "source": "测试规则.pdf", "section": "第2条", "text": "省间日前现货出清电价上限为760元/MWh。"},
            ],
        )

        async def _fake_execute(req):
            return result

        pipeline = _FakePipelineWithStages.__new__(_FakePipelineWithStages)
        pipeline.execute = _fake_execute

        model = MockJudgeModel([
            '{"claims": [{"claim": "超限", "supported": true, "reason": "r"}]}',
            '{"score": 0.9, "reason": "ok"}',
            '{"verdicts": [{"index": 0, "relevant": true, "reason": "r"}]}',
            '{"sentences": [{"sentence": "s1", "supported": true, "reason": "r"}]}',
        ])
        runner = EvalRunner(pipeline, judge_model=model)
        import asyncio
        report: EvalReport = asyncio.run(runner.run([case], top_k=10))

        assert report.success_cases == 1
        assert report.ragas_metrics_count == 1
        assert report.avg_faithfulness == 1.0
        assert report.avg_answer_relevancy == 0.9
        assert report.avg_context_precision == 1.0
        assert report.avg_context_recall == 1.0

        m = report.details[0]
        assert m.faithfulness == 1.0
        assert m.answer_relevancy == 0.9
        assert m.ragas_skipped is False
        assert m.ragas_details["judge_latency_ms"] is not None

    def test_judge_exception_degrades_to_none(self):
        """judge 全程异常 → 指标 None、报告不崩、avg 为 None。"""
        case = TestCase(
            id="tc-fail",
            question="Q",
            reference_answer="参考",
        )
        result = _make_stage_result(
            decision="不符合",
            reason="reason",
            evidence_texts=["证据"],
            retrieved_chunks_full=[{"chunk_id": "c1", "text": "规则文本"}],
        )

        async def _fake_execute(req):
            return result

        pipeline = _FakePipelineWithStages.__new__(_FakePipelineWithStages)
        pipeline.execute = _fake_execute

        model = MockJudgeModel([])  # 空队列 → 每项指标调用都抛异常
        runner = EvalRunner(pipeline, judge_model=model)
        import asyncio
        report = asyncio.run(runner.run([case], top_k=10))

        m = report.details[0]
        assert m.faithfulness is None
        assert m.answer_relevancy is None
        assert m.context_precision is None
        assert m.context_recall is None
        assert report.avg_faithfulness is None
        assert report.avg_context_recall is None
        assert report.ragas_metrics_count == 0

    def test_missing_reference_avg_is_none(self):
        """用例无 reference_answer → precision/recall 均值为 None,其余正常。"""
        case = TestCase(
            id="tc-noref",
            question="Q",
            expected_decision="不符合",
        )
        result = _make_stage_result(
            decision="不符合",
            reason="reason",
            evidence_texts=["证据"],
            retrieved_chunks_full=[{"chunk_id": "c1", "text": "规则文本"}],
        )

        async def _fake_execute(req):
            return result

        pipeline = _FakePipelineWithStages.__new__(_FakePipelineWithStages)
        pipeline.execute = _fake_execute

        model = MockJudgeModel([
            '{"claims": [{"claim": "c", "supported": true, "reason": "r"}]}',
            '{"score": 0.7, "reason": "ok"}',
        ])
        runner = EvalRunner(pipeline, judge_model=model)
        import asyncio
        report = asyncio.run(runner.run([case], top_k=10))

        assert report.avg_faithfulness == 1.0
        assert report.avg_answer_relevancy == 0.7
        assert report.avg_context_precision is None
        assert report.avg_context_recall is None
        assert report.ragas_metrics_count == 1

    def test_ragas_disabled_flag(self):
        """ragas_enabled=False → 不调用模型,指标保持默认 None。"""
        case = TestCase(id="tc-off", question="Q")
        result = _make_stage_result(
            decision="不符合",
            reason="reason",
            evidence_texts=[],
            retrieved_chunks_full=[{"chunk_id": "c1", "text": "规则文本"}],
        )

        async def _fake_execute(req):
            return result

        pipeline = _FakePipelineWithStages.__new__(_FakePipelineWithStages)
        pipeline.execute = _fake_execute

        model = MockJudgeModel([])
        runner = EvalRunner(pipeline, judge_model=model, ragas_enabled=False)
        import asyncio
        report = asyncio.run(runner.run([case], top_k=10))

        m = report.details[0]
        assert m.faithfulness is None
        assert m.ragas_skipped is True
        assert "已关闭" in m.ragas_skip_reason
        assert model.call_count == 0

    def test_save_load_report_roundtrip(self, tmp_path):
        """save_report → load_report 新字段完整往返。"""
        case = TestCase(id="tc-round", question="Q")
        result = _make_stage_result(
            decision="不符合",
            reason="reason",
            evidence_texts=[],
            retrieved_chunks_full=[{"chunk_id": "c1", "text": "规则文本"}],
        )

        async def _fake_execute(req):
            return result

        pipeline = _FakePipelineWithStages.__new__(_FakePipelineWithStages)
        pipeline.execute = _fake_execute

        model = MockJudgeModel([
            '{"claims": [{"claim": "c", "supported": true, "reason": "r"}]}',
            '{"score": 0.6, "reason": "ok"}',
        ])
        runner = EvalRunner(pipeline, judge_model=model)
        import asyncio
        report = asyncio.run(runner.run([case], top_k=10))
        report.timestamp = "2026-08-08T00:00:00Z"

        path = tmp_path / "report.json"
        EvalRunner.save_report(report, str(path))
        loaded = EvalRunner.load_report(str(path))

        assert loaded.avg_faithfulness == 1.0
        assert loaded.avg_answer_relevancy == 0.6
        assert loaded.ragas_metrics_count == 1
        assert loaded.details[0].ragas_details["judge_latency_ms"] is not None

    def test_runner_defaults(self):
        """默认构造:ragas_enabled=True、judge_model=None。"""
        runner = EvalRunner(None)
        assert runner._ragas_enabled is True
        assert runner._judge_model is None


# ---------------------------------------------------------------------------
# TestCase reference_answer
# ---------------------------------------------------------------------------


class TestTestCaseReferenceAnswer:
    def test_save_load_roundtrip(self, tmp_path):
        path = tmp_path / "cases.json"
        manager = TestCaseManager(str(path))
        manager.save([
            TestCase(id="tc-1", question="Q", reference_answer="参考答案内容。"),
        ])
        loaded = TestCaseManager(str(path)).load()
        assert loaded[0].reference_answer == "参考答案内容。"

    def test_legacy_json_without_field_loads_empty(self, tmp_path):
        path = tmp_path / "cases.json"
        path.write_text(
            json.dumps([{"id": "tc-1", "question": "Q"}], ensure_ascii=False),
            encoding="utf-8",
        )
        loaded = TestCaseManager(str(path)).load()
        assert loaded[0].reference_answer == ""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCliMain:
    def test_main_runs_and_saves_report(self, tmp_path, monkeypatch):
        """CLI 全流程:mock pipeline 与 runner,断言报告文件产出与返回码。"""
        import sys

        import src.rule_review.evaluation as ev

        cases = [TestCase(id="tc-cli", question="Q")]
        cases_path = tmp_path / "cases.json"
        ev.TestCaseManager(str(cases_path)).save(cases)
        report_path = tmp_path / "eval.json"

        # mock get_default_pipeline（main() 内从 pipeline 模块导入）→ 固定返回的假 pipeline
        fake_pipeline = _FakePipelineWithStages(_make_stage_result(
            decision="不符合",
            reason="reason",
            evidence_texts=[],
            retrieved_chunks_full=[{"chunk_id": "c1", "text": "规则文本"}],
        ))
        import src.rule_review.pipeline as rrp
        monkeypatch.setattr(rrp, "get_default_pipeline", lambda: fake_pipeline)

        # mock 评测模型:直接替换 EvalRunner 构造后的 judge_model
        original_init = ev.EvalRunner.__init__

        def patched_init(self, pipeline=None, judge_model=None, ragas_enabled=True):
            original_init(self, pipeline, judge_model=MockJudgeModel([
                '{"claims": [{"claim": "c", "supported": true, "reason": "r"}]}',
                '{"score": 0.8, "reason": "ok"}',
            ]), ragas_enabled=ragas_enabled)
        monkeypatch.setattr(ev.EvalRunner, "__init__", patched_init)

        rc = ev.main([
            "--cases", str(cases_path),
            "--report", str(report_path),
            "--no-ragas",
        ])
        assert rc == 0
        assert report_path.exists()

        data = json.loads(report_path.read_text(encoding="utf-8"))
        assert data["total_cases"] == 1
        assert data["decision_accuracy"] == 1.0

    def test_main_empty_cases_returns_1(self, tmp_path, monkeypatch):
        import src.rule_review.evaluation as ev

        empty_path = tmp_path / "empty.json"
        empty_path.write_text("[]", encoding="utf-8")

        rc = ev.main(["--cases", str(empty_path)])
        assert rc == 1
