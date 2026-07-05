"""
规则审查系统 Judge 校验模块单元测试

覆盖 src/rule_review/judge.py 的：
- Judge 提示词构建
- Judge 正常校验流程（LLM 输出 → 校验 → 修正）
- 幻觉检测、逻辑自检、遗漏检查
- Judge 异常兜底（超时、API 错误、格式异常、重试）
- not_found 跳过
- Pipeline 集成函数 verify_with_fallback
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from src.rule_review.judge import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
    JudgeResult,
    RuleReviewJudge,
    get_default_judge,
    verify_with_fallback,
)
from src.rule_review.schemas import LLMOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm_output(
    decision="不符合",
    reason="实际电价800元/MWh超过上限760元/MWh",
    not_found=False,
) -> LLMOutput:
    return LLMOutput(
        decision=decision,
        reason=reason,
        evidence=[
            {
                "source": "测试规则.pdf",
                "section": "第2条 价格上限",
                "page": 2,
                "text": "省间日前现货出清电价上限为760元/MWh。",
            }
        ],
        confidence=0.95,
        not_found=not_found,
    )


def _make_judge_response(verified=True, corrections=None) -> str:
    """生成 Judge 的标准 JSON 输出。"""
    return json.dumps(
        {
            "verified": verified,
            "corrections": corrections or [],
            "hallucinated_evidence": [],
            "missing_rules": [],
            "final_decision": "不符合",
            "final_reason": "实际电价800元/MWh超过上限760元/MWh，确认判断正确。",
            "final_evidence": [
                {
                    "source": "测试规则.pdf",
                    "section": "第2条 价格上限",
                    "page": 2,
                    "text": "省间日前现货出清电价上限为760元/MWh。",
                }
            ],
            "confidence": 0.98,
        },
        ensure_ascii=False,
    )


def _make_invalid_response() -> str:
    return "这不是有效的 JSON 响应"


class MockJudgeModel:
    """可编程的 Mock 模型，用于测试各种 Judge 场景。"""

    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or [_make_judge_response()]
        self.call_count = 0

    async def ainvoke(self, messages, **kwargs) -> AIMessage:
        idx = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        return AIMessage(content=self.responses[idx])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_chunks() -> list[dict]:
    return [
        {
            "text": "省间日前现货出清电价上限为760元/MWh，各省按此标准执行。",
            "source": "测试规则.pdf",
            "section": "第2条 价格上限",
            "page": 2,
        },
        {
            "text": "违反价格上限将被处以罚款。",
            "source": "测试规则.pdf",
            "section": "第4条 违规处罚",
            "page": 4,
        },
    ]


@pytest.fixture
def llm_output() -> LLMOutput:
    return _make_llm_output()


# ---------------------------------------------------------------------------
# Judge 提示词测试
# ---------------------------------------------------------------------------


class TestJudgePrompts:
    def test_system_prompt_contains_key_rules(self):
        assert "电力交易规则审查结果校验专家" in JUDGE_SYSTEM_PROMPT
        assert "幻觉检测" in JUDGE_SYSTEM_PROMPT
        assert "逻辑自检" in JUDGE_SYSTEM_PROMPT
        assert "遗漏检查" in JUDGE_SYSTEM_PROMPT
        assert "verified" in JUDGE_SYSTEM_PROMPT
        assert "hallucinated_evidence" in JUDGE_SYSTEM_PROMPT

    def test_user_template_has_placeholders(self):
        assert "{original_query}" in JUDGE_USER_TEMPLATE
        assert "{context}" in JUDGE_USER_TEMPLATE
        assert "{llm_output}" in JUDGE_USER_TEMPLATE
        assert "{tool_logs}" in JUDGE_USER_TEMPLATE


# ---------------------------------------------------------------------------
# JudgeResult 数据模型测试
# ---------------------------------------------------------------------------


class TestJudgeResult:
    def test_default_values(self):
        r = JudgeResult()
        assert r.verified is False
        assert r.judge_skipped is False
        assert r.corrections == []
        assert r.hallucinated_evidence == []

    def test_skip_result(self):
        r = JudgeResult(judge_skipped=True, judge_skipped_reason="timeout")
        assert r.judge_skipped is True
        assert r.judge_skipped_reason == "timeout"


# ---------------------------------------------------------------------------
# RuleReviewJudge 核心流程测试
# ---------------------------------------------------------------------------


class TestRuleReviewJudge:
    @pytest.mark.asyncio
    async def test_verify_normal_flow(self, llm_output, sample_chunks):
        judge = RuleReviewJudge(model=MockJudgeModel())
        result = await judge.verify(llm_output, "测试问题", sample_chunks)

        assert isinstance(result, JudgeResult)
        assert result.verified is True
        assert result.final_decision == "不符合"
        assert result.confidence == 0.98
        assert result.judge_skipped is False

    @pytest.mark.asyncio
    async def test_verify_not_found_skips(self, sample_chunks):
        llm_out = _make_llm_output(not_found=True)
        judge = RuleReviewJudge(model=MockJudgeModel())
        result = await judge.verify(llm_out, "测试问题", sample_chunks)

        assert result.judge_skipped is True
        assert "not_found" in result.judge_skipped_reason.lower() or "文档" in result.judge_skipped_reason

    @pytest.mark.asyncio
    async def test_verify_retry_on_invalid_format(
        self, llm_output, sample_chunks
    ):
        """首次输出无效 JSON，重试后返回有效结果。"""
        model = MockJudgeModel([
            _make_invalid_response(),
            _make_judge_response(),
        ])
        judge = RuleReviewJudge(model=model)
        result = await judge.verify(llm_output, "测试问题", sample_chunks)

        assert result.verified is True
        assert model.call_count == 2

    @pytest.mark.asyncio
    async def test_verify_all_retries_fail(self, llm_output, sample_chunks):
        """两次都返回无效格式，应跳过。"""
        model = MockJudgeModel(["bad1", "bad2"])
        judge = RuleReviewJudge(model=model)
        result = await judge.verify(llm_output, "测试问题", sample_chunks)

        assert result.judge_skipped is True
        assert "格式" in result.judge_skipped_reason

    @pytest.mark.asyncio
    async def test_verify_timeout(self, llm_output, sample_chunks):
        """模拟超时场景。"""
        slow_model = MagicMock()
        async def _slow(*args, **kwargs):
            import asyncio
            await asyncio.sleep(999)  # won't actually sleep due to wait_for
            return AIMessage(content="{}")
        slow_model.ainvoke = _slow

        judge = RuleReviewJudge(model=slow_model)
        with patch("src.rule_review.judge._JUDGE_TIMEOUT", 0.001):
            result = await judge.verify(llm_output, "测试问题", sample_chunks)

        assert result.judge_skipped is True
        assert "超时" in result.judge_skipped_reason

    @pytest.mark.asyncio
    async def test_verify_api_error(self, llm_output, sample_chunks):
        """模拟 API 调用异常。"""
        bad_model = MagicMock()
        bad_model.ainvoke = AsyncMock(side_effect=RuntimeError("API 500"))

        judge = RuleReviewJudge(model=bad_model)
        result = await judge.verify(llm_output, "测试问题", sample_chunks)

        assert result.judge_skipped is True
        assert "API" in result.judge_skipped_reason or "RuntimeError" in result.judge_skipped_reason

    @pytest.mark.asyncio
    async def test_verify_with_hallucinations(self, llm_output, sample_chunks):
        """Judge 检测到幻觉时应有正确的输出。"""
        response = json.dumps(
            {
                "verified": False,
                "corrections": [],
                "hallucinated_evidence": [
                    {"index": 0, "reason": "无法在原文中找到对应文本"}
                ],
                "missing_rules": [],
                "final_decision": "无法判断",
                "final_reason": "证据不可靠，无法做出判断",
                "final_evidence": [],
                "confidence": 0.3,
            },
            ensure_ascii=False,
        )
        judge = RuleReviewJudge(model=MockJudgeModel([response]))
        result = await judge.verify(llm_output, "测试问题", sample_chunks)

        assert result.verified is False
        assert len(result.hallucinated_evidence) == 1
        assert result.final_decision == "无法判断"
        assert result.confidence == 0.3

    @pytest.mark.asyncio
    async def test_verify_with_tool_logs(self, llm_output, sample_chunks):
        """带工具日志的校验应正常工作。"""
        tool_logs = [
            {"tool": "extract_table_data", "result": {"value": 760}, "args": {}}
        ]
        judge = RuleReviewJudge(model=MockJudgeModel())
        result = await judge.verify(
            llm_output, "测试问题", sample_chunks, tool_logs=tool_logs
        )
        assert result.verified is True

    @pytest.mark.asyncio
    async def test_verify_with_corrections(self, llm_output, sample_chunks):
        """Judge 对结果有修正时返回 corrections。"""
        response = json.dumps(
            {
                "verified": True,
                "corrections": [
                    {
                        "field": "reason",
                        "issue": "理由不够详细",
                        "suggestion": "补充具体超出值",
                    }
                ],
                "hallucinated_evidence": [],
                "missing_rules": [],
                "final_decision": "不符合",
                "final_reason": "实际电价800元/MWh超出上限760元/MWh，超出40元/MWh",
                "final_evidence": [e.model_dump() if hasattr(e, "model_dump") else dict(e) for e in llm_output.evidence],
                "confidence": 0.96,
            },
            ensure_ascii=False,
        )
        judge = RuleReviewJudge(model=MockJudgeModel([response]))
        result = await judge.verify(llm_output, "测试问题", sample_chunks)

        assert len(result.corrections) == 1
        assert result.final_reason != llm_output.reason


# ---------------------------------------------------------------------------
# verify_with_fallback 集成函数测试
# ---------------------------------------------------------------------------


class TestVerifyWithFallback:
    @pytest.mark.asyncio
    async def test_normal_flow(self, llm_output, sample_chunks):
        judge = RuleReviewJudge(model=MockJudgeModel())
        result = await verify_with_fallback(
            judge, llm_output, "测试问题", sample_chunks
        )

        assert isinstance(result, dict)
        assert result["decision"] == "不符合"
        assert result["judge_verified"] is True

    @pytest.mark.asyncio
    async def test_not_found_skips(self, sample_chunks):
        llm_out = _make_llm_output(not_found=True)
        judge = RuleReviewJudge(model=MockJudgeModel())
        result = await verify_with_fallback(
            judge, llm_out, "测试问题", sample_chunks
        )

        assert result["judge_skipped"] is True
        assert result["not_found"] is True

    @pytest.mark.asyncio
    async def test_judge_skipped_preserves_original(
        self, llm_output, sample_chunks
    ):
        """Judge 跳过时保留原始 LLM 输出。"""
        bad_model = MagicMock()
        bad_model.ainvoke = AsyncMock(side_effect=RuntimeError("down"))

        judge = RuleReviewJudge(model=bad_model)
        result = await verify_with_fallback(
            judge, llm_output, "测试问题", sample_chunks
        )

        assert result["judge_skipped"] is True
        assert result["decision"] == llm_output.decision
        assert result["reason"] == llm_output.reason


# ---------------------------------------------------------------------------
# 上下文与工具日志构建测试
# ---------------------------------------------------------------------------


class TestContextBuilding:
    def test_build_context_text(self):
        chunks = [
            {"text": "规则A", "section": "第1条", "page": 1},
            {"text": "规则B", "section": "第2条", "page": 2},
        ]
        text = RuleReviewJudge._build_context_text(chunks)
        assert "[Chunk 1]" in text
        assert "[Chunk 2]" in text
        assert "第1条" in text
        assert "规则A" in text
        assert "---" in text

    def test_build_context_text_empty(self):
        text = RuleReviewJudge._build_context_text([])
        assert "无上下文" in text

    def test_build_tool_logs_text_with_data(self):
        logs = [
            {"tool": "test_tool", "result": {"ok": True}},
            {"tool": "other_tool", "result": {"value": 42}},
        ]
        text = RuleReviewJudge._build_tool_logs_text(logs)
        assert "test_tool" in text
        assert "other_tool" in text

    def test_build_tool_logs_text_empty(self):
        text = RuleReviewJudge._build_tool_logs_text([])
        assert "无工具调用" in text


# ---------------------------------------------------------------------------
# 内容提取测试
# ---------------------------------------------------------------------------


class TestContentExtraction:
    def test_extract_string(self):
        msg = AIMessage(content="hello")
        assert RuleReviewJudge._extract_content(msg) == "hello"

    def test_extract_list(self):
        msg = AIMessage(content=[{"text": "a"}, {"text": "b"}])
        assert RuleReviewJudge._extract_content(msg) == "ab"


# ---------------------------------------------------------------------------
# 默认单例测试
# ---------------------------------------------------------------------------


class TestDefaultJudge:
    def test_singleton(self):
        j1 = get_default_judge()
        j2 = get_default_judge()
        assert j1 is j2
