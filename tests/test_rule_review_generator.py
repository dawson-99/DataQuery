"""
规则审查系统 Prompt 与 LLM 推理模块单元测试

覆盖：
- src/rule_review/prompts.py：SYSTEM_PROMPT、build_rag_context_prompt、build_messages
- src/rule_review/generator.py：parse_llm_output、RuleReviewGenerator 非流式/流式生成
"""

from __future__ import annotations

import json
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from src.rule_review.generator import (
    RuleReviewGenerator,
    _extract_content,
    _extract_chunk_content,
    get_default_generator,
    parse_llm_output,
)
from src.rule_review.prompts import (
    SYSTEM_PROMPT,
    TERMS_SECTION,
    build_messages,
    build_rag_context_prompt,
)
from src.rule_review.schemas import LLMOutput


# ---------------------------------------------------------------------------
# Helper: Mock 模型
# ---------------------------------------------------------------------------


def _make_valid_llm_json(decision="不符合", reason="测试理由", not_found=False) -> str:
    """生成标准格式的 LLM JSON 输出。"""
    return json.dumps(
        {
            "decision": decision,
            "reason": reason,
            "evidence": [
                {
                    "source": "测试规则.pdf",
                    "section": "第2条 价格上限",
                    "page": 2,
                    "text": "省间日前现货出清电价上限为760元/MWh。",
                }
            ],
            "confidence": 0.95,
            "not_found": not_found,
        },
        ensure_ascii=False,
    )


class MockChatModel:
    """模拟 ChatQwen / ProxyChatModel，返回 deterministic 输出。"""

    def __init__(self, response: str | None = None, stream_chunks: list[str] | None = None):
        self.response = response or _make_valid_llm_json()
        self.stream_chunks = stream_chunks or list(self.response)  # 逐字符流式

    async def ainvoke(self, messages, **kwargs) -> AIMessage:
        return AIMessage(content=self.response)

    async def astream(self, messages, **kwargs) -> AsyncGenerator[AIMessageChunk, None]:
        for ch in self.stream_chunks:
            yield AIMessageChunk(content=ch)


class MockModelStore:
    """可动态更新 response 的 mock 模型（用于测试重试逻辑）。"""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_count = 0

    async def ainvoke(self, messages, **kwargs) -> AIMessage:
        idx = min(self.call_count, len(self.responses) - 1)
        resp = self.responses[idx]
        self.call_count += 1
        return AIMessage(content=resp)

    async def astream(self, messages, **kwargs) -> AsyncGenerator[AIMessageChunk, None]:
        idx = min(self.call_count, len(self.responses) - 1)
        resp = self.responses[idx]
        self.call_count += 1
        for ch in resp:
            yield AIMessageChunk(content=ch)


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
            "text": "违反价格上限规则的交易主体将被处罚。",
            "source": "测试规则.pdf",
            "section": "第4条 违规处罚",
            "page": 4,
        },
    ]


@pytest.fixture
def mock_model():
    return MockChatModel()


# ---------------------------------------------------------------------------
# prompts.py 测试
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_system_prompt_contains_key_sections(self):
        assert "电力交易规则审查专家" in SYSTEM_PROMPT
        assert "核心原则" in SYSTEM_PROMPT
        assert "输出格式" in SYSTEM_PROMPT
        assert "决策指南" in SYSTEM_PROMPT
        assert "符合" in SYSTEM_PROMPT
        assert "不符合" in SYSTEM_PROMPT

    def test_system_prompt_includes_current_date(self):
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        assert today in SYSTEM_PROMPT


class TestTermsSection:
    def test_terms_section_not_empty(self):
        assert len(TERMS_SECTION) > 0
        assert "日前现货出清电价" in TERMS_SECTION or "术语" in TERMS_SECTION


class TestBuildRagContextPrompt:
    def test_build_includes_query(self):
        chunks = [{"text": "规则内容", "source": "文档.pdf", "section": "第1条", "page": 1}]
        prompt = build_rag_context_prompt("测试问题", chunks)
        assert "测试问题" in prompt

    def test_build_includes_chunk_text(self):
        chunks = [{"text": "规则内容", "source": "文档.pdf", "section": "第1条", "page": 1}]
        prompt = build_rag_context_prompt("测试问题", chunks)
        assert "规则内容" in prompt

    def test_build_includes_source_info(self):
        chunks = [{"text": "内容", "source": "测试.pdf", "section": "第3条", "page": 5}]
        prompt = build_rag_context_prompt("query", chunks)
        assert "第3条" in prompt
        assert "第5页" in prompt

    def test_build_multiple_chunks(self):
        chunks = [
            {"text": "规则A", "source": "doc.pdf", "section": "第1条", "page": 1},
            {"text": "规则B", "source": "doc.pdf", "section": "第2条", "page": 2},
        ]
        prompt = build_rag_context_prompt("query", chunks)
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "---" in prompt

    def test_build_terms_section_default(self):
        chunks = [{"text": "测试", "source": "d.pdf"}]
        prompt = build_rag_context_prompt("q", chunks)
        assert "术语参考" in prompt

    def test_build_custom_terms_section(self):
        chunks = [{"text": "测试", "source": "d.pdf"}]
        prompt = build_rag_context_prompt("q", chunks, terms_section="自定义术语")
        assert "自定义术语" in prompt


class TestBuildMessages:
    def test_build_messages_structure(self, sample_chunks):
        msgs = build_messages("测试问题", sample_chunks)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "测试问题" in msgs[1]["content"]

    def test_build_messages_with_tool_results(self, sample_chunks):
        tool_results = [
            {"tool": "extract_table_data", "args": {}, "result": {"value": 760}}
        ]
        msgs = build_messages("测试问题", sample_chunks, tool_results=tool_results)
        assert "工具执行结果" in msgs[1]["content"]
        assert "extract_table_data" in msgs[1]["content"]
        assert "760" in msgs[1]["content"]

    def test_build_messages_custom_system(self, sample_chunks):
        custom_sys = "自定义 System Prompt"
        msgs = build_messages("q", sample_chunks, system_prompt=custom_sys)
        assert msgs[0]["content"] == custom_sys


# ---------------------------------------------------------------------------
# generator.py: parse_llm_output 测试
# ---------------------------------------------------------------------------


class TestParseLLMOutput:
    def test_parse_valid_json(self):
        raw = _make_valid_llm_json()
        result = parse_llm_output(raw)
        assert result is not None
        assert result.decision == "不符合"
        assert result.reason == "测试理由"
        assert len(result.evidence) == 1
        assert result.confidence == 0.95
        assert result.not_found is False

    def test_parse_json_in_code_block(self):
        raw = "```json\n" + _make_valid_llm_json() + "\n```"
        result = parse_llm_output(raw)
        assert result is not None
        assert result.decision == "不符合"

    def test_parse_not_found_true(self):
        raw = json.dumps({
            "decision": "无法判断",
            "reason": "无相关规则",
            "evidence": [],
            "confidence": 0.0,
            "not_found": True,
        }, ensure_ascii=False)
        result = parse_llm_output(raw)
        assert result is not None
        assert result.not_found is True

    def test_parse_invalid_json_returns_none(self):
        result = parse_llm_output("这不是有效的 JSON 内容")
        assert result is None

    def test_parse_empty_string_returns_none(self):
        result = parse_llm_output("")
        assert result is None

    def test_parse_with_tool_calls(self):
        raw = json.dumps({
            "decision": "不符合",
            "reason": "超出上限",
            "evidence": [],
            "confidence": 0.9,
            "tool_calls": [
                {"tool": "arithmetic_compare", "args": {"actual": 800, "operator": "gt", "threshold": 760}}
            ],
        }, ensure_ascii=False)
        result = parse_llm_output(raw)
        assert result is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["tool"] == "arithmetic_compare"


# ---------------------------------------------------------------------------
# generator.py: RuleReviewGenerator 测试
# ---------------------------------------------------------------------------


class TestRuleReviewGenerator:
    @pytest.mark.asyncio
    async def test_generate_returns_llm_output(self, sample_chunks):
        gen = RuleReviewGenerator(model=MockChatModel())
        result = await gen.generate("测试问题", sample_chunks)
        assert result is not None
        assert isinstance(result, LLMOutput)
        assert result.decision == "不符合"

    @pytest.mark.asyncio
    async def test_generate_empty_chunks(self):
        gen = RuleReviewGenerator(model=MockChatModel())
        result = await gen.generate("测试问题", [])
        assert result is not None

    @pytest.mark.asyncio
    async def test_generate_stream_yields_text(self, sample_chunks):
        response = _make_valid_llm_json()
        gen = RuleReviewGenerator(model=MockChatModel(
            response=response,
            stream_chunks=[response[i:i + 10] for i in range(0, len(response), 10)],
        ))

        chunks = []
        async for text in gen.generate_stream("测试问题", sample_chunks):
            chunks.append(text)

        full = "".join(chunks)
        parsed = json.loads(full)
        assert parsed["decision"] == "不符合"

    @pytest.mark.asyncio
    async def test_generate_stream_error_yields_fallback(self, sample_chunks):
        """LLM 流式调用异常时，应 yield 一个 fallback JSON。"""
        bad_model = MagicMock()
        bad_model.astream = AsyncMock(side_effect=RuntimeError("fake error"))

        gen = RuleReviewGenerator(model=bad_model)
        chunks = []
        async for text in gen.generate_stream("测试问题", sample_chunks):
            chunks.append(text)

        full = "".join(chunks)
        parsed = json.loads(full)
        assert "error" in parsed
        assert parsed["decision"] == "无法判断"

    @pytest.mark.asyncio
    async def test_generate_retry_on_parse_failure(self, sample_chunks):
        """第一次返回无效 JSON，第二次返回有效 JSON 时自动重试。"""
        model = MockModelStore([
            "invalid response",
            _make_valid_llm_json(),
        ])
        gen = RuleReviewGenerator(model=model)
        result = await gen.generate("测试问题", sample_chunks, max_retries=1)
        assert result is not None
        assert result.decision == "不符合"
        assert model.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_with_tool_results(self, sample_chunks):
        gen = RuleReviewGenerator(model=MockChatModel())
        tool_results = [
            {"tool": "test", "args": {}, "result": {"ok": True}}
        ]
        result = await gen.generate("测试问题", sample_chunks, tool_results=tool_results)
        assert result is not None

    @pytest.mark.asyncio
    async def test_generate_all_retries_fail(self, sample_chunks):
        """全部重试都返回无效 JSON，应返回 None。"""
        model = MockModelStore(["bad", "still bad"])
        gen = RuleReviewGenerator(model=model)
        result = await gen.generate("测试问题", sample_chunks, max_retries=1)
        assert result is None

    @pytest.mark.asyncio
    async def test_generate_raw(self):
        gen = RuleReviewGenerator(model=MockChatModel(response="原始文本"))
        result = await gen.generate_raw([{"role": "user", "content": "测试"}])
        assert result == "原始文本"

    def test_model_property_lazy_init(self):
        """model 属性在第一次访问时才创建模型实例。"""
        gen = RuleReviewGenerator()
        assert gen._model is None
        # 不实际触发远程调用，只验证惰性创建逻辑
        # 直接设置 _model 跳过工厂逻辑
        mock = MagicMock()
        gen._model = mock
        assert gen.model is mock

    def test_default_generator_singleton(self):
        g1 = get_default_generator()
        g2 = get_default_generator()
        assert g1 is g2


# ---------------------------------------------------------------------------
# 辅助函数测试
# ---------------------------------------------------------------------------


class TestContentExtraction:
    def test_extract_content_string(self):
        msg = AIMessage(content="hello")
        assert _extract_content(msg) == "hello"

    def test_extract_content_list(self):
        msg = AIMessage(content=[{"text": "part1"}, {"text": "part2"}])
        assert _extract_content(msg) == "part1part2"

    def test_extract_chunk_content_string(self):
        chunk = AIMessageChunk(content="delta")
        assert _extract_chunk_content(chunk) == "delta"


# ---------------------------------------------------------------------------
# generator.py: to_langchain_messages 测试
# ---------------------------------------------------------------------------


class TestToLangchainMessages:
    def test_conversion(self):
        msgs = [
            {"role": "system", "content": "系统提示"},
            {"role": "user", "content": "用户问题"},
            {"role": "assistant", "content": "助手回复"},
        ]
        gen = RuleReviewGenerator(model=MockChatModel())
        lc_msgs = gen._to_langchain_messages(msgs)
        assert len(lc_msgs) == 3
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        assert isinstance(lc_msgs[0], SystemMessage)
        assert isinstance(lc_msgs[1], HumanMessage)
        assert isinstance(lc_msgs[2], AIMessage)
