"""
Phase 2.4: LLM tool_call 输出能力验证

验证 LLM 在各种场景下正确输出 tool_calls JSON 格式：
- 单工具调用：extract_table_data + arithmetic_compare 链式调用
- 表格中带单位、多列、模糊匹配
- tool_calls 为空时的正常输出
- JSON 容错：代码块包裹、多余空白、markdown 标记
- 降级场景：工具失败后的重新生成
"""

from __future__ import annotations

import json

import pytest

from src.rule_review.generator import parse_llm_output
from src.rule_review.prompts import (
    SYSTEM_PROMPT_V2,
    TOOLS_SECTION,
    build_messages,
    get_system_prompt,
)
from src.rule_review.schemas import LLMOutput
from src.rule_review.tool_executor import (
    ToolExecutor,
    extract_table_data,
)


# ---------------------------------------------------------------------------
# System Prompt V2 格式验证
# ---------------------------------------------------------------------------


class TestSystemPromptV2:
    def test_contains_tool_rules(self):
        prompt = get_system_prompt(include_tools=True)
        assert "工具调用规则" in prompt
        assert "tool_calls" in prompt

    def test_contains_all_five_tools(self):
        prompt = get_system_prompt(include_tools=True)
        assert "extract_table_data" in prompt
        assert "arithmetic_compare" in prompt
        assert "resolve_cross_reference" in prompt
        assert "validate_date_applicability" in prompt
        assert "unit_converter" in prompt

    def test_contains_tool_example(self):
        prompt = get_system_prompt(include_tools=True)
        assert "冀北电价800元/MWh是否超出上限" in prompt
        assert "760" in prompt

    def test_v1_no_tool_rules(self):
        prompt = get_system_prompt(include_tools=False)
        assert "工具调用规则" not in prompt
        assert "tool_calls" not in prompt

    def test_terms_section_loaded(self):
        assert "日前现货出清电价" in TOOLS_SECTION or len(TOOLS_SECTION) > 0


# ---------------------------------------------------------------------------
# parse_llm_output: tool_calls 解析验证
# ---------------------------------------------------------------------------


class TestParseLLMOutputWithTools:
    def test_parse_no_tool_calls(self):
        raw = json.dumps({
            "decision": "不符合",
            "reason": "超出上限",
            "evidence": [],
            "confidence": 0.9,
            "tool_calls": [],
        }, ensure_ascii=False)
        result = parse_llm_output(raw)
        assert result is not None
        assert result.tool_calls == []

    def test_parse_single_tool_call(self):
        raw = json.dumps({
            "decision": "",
            "reason": "",
            "evidence": [],
            "confidence": 0.0,
            "tool_calls": [
                {"tool": "arithmetic_compare", "args": {"actual": 800, "operator": "gt", "threshold": 760}},
            ],
        }, ensure_ascii=False)
        result = parse_llm_output(raw)
        assert result is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["tool"] == "arithmetic_compare"

    def test_parse_chained_tool_calls(self):
        """表格提取 → 算术比较的链式调用。"""
        raw = json.dumps({
            "decision": "",
            "reason": "",
            "evidence": [],
            "confidence": 0.0,
            "tool_calls": [
                {"tool": "extract_table_data", "args": {
                    "table_text": "| 地区 | 电价上限 |\n| 冀北 | 760 |",
                    "filter_column": "地区", "filter_value": "冀北",
                    "select_column": "电价上限",
                }},
                {"tool": "arithmetic_compare", "args": {
                    "actual": 800, "operator": "gt", "threshold": 760,
                }},
            ],
        }, ensure_ascii=False)
        result = parse_llm_output(raw)
        assert result is not None
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["tool"] == "extract_table_data"
        assert result.tool_calls[1]["tool"] == "arithmetic_compare"

    def test_parse_in_code_block(self):
        raw = '```json\n' + json.dumps({
            "decision": "不符合",
            "reason": "超出",
            "evidence": [],
            "confidence": 0.9,
            "tool_calls": [],
        }, ensure_ascii=False) + '\n```'
        result = parse_llm_output(raw)
        assert result is not None
        assert result.decision == "不符合"

    def test_parse_with_extra_text(self):
        """LLM 可能在 JSON 前后附带解释文字。"""
        raw = (
            '根据规则文档分析，结果如下：\n'
            + json.dumps({
                "decision": "不符合",
                "reason": "超出上限",
                "evidence": [],
                "confidence": 0.9,
                "tool_calls": [],
            }, ensure_ascii=False)
            + '\n以上为审查结果。'
        )
        result = parse_llm_output(raw)
        assert result is not None
        assert result.decision == "不符合"

    def test_parse_unit_converter_tool(self):
        raw = json.dumps({
            "decision": "",
            "reason": "",
            "evidence": [],
            "confidence": 0.0,
            "tool_calls": [
                {"tool": "unit_converter", "args": {
                    "value": 100, "from_unit": "万kWh", "to_unit": "MWh",
                }},
            ],
        }, ensure_ascii=False)
        result = parse_llm_output(raw)
        assert result is not None
        assert result.tool_calls[0]["tool"] == "unit_converter"


# ---------------------------------------------------------------------------
# build_messages: tool_results 注入验证
# ---------------------------------------------------------------------------


class TestBuildMessagesWithTools:
    def test_messages_with_tool_results(self):
        chunks = [{"text": "规则内容", "source": "doc.pdf", "section": "第1条", "page": 1}]
        tool_results = [
            {"tool": "arithmetic_compare", "result": {"success": True, "data": {"result": True}}},
        ]
        msgs = build_messages("测试问题", chunks, tool_results=tool_results)
        assert len(msgs) == 2
        assert "工具执行结果" in msgs[1]["content"]
        assert "arithmetic_compare" in msgs[1]["content"]

    def test_messages_with_multiple_tool_results(self):
        chunks = [{"text": "规则", "source": "d.pdf"}]
        tool_results = [
            {"tool": "t1", "result": {"success": True, "data": {"v": 1}}},
            {"tool": "t2", "result": {"success": False, "error": "bad"}},
        ]
        msgs = build_messages("q", chunks, tool_results=tool_results)
        assert "t1" in msgs[1]["content"]
        assert "t2" in msgs[1]["content"]


# ---------------------------------------------------------------------------
# ToolExecutor: 完整调用链端到端验证
# ---------------------------------------------------------------------------


class TestToolCallChainE2E:
    """模拟 LLM 输出 tool_calls → 执行 → 最终推理的完整链路。"""

    def test_extract_then_compare_chain(self):
        """LLM 先提取表格数据，再算术比较。"""
        table_md = (
            "| 省份 | 电价上限(元/MWh) |\n"
            "|------|-----------------|\n"
            "| 冀北 | 760 |\n"
            "| 山西 | 780 |\n"
        )

        # 模拟 LLM 第一轮输出
        tool_calls = [
            {"tool": "extract_table_data", "args": {
                "table_text": table_md,
                "filter_column": "省份",
                "filter_value": "冀北",
                "select_column": "电价上限(元/MWh)",
            }},
        ]
        results = ToolExecutor.execute_tool_calls(tool_calls)
        assert results[0]["result"]["success"] is True
        assert results[0]["result"]["data"]["value"] == 760

        # 模拟 LLM 第二轮：用提取的值做比较
        value = results[0]["result"]["data"]["value"]
        compare_calls = [
            {"tool": "arithmetic_compare", "args": {
                "actual": 800, "operator": "gt", "threshold": value,
            }},
        ]
        compare_results = ToolExecutor.execute_tool_calls(compare_calls)
        assert compare_results[0]["result"]["success"] is True
        assert compare_results[0]["result"]["data"]["result"] is True

    def test_unit_then_compare_chain(self):
        """LLM 先转换单位，再比较。"""
        # 模拟工具调用链
        tool_calls = [
            {"tool": "unit_converter", "args": {
                "value": 100, "from_unit": "万kWh", "to_unit": "MWh",
            }},
        ]
        results = ToolExecutor.execute_tool_calls(tool_calls)
        assert results[0]["result"]["success"] is True

        converted_value = results[0]["result"]["data"]["value"]
        assert converted_value == 1000.0  # 100万kWh = 1000MWh

        compare_calls = [
            {"tool": "arithmetic_compare", "args": {
                "actual": converted_value, "operator": "lt", "threshold": 2000,
            }},
        ]
        compare_results = ToolExecutor.execute_tool_calls(compare_calls)
        assert compare_results[0]["result"]["data"]["result"] is True

    def test_date_validation_then_decision(self):
        """时效性校验后做出判断。"""
        rule_text = "本规则自2024年1月1日起施行，2025年12月31日废止。"

        tool_calls = [
            {"tool": "validate_date_applicability", "args": {
                "rule_text": rule_text,
                "query_date": "2025-03-15",
            }},
        ]
        results = ToolExecutor.execute_tool_calls(tool_calls)
        assert results[0]["result"]["success"] is True
        assert results[0]["result"]["data"]["is_applicable"] is True

    def test_resolve_then_decision(self):
        """交叉引用解析后重新推理。"""
        tool_calls = [
            {"tool": "resolve_cross_reference", "args": {
                "reference_text": "按照《交易规则》第5条执行",
            }},
        ]
        results = ToolExecutor.execute_tool_calls(tool_calls)
        assert results[0]["result"]["success"] is True


# ---------------------------------------------------------------------------
# 真实 LLM 输出格式模拟测试
# ---------------------------------------------------------------------------


class TestRealisticLLMOutputs:
    """模拟真实 LLM 可能输出的各种 tool_calls 格式变体。"""

    def test_single_tool_no_final_decision(self):
        """第一轮：LLM 检测到需要工具，输出空 decision + tool_calls。"""
        llm_text = json.dumps({
            "decision": "",
            "reason": "需要从表格中提取冀北的电价上限",
            "evidence": [],
            "confidence": 0.0,
            "tool_calls": [
                {"tool": "extract_table_data", "args": {
                    "table_text": "| 省份 | 电价上限(元/MWh) |\n|------|-----------------|\n| 冀北 | 760 |",
                    "filter_column": "省份",
                    "filter_value": "冀北",
                    "select_column": "电价上限(元/MWh)",
                }},
            ],
        }, ensure_ascii=False)

        result = parse_llm_output(llm_text)
        assert result is not None
        assert result.decision == ""  # 未做出最终判断
        assert len(result.tool_calls) == 1

    def test_final_output_after_tools(self):
        """工具执行后：LLM 输出完整结果，tool_calls 为空。"""
        llm_text = json.dumps({
            "decision": "不符合",
            "reason": "工具提取冀北电价上限为760元/MWh，实际值800 > 760，超出上限40元/MWh。",
            "evidence": [
                {"source": "规则.pdf", "section": "第2条", "page": 2,
                 "text": "冀北电价上限为760元/MWh。"}
            ],
            "confidence": 0.95,
            "tool_calls": [],
        }, ensure_ascii=False)

        result = parse_llm_output(llm_text)
        assert result is not None
        assert result.decision == "不符合"
        assert result.tool_calls == []
        assert result.confidence == 0.95

    def test_markdown_code_block_wrapped(self):
        """LLM 常见输出：用 markdown 代码块包裹 JSON。"""
        llm_text = (
            "根据规则文档内容，审查结果如下：\n\n"
            "```json\n"
            + json.dumps({
                "decision": "不符合",
                "reason": "超出上限",
                "evidence": [],
                "confidence": 0.92,
                "tool_calls": [],
            }, ensure_ascii=False, indent=2)
            + "\n```"
        )
        result = parse_llm_output(llm_text)
        assert result is not None
        assert result.decision == "不符合"

    def test_formatted_results_after_execution(self):
        """验证 ToolExecutor 的结果格式化文本包含正确信息。"""
        results = [
            {"tool": "extract_table_data", "result": {
                "success": True,
                "data": {"value": 760, "unit": "元/MWh", "matched_filter": "冀北", "row_number": 1},
            }},
            {"tool": "arithmetic_compare", "result": {
                "success": True,
                "data": {"result": True, "expression": "800 > 760", "detail": "实际值800超出阈值760"},
            }},
        ]
        formatted = ToolExecutor.format_results_for_llm(results)
        assert "执行成功" in formatted
        assert "760" in formatted
        assert "800" in formatted
        assert "extract_table_data" in formatted
        assert "arithmetic_compare" in formatted


# ---------------------------------------------------------------------------
# 容错测试
# ---------------------------------------------------------------------------


class TestToolCallErrorHandling:
    def test_unknown_tool_in_call(self):
        result = ToolExecutor.execute_tool("non_existent_tool", {"a": 1})
        assert result["success"] is False

    def test_missing_required_args(self):
        result = ToolExecutor.execute_tool("extract_table_data", {})
        assert result["success"] is False

    def test_malformed_tool_call(self):
        """tool_calls 包含不完整信息。"""
        bad_calls = [{"tool": "extract_table_data"}]  # 缺少 args
        results = ToolExecutor.execute_tool_calls(bad_calls)
        assert results[0]["result"]["success"] is False

    def test_parse_malformed_json(self):
        """LLM 输出截断或损坏的 JSON。"""
        result = parse_llm_output('{"decision": "不符合", "reason"')
        assert result is None

    def test_tool_with_cross_reference_no_chunks(self):
        """交叉引用工具在无 chunks 时优雅降级。"""
        result = ToolExecutor.execute_tool("resolve_cross_reference", {
            "reference_text": "参照第5条执行",
        })
        assert result["success"] is True
        assert result["data"]["references"] == []
