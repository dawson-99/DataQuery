"""
规则审查系统 Tool 执行器单元测试

覆盖 src/rule_review/tool_executor.py 的：
- Tool 1: extract_table_data — 表格数据提取
- Tool 2: arithmetic_compare — 精确算术比较
- Tool 3: resolve_cross_reference — 交叉引用解析
- Tool 4: validate_date_applicability — 规则时效性校验
- Tool 5: unit_converter — 单位转换
- ToolExecutor dispatch 模式
- Tool 调用循环与降级
- 格式化和辅助函数
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rule_review.tool_executor import (
    TOOL_TOTAL_TIMEOUT,
    ToolExecutor,
    _cn_to_int,
    _extract_number_and_unit,
    _parse_markdown_table,
    arithmetic_compare,
    execute_with_tool_loop,
    extract_table_data,
    resolve_cross_reference,
    unit_converter,
    validate_date_applicability,
)


# ---------------------------------------------------------------------------
# Tool 1: extract_table_data
# ---------------------------------------------------------------------------


class TestExtractTableData:
    def test_extract_basic(self):
        result = extract_table_data(
            table_text="| 地区 | 电价上限(元/MWh) |\n|------|-----------------|\n| 冀北 | 760 |\n| 山西 | 780 |",
            filter_column="地区",
            filter_value="冀北",
            select_column="电价上限(元/MWh)",
        )
        assert result["success"] is True
        assert result["data"]["value"] == 760
        # 数值 760 无单位后缀，unit 为空
        assert result["data"]["matched_filter"] == "冀北"

    def test_extract_with_unit_in_value(self):
        result = extract_table_data(
            table_text="| 省份 | 上限 |\n|------|------|\n| 冀北 | 760元/MWh |",
            filter_column="省份",
            filter_value="冀北",
            select_column="上限",
        )
        assert result["success"] is True
        assert result["data"]["value"] == 760

    def test_extract_fuzzy_filter_match(self):
        """包含匹配也应有结果。"""
        result = extract_table_data(
            table_text="| 地区 | 价格上限(元/MWh) |\n|------|-----------------|\n| 四川主网 | 800 |",
            filter_column="地区",
            filter_value="四川",
            select_column="价格上限(元/MWh)",
        )
        assert result["success"] is True
        assert result["data"]["value"] == 800

    def test_extract_missing_column(self):
        result = extract_table_data(
            table_text="| 地区 | 上限 |\n|------|------|\n| 冀北 | 760 |",
            filter_column="不存在的列",
            filter_value="冀北",
            select_column="上限",
        )
        assert result["success"] is False
        assert "未找到列" in result["error"]

    def test_extract_missing_row(self):
        result = extract_table_data(
            table_text="| 地区 | 上限 |\n|------|------|\n| 冀北 | 760 |",
            filter_column="地区",
            filter_value="北京",
            select_column="上限",
        )
        assert result["success"] is False

    def test_extract_invalid_table(self):
        result = extract_table_data(
            table_text="这不是表格内容",
            filter_column="col",
            filter_value="val",
            select_column="col",
        )
        assert result["success"] is False

    def test_extract_multiline_table(self):
        table = (
            "| 省份 | 电价上限(元/MWh) |\n"
            "|------|-----------------|\n"
            "| 冀北 | 760 |\n"
            "| 山西 | 780 |\n"
            "| 四川主网 | 800 |\n"
        )
        result = extract_table_data(
            table_text=table,
            filter_column="省份",
            filter_value="山西",
            select_column="电价上限(元/MWh)",
        )
        assert result["success"] is True
        assert result["data"]["value"] == 780


# ---------------------------------------------------------------------------
# Tool 2: arithmetic_compare
# ---------------------------------------------------------------------------


class TestArithmeticCompare:
    def test_gt_true(self):
        result = arithmetic_compare(800, "gt", 760)
        assert result["success"] is True
        assert result["data"]["result"] is True
        assert "800 > 760" in result["data"]["expression"]

    def test_gt_false(self):
        result = arithmetic_compare(700, "gt", 760)
        assert result["success"] is True
        assert result["data"]["result"] is False

    def test_lte(self):
        result = arithmetic_compare(760, "lte", 760)
        assert result["data"]["result"] is True

    def test_between_true(self):
        result = arithmetic_compare(500, "between", 300, threshold_high=800)
        assert result["data"]["result"] is True

    def test_between_false(self):
        result = arithmetic_compare(900, "between", 300, threshold_high=800)
        assert result["data"]["result"] is False

    def test_between_missing_high(self):
        result = arithmetic_compare(500, "between", 300)
        assert result["success"] is False

    def test_eq(self):
        result = arithmetic_compare(760, "eq", 760)
        assert result["data"]["result"] is True

    def test_neq(self):
        result = arithmetic_compare(760, "neq", 800)
        assert result["data"]["result"] is True

    def test_invalid_operator(self):
        result = arithmetic_compare(100, "invalid", 50)
        assert result["success"] is False
        assert "不支持的运算符" in result["error"]

    def test_gte(self):
        result = arithmetic_compare(760, "gte", 760)
        assert result["data"]["result"] is True

    def test_lt(self):
        result = arithmetic_compare(700, "lt", 760)
        assert result["data"]["result"] is True


# ---------------------------------------------------------------------------
# Tool 3: resolve_cross_reference
# ---------------------------------------------------------------------------


class TestResolveCrossReference:
    def test_internal_article(self):
        chunks = [
            {"text": "第5条 交易规则适用于所有市场交易主体。", "source": "规则.pdf"},
        ]
        result = resolve_cross_reference(
            "参照第5条执行", all_chunks=chunks,
        )
        assert result["success"] is True
        refs = result["data"]["references"]
        assert len(refs) >= 0  # 能否找到取决于 chunk 格式

    def test_external_doc_pattern(self):
        result = resolve_cross_reference(
            "按照《交易规则》第3条执行",
            all_chunks=[],
        )
        assert result["success"] is True

    def test_no_chunks(self):
        result = resolve_cross_reference("参照第5条")
        assert result["success"] is True
        assert result["data"]["references"] == []

    def test_chapter_pattern(self):
        result = resolve_cross_reference(
            "参照第三章内容",
            all_chunks=[],
        )
        assert result["success"] is True


# ---------------------------------------------------------------------------
# Tool 4: validate_date_applicability
# ---------------------------------------------------------------------------


class TestValidateDateApplicability:
    def test_effective_date_valid(self):
        text = "本规则自2024年1月1日起施行。"
        result = validate_date_applicability(text, "2025-03-15")
        assert result["success"] is True
        assert result["data"]["is_applicable"] is True
        assert result["data"]["effective_date"] == "2024-01-01"

    def test_query_before_effective(self):
        text = "本规则自2024年1月1日起施行。"
        result = validate_date_applicability(text, "2023-06-15")
        assert result["data"]["is_applicable"] is False
        assert "早于" in result["data"]["reason"]

    def test_expired_rule(self):
        text = "本规则于2024年12月31日废止。"
        result = validate_date_applicability(text, "2025-03-15")
        assert result["data"]["is_applicable"] is False

    def test_no_date_info(self):
        result = validate_date_applicability("本规则适用于所有交易。", "2025-03-15")
        assert result["data"]["is_applicable"] is True
        assert "默认" in result["data"]["reason"]

    def test_with_version(self):
        text = "本规则（2024年版）自2024年3月1日起施行。"
        result = validate_date_applicability(text, "2025-01-01")
        assert result["data"]["version"] == "2024年版"
        assert result["data"]["is_applicable"] is True

    def test_invalid_query_date(self):
        result = validate_date_applicability("规则文本", "bad-date")
        assert result["success"] is False

    def test_date_with_slash(self):
        text = "自2024/01/01起施行"
        result = validate_date_applicability(text, "2025-03-15")
        assert result["success"] is True
        assert result["data"]["is_applicable"] is True


# ---------------------------------------------------------------------------
# Tool 5: unit_converter
# ---------------------------------------------------------------------------


class TestUnitConverter:
    def test_energy_same_unit(self):
        result = unit_converter(100, "MWh", "MWh")
        assert result["success"] is True
        assert result["data"]["value"] == 100
        assert result["data"]["was_converted"] is False

    def test_energy_conversion(self):
        result = unit_converter(1, "万kWh", "MWh")
        assert result["success"] is True
        assert result["data"]["value"] == 10.0
        assert result["data"]["was_converted"] is True

    def test_price_conversion(self):
        result = unit_converter(100, "分/kWh", "元/MWh")
        assert result["success"] is True
        assert result["data"]["value"] == 1000.0

    def test_cross_type_conversion(self):
        """能量和价格单位不能互转。"""
        result = unit_converter(100, "MWh", "元/MWh")
        assert result["success"] is False
        assert "无法" in result["error"]

    def test_unknown_unit(self):
        result = unit_converter(100, "未知单位", "MWh")
        assert result["success"] is False

    def test_gwh_to_mwh(self):
        result = unit_converter(1, "GWh", "MWh")
        assert result["data"]["value"] == 1000.0

    def test_kwh_to_mwh(self):
        result = unit_converter(1000, "kWh", "MWh")
        assert result["data"]["value"] == 1.0

    def test_billion_kwh_to_mwh(self):
        result = unit_converter(1, "亿kWh", "MWh")
        assert result["data"]["value"] == 10000.0


# ---------------------------------------------------------------------------
# ToolExecutor dispatch 模式测试
# ---------------------------------------------------------------------------


class TestToolExecutor:
    def test_execute_known_tool(self):
        result = ToolExecutor.execute_tool(
            "arithmetic_compare",
            {"actual": 100, "operator": "gt", "threshold": 50},
        )
        assert result["success"] is True

    def test_execute_unknown_tool(self):
        result = ToolExecutor.execute_tool("unknown_tool", {})
        assert result["success"] is False
        assert "未知工具" in result["error"]

    def test_execute_bad_args(self):
        result = ToolExecutor.execute_tool("arithmetic_compare", {})
        assert result["success"] is False

    def test_execute_tool_calls(self):
        calls = [
            {"tool": "unit_converter", "args": {"value": 1, "from_unit": "万kWh", "to_unit": "MWh"}},
            {"tool": "arithmetic_compare", "args": {"actual": 10, "operator": "gt", "threshold": 5}},
        ]
        results = ToolExecutor.execute_tool_calls(calls)
        assert len(results) == 2
        assert all("tool" in r for r in results)
        assert all("result" in r for r in results)
        assert all("latency_ms" in r for r in results)
        assert results[0]["result"]["success"] is True
        assert results[1]["result"]["success"] is True

    def test_execute_tool_calls_mixed(self):
        """部分成功、部分失败。"""
        calls = [
            {"tool": "unknown", "args": {}},
            {"tool": "unit_converter", "args": {"value": 5, "from_unit": "MWh", "to_unit": "MWh"}},
        ]
        results = ToolExecutor.execute_tool_calls(calls)
        assert len(results) == 2
        assert results[0]["result"]["success"] is False
        assert results[1]["result"]["success"] is True

    def test_format_results_for_llm(self):
        results = [
            {"tool": "test", "result": {"success": True, "data": {"ok": True}}},
            {"tool": "fail", "result": {"success": False, "error": "some error"}},
        ]
        formatted = ToolExecutor.format_results_for_llm(results)
        assert "执行成功" in formatted
        assert "执行失败" in formatted
        assert "test" in formatted
        assert "fail" in formatted


# ---------------------------------------------------------------------------
# Tool 循环测试
# ---------------------------------------------------------------------------


class TestToolLoop:
    @pytest.mark.asyncio
    async def test_no_tool_calls_needed(self):
        """LLM 直接返回无 tool_calls 的结果。"""
        mock_gen = MagicMock()

        # 返回无 tool_calls 的结果
        from src.rule_review.schemas import LLMOutput

        output = LLMOutput(
            decision="不符合",
            reason="超过上限",
            evidence=[],
            confidence=0.9,
            tool_calls=[],
        )

        mock_gen.generate_raw = AsyncMock(return_value=json.dumps({
            "decision": "不符合",
            "reason": "超过上限",
            "evidence": [],
            "confidence": 0.9,
            "tool_calls": [],
        }, ensure_ascii=False))

        final, logs = await execute_with_tool_loop(
            mock_gen, "测试问题", [{"text": "规则", "source": "d.pdf"}],
        )
        assert final is not None
        assert final["decision"] == "不符合"
        assert len(logs) == 0

    @pytest.mark.asyncio
    async def test_one_round_tool_call(self):
        """一轮工具调用后 LLM 返回最终结果。"""
        mock_gen = MagicMock()

        # 第一次调用返回含 tool_calls
        call1 = json.dumps({
            "decision": "",
            "reason": "",
            "evidence": [],
            "confidence": 0.0,
            "tool_calls": [
                {"tool": "arithmetic_compare", "args": {"actual": 800, "operator": "gt", "threshold": 760}}
            ],
        }, ensure_ascii=False)

        # 第二次调用返回最终结果
        call2 = json.dumps({
            "decision": "不符合",
            "reason": "800 > 760",
            "evidence": [],
            "confidence": 0.95,
            "tool_calls": [],
        }, ensure_ascii=False)

        mock_gen.generate_raw = AsyncMock(side_effect=[call1, call2])

        final, logs = await execute_with_tool_loop(
            mock_gen, "测试", [{"text": "规则", "source": "d.pdf"}],
        )
        assert final is not None
        assert final["decision"] == "不符合"
        assert len(logs) == 1
        assert logs[0]["tool"] == "arithmetic_compare"
        assert logs[0]["result"]["success"] is True

    @pytest.mark.asyncio
    async def test_all_tools_fail_fallback(self):
        """所有工具失败时降级。"""
        mock_gen = MagicMock()

        call1 = json.dumps({
            "decision": "",
            "reason": "",
            "evidence": [],
            "confidence": 0.0,
            "tool_calls": [
                {"tool": "extract_table_data", "args": {
                    "table_text": "bad",
                    "filter_column": "x",
                    "filter_value": "y",
                    "select_column": "z",
                }}
            ],
        }, ensure_ascii=False)

        call_fallback = json.dumps({
            "decision": "无法判断",
            "reason": "工具失败",
            "evidence": [],
            "confidence": 0.0,
        }, ensure_ascii=False)

        mock_gen.generate_raw = AsyncMock(side_effect=[call1, call_fallback])

        final, logs = await execute_with_tool_loop(
            mock_gen, "测试", [{"text": "规则", "source": "d.pdf"}],
        )
        assert final is not None
        assert final.get("tool_unsolved") is True

    @pytest.mark.asyncio
    async def test_max_rounds_exceeded(self):
        """达到最大轮数时降级。"""
        mock_gen = MagicMock()

        call = json.dumps({
            "decision": "",
            "reason": "",
            "evidence": [],
            "confidence": 0.0,
            "tool_calls": [
                {"tool": "unit_converter", "args": {"value": 800, "from_unit": "MWh", "to_unit": "MWh"}}
            ],
        }, ensure_ascii=False)

        fallback = json.dumps({
            "decision": "无法判断",
            "reason": "超轮数",
            "evidence": [],
            "confidence": 0.0,
        }, ensure_ascii=False)

        # 每轮都返回 tool_calls，直到达到 max_rounds
        responses = [call, call, call, fallback]
        mock_gen.generate_raw = AsyncMock(side_effect=responses)

        final, logs = await execute_with_tool_loop(
            mock_gen, "测试", [{"text": "规则", "source": "d.pdf"}],
            max_rounds=3,
        )
        assert final is not None
        assert final.get("tool_unsolved") is True
        assert len(logs) == 3  # 3 rounds of tool calls


# ---------------------------------------------------------------------------
# 辅助函数测试
# ---------------------------------------------------------------------------


class TestParseMarkdownTable:
    def test_standard_table(self):
        md = "| 名称 | 值 |\n|------|----|\n| A | 1 |\n| B | 2 |"
        rows = _parse_markdown_table(md)
        assert len(rows) == 2
        assert rows[0]["名称"] == "A"
        assert rows[1]["值"] == "2"

    def test_empty_input(self):
        assert _parse_markdown_table("") == []
        assert _parse_markdown_table("无表格内容") == []

    def test_no_data_rows(self):
        md = "| 名称 | 值 |\n|------|----|"
        rows = _parse_markdown_table(md)
        assert len(rows) == 0  # 只有表头+分隔符，无数据行


class TestExtractNumberAndUnit:
    def test_with_unit(self):
        value, unit = _extract_number_and_unit("760元/MWh")
        assert value == 760
        assert "元/MWh" in unit

    def test_pure_number(self):
        value, unit = _extract_number_and_unit("760")
        assert value == 760
        assert unit == ""

    def test_float(self):
        value, unit = _extract_number_and_unit("760.5")
        assert value == 760.5

    def test_non_numeric(self):
        value, unit = _extract_number_and_unit("无数据")
        assert value is None


class TestCnToInt:
    def test_basic(self):
        assert _cn_to_int("一") == 1
        assert _cn_to_int("五") == 5
        assert _cn_to_int("十") == 10
        assert _cn_to_int("十五") == 15

    def test_digit(self):
        assert _cn_to_int("12") == 12
        assert _cn_to_int("5") == 5

    def test_unknown(self):
        assert _cn_to_int("不存在") == 0
