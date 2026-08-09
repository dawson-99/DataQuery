"""
工具参数 JSON Schema 校验测试（方向八）

覆盖 src/rule_review/tool_schema.py：
- required 缺失 / 类型错误 / enum 越界拦截
- 合法参数放行
- 未注册 schema 工具向后兼容
- ToolExecutor.execute_tool 集成：schema_error 信号
- execute_with_tool_loop 中 schema_error 不算执行失败（可修正重试）
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from src.rule_review.generator import parse_llm_output
from src.rule_review.schemas import LLMOutput
from src.rule_review.tool_executor import (
    ToolExecutor,
    execute_with_tool_loop,
)
from src.rule_review.tool_schema import validate_tool_args


# ---------------------------------------------------------------------------
# validate_tool_args 单元测试
# ---------------------------------------------------------------------------


class TestValidateToolArgs:
    def test_valid_args_pass(self):
        ok, err = validate_tool_args(
            "arithmetic_compare",
            {"actual": 800, "operator": "gt", "threshold": 760},
        )
        assert ok, err
        assert err == ""

    def test_missing_required_rejected(self):
        ok, err = validate_tool_args(
            "arithmetic_compare",
            {"actual": 800, "threshold": 760},  # 缺 operator
        )
        assert not ok
        assert "operator" in err

    def test_type_error_rejected(self):
        ok, err = validate_tool_args(
            "arithmetic_compare",
            {"actual": "800", "operator": "gt", "threshold": 760},  # actual 是字符串
        )
        assert not ok
        assert "类型错误" in err

    def test_enum_out_of_range_rejected(self):
        ok, err = validate_tool_args(
            "arithmetic_compare",
            {"actual": 800, "operator": "foo", "threshold": 760},  # operator 非法
        )
        assert not ok
        assert "取值非法" in err

    def test_int_ok_for_number(self):
        ok, err = validate_tool_args(
            "unit_converter",
            {"value": 800, "from_unit": "元/MWh", "to_unit": "分/kWh"},  # int 可作 number
        )
        assert ok, err

    def test_bool_not_ok_for_number(self):
        ok, err = validate_tool_args(
            "unit_converter",
            {"value": True, "from_unit": "元/MWh", "to_unit": "分/kWh"},  # bool 不是 number
        )
        assert not ok

    def test_args_not_dict_rejected(self):
        ok, err = validate_tool_args("arithmetic_compare", ["not", "dict"])
        assert not ok
        assert "JSON 对象" in err

    def test_unregistered_tool_passes(self):
        """未注册 schema 的工具放行（向后兼容）。"""
        ok, err = validate_tool_args("some_future_tool", {"anything": 1})
        assert ok
        assert err == ""

    def test_unknown_param_ignored(self):
        """schema 中未声明的参数键忽略，不报错。"""
        ok, err = validate_tool_args(
            "arithmetic_compare",
            {"actual": 800, "operator": "gt", "threshold": 760, "extra": "x"},
        )
        assert ok, err


# ---------------------------------------------------------------------------
# ToolExecutor.execute_tool 集成
# ---------------------------------------------------------------------------


class TestExecuteToolWithSchema:
    def test_invalid_args_returns_schema_error(self):
        result = ToolExecutor.execute_tool(
            "arithmetic_compare",
            {"actual": "800", "operator": "gt", "threshold": 760},
        )
        assert result["success"] is False
        assert result.get("schema_error") is True
        assert "参数校验失败" in result["error"]

    def test_valid_args_executes_normally(self):
        result = ToolExecutor.execute_tool(
            "arithmetic_compare",
            {"actual": 800, "operator": "gt", "threshold": 760},
        )
        assert result["success"] is True
        assert result["data"]["result"] is True

    def test_unknown_tool_behavior_unchanged(self):
        result = ToolExecutor.execute_tool("no_such_tool", {})
        assert result["success"] is False
        assert "未知工具" in result["error"]
        assert "schema_error" not in result

    def test_tool_without_schema_still_works(self):
        """注册了函数但 schema 缺失（兼容场景）也应正常执行。"""
        result = ToolExecutor.execute_tool(
            "extract_table_data",
            {
                "table_text": "| 地区 | 上限 |\n| 冀北 | 760 |",
                "filter_column": "地区",
                "filter_value": "冀北",
                "select_column": "上限",
            },
        )
        assert result["success"] is True
        assert result["data"]["value"] == 760


# ---------------------------------------------------------------------------
# execute_with_tool_loop：schema_error 不算执行失败
# ---------------------------------------------------------------------------


class TestToolLoopSchemaError:
    @pytest.mark.asyncio
    async def test_schema_error_not_trigger_fallback(self):
        """参数校验失败仍注入结果让 LLM 修正，而不是直接降级。"""

        class MockGenerator:
            def __init__(self, rounds_output):
                self.rounds = rounds_output
                self.calls = 0

            async def generate_raw(self, messages):
                raw = self.rounds[self.calls]
                self.calls += 1
                return raw

            async def generate(self, *args, **kwargs):
                return None

        # 第 1 轮：带非法参数的工具调用（operator 越界）
        round1 = json.dumps(
            {
                "decision": "",
                "reason": "",
                "evidence": [],
                "confidence": 0.0,
                "tool_calls": [
                    {
                        "tool": "arithmetic_compare",
                        "args": {"actual": 800, "operator": "invalid", "threshold": 760},
                    }
                ],
                "not_found": False,
            },
            ensure_ascii=False,
        )
        # 第 2 轮：修正参数后正常输出
        round2 = json.dumps(
            {
                "decision": "不符合",
                "reason": "800元/MWh超过上限760元/MWh",
                "evidence": [],
                "confidence": 0.9,
                "tool_calls": [],
                "not_found": False,
            },
            ensure_ascii=False,
        )

        gen = MockGenerator([round1, round2])
        final, tool_logs = await execute_with_tool_loop(gen, "测试问题", [])

        assert gen.calls == 2
        assert final is not None
        assert final["decision"] == "不符合"
        # 正常终止：tool_unsolved 为 False（非降级产物）
        assert final.get("tool_unsolved") is False
        # 日志记录了 schema_error 结果
        assert tool_logs[0]["result"].get("schema_error") is True
        assert len(tool_logs) == 1

    @pytest.mark.asyncio
    async def test_all_execution_failures_still_fallback(self):
        """全部为执行类错误（非 schema_error）时仍降级。"""

        class MockGenerator:
            def __init__(self):
                self.calls = 0

            async def generate_raw(self, messages):
                self.calls += 1
                return json.dumps(
                    {
                        "decision": "",
                        "reason": "",
                        "evidence": [],
                        "confidence": 0.0,
                        "tool_calls": [{"tool": "no_such_tool", "args": {}}],
                        "not_found": False,
                    },
                    ensure_ascii=False,
                )

            async def generate(self, *args, **kwargs):
                return None

        gen = MockGenerator()
        final, tool_logs = await execute_with_tool_loop(gen, "测试问题", [])

        assert final is not None
        assert final.get("tool_unsolved") is True
