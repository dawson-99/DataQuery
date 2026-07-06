"""
规则审查系统 - 沙箱辅助模块单元测试

覆盖：
- src/rule_review/sandbox_utils.py：safe_parse_number、safe_divide、safe_percentage、clamp、is_numeric
- 中文数字解析：_try_parse_chinese_number
- ToolSandbox：execute、execute_function、validate_tool_result
- execute_with_timeout：超时执行器
"""

from __future__ import annotations

import asyncio
import time

import pytest

from src.rule_review.sandbox_utils import (
    ToolSandbox,
    _try_parse_chinese_number,
    clamp,
    execute_with_timeout,
    get_default_tool_sandbox,
    is_numeric,
    safe_divide,
    safe_parse_number,
    safe_percentage,
)


# ---------------------------------------------------------------------------
# safe_parse_number 测试
# ---------------------------------------------------------------------------


class TestSafeParseNumber:
    def test_int(self):
        assert safe_parse_number(42) == 42.0

    def test_float(self):
        assert safe_parse_number(3.14) == 3.14

    def test_none(self):
        assert safe_parse_number(None) is None

    def test_bool(self):
        """bool 是 int 的子类但不应该被当作数字。"""
        assert safe_parse_number(True) is None
        assert safe_parse_number(False) is None

    def test_numeric_string(self):
        assert safe_parse_number("123") == 123.0

    def test_negative_string(self):
        assert safe_parse_number("-45.6") == -45.6

    def test_string_with_unit(self):
        """带单位的字符串提取数值部分。"""
        assert safe_parse_number("760元/MWh") == 760.0
        assert safe_parse_number("800元每兆瓦时") == 800.0

    def test_string_with_prefix(self):
        assert safe_parse_number("约760元") == 760.0

    def test_non_numeric_string(self):
        assert safe_parse_number("abc") is None

    def test_empty_string(self):
        assert safe_parse_number("") is None

    def test_zero(self):
        assert safe_parse_number(0) == 0.0
        assert safe_parse_number("0") == 0.0

    def test_scientific_notation(self):
        assert safe_parse_number("1.5e3") == 1500.0

    def test_complex_number_rejected(self):
        """复数不被视为简单数字。"""
        assert safe_parse_number(1 + 2j) is None


# ---------------------------------------------------------------------------
# safe_divide 测试
# ---------------------------------------------------------------------------


class TestSafeDivide:
    def test_normal_division(self):
        assert safe_divide(10, 2) == 5.0

    def test_divide_by_zero(self):
        assert safe_divide(10, 0) == 0.0

    def test_divide_by_zero_custom_default(self):
        assert safe_divide(10, 0, default=-1.0) == -1.0

    def test_zero_divided_by_number(self):
        assert safe_divide(0, 5) == 0.0

    def test_negative_division(self):
        assert safe_divide(-10, 2) == -5.0


# ---------------------------------------------------------------------------
# safe_percentage 测试
# ---------------------------------------------------------------------------


class TestSafePercentage:
    def test_normal_percentage(self):
        assert safe_percentage(25, 100) == 25.0

    def test_percentage_zero_whole(self):
        assert safe_percentage(10, 0) == 0.0

    def test_percentage_custom_default(self):
        assert safe_percentage(10, 0, default=100.0) == 100.0

    def test_percentage_half(self):
        assert safe_percentage(1, 2) == 50.0


# ---------------------------------------------------------------------------
# clamp 测试
# ---------------------------------------------------------------------------


class TestClamp:
    def test_value_in_range(self):
        assert clamp(5, 0, 10) == 5

    def test_value_below_range(self):
        assert clamp(-5, 0, 10) == 0

    def test_value_above_range(self):
        assert clamp(15, 0, 10) == 10

    def test_value_at_lower_bound(self):
        assert clamp(0, 0, 10) == 0

    def test_value_at_upper_bound(self):
        assert clamp(10, 0, 10) == 10

    def test_negative_range(self):
        assert clamp(3, -10, -1) == -1


# ---------------------------------------------------------------------------
# is_numeric 测试
# ---------------------------------------------------------------------------


class TestIsNumeric:
    def test_int(self):
        assert is_numeric(1) is True

    def test_float(self):
        assert is_numeric(1.5) is True

    def test_bool(self):
        assert is_numeric(True) is False
        assert is_numeric(False) is False

    def test_string(self):
        assert is_numeric("123") is False

    def test_none(self):
        assert is_numeric(None) is False


# ---------------------------------------------------------------------------
# 中文数字解析测试
# ---------------------------------------------------------------------------


class TestChineseNumber:
    def test_simple_digit(self):
        assert _try_parse_chinese_number("五") == 5.0

    def test_ten(self):
        assert _try_parse_chinese_number("十") == 10.0

    def test_eleven(self):
        assert _try_parse_chinese_number("十一") == 11.0

    def test_twenty(self):
        assert _try_parse_chinese_number("二十") == 20.0

    def test_twenty_five(self):
        assert _try_parse_chinese_number("二十五") == 25.0

    def test_hundred(self):
        assert _try_parse_chinese_number("一百") == 100.0

    def test_hundred_twenty_three(self):
        assert _try_parse_chinese_number("一百二十三") == 123.0

    def test_thousand(self):
        assert _try_parse_chinese_number("一千") == 1000.0

    def test_ten_thousand(self):
        assert _try_parse_chinese_number("一万") == 10000.0

    def test_large_number(self):
        assert _try_parse_chinese_number("五万三千二百一十") == 53210.0

    def test_zero(self):
        assert _try_parse_chinese_number("零") == 0.0

    def test_decimal(self):
        assert _try_parse_chinese_number("三点五") == 3.5

    def test_negative(self):
        assert _try_parse_chinese_number("负五") == -5.0

    def test_mixed_text(self):
        """只提取中文数字部分。"""
        assert _try_parse_chinese_number("第二百三十四条") == 234.0

    def test_empty(self):
        assert _try_parse_chinese_number("") is None

    def test_non_cn_text(self):
        assert _try_parse_chinese_number("hello") is None

    def test_liang(self):
        """两 = 二。"""
        assert _try_parse_chinese_number("两万") == 20000.0


# ---------------------------------------------------------------------------
# execute_with_timeout 测试
# ---------------------------------------------------------------------------


class TestExecuteWithTimeout:
    @pytest.mark.asyncio
    async def test_sync_function(self):
        def add(a, b):
            return a + b

        result = await execute_with_timeout(add, 1, 2)
        assert result["success"] is True
        assert result["data"] == 3

    @pytest.mark.asyncio
    async def test_async_function(self):
        async def async_add(a, b):
            await asyncio.sleep(0.01)
            return a + b

        result = await execute_with_timeout(async_add, 1, 2, timeout=5)
        assert result["success"] is True
        assert result["data"] == 3

    @pytest.mark.asyncio
    async def test_timeout(self):
        def slow_func():
            time.sleep(10)
            return "done"

        result = await execute_with_timeout(slow_func, timeout=0.1)
        assert result["success"] is False
        assert "超时" in result["error"]

    @pytest.mark.asyncio
    async def test_exception(self):
        def failing_func():
            raise ValueError("测试异常")

        result = await execute_with_timeout(failing_func)
        assert result["success"] is False
        assert "异常" in result["error"]

    @pytest.mark.asyncio
    async def test_keyword_arguments(self):
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}"

        result = await execute_with_timeout(greet, name="World", greeting="Hi")
        assert result["data"] == "Hi, World"


# ---------------------------------------------------------------------------
# ToolSandbox 测试
# ---------------------------------------------------------------------------


class TestToolSandboxExecute:
    @pytest.mark.asyncio
    async def test_execute_simple_code(self):
        sandbox = ToolSandbox()
        result = await sandbox.execute(
            "analysis_result = {'sum': safe_parse_number('100') + safe_parse_number('200')}"
        )
        assert result["success"] is True
        assert result["data"]["data"]["sum"] == 300.0

    @pytest.mark.asyncio
    async def test_execute_with_safe_helpers(self):
        sandbox = ToolSandbox()
        result = await sandbox.execute("""
value = safe_parse_number("760元/MWh")
result = safe_divide(value, 2)
analysis_result = {"half": result}
""")
        assert result["success"] is True
        assert result["data"]["data"]["half"] == 380.0

    @pytest.mark.asyncio
    async def test_execute_with_context(self):
        sandbox = ToolSandbox()
        result = await sandbox.execute(
            "analysis_result = {'actual': actual, 'threshold': threshold, 'exceeds': actual > threshold}",
            context={"actual": 800, "threshold": 760},
        )
        assert result["success"] is True
        assert result["data"]["data"]["exceeds"] is True

    @pytest.mark.asyncio
    async def test_execute_with_clamp(self):
        sandbox = ToolSandbox()
        result = await sandbox.execute(
            "analysis_result = {'clamped': clamp(150, 0, 100)}"
        )
        assert result["success"] is True
        assert result["data"]["data"]["clamped"] == 100

    @pytest.mark.asyncio
    async def test_execute_code_error(self):
        sandbox = ToolSandbox()
        result = await sandbox.execute(
            "analysis_result = undefined_var"
        )
        assert result["success"] is True  # execute_with_timeout 不会报错
        # 内层沙箱执行应该失败
        assert not result["data"]["success"]

    @pytest.mark.asyncio
    async def test_execute_with_date_tools(self):
        sandbox = ToolSandbox()
        result = await sandbox.execute(
            "analysis_result = {'today': str(date(2025, 3, 15))}"
        )
        assert result["success"] is True
        assert result["data"]["data"]["today"] == "2025-03-15"


class TestToolSandboxValidate:
    def test_valid_result(self):
        sandbox = ToolSandbox()
        r = sandbox.validate_tool_result(
            {"value": 760, "unit": "元/MWh"},
            expected_keys=["value", "unit"],
        )
        assert r["valid"] is True
        assert r["errors"] == []

    def test_missing_keys(self):
        sandbox = ToolSandbox()
        r = sandbox.validate_tool_result(
            {"value": 760},
            expected_keys=["value", "unit"],
        )
        assert r["valid"] is False
        assert "unit" in r["errors"][0]

    def test_none_result(self):
        sandbox = ToolSandbox()
        r = sandbox.validate_tool_result(None, expected_keys=["value"])
        assert r["valid"] is False
        assert "None" in r["errors"][0]

    def test_not_dict(self):
        sandbox = ToolSandbox()
        r = sandbox.validate_tool_result("string", expected_keys=["value"])
        assert r["valid"] is False

    def test_no_expected_keys(self):
        sandbox = ToolSandbox()
        r = sandbox.validate_tool_result({"anything": "goes"})
        assert r["valid"] is True

    def test_empty_expected_keys(self):
        sandbox = ToolSandbox()
        r = sandbox.validate_tool_result({"a": 1}, expected_keys=[])
        assert r["valid"] is True


class TestToolSandboxExecuteFunction:
    @pytest.mark.asyncio
    async def test_sync_function(self):
        sandbox = ToolSandbox()
        result = await sandbox.execute_function(lambda x, y: x + y, 3, 4)
        assert result["success"] is True
        assert result["data"] == 7

    @pytest.mark.asyncio
    async def test_async_function(self):
        sandbox = ToolSandbox()

        async def compute(a, b):
            await asyncio.sleep(0.01)
            return a * b

        result = await sandbox.execute_function(compute, 6, 7)
        assert result["success"] is True
        assert result["data"] == 42


# ---------------------------------------------------------------------------
# 默认工厂
# ---------------------------------------------------------------------------


class TestDefaultToolSandbox:
    def test_singleton(self):
        s1 = get_default_tool_sandbox()
        s2 = get_default_tool_sandbox()
        assert s1 is s2

    def test_instance_type(self):
        sandbox = get_default_tool_sandbox()
        assert isinstance(sandbox, ToolSandbox)


# ---------------------------------------------------------------------------
# 工具函数集成测试
# ---------------------------------------------------------------------------


class TestSandboxIntegration:
    """验证沙箱中可用的安全函数与工具上下文。"""

    @pytest.mark.asyncio
    async def test_tool_chain_simulation(self):
        """模拟典型工具链：解析数值 → 计算 → 校验结果。"""
        sandbox = ToolSandbox()

        # 模拟 extract_table_data 后的算术比较
        code = """
price = safe_parse_number("800元/MWh")
limit = 760

exceeds = price > limit
excess_amount = price - limit
excess_pct = safe_percentage(excess_amount, limit)

analysis_result = {
    "exceeds": exceeds,
    "excess_amount": safe_parse_number(str(excess_amount)),
    "excess_pct": safe_parse_number(str(round(excess_pct, 2))),
}
"""
        result = await sandbox.execute(code)
        assert result["success"] is True
        data = result["data"]["data"]
        assert data["exceeds"] is True
        assert data["excess_amount"] == 40.0
        assert data["excess_pct"] == 5.26

    @pytest.mark.asyncio
    async def test_unit_conversion_simulation(self):
        """模拟单位换算后比较。"""
        sandbox = ToolSandbox()

        code = """
# 万kWh → MWh: * 10
actual_mwh = safe_parse_number("80") * 10  # 80万kWh = 800 MWh
threshold_mwh = 760

analysis_result = {
    "actual_mwh": actual_mwh,
    "threshold": threshold_mwh,
    "exceeds": actual_mwh > threshold_mwh,
}
"""
        result = await sandbox.execute(code)
        assert result["success"] is True
        data = result["data"]["data"]
        assert data["exceeds"] is True
        assert data["actual_mwh"] == 800.0

    @pytest.mark.asyncio
    async def test_date_applicability_simulation(self):
        """模拟规则时效性校验。"""
        sandbox = ToolSandbox()

        code = """
query_date = date(2025, 3, 15)
effective_date = date(2024, 1, 1)

is_applicable = query_date >= effective_date

analysis_result = {
    "is_applicable": is_applicable,
    "effective_date": str(effective_date),
    "query_date": str(query_date),
}
"""
        result = await sandbox.execute(code)
        assert result["success"] is True
        data = result["data"]["data"]
        assert data["is_applicable"] is True
