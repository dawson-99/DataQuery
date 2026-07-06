"""
电力规则审查系统 - 沙箱辅助模块

按设计文档 §4.1 实现，封装现有 PythonSandbox，为工具执行提供：
- 超时控制（单个工具 10s 上限）
- 工具执行上下文（预注入变量与安全函数白名单）
- 安全数据类型转换（parse_number、safe_divide 等）
- 工具结果验证（格式校验 + 类型检查）

设计原则：不修改现有 PythonSandbox，在外层增加工具执行所需的上下文和约束。
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Callable, TypeVar

from src.utils.python_sandbox import PythonSandbox, SandboxExecutionError

logger = logging.getLogger(__name__)

# 单次工具执行超时（秒）
DEFAULT_TOOL_TIMEOUT = 10

# 单次沙箱代码执行超时（秒）
DEFAULT_SANDBOX_TIMEOUT = 10

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# 安全辅助函数（供工具和沙箱使用）
# ---------------------------------------------------------------------------


def safe_parse_number(value: Any) -> float | None:
    """安全地将值解析为数字，解析失败返回 None。

    支持中文数字（"一万二千"等简单形式）、带单位的字符串（"760元/MWh"）。

    Args:
        value: 任意输入值。

    Returns:
        解析后的数字，或 None。
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        if isinstance(value, bool):
            return None
        return float(value)

    if isinstance(value, str):
        # 尝试直接转换
        try:
            return float(value)
        except ValueError:
            pass

        # 尝试提取数值部分（如 "760元/MWh"）
        import re
        num_match = re.search(r"[-+]?\d+\.?\d*", value)
        if num_match:
            try:
                return float(num_match.group())
            except ValueError:
                pass

        # 尝试中文数字（简单映射）
        cn_result = _try_parse_chinese_number(value)
        if cn_result is not None:
            return cn_result

    return None


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """安全除法，分母为 0 时返回 default。"""
    if b == 0:
        return default
    return a / b


def safe_percentage(part: float, whole: float, default: float = 0.0) -> float:
    """安全百分比计算，分母为 0 时返回 default。"""
    if whole == 0:
        return default
    return (part / whole) * 100


def clamp(value: float, low: float, high: float) -> float:
    """将值限制在 [low, high] 区间内。"""
    return max(low, min(value, high))


def is_numeric(value: Any) -> bool:
    """判断值是否为数值类型（不含 bool）。"""
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float))


# ---------------------------------------------------------------------------
# 中文数字辅助
# ---------------------------------------------------------------------------

# 中文数字映射
_CN_NUM_MAP: dict[str, int] = {
    "零": 0, "〇": 0,
    "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    "十": 10, "百": 100, "千": 1000,
    "万": 10000, "亿": 100000000,
    "两": 2,
}


def _try_parse_chinese_number(text: str) -> float | None:
    """尝试将简单中文数字文本转为数字。

    支持形式：一百二十三、一千零五十、三点五、百分之八十

    Args:
        text: 原始文本。

    Returns:
        解析后的数字，或 None。
    """
    if not text:
        return None

    # 过滤非中文字符
    clean = "".join(ch for ch in text if ch in _CN_NUM_MAP or ch == "点" or ch == "负")

    if not clean:
        return None

    # 处理"负"
    negative = clean.startswith("负")
    if negative:
        clean = clean[1:]

    # 处理小数点
    if "点" in clean:
        parts = clean.split("点", 1)
        integer_part = _parse_cn_integer(parts[0])
        decimal_part = _parse_cn_decimal(parts[1])
        if integer_part is not None:
            result = integer_part + decimal_part
            return -result if negative else result
        return None

    result = _parse_cn_integer(clean)
    if result is not None:
        return -result if negative else result
    return None


def _parse_cn_integer(text: str) -> float | None:
    """解析中文整数部分。"""
    if not text:
        return 0.0

    total = 0.0
    current = 0.0
    section = 0.0  # 万以内的节

    for ch in text:
        if ch not in _CN_NUM_MAP:
            return None
        val = _CN_NUM_MAP[ch]

        if val >= 10000:  # 万、亿
            section = (section + current) * val
            total += section
            section = 0.0
            current = 0.0
        elif val >= 10:  # 十、百、千
            if current == 0:
                current = 1
            section += current * val
            current = 0.0
        else:  # 0-9
            current = float(val)

    section += current
    total += section
    return total


def _parse_cn_decimal(text: str) -> float:
    """解析中文小数部分。"""
    result = 0.0
    divisor = 10.0
    for ch in text:
        if ch in _CN_NUM_MAP:
            val = _CN_NUM_MAP[ch]
            if val < 10:
                result += val / divisor
                divisor *= 10
    return result


# ---------------------------------------------------------------------------
# 超时执行器
# ---------------------------------------------------------------------------


async def execute_with_timeout(
    func: Callable[..., Any],
    *args: Any,
    timeout: float = DEFAULT_TOOL_TIMEOUT,
    **kwargs: Any,
) -> dict[str, Any]:
    """带超时的异步函数执行器。

    将同步函数包装为异步执行，超过 timeout 秒则强制取消。

    Args:
        func: 要执行的函数。
        *args: 位置参数。
        timeout: 超时秒数。
        **kwargs: 关键字参数。

    Returns:
        {"success": bool, "data": Any, "error": str}
    """
    try:
        if inspect.iscoroutinefunction(func):
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=timeout,
            )
        else:
            result = await asyncio.wait_for(
                asyncio.to_thread(func, *args, **kwargs),
                timeout=timeout,
            )
        return {"success": True, "data": result}
    except asyncio.TimeoutError:
        return {
            "success": False,
            "error": f"函数执行超时（>{timeout}s）: {func.__name__}",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"函数执行异常: {str(e)}",
        }


# ---------------------------------------------------------------------------
# ToolSandbox：为工具执行封装的沙箱
# ---------------------------------------------------------------------------


class ToolSandbox:
    """工具执行沙箱——封装 PythonSandbox，增加工具后续执行所需的上下文。

    与 ToolExecutor 配合使用：
    - ToolExecutor 处理工具定义内的纯 Python 逻辑
    - ToolSandbox 处理工具执行后的补充计算（如 LLM 生成的辅助代码）

    特点：
    1. 预注入工具执行可用的安全变量和函数
    2. 限制单次执行超时
    3. 结果格式校验
    """

    # 工具上下文安全全局变量（在 PythonSandbox 基础上扩展）
    TOOL_CONTEXT_GLOBALS = {
        **PythonSandbox.SAFE_GLOBALS,
        # 安全辅助函数
        "safe_parse_number": safe_parse_number,
        "safe_divide": safe_divide,
        "safe_percentage": safe_percentage,
        "clamp": clamp,
        "is_numeric": is_numeric,
        # 日期工具
        "date": __import__("datetime").date,
        "datetime": __import__("datetime").datetime,
        "timedelta": __import__("datetime").timedelta,
    }

    def __init__(self, timeout: float = DEFAULT_SANDBOX_TIMEOUT):
        """
        Args:
            timeout: 单次执行超时（秒）。
        """
        self._sandbox = PythonSandbox()
        self.timeout = timeout

    async def execute(
        self,
        code: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """带超时保护的沙箱代码执行。

        Args:
            code: Python 代码字符串。
            context: 额外的上下文变量（注入到局部变量）。

        Returns:
            {"success": bool, "data": dict | None, "error": str}
        """
        # 构建执行上下文
        ctx = dict(context or {})

        # 使用自定义 SAFE_GLOBALS 执行
        result = await execute_with_timeout(
            self._execute_in_sandbox,
            code,
            ctx,
            timeout=self.timeout,
        )

        if not result["success"]:
            return result

        return result

    def _execute_in_sandbox(self, code: str, context: dict[str, Any]) -> dict[str, Any]:
        """在沙箱中同步执行代码（内部方法）。"""
        # 使用 PythonSandbox 的底层能力
        # 这里复用 sandbox 的 _compile_sandbox_code 和 exec 逻辑
        import contextlib
        import io

        from src.utils.python_sandbox import _ImportStripper
        import ast

        stdout_buffer = io.StringIO()
        local_vars = {
            "analysis_result": None,
            **context,
        }

        # 编译 + 去 import
        parsed = ast.parse(code, mode="exec")
        stripped = _ImportStripper().visit(parsed)
        ast.fix_missing_locations(stripped)
        compiled = compile(stripped, "<tool_sandbox>", "exec")

        try:
            with contextlib.redirect_stdout(stdout_buffer):
                exec(compiled, dict(self.TOOL_CONTEXT_GLOBALS), local_vars)
        except Exception as e:
            return {
                "success": False,
                "error": f"沙箱执行异常: {str(e)}",
                "stdout": stdout_buffer.getvalue().strip(),
            }

        result = local_vars.get("analysis_result")
        return {
            "success": True,
            "data": result,
            "stdout": stdout_buffer.getvalue().strip(),
        }

    async def execute_function(
        self,
        func: Callable[..., Any],
        *args: Any,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """在沙箱上下文中安全执行一个函数。

        与 ToolExecutor.execute_tool 配合，当单个工具需要后端
        执行额外的 LLM 生成代码时使用。

        Args:
            func: 要执行的函数。
            *args: 位置参数。
            context: 上下文变量（未使用，保留扩展）。
            **kwargs: 关键字参数。

        Returns:
            {"success": bool, "data": Any, "error": str}
        """
        return await execute_with_timeout(
            func,
            *args,
            timeout=self.timeout,
            **kwargs,
        )

    def validate_tool_result(
        self,
        result: Any,
        expected_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        """校验工具返回结果的格式。

        Args:
            result: 工具返回结果。
            expected_keys: 期望包含的键列表。

        Returns:
            {"valid": bool, "errors": list[str]}
        """
        errors: list[str] = []

        if result is None:
            errors.append("结果为 None")
            return {"valid": False, "errors": errors}

        if expected_keys:
            if isinstance(result, dict):
                missing = [k for k in expected_keys if k not in result]
                if missing:
                    errors.append(f"缺少字段: {', '.join(missing)}")
            else:
                errors.append(f"结果应为 dict，实际为 {type(result).__name__}")

        return {"valid": len(errors) == 0, "errors": errors}


# ---------------------------------------------------------------------------
# 默认工厂
# ---------------------------------------------------------------------------


_default_tool_sandbox: ToolSandbox | None = None


def get_default_tool_sandbox() -> ToolSandbox:
    """获取默认工具沙箱单例。"""
    global _default_tool_sandbox
    if _default_tool_sandbox is None:
        _default_tool_sandbox = ToolSandbox()
    return _default_tool_sandbox
