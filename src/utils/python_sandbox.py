import ast
import contextlib
import io
import json
import math
import statistics
import traceback
from typing import Any

import pandas as pd


class SandboxExecutionError(Exception):
    pass


class _ImportStripper(ast.NodeTransformer):
    """Remove import statements from generated sandbox code."""

    def visit_Import(self, node: ast.Import) -> None:
        return None

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        return None


def _safe_extreme(func, *args, **kwargs):
    """Return None instead of raising on empty single-iterable max/min calls."""
    if len(args) != 1 or "default" in kwargs:
        return func(*args, **kwargs)

    try:
        values = list(args[0])
    except TypeError:
        return func(*args, **kwargs)

    if not values:
        return None
    return func(values, **kwargs)


def _safe_max(*args, **kwargs):
    return _safe_extreme(max, *args, **kwargs)


def _safe_min(*args, **kwargs):
    return _safe_extreme(min, *args, **kwargs)


class PythonSandbox:
    """轻量 Python 沙箱执行器。

    约定：传入的生成代码必须将最终结果写入 `analysis_result` 变量。
    """

    SAFE_BUILTINS = {
        "len": len,
        "min": _safe_min,
        "max": _safe_max,
        "sum": sum,
        "sorted": sorted,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "abs": abs,
        "round": round,
        "any": any,
        "all": all,
        "isinstance": isinstance,
        "type": type,
        "print": print,
    }

    SAFE_GLOBALS = {
        "__builtins__": SAFE_BUILTINS,
        "json": json,
        "math": math,
        "statistics": statistics,
        "pd": pd,
        "pandas": pd,
    }

    @staticmethod
    def _compile_sandbox_code(code: str):
        parsed = ast.parse(code, mode="exec")
        stripped = _ImportStripper().visit(parsed)
        ast.fix_missing_locations(stripped)
        return compile(stripped, "<trend_analysis>", "exec")

    @staticmethod
    def _normalize_dataset(dataset: Any) -> tuple[Any, list[dict[str, Any]]]:
        if isinstance(dataset, list):
            records = [item for item in dataset if isinstance(item, dict)]
            if records:
                return records, records
            return dataset, []

        if isinstance(dataset, dict):
            for key in ("grouped_data", "extreme_scope_data", "data"):
                nested = dataset.get(key)
                if isinstance(nested, list):
                    normalized, records = PythonSandbox._normalize_dataset(nested)
                    if records:
                        return normalized, records
            return [dataset], [dataset]

        return dataset, []

    def execute(self, code: str, dataset: Any) -> dict[str, Any]:
        stdout_buffer = io.StringIO()
        normalized_data, records = self._normalize_dataset(dataset)
        local_vars = {
            "data": normalized_data,
            "records": records,
            "raw_data": dataset,
            "analysis_result": None,
        }

        try:
            compiled = self._compile_sandbox_code(code)
            with contextlib.redirect_stdout(stdout_buffer):
                exec(compiled, dict(self.SAFE_GLOBALS), local_vars)
        except Exception as exc:
            raise SandboxExecutionError(traceback.format_exc()) from exc

        analysis_result = local_vars.get("analysis_result")
        if analysis_result is None:
            raise SandboxExecutionError("生成代码未写入 analysis_result 结果变量")
        if not isinstance(analysis_result, dict):
            raise SandboxExecutionError("analysis_result 必须是可 JSON 序列化的 dict")

        try:
            json.dumps(analysis_result, ensure_ascii=False)
        except TypeError as exc:
            raise SandboxExecutionError(f"analysis_result 不可 JSON 序列化: {exc}") from exc

        return {
            "analysis_result": analysis_result,
            "stdout": stdout_buffer.getvalue().strip(),
        }
