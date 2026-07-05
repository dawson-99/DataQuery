"""
电力规则审查系统 - Tool 执行器

按设计文档 §8 Tool 系统设计实现 Phase 2：
- 5 个纯 Python 工具函数（不调 LLM）
- ToolExecutor dispatch 模式（复用 AggregationAgent 模式）
- Tool 调用循环：LLM → tool_calls → 执行 → 结果注入 → 重新推理
- 终止条件：3 轮限制 + 30s 超时 + 全部失败降级

遵循现有项目模式：
- 工具定义在 JSON 配置文件中（非 @tool 装饰器）
- dispatch 映射表：工具名 → 纯 Python 函数
- 与 Pipeline 集成：结果注入 messages → LLM 继续生成
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import date, datetime
from typing import Any

logger = logging.getLogger(__name__)

# 工具循环限制
MAX_TOOL_ROUNDS = 3
TOOL_TOTAL_TIMEOUT = 30  # 秒

# 单位换算表
_UNIT_TO_MWH: dict[str, float] = {
    "MWh": 1.0,
    "万kWh": 10.0,
    "亿kWh": 10000.0,
    "GWh": 1000.0,
    "kWh": 0.001,
}

_UNIT_TO_YUAN_PER_MWH: dict[str, float] = {
    "元/MWh": 1.0,
    "元/千度": 1.0,
    "元/万kWh": 0.1,
    "分/kWh": 10.0,
    "元/MWh万": 1.0,
}

_ALL_ENERGY_UNITS = {**_UNIT_TO_MWH, **_UNIT_TO_YUAN_PER_MWH}

# 条款号转换映射（中文数字 → 阿拉伯数字）
_CN_NUM_MAP: dict[str, int] = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    "十一": 11, "十二": 12, "十三": 13, "十四": 14, "十五": 15,
    "十六": 16, "十七": 17, "十八": 18, "十九": 19, "二十": 20,
    "零": 0,
}


# ---------------------------------------------------------------------------
# Tool 1: extract_table_data
# ---------------------------------------------------------------------------

def extract_table_data(
    table_text: str,
    filter_column: str,
    filter_value: str,
    select_column: str,
) -> dict:
    """从 Markdown 表格中精确提取指定行列的数值。

    实现：
    1. 正则解析 Markdown table → List[dict]
    2. 模糊匹配 filter_value 定位目标行
    3. 提取 select_column 的值
    4. 返回 {value, unit, matched_filter, row_number}
    """
    rows = _parse_markdown_table(table_text)
    if not rows:
        return {"success": False, "error": "无法解析表格，请确认 table_text 是有效的 Markdown 表格格式"}

    # 标准化列名
    columns = list(rows[0].keys())

    # 模糊匹配列名
    matched_filter_col = _fuzzy_match_column(filter_column, columns)
    if matched_filter_col is None:
        return {
            "success": False,
            "error": f"未找到列 '{filter_column}'，可用列: {', '.join(columns)}",
        }

    matched_select_col = _fuzzy_match_column(select_column, columns)
    if matched_select_col is None:
        return {
            "success": False,
            "error": f"未找到列 '{select_column}'，可用列: {', '.join(columns)}",
        }

    # 模糊匹配 filter_value 定位目标行
    best_row_idx = -1
    for i, row in enumerate(rows):
        cell_value = str(row.get(matched_filter_col, "")).strip()
        if _fuzzy_match_value(filter_value, cell_value):
            best_row_idx = i
            break

    if best_row_idx < 0:
        return {
            "success": False,
            "error": f"未找到 {filter_column}='{filter_value}' 的行",
        }

    target_row = rows[best_row_idx]
    raw_value = target_row.get(matched_select_col, "").strip()

    # 尝试提取数值和单位
    value, unit = _extract_number_and_unit(raw_value)

    return {
        "success": True,
        "data": {
            "value": value,
            "unit": unit,
            "raw_value": raw_value,
            "matched_filter": target_row.get(matched_filter_col, "").strip(),
            "row_number": best_row_idx + 1,  # 1-indexed
        },
    }


# ---------------------------------------------------------------------------
# Tool 2: arithmetic_compare
# ---------------------------------------------------------------------------

def arithmetic_compare(
    actual: float,
    operator: str,
    threshold: float,
    threshold_high: float | None = None,
) -> dict:
    """精确算术比较。假定输入值已完成单位统一。

    operator 支持: gt / gte / lt / lte / eq / neq / between
    """
    op = operator.lower().strip()
    detail = ""

    try:
        if op == "gt":
            result = actual > threshold
            expression = f"{actual} > {threshold}"
        elif op == "gte":
            result = actual >= threshold
            expression = f"{actual} >= {threshold}"
        elif op == "lt":
            result = actual < threshold
            expression = f"{actual} < {threshold}"
        elif op == "lte":
            result = actual <= threshold
            expression = f"{actual} <= {threshold}"
        elif op == "eq":
            result = abs(actual - threshold) < 1e-9
            expression = f"{actual} == {threshold}"
        elif op == "neq":
            result = abs(actual - threshold) >= 1e-9
            expression = f"{actual} != {threshold}"
        elif op == "between":
            if threshold_high is None:
                return {
                    "success": False,
                    "error": "between 操作需要提供 threshold_high 参数",
                }
            result = threshold <= actual <= threshold_high
            expression = f"{threshold} <= {actual} <= {threshold_high}"
        else:
            return {
                "success": False,
                "error": f"不支持的运算符 '{operator}'，支持: gt/gte/lt/lte/eq/neq/between",
            }

        # 生成中文描述
        if op in ("gt", "gte"):
            if result:
                diff = actual - threshold
                detail = f"实际值{actual}超出阈值{threshold}，超出{diff}"
            else:
                detail = f"实际值{actual}未超出阈值{threshold}"
        elif op in ("lt", "lte"):
            if result:
                detail = f"实际值{actual}低于阈值{threshold}"
            else:
                diff = actual - threshold
                detail = f"实际值{actual}超出阈值{threshold}，超出{diff}"
        elif op == "between":
            if result:
                detail = f"实际值{actual}在区间[{threshold}, {threshold_high}]内"
            else:
                detail = f"实际值{actual}不在区间[{threshold}, {threshold_high}]内"

        return {
            "success": True,
            "data": {
                "result": result,
                "expression": expression,
                "detail": detail,
            },
        }
    except (TypeError, ValueError) as e:
        return {"success": False, "error": f"算术比较失败: {str(e)}"}


# ---------------------------------------------------------------------------
# Tool 3: resolve_cross_reference
# ---------------------------------------------------------------------------

# 交叉引用正则模式
_CROSS_REF_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("internal_article", re.compile(r"第([一二三四五六七八九十0-9]+)条\s*(?:第([一二三四五六七八九十0-9]+)款)?")),
    ("internal_chapter", re.compile(r"第([一二三四五六七八九十0-9]+)章\s*")),
    ("internal_section", re.compile(r"第([一二三四五六七八九十0-9]+)节\s*")),
    ("external_doc", re.compile(r"(?:按照|参照|依据|根据)《([^》]+)》")),
    ("external_article", re.compile(r"《([^》]+)》\s*第([一二三四五六七八九十0-9]+)条")),
]


def resolve_cross_reference(
    reference_text: str,
    all_chunks: list[dict] | None = None,
    current_doc_id: str | None = None,
) -> dict:
    """解析交叉引用，找到被引条款原文。

    支持两种引用模式：
    - 内部引用："第X条(第Y款)"、"第X章第Y节" → 在当前 doc 的 chunks 中搜索
    - 外部引用："按照《XXX》第X条" → 在所有 docs 中搜索
    """
    if all_chunks is None:
        all_chunks = []

    references: list[dict] = []

    # 检测各种交叉引用模式
    for pattern_type, pattern in _CROSS_REF_PATTERNS:
        for match in pattern.finditer(reference_text):
            if pattern_type in ("internal_article", "internal_chapter", "internal_section"):
                # 提取条款号并标准化
                article_num_cn = match.group(1)
                article_num = _cn_to_int(article_num_cn)
                search_term = f"第{article_num}条" if pattern_type == "internal_article" else None
                if search_term is None:
                    search_term = f"第{article_num}{'章' if 'chapter' in pattern_type else '节'}"

                # 在当前 chunks 中搜索
                resolved = _find_in_chunks(search_term, all_chunks, current_doc_id)
                if resolved:
                    references.append({
                        "type": pattern_type,
                        "pattern": match.group(0),
                        "resolved_text": resolved.get("text", "")[:300],
                        "source": resolved.get("source", ""),
                    })
            elif pattern_type in ("external_doc", "external_article"):
                doc_name = match.group(1)
                resolved = _find_in_chunks(doc_name, all_chunks)
                if resolved:
                    references.append({
                        "type": pattern_type,
                        "pattern": match.group(0),
                        "resolved_text": resolved.get("text", "")[:300],
                        "source": resolved.get("source", ""),
                    })

    if not references:
        return {
            "success": True,
            "data": {
                "references": [],
                "message": "未在可用 chunks 中解析到交叉引用",
            },
        }

    return {
        "success": True,
        "data": {"references": references},
    }


# ---------------------------------------------------------------------------
# Tool 4: validate_date_applicability
# ---------------------------------------------------------------------------

_DATE_PATTERNS_VALIDATE: list[tuple[str, re.Pattern]] = [
    ("effective", re.compile(r"(?:自|从|于)(\d{4})年(\d{1,2})月(\d{1,2})日(?:起)?施行")),
    ("effective", re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日起?施行")),
    ("effective", re.compile(r"(?:自|从|于)(\d{4})[/-](\d{1,2})[/-](\d{1,2})(?:起)?施行")),
    ("expiry", re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日?废[止除]")),
    ("expiry", re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日?失效")),
    ("version", re.compile(r"[（(](\d{4})年?版[）)]")),
]


def validate_date_applicability(rule_text: str, query_date: str) -> dict:
    """判断规则在给定日期是否有效。

    返回 {is_applicable, effective_date, expiry_date, version, reason}
    """
    effective_date: str | None = None
    expiry_date: str | None = None
    version: str | None = None

    for label, pattern in _DATE_PATTERNS_VALIDATE:
        match = pattern.search(rule_text)
        if match:
            try:
                if label == "effective":
                    y, m, d = int(match.group(1)), int(match.group(2)), int(match.group(3))
                    effective_date = f"{y:04d}-{m:02d}-{d:02d}"
                elif label == "expiry":
                    y, m, d = int(match.group(1)), int(match.group(2)), int(match.group(3))
                    expiry_date = f"{y:04d}-{m:02d}-{d:02d}"
                elif label == "version":
                    version = f"{match.group(1)}年版"
            except (ValueError, IndexError):
                continue

    # 解析 query_date
    try:
        qd = date.fromisoformat(query_date)
    except (ValueError, TypeError):
        return {
            "success": False,
            "error": f"query_date 格式无效: '{query_date}'，需要 YYYY-MM-DD",
        }

    # 判断是否有效
    if effective_date is None and expiry_date is None:
        return {
            "success": True,
            "data": {
                "is_applicable": True,
                "effective_date": None,
                "expiry_date": None,
                "version": version,
                "reason": "规则文本中未找到明确的施行/废止日期，默认为有效",
            },
        }

    reasons: list[str] = []

    if effective_date:
        ed = date.fromisoformat(effective_date)
        if qd < ed:
            reasons.append(f"规则于 {effective_date} 施行，查询日期 {query_date} 早于施行日期")

    if expiry_date:
        xd = date.fromisoformat(expiry_date)
        if qd > xd:
            reasons.append(f"规则于 {expiry_date} 废止，查询日期 {query_date} 晚于废止日期")

    is_applicable = len(reasons) == 0

    return {
        "success": True,
        "data": {
            "is_applicable": is_applicable,
            "effective_date": effective_date,
            "expiry_date": expiry_date,
            "version": version,
            "reason": "；".join(reasons) if reasons else "规则在查询日期有效",
        },
    }


# ---------------------------------------------------------------------------
# Tool 5: unit_converter
# ---------------------------------------------------------------------------

def unit_converter(value: float, from_unit: str, to_unit: str) -> dict:
    """单位转换。先转到基准单位，再转到目标单位。

    支持的能量单位: MWh, 万kWh, 亿kWh, GWh, kWh
    支持的价格单位: 元/MWh, 元/千度, 元/万kWh, 分/kWh
    """
    from_unit = from_unit.strip()
    to_unit = to_unit.strip()

    if from_unit == to_unit:
        return {
            "success": True,
            "data": {"value": value, "unit": to_unit, "was_converted": False},
        }

    # 查找单位类型
    if from_unit in _UNIT_TO_MWH and to_unit in _UNIT_TO_MWH:
        # 能量单位互转
        base = value * _UNIT_TO_MWH[from_unit]  # 转为 MWh
        result = base / _UNIT_TO_MWH[to_unit]
    elif from_unit in _UNIT_TO_YUAN_PER_MWH and to_unit in _UNIT_TO_YUAN_PER_MWH:
        # 价格单位互转
        base = value * _UNIT_TO_YUAN_PER_MWH[from_unit]  # 转为 元/MWh
        result = base / _UNIT_TO_YUAN_PER_MWH[to_unit]
    elif from_unit in _UNIT_TO_MWH and to_unit in _UNIT_TO_YUAN_PER_MWH:
        return {
            "success": False,
            "error": f"无法在能量单位 '{from_unit}' 和价格单位 '{to_unit}' 之间转换（不是同类单位）",
        }
    elif from_unit in _UNIT_TO_YUAN_PER_MWH and to_unit in _UNIT_TO_MWH:
        return {
            "success": False,
            "error": f"无法在价格单位 '{from_unit}' 和能量单位 '{to_unit}' 之间转换（不是同类单位）",
        }
    else:
        known = list(_ALL_ENERGY_UNITS.keys())
        return {
            "success": False,
            "error": f"不支持的单位: from='{from_unit}', to='{to_unit}'。已知单位: {', '.join(known)}",
        }

    return {
        "success": True,
        "data": {
            "value": round(result, 6),
            "unit": to_unit,
            "original_value": value,
            "original_unit": from_unit,
            "was_converted": True,
        },
    }


# ---------------------------------------------------------------------------
# ToolExecutor: Dispatch 运行时
# ---------------------------------------------------------------------------


class ToolExecutor:
    """工具调用运行时。遵循现有 AggregationAgent 的 dispatch 模式。

    与 Phase 1 Pipeline 集成，提供 tool_call 检测 → 执行 → 结果注入 的完整循环。
    """

    TOOL_MAP: dict[str, Any] = {
        "extract_table_data": extract_table_data,
        "arithmetic_compare": arithmetic_compare,
        "resolve_cross_reference": resolve_cross_reference,
        "validate_date_applicability": validate_date_applicability,
        "unit_converter": unit_converter,
    }

    @staticmethod
    def execute_tool(tool_name: str, args: dict) -> dict:
        """执行单个工具调用。"""
        func = ToolExecutor.TOOL_MAP.get(tool_name)
        if not func:
            return {"success": False, "error": f"未知工具: {tool_name}"}
        try:
            return func(**args)
        except TypeError as e:
            return {"success": False, "error": f"参数错误: {str(e)}"}
        except Exception as e:
            logger.warning("[ToolExecutor] 工具 %s 执行异常: %s", tool_name, e)
            return {"success": False, "error": str(e)}

    @staticmethod
    def execute_tool_calls(tool_calls: list[dict]) -> list[dict]:
        """批量执行 tool_calls，返回带元信息的结果列表。"""
        results = []
        for call in tool_calls:
            tool_name = call.get("tool", "")
            args = call.get("args", {})
            t0 = time.monotonic()
            result = ToolExecutor.execute_tool(tool_name, args)
            elapsed_ms = (time.monotonic() - t0) * 1000
            results.append({
                "tool": tool_name,
                "args": args,
                "result": result,
                "latency_ms": round(elapsed_ms, 2),
            })
        return results

    @staticmethod
    def format_results_for_llm(tool_results: list[dict]) -> str:
        """将工具结果格式化为 LLM 可读的文本。

        返回的文本会注入到 messages 中供 LLM 继续推理。
        """
        lines = ["## 工具执行结果\n"]
        for i, tr in enumerate(tool_results):
            tool_name = tr.get("tool", "unknown")
            result = tr.get("result", {})
            success = result.get("success", False)

            lines.append(f"### 工具 {i + 1}: {tool_name}")
            if success:
                data = result.get("data", result)
                lines.append(f"执行成功: {json.dumps(data, ensure_ascii=False, indent=2)}")
            else:
                error = result.get("error", "未知错误")
                lines.append(f"执行失败: {error}")
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 调用循环（供 Pipeline 使用）
# ---------------------------------------------------------------------------


async def execute_with_tool_loop(
    generator: Any,  # RuleReviewGenerator
    query: str,
    context_chunks: list[dict],
    max_rounds: int = MAX_TOOL_ROUNDS,
    total_timeout: float = TOOL_TOTAL_TIMEOUT,
) -> tuple[dict | None, list[dict]]:
    """带终止条件的 Tool 调用循环。

    流程:
    1. LLM 生成 → 检测 tool_calls
    2. tool_calls 非空 → 执行工具 → 结果注入 → 返回 step 1
    3. tool_calls 为空 → 正常终止
    4. 达到 max_rounds 或超时 → 降级处理

    Args:
        generator: RuleReviewGenerator 实例。
        query: 用户问题。
        context_chunks: 检索 chunks。
        max_rounds: 最大循环轮数，默认 3。
        total_timeout: 总超时秒数，默认 30。

    Returns:
        (final_output, tool_logs)
        - final_output: LLM 最终输出 dict，None 表示降级失败
        - tool_logs: 工具调用日志列表
    """
    from src.rule_review.generator import parse_llm_output
    from src.rule_review.prompts import build_messages

    tool_logs: list[dict] = []
    messages = build_messages(query, context_chunks)
    round_start = time.monotonic()

    for round_num in range(1, max_rounds + 1):
        # 超时检查
        elapsed = time.monotonic() - round_start
        if elapsed > total_timeout:
            logger.warning("[Tool] 总超时 %.1fs，强制终止", elapsed)
            final = await _fallback_generate(
                generator, messages, query, context_chunks,
                tool_unsolved_reason=f"总超时 {total_timeout}s",
            )
            return final, tool_logs

        # LLM 生成
        raw_text = await generator.generate_raw(messages)
        llm_output = parse_llm_output(raw_text)

        if llm_output is None:
            logger.warning("[Tool] 第 %d 轮 LLM 输出解析失败", round_num)
            final = await _fallback_generate(
                generator, messages, query, context_chunks,
                tool_unsolved_reason="LLM 输出格式异常",
            )
            return final, tool_logs

        tool_calls = llm_output.tool_calls
        if not tool_calls:
            # 正常终止：LLM 不需要工具
            logger.info("[Tool] 第 %d 轮 LLM 无 tool_calls，正常终止", round_num)
            return llm_output.model_dump(), tool_logs

        # 执行工具
        logger.info("[Tool] 第 %d 轮执行 %d 个工具", round_num, len(tool_calls))
        round_results = ToolExecutor.execute_tool_calls(tool_calls)
        for r in round_results:
            tool_logs.append({
                "round": round_num,
                "tool": r["tool"],
                "args": r["args"],
                "result": r["result"],
                "latency_ms": r.get("latency_ms", 0),
                "timestamp": datetime.now().isoformat(),
            })

        # 全部失败 → 降级
        if all(not r["result"].get("success", False) for r in round_results):
            logger.warning("[Tool] 第 %d 轮全部工具失败，降级", round_num)
            final = await _fallback_generate(
                generator, messages, query, context_chunks,
                tool_results=round_results,
                tool_unsolved_reason="all_tools_failed",
            )
            return final, tool_logs

        # 注入结果，继续下一轮
        tool_text = ToolExecutor.format_results_for_llm(round_results)
        messages.append({"role": "user", "content": tool_text})
        messages.append({
            "role": "user",
            "content": "请基于以上工具执行结果，重新生成完整的审查结果 JSON。如果还需要更多工具调用，请在 tool_calls 中列出。",
        })

    # 达到最大轮数 → 降级
    logger.warning("[Tool] %d 轮后仍未解决，降级", max_rounds)
    final = await _fallback_generate(
        generator, messages, query, context_chunks,
        tool_unsolved_reason=f"exceeded {max_rounds} rounds",
    )
    return final, tool_logs


async def _fallback_generate(
    generator: Any,
    messages: list[dict],
    query: str,
    context_chunks: list[dict],
    tool_results: list[dict] | None = None,
    tool_unsolved_reason: str = "",
) -> dict | None:
    """降级生成：让 LLM 基于已有信息做最好判断。"""
    from src.rule_review.generator import parse_llm_output

    fallback_prompt = (
        "## 注意\n"
        "工具调用未能完成所有计算。请基于目前已检索到的规则原文，"
        "尽力做出最好的判断。如果信息不足以做出确定判断，请标记为'无法判断'。"
    )
    messages.append({"role": "user", "content": fallback_prompt})

    # 如果有工具结果，也注入
    if tool_results:
        tool_text = ToolExecutor.format_results_for_llm(tool_results)
        messages.append({"role": "user", "content": tool_text})

    raw_text = await generator.generate_raw(messages)
    output = parse_llm_output(raw_text)

    if output is None:
        return {
            "decision": "无法判断",
            "reason": "工具调用未能完成且 LLM 无法做出判断。",
            "evidence": [],
            "confidence": 0.0,
            "tool_unsolved": True,
            "tool_unsolved_reason": tool_unsolved_reason,
        }

    result = output.model_dump()
    result["tool_unsolved"] = True
    result["tool_unsolved_reason"] = tool_unsolved_reason
    return result


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _parse_markdown_table(text: str) -> list[dict]:
    """解析 Markdown 表格为 List[dict]。"""
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]

    # 找到表头行
    header_idx = -1
    for i, line in enumerate(lines):
        if line.startswith("|") and line.endswith("|"):
            header_idx = i
            break

    if header_idx < 0:
        return []

    # 解析表头
    headers = [h.strip() for h in lines[header_idx].split("|")[1:-1]]
    if not headers:
        return []

    # 跳过分隔符行
    data_start = header_idx + 1
    if data_start < len(lines) and re.match(r"^[\|\s\-:]+$", lines[data_start]):
        data_start += 1

    # 解析数据行
    rows = []
    for line in lines[data_start:]:
        if line.startswith("|") and line.endswith("|"):
            cells = [c.strip() for c in line.split("|")[1:-1]]
            row_dict = {}
            for j, cell in enumerate(cells):
                if j < len(headers):
                    row_dict[headers[j]] = cell
            rows.append(row_dict)

    return rows


def _fuzzy_match_column(target: str, columns: list[str]) -> str | None:
    """模糊匹配列名。先精确匹配，再包含匹配。"""
    # 精确匹配
    target_clean = target.strip()
    for col in columns:
        if col == target_clean:
            return col

    # 包含匹配
    for col in columns:
        if target_clean in col or col in target_clean:
            return col

    return None


def _fuzzy_match_value(target: str, cell_value: str) -> bool:
    """模糊匹配值。支持精确匹配和包含匹配。"""
    target_clean = target.strip()
    cell_clean = cell_value.strip()

    if target_clean == cell_clean:
        return True
    if target_clean in cell_clean or cell_clean in target_clean:
        return True

    return False


def _extract_number_and_unit(raw: str) -> tuple[float | None, str]:
    """从字符串中提取数值和单位。"""
    # 匹配 "数字 + 可选空格 + 单位"
    match = re.match(r"^\s*([+-]?\d+\.?\d*)\s*(.*)$", raw)
    if match:
        value = float(match.group(1))
        unit = match.group(2).strip()
        return value, unit

    return None, raw


def _cn_to_int(cn: str) -> int:
    """中文数字 → 阿拉伯数字。"""
    if cn.isdigit():
        return int(cn)
    return _CN_NUM_MAP.get(cn, 0)


def _find_in_chunks(
    keyword: str,
    chunks: list[dict],
    doc_filter: str | None = None,
) -> dict | None:
    """在 chunks 中搜索关键词，返回最佳匹配。"""
    for chunk in chunks:
        text = chunk.get("text", "")
        doc_id = chunk.get("doc_id", "")
        if doc_filter and doc_id and doc_id != doc_filter:
            continue
        if keyword in text:
            return chunk
    return None
