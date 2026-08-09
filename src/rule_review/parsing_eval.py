"""
电力规则审查系统 - 解析精度评测（表格单元格识别准确率）

背景（docs/rule-review.md §4.3「解析分层」）:
- 重要政策问答表格 → 手动入库（手工整理 markdown，识别准确率 100%）
- 相对不重要的文档 → 本地 MinerU 解析（表格可完整识别，优于已弃用的 API 方案）
- 目标: 表格单元格识别准确率 ≥ TABLE_CELL_ACCURACY_TARGET (0.98)

评测口径:
- 以手工标注的期望表为基准，正确单元格数 / 期望表单元格总数
- 实际表缺行/缺列/单元格缺失记错误；实际表多出的行/列不计入分母
- 单元格按 (row, col) 索引对齐，strip 后逐格比较

数据: data/evaluation/table_parse_cases.json
CLI: python -m src.rule_review.parsing_eval [--cases PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

from src.rule_review.parsers import markdown_to_page_content

logger = logging.getLogger(__name__)

DEFAULT_TABLE_PARSE_CASES_PATH = "data/evaluation/table_parse_cases.json"

# 解析精度目标：表格单元格识别准确率（手动入库 100%、MinerU ≥98% 的评测断言阈值）
TABLE_CELL_ACCURACY_TARGET = 0.98


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


@dataclass
class TableParseCase:
    """表格解析精度评测用例。"""

    id: str
    name: str = ""
    markdown_table: str = ""  # 解析器输出（markdown 表格文本）
    expected_rows: list[list[str]] = field(default_factory=list)  # 手工标注的期望表


@dataclass
class TableCellEvalResult:
    """单表解析精度结果。"""

    case_id: str
    total_cells: int
    correct_cells: int
    accuracy: float  # correct / total
    errors: list[dict] = field(default_factory=list)  # [{"row", "col", "expected", "actual"}]


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------


def parse_markdown_table(markdown: str) -> list[list[str]]:
    """从 markdown 表格文本提取行数据。

    复用 parsers.markdown_to_page_content 取 table_blocks[0].rows——
    与生产解析路径（MinerU 输出 / 手动入库）同一拆分口径，杜绝「评测用另一套解析」。
    无表格块时返回空列表。
    """
    page = markdown_to_page_content(markdown)
    if not page.table_blocks:
        return []
    return page.table_blocks[0].rows


def compute_table_cell_accuracy(
    actual_rows: list[list[str]],
    expected_rows: list[list[str]],
) -> TableCellEvalResult:
    """表格单元格识别准确率 = 正确单元格数 / 期望表单元格总数。

    对齐规则（以手工标注的期望表为基准）:
    - 实际表按 (row, col) 索引对齐比较，strip 后相等即正确；
    - 实际表缺行 / 缺列 / 某行列数不足 → 缺失单元格记错误；
    - 实际表多出的行 / 列不计入分母（不惩罚解析器多识别内容）。
    """
    total_cells = sum(len(row) for row in expected_rows)
    correct_cells = 0
    errors: list[dict] = []

    for r_idx, expected_row in enumerate(expected_rows):
        actual_row = actual_rows[r_idx] if r_idx < len(actual_rows) else []
        for c_idx, expected_cell in enumerate(expected_row):
            actual_cell = (
                actual_row[c_idx].strip() if c_idx < len(actual_row) else ""
            )
            if actual_cell == expected_cell.strip():
                correct_cells += 1
            else:
                errors.append(
                    {
                        "row": r_idx,
                        "col": c_idx,
                        "expected": expected_cell,
                        "actual": actual_cell,
                    }
                )

    accuracy = correct_cells / total_cells if total_cells else 0.0
    return TableCellEvalResult(
        case_id="",
        total_cells=total_cells,
        correct_cells=correct_cells,
        accuracy=accuracy,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# 评测运行
# ---------------------------------------------------------------------------


def load_table_parse_cases(path: str | None = None) -> list[TableParseCase]:
    """从 JSON 文件加载评测用例。"""
    p = Path(path or DEFAULT_TABLE_PARSE_CASES_PATH)
    if not p.exists():
        logger.warning("[ParsingEval] 评测数据不存在: %s", p)
        return []

    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)

    cases: list[TableParseCase] = []
    for item in raw:
        try:
            cases.append(
                TableParseCase(
                    id=item["id"],
                    name=item.get("name", ""),
                    markdown_table=item.get("markdown_table", ""),
                    expected_rows=item.get("expected_rows", []),
                )
            )
        except KeyError as e:
            logger.warning("[ParsingEval] 跳过无效用例 %s: 缺少字段 %s", item.get("id", "?"), e)
    return cases


def evaluate_table_parsing(
    cases: list[TableParseCase],
) -> tuple[list[TableCellEvalResult], float]:
    """批量评测，返回 (逐表结果, 总体准确率)。

    实际行来自 parse_markdown_table(case.markdown_table)——即把解析器输出
    （MinerU markdown 或手动入库原文）按生产口径还原为表格行后与期望表比较。
    """
    results: list[TableCellEvalResult] = []
    for case in cases:
        actual_rows = parse_markdown_table(case.markdown_table)
        result = compute_table_cell_accuracy(actual_rows, case.expected_rows)
        result.case_id = case.id
        results.append(result)

    total_cells = sum(r.total_cells for r in results)
    correct_cells = sum(r.correct_cells for r in results)
    overall = correct_cells / total_cells if total_cells else 0.0
    return results, overall


def main(argv: list[str] | None = None) -> int:
    """CLI: python -m src.rule_review.parsing_eval [--cases PATH]

    打印逐表与总体准确率；总体准确率 ≥ TABLE_CELL_ACCURACY_TARGET 时 PASS，
    否则 FAIL 并返回退出码 1。
    """
    parser = argparse.ArgumentParser(
        description="规则审查系统解析精度评测（表格单元格识别准确率）"
    )
    parser.add_argument(
        "--cases",
        default=DEFAULT_TABLE_PARSE_CASES_PATH,
        help="评测数据 JSON 路径",
    )
    args = parser.parse_args(argv)

    cases = load_table_parse_cases(args.cases)
    if not cases:
        print(f"[ParsingEval] 评测数据为空或加载失败: {args.cases}")
        return 1

    results, overall = evaluate_table_parsing(cases)
    print(f"\n[ParsingEval] 共 {len(cases)} 张表 | 目标准确率 ≥ {TABLE_CELL_ACCURACY_TARGET:.0%}")
    for r in results:
        status = "PASS" if r.accuracy >= TABLE_CELL_ACCURACY_TARGET else "FAIL"
        print(
            f"  [{status}] {r.case_id}: 准确率 {r.accuracy:.2%} "
            f"({r.correct_cells}/{r.total_cells} 单元格)"
        )
        for err in r.errors[:5]:
            print(f"      行{err['row']} 列{err['col']}: 期望「{err['expected']}」实际「{err['actual']}」")
        if len(r.errors) > 5:
            print(f"      ... 共 {len(r.errors)} 处错误")

    passed = overall >= TABLE_CELL_ACCURACY_TARGET
    print(
        f"\n[ParsingEval] 总体准确率 {overall:.2%}"
        f" | {'PASS' if passed else 'FAIL'}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
