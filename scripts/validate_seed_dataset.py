#!/usr/bin/env python3
"""
种子数据集校验 + 覆盖统计
==========================
对 data/evaluation/rule_review_seed_dataset.json 做双层校验：

1. 数据集层（每条种子）：
   - query 无 `{` 占位符残留
   - expected_tools 工具名 ∈ tools_config.json 白名单
   - expected_decision ∈ 4 枚举
   - decision_rule=gt760_bad 的条目：决策与数值基线一致性重算
   - expected_workflow 提及的工具集合 == expected_tools 集合

2. 规格层（rule_review_seed_spec.json）：
   - 模板槽位引用存在于 slot_vocab
   - decision_rule 合法
   - 槽位词表实体多样性统计

使用：
  python scripts/validate_seed_dataset.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DEFAULT_DATASET_PATH = "data/evaluation/rule_review_seed_dataset.json"
DEFAULT_SPEC_PATH = "data/evaluation/rule_review_seed_spec.json"
TOOLS_CONFIG_PATH = "data/env_variables/tools_config.json"

VALID_DECISIONS = ("符合", "不符合", "部分符合", "无法判断")
VALID_DECISION_RULES = ("fixed", "gt760_bad", "gte_10000_good")

# expected_workflow 里的工具名以 "工具名(" 形式出现
_TOOL_CALL_RE = re.compile(r"([a-z_]+)\(")


def load_tool_names() -> set[str]:
    with open(TOOLS_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    return set(config.get("tools", {}).keys())


# ============================================================================
# 数据集层校验
# ============================================================================


def _recompute_decision(evidence: dict) -> str | None:
    """按决策依据重算决策，用于一致性比对。"""
    rule = evidence.get("rule")
    if rule == "gt760_bad":
        return "不符合" if evidence["mwh"] > evidence["baseline"] else "符合"
    if rule == "gte_10000_good":
        return "符合" if evidence["mwh"] >= evidence["baseline"] else "不符合"
    return None


def validate_variant(variant: dict, tool_names: set[str]) -> list[str]:
    """校验单条种子，返回错误列表（空 = 合法）。"""
    errors: list[str] = []
    vid = variant.get("variant_id", "?")

    query = variant.get("query", "")
    if not query:
        errors.append(f"[{vid}] query 为空")
    elif "{" in query or "}" in query:
        errors.append(f"[{vid}] query 含占位符残留: {query!r}")

    decision = variant.get("expected_decision", "")
    if decision not in VALID_DECISIONS:
        errors.append(f"[{vid}] expected_decision 非法: {decision!r}")

    tools = variant.get("expected_tools", [])
    for tool in tools:
        if tool not in tool_names:
            errors.append(f"[{vid}] 未知工具: {tool!r}")

    # workflow 提及工具与 expected_tools 一致（负样本两者皆空）
    workflow = variant.get("expected_workflow", "")
    mentioned = set(_TOOL_CALL_RE.findall(workflow))
    if mentioned != set(tools):
        errors.append(
            f"[{vid}] expected_workflow 提及工具 {sorted(mentioned)} "
            f"≠ expected_tools {sorted(tools)}"
        )

    # 决策与数值基线一致性：按 decision_evidence 重算比对
    evidence = variant.get("decision_evidence")
    if evidence and evidence.get("rule") != "fixed":
        recomputed = _recompute_decision(evidence)
        if recomputed is not None and decision != recomputed:
            errors.append(
                f"[{vid}] expected_decision {decision!r} 与数值基线不一致"
                f"（应 {recomputed!r}，依据 {evidence}）"
            )
    return errors


def validate_dataset(
    dataset: dict, tool_names: set[str]
) -> tuple[list[str], list[dict]]:
    """校验全部种子，返回（错误列表, 合法种子列表）。"""
    errors: list[str] = []
    valid = []
    for variant in dataset.get("seeds", []):
        errs = validate_variant(variant, tool_names)
        if errs:
            errors.extend(errs)
        else:
            valid.append(variant)
    return errors, valid


# ============================================================================
# 覆盖统计
# ============================================================================


def build_stats(seeds: list[dict]) -> dict:
    """按 category / 工具 / 决策 / 句式 / query 多样性统计。"""
    by_category: dict[str, int] = {}
    by_tool: dict[str, int] = {}
    by_decision: dict[str, int] = {}
    template_ids: set[str] = set()
    queries: set[str] = set()
    negative = 0

    for seed in seeds:
        cat = seed.get("category", "")
        by_category[cat] = by_category.get(cat, 0) + 1
        tools = seed.get("expected_tools", [])
        if not tools:
            negative += 1
        for tool in tools:
            by_tool[tool] = by_tool.get(tool, 0) + 1
        dec = seed.get("expected_decision", "")
        by_decision[dec] = by_decision.get(dec, 0) + 1
        template_ids.add(seed.get("template_id", ""))
        queries.add(seed.get("query", ""))

    total = len(seeds)
    return {
        "total": total,
        "by_category": dict(sorted(by_category.items(), key=lambda x: -x[1])),
        "by_tool": dict(sorted(by_tool.items(), key=lambda x: -x[1])),
        "by_decision": by_decision,
        "negative_ratio": round(negative / total, 3) if total else 0.0,
        "template_count": len(template_ids),
        "unique_query_count": len(queries),
    }


# ============================================================================
# 规格层校验
# ============================================================================


def validate_spec(spec: dict) -> list[str]:
    """校验句式模板与槽位词表，返回错误列表。"""
    errors: list[str] = []
    slot_vocab = spec.get("slot_vocab", {})
    templates = spec.get("templates", [])
    tool_names = load_tool_names()

    for template in templates:
        tid = template.get("template_id", "?")
        rule = template.get("decision_rule", "fixed")
        if rule not in VALID_DECISION_RULES:
            errors.append(f"[{tid}] decision_rule 非法: {rule!r}")
        if rule == "fixed" and template.get("expected_decision") not in VALID_DECISIONS:
            errors.append(f"[{tid}] fixed 决策非法: {template.get('expected_decision')!r}")
        for slot in template.get("slots", []):
            if slot not in slot_vocab:
                errors.append(f"[{tid}] 槽位 {slot!r} 在 slot_vocab 中不存在")
        for tool in template.get("expected_tools", []):
            if tool not in tool_names:
                errors.append(f"[{tid}] 未知工具: {tool!r}")
        query_template = template.get("query_template", "")
        used = set(re.findall(r"{(\w+)}", query_template))
        declared = set(template.get("slots", []))
        if used != declared:
            errors.append(
                f"[{tid}] query_template 槽位 {sorted(used)} ≠ slots 声明 {sorted(declared)}"
            )
    return errors


def spec_vocab_stats(spec: dict) -> dict:
    """槽位词表实体多样性统计。"""
    result = {}
    for name, items in spec.get("slot_vocab", {}).items():
        result[name] = len(items)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="种子数据集校验 + 覆盖统计")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--spec", default=DEFAULT_SPEC_PATH)
    args = parser.parse_args()

    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    with open(args.spec, "r", encoding="utf-8") as f:
        spec = json.load(f)

    tool_names = load_tool_names()
    spec_errors = validate_spec(spec)
    dataset_errors, valid = validate_dataset(dataset, tool_names)

    print("=== 规格层校验 ===")
    if spec_errors:
        for e in spec_errors:
            print(f"  ✗ {e}")
    else:
        print("  ✓ 句式模板与槽位词表合法")

    print("\n=== 数据集层校验 ===")
    if dataset_errors:
        for e in dataset_errors[:30]:
            print(f"  ✗ {e}")
        print(f"  ...共 {len(dataset_errors)} 条错误")
    else:
        print(f"  ✓ {len(valid)} 条种子全部合法")

    print("\n=== 覆盖统计 ===")
    stats = build_stats(valid)
    print(f"  总条数: {stats['total']}（句式模板 {stats['template_count']} 个，去重 query {stats['unique_query_count']} 条）")
    print(f"  按类别: {stats['by_category']}")
    print(f"  按工具: {stats['by_tool']}")
    print(f"  按决策: {stats['by_decision']}")
    print(f"  负样本占比: {stats['negative_ratio']:.1%}")
    print(f"  词表实体数: {spec_vocab_stats(spec)}")

    if spec_errors or dataset_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
