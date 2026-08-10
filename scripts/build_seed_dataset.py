#!/usr/bin/env python3
"""
句式模板 × 槽位词表 → 完整种子数据集
======================================
输入：data/evaluation/rule_review_seed_spec.json（v2：facts + slot_vocab + templates）
输出：data/evaluation/rule_review_seed_dataset.json（完整种子，无占位符）

扩量机制（两步走）：
  ① 句式改写 —— 同一语义场景写成多种问法（templates 手工编写，覆盖 9 工具）
  ② 槽位替换 —— {region}/{price}/{date}... 填充全局词表（slot_vocab）不同候选词

决策自动重判：
  decision_rule = "gt760_bad" → 填入槽位的数值 mwh > facts.price_cap_mwh(760) → 不符合，否则符合
  decision_rule = "fixed"     → 模板固定 expected_decision 值

使用：
  python scripts/build_seed_dataset.py [--spec ...] [--output ...] [--target-count 1000]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DEFAULT_SPEC_PATH = "data/evaluation/rule_review_seed_spec.json"
DEFAULT_OUTPUT_PATH = "data/evaluation/rule_review_seed_dataset.json"
# 每个句式模板的变体上限：防止组合爆炸的模板（如 region×price×date）
# 挤占小组合模板的分布，保证 9 工具训练数据均衡
TEMPLATE_VARIANT_CAP = 25

VALID_DECISIONS = ("符合", "不符合", "部分符合", "无法判断")


# ============================================================================
# 槽位词表解析
# ============================================================================


def resolve_slot_values(slot_name: str, slot_vocab: dict) -> list[dict]:
    """把词表条目统一为 {text, mwh?} 对象（支持 str 或 dict 两种写法）。"""
    items = slot_vocab.get(slot_name, [])
    resolved = []
    for item in items:
        if isinstance(item, str):
            resolved.append({"text": item})
        else:
            resolved.append({"text": item["text"], "mwh": item.get("mwh")})
    return resolved


def has_mwh(assignment: dict) -> int | None:
    """从槽位赋值中找第一个带数值的（用于自动重判）。"""
    for _, value in assignment.items():
        mwh = value.get("mwh")
        if mwh is not None:
            return mwh
    return None


# ============================================================================
# 决策重判
# ============================================================================


def compute_decision(
    template: dict, assignment: dict, facts: dict, slot_vocab: dict
) -> str:
    """按模板的 decision_rule 重判决策。

    - "gt760_bad"：价格上限语义，mwh > 基线(760) → 不符合；等于/低于 → 符合
    - "gte_10000_good"：交易量下限语义，kwh >= 基线(10000) → 符合；否则不符合
    - "fixed"：模板固定 expected_decision
    """
    rule = template.get("decision_rule", "fixed")
    if rule == "gt760_bad":
        baseline = facts.get("price_cap_mwh", 760)
        mwh = has_mwh(assignment)
        if mwh is None:
            raise ValueError(
                f"模板 {template['template_id']} decision_rule=gt760_bad "
                "但槽位赋值中没有带 mwh 的候选词"
            )
        return "不符合" if mwh > baseline else "符合"
    if rule == "gte_10000_good":
        baseline = facts.get("min_quantity_kwh", 10000)
        mwh = has_mwh(assignment)
        if mwh is None:
            raise ValueError(
                f"模板 {template['template_id']} decision_rule=gte_10000_good "
                "但槽位赋值中没有带 mwh 的候选词"
            )
        return "符合" if mwh >= baseline else "不符合"
    if rule == "fixed":
        return template.get("expected_decision", "无法判断")
    raise ValueError(f"未知 decision_rule: {rule!r}")


# ============================================================================
# 变体生成
# ============================================================================


def render_query(query_template: str, assignment: dict) -> str:
    """槽位占位符替换为具体词（query_template 中只允许已声明槽位）。"""
    kwargs = {name: value["text"] for name, value in assignment.items()}
    return query_template.format(**kwargs)


def generate_variants(
    template: dict, slot_vocab: dict, facts: dict, max_variants: int | None = None
) -> list[dict]:
    """单个句式模板 × 槽位词表 → 变体列表（笛卡尔组合）。"""
    slot_names = template.get("slots", [])
    candidates = {
        name: resolve_slot_values(name, slot_vocab) for name in slot_names
    }
    combos = [
        dict(zip(candidates.keys(), values))
        for values in itertools.product(*candidates.values())
    ]
    if max_variants is not None and len(combos) > max_variants:
        # 均匀步长截取，保持词表覆盖且结果确定（不用随机）
        step = math.ceil(len(combos) / max_variants)
        combos = combos[::step][:max_variants]

    variants = []
    for i, combo in enumerate(combos, 1):
        query = render_query(template["query_template"], combo)
        workflow = template.get("expected_workflow", "")
        if workflow:
            try:
                workflow = render_query(workflow, combo)
            except (KeyError, IndexError):
                pass  # workflow 含非槽位占位符时保持原样
        variants.append({
            "variant_id": f"{template['template_id']}_{i:03d}",
            "template_id": template["template_id"],
            "category": template.get("category", ""),
            "query": query,
            "expected_tools": list(template.get("expected_tools", [])),
            "expected_workflow": workflow,
            "expected_decision": compute_decision(template, combo, facts, slot_vocab),
            "decision_rule": template.get("decision_rule", "fixed"),
        })
    return variants


# ============================================================================
# 数据集构建
# ============================================================================


def trim_by_template(
    variants: list[dict], target: int, min_per_template: int = 1
) -> list[dict]:
    """按模板分组按比例取整裁剪（保底每模板 ≥1 条），不足 target 时逐条补差。

    结果确定（无随机），分布近似保留。
    """
    groups: dict[str, list[dict]] = {}
    for v in variants:
        groups.setdefault(v["template_id"], []).append(v)
    ratio = target / len(variants)
    picked: dict[str, list[dict]] = {}
    for gid, vs in groups.items():
        n = max(min_per_template, min(len(vs), round(len(vs) * ratio)))
        if len(vs) <= n:
            picked[gid] = list(vs)
        else:
            step = math.ceil(len(vs) / n)
            picked[gid] = vs[::step][:n]

    total = sum(len(v) for v in picked.values())
    # 补差：优先给剩余空间最大的组补未选条目（组内按序扫描）
    while total < target:
        best_gid: str | None = None
        best_added: dict | None = None
        for gid, vs in groups.items():
            cur = picked[gid]
            if len(cur) >= len(vs):
                continue
            for v in vs:
                if v not in cur:
                    best_gid, best_added = gid, v
                    break
        if best_gid is None:
            break
        picked[best_gid].append(best_added)
        total += 1

    return [v for vs in picked.values() for v in vs]


def build_dataset(
    spec_path: str = DEFAULT_SPEC_PATH,
    target_count: int = 1000,
    negative_ceiling: int | None = None,
) -> tuple[list[dict], dict]:
    """生成完整种子数据集。

    裁剪策略：
    - 负样本组（expected_tools 为空）优先保留到 negative_ceiling
      （默认 target_count × 13%，即 10-15% 占比目标的上沿），
      避免无槽位固定负样本在整体均匀裁剪时被清掉；
    - 其余模板按比例裁剪，保底每模板 ≥1 条。
    """
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    facts = spec.get("facts", {})
    slot_vocab = spec.get("slot_vocab", {})
    templates = spec.get("templates", [])

    all_variants: list[dict] = []
    per_template: dict[str, int] = {}
    for template in templates:
        variants = generate_variants(
            template, slot_vocab, facts, max_variants=TEMPLATE_VARIANT_CAP
        )
        all_variants.extend(variants)
        per_template[template["template_id"]] = len(variants)

    if target_count and len(all_variants) > target_count:
        ceiling = (
            negative_ceiling
            if negative_ceiling is not None
            else max(100, int(target_count * 0.13))
        )
        neg = [v for v in all_variants if not v["expected_tools"]]
        non_neg = [v for v in all_variants if v["expected_tools"]]
        if len(neg) > ceiling:
            neg = trim_by_template(neg, ceiling)
        remaining = target_count - len(neg)
        if len(non_neg) > remaining:
            non_neg = trim_by_template(non_neg, remaining)
        all_variants = neg + non_neg

    stats = {
        "templates": len(templates),
        "generated_total": sum(per_template.values()),
        "final_count": len(all_variants),
        "negative_count": sum(1 for v in all_variants if not v["expected_tools"]),
        "per_template": per_template,
    }
    return all_variants, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="句式×槽位 → 完整种子数据集")
    parser.add_argument("--spec", default=DEFAULT_SPEC_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--target-count", type=int, default=1000)
    args = parser.parse_args()

    variants, stats = build_dataset(args.spec, args.target_count)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"version": 2, "count": len(variants), "seeds": variants},
                  f, ensure_ascii=False, indent=2)

    print(f"句式模板: {stats['templates']} 个")
    print(f"全量组合: {stats['generated_total']} 条 → 裁剪后: {stats['final_count']} 条 → {out_path}")


if __name__ == "__main__":
    main()
