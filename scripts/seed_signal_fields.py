#!/usr/bin/env python3
"""
expected_keywords / expected_sources 派生与白名单（build 与 validate 共用）
============================================================================
RL 纯规则 verifier 需要每条种子携带证据信号字段（判分真值）：

- expected_keywords：静态关键词（模板声明：规则主题词 + 基线数值）
  + 槽位派生（填充文本 + mwh 数值），供 keyword_recall 判分
- expected_sources：静态来源（模板声明）+ 槽位派生（doc_name/other_doc_name
  填充文本），供 source_recall 判分

与 decision_evidence 同构：build 时记录 keyword_evidence / source_evidence
（{static, derived} 双轨），validate 时按派生规则重算比对，防手改 spec 漂移。

派生排除槽位：out_region / no_such_article —— 语义是「未命中 → 修正/反馈」，
强制模型复述原词会误罚 recovery 场景。
"""

from __future__ import annotations

# 不派生关键词的槽位：
# - out_region / no_such_article：未命中语义，强制复述会误罚 recovery
# - citation_text：整句引用（超关键词长度限制），核验由工具结果驱动而非关键词
EXCLUDED_KEYWORD_SLOTS = {"out_region", "no_such_article", "citation_text"}
# 来源派生槽位：填充文本并入 expected_sources
SOURCE_SLOTS = {"doc_name", "other_doc_name"}


def _dedup(items: list[str]) -> list[str]:
    """去重保序。"""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def derive_keywords(assignment: dict, slot_names: list[str]) -> list[str]:
    """槽位派生关键词：每个填充文本 + 带 mwh 条目的数值字符串。

    exclude 槽位跳过；结果去重保序。
    """
    kws: list[str] = []
    for name in slot_names:
        if name in EXCLUDED_KEYWORD_SLOTS:
            continue
        value = assignment[name]
        if value.get("text"):
            kws.append(value["text"])
        if value.get("mwh") is not None:
            kws.append(str(value["mwh"]))
    return _dedup(kws)


def derive_sources(assignment: dict, slot_names: list[str]) -> list[str]:
    """槽位派生来源：doc_name / other_doc_name 填充文本。"""
    return _dedup(
        assignment[n]["text"] for n in slot_names if n in SOURCE_SLOTS
    )


def merge_signals(
    static_keywords: list[str],
    static_sources: list[str],
    assignment: dict,
    slot_names: list[str],
) -> dict:
    """静态 + 派生合并（static 在前 derived 在后，去重保序）。

    返回 {expected_keywords, keyword_evidence, expected_sources, source_evidence}，
    evidence 字段供 validator 重算比对（同 decision_evidence 模式）。
    """
    derived_kws = derive_keywords(assignment, slot_names)
    derived_srcs = derive_sources(assignment, slot_names)
    return {
        "expected_keywords": _dedup(list(static_keywords) + derived_kws),
        "keyword_evidence": {
            "static": list(static_keywords),
            "derived": derived_kws,
        },
        "expected_sources": _dedup(list(static_sources) + derived_srcs),
        "source_evidence": {
            "static": list(static_sources),
            "derived": derived_srcs,
        },
    }


def source_whitelist(facts: dict, slot_vocab: dict) -> list[str]:
    """来源白名单：facts.docs 去 .pdf 后缀 ∪ doc_name ∪ other_doc_name。"""
    names: set[str] = set()
    for doc in facts.get("docs", []):
        names.add(doc.removesuffix(".pdf") if doc.endswith(".pdf") else doc)
    names.update(slot_vocab.get("doc_name", []))
    names.update(slot_vocab.get("other_doc_name", []))
    return sorted(names)
