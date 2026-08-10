"""
种子数据集生成与校验测试

覆盖 scripts/build_seed_dataset.py 与 scripts/validate_seed_dataset.py：
- 槽位词表解析（str / dict 两种写法）
- 决策自动重判（gt760_bad：mwh > 760 → 不符合）
- 变体生成（笛卡尔组合、max_variants 均匀截取、占位符替换）
- 数据集校验（占位符残留/未知工具/决策枚举/workflow 一致性/负样本）
- 规格层校验（槽位引用/decision_rule/占位符与声明一致）
- 覆盖统计（负样本占比、工具分布）
- 真实 spec 文件集成：生成 → 校验全过 + 数量接近 1000
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.build_seed_dataset import (  # noqa: E402
    compute_decision,
    generate_variants,
    render_query,
    resolve_slot_values,
)
from scripts.seed_signal_fields import (  # noqa: E402
    _dedup,
    derive_keywords,
    derive_sources,
    merge_signals,
    source_whitelist,
)
from scripts.validate_seed_dataset import (  # noqa: E402
    build_stats,
    validate_dataset,
    validate_spec,
    validate_variant,
)

_SPEC_PATH = "data/evaluation/rule_review_seed_spec.json"
_DATASET_PATH = "data/evaluation/rule_review_seed_dataset.json"
_TOOL_NAMES = {"extract_table_data", "arithmetic_compare", "locate_clause"}


# ============================================================================
# 槽位词表解析
# ============================================================================


class TestResolveSlotValues:
    def test_str_item(self):
        values = resolve_slot_values("region", {"region": ["冀北", "山西"]})
        assert values == [{"text": "冀北"}, {"text": "山西"}]

    def test_dict_item_with_mwh(self):
        values = resolve_slot_values(
            "price", {"price": [{"text": "800元/MWh", "mwh": 800}]}
        )
        assert values == [{"text": "800元/MWh", "mwh": 800}]

    def test_missing_slot_returns_empty(self):
        assert resolve_slot_values("nope", {}) == []


# ============================================================================
# 决策重判
# ============================================================================


class TestComputeDecision:
    def test_gt760_bad_above(self):
        tpl = {"template_id": "t", "decision_rule": "gt760_bad"}
        assert compute_decision(tpl, {"price": {"text": "800元/MWh", "mwh": 800}},
                                {"price_cap_mwh": 760}, {}) == "不符合"

    def test_gt760_bad_equal(self):
        tpl = {"template_id": "t", "decision_rule": "gt760_bad"}
        assert compute_decision(tpl, {"price": {"text": "760元/MWh", "mwh": 760}},
                                {"price_cap_mwh": 760}, {}) == "符合"

    def test_gt760_bad_below(self):
        tpl = {"template_id": "t", "decision_rule": "gt760_bad"}
        assert compute_decision(tpl, {"price": {"text": "620元/MWh", "mwh": 620}},
                                {"price_cap_mwh": 760}, {}) == "符合"

    def test_unit_converted_value(self):
        # 0.8元/kWh = 800元/MWh，重判应为不符合
        tpl = {"template_id": "t", "decision_rule": "gt760_bad"}
        assert compute_decision(tpl, {"price": {"text": "0.8元/kWh", "mwh": 800}},
                                {"price_cap_mwh": 760}, {}) == "不符合"

    def test_fixed(self):
        tpl = {"template_id": "t", "decision_rule": "fixed",
               "expected_decision": "无法判断"}
        assert compute_decision(tpl, {}, {"price_cap_mwh": 760}, {}) == "无法判断"

    def test_gt760_bad_missing_mwh_raises(self):
        tpl = {"template_id": "t", "decision_rule": "gt760_bad"}
        with pytest.raises(ValueError):
            compute_decision(tpl, {"region": {"text": "冀北"}},
                             {"price_cap_mwh": 760}, {})


# ============================================================================
# 变体生成
# ============================================================================


class TestGenerateVariants:
    SPEC = {
        "facts": {"price_cap_mwh": 760},
        "slot_vocab": {
            "region": ["冀北", "山西"],
            "price": [
                {"text": "800元/MWh", "mwh": 800},
                {"text": "620元/MWh", "mwh": 620},
            ],
        },
    }

    def test_cartesian_combination(self):
        tpl = {
            "template_id": "compare_001",
            "category": "compare",
            "query_template": "{region}的日前现货出清电价{price}是否超过价格上限？",
            "expected_tools": ["extract_table_data", "arithmetic_compare"],
            "expected_workflow": "extract_table_data({region}, 上限) → arithmetic_compare({price} gt 760)",
            "decision_rule": "gt760_bad",
            "slots": ["region", "price"],
        }
        variants = generate_variants(tpl, self.SPEC["slot_vocab"],
                                     self.SPEC["facts"])
        assert len(variants) == 4  # 2 region × 2 price

        queries = {v["query"] for v in variants}
        assert "冀北的日前现货出清电价800元/MWh是否超过价格上限？" in queries
        assert "山西的日前现货出清电价620元/MWh是否超过价格上限？" in queries
        # 无占位符残留
        assert all("{" not in v["query"] for v in variants)

        by_query = {v["query"]: v for v in variants}
        assert by_query["冀北的日前现货出清电价800元/MWh是否超过价格上限？"]["expected_decision"] == "不符合"
        assert by_query["冀北的日前现货出清电价620元/MWh是否超过价格上限？"]["expected_decision"] == "符合"
        # 工具链继承
        assert variants[0]["expected_tools"] == ["extract_table_data", "arithmetic_compare"]
        assert variants[0]["template_id"] == "compare_001"
        # 决策依据记录（供校验重算）
        assert variants[0]["decision_evidence"] == {
            "rule": "gt760_bad", "mwh": 800, "baseline": 760
        }

    def test_max_variants_uniform_slice(self):
        tpl = {
            "template_id": "t",
            "query_template": "{price}",
            "expected_tools": [],
            "expected_workflow": "",
            "decision_rule": "fixed",
            "expected_decision": "无法判断",
            "slots": ["price"],
        }
        variants = generate_variants(tpl, self.SPEC["slot_vocab"],
                                     self.SPEC["facts"], max_variants=1)
        assert len(variants) == 1

    def test_no_slots_template(self):
        tpl = {
            "template_id": "neg_001",
            "category": "negative",
            "query_template": "电力交易为什么要有价格上限？",
            "expected_tools": [],
            "expected_workflow": "",
            "decision_rule": "fixed",
            "expected_decision": "无法判断",
            "slots": [],
        }
        variants = generate_variants(tpl, self.SPEC["slot_vocab"],
                                     self.SPEC["facts"])
        assert len(variants) == 1
        assert variants[0]["query"] == "电力交易为什么要有价格上限？"
        assert variants[0]["expected_tools"] == []
        assert variants[0]["expected_decision"] == "无法判断"


class TestRenderQuery:
    def test_render(self):
        out = render_query("{region}的{price}", {"region": {"text": "冀北"}, "price": {"text": "800元/MWh"}})
        assert out == "冀北的800元/MWh"


# ============================================================================
# 数据集层校验
# ============================================================================


class TestValidateVariant:
    def test_valid(self):
        v = {
            "variant_id": "v1",
            "query": "冀北的日前现货出清电价800元/MWh是否超过价格上限？",
            "expected_tools": ["extract_table_data", "arithmetic_compare"],
            "expected_workflow": "extract_table_data(冀北, 上限) → arithmetic_compare(800 gt 760)",
            "expected_decision": "不符合",
        }
        assert validate_variant(v, _TOOL_NAMES) == []

    def test_placeholder_remaining(self):
        v = {
            "variant_id": "v1",
            "query": "{region}的日前现货出清电价上限是多少？",
            "expected_tools": ["extract_table_data"],
            "expected_workflow": "extract_table_data(冀北, 上限)",
            "expected_decision": "无法判断",
        }
        errors = validate_variant(v, _TOOL_NAMES)
        assert any("占位符" in e for e in errors)

    def test_unknown_tool(self):
        v = {
            "variant_id": "v1",
            "query": "q",
            "expected_tools": ["no_such_tool"],
            "expected_workflow": "no_such_tool(x)",
            "expected_decision": "无法判断",
        }
        errors = validate_variant(v, _TOOL_NAMES)
        assert any("未知工具" in e for e in errors)

    def test_invalid_decision(self):
        v = {
            "variant_id": "v1",
            "query": "q",
            "expected_tools": [],
            "expected_workflow": "",
            "expected_decision": "也许吧",
        }
        errors = validate_variant(v, _TOOL_NAMES)
        assert any("expected_decision" in e for e in errors)

    def test_workflow_tool_mismatch(self):
        v = {
            "variant_id": "v1",
            "query": "q",
            "expected_tools": ["extract_table_data"],
            "expected_workflow": "arithmetic_compare(800 gt 760)",
            "expected_decision": "不符合",
        }
        errors = validate_variant(v, _TOOL_NAMES)
        assert any("≠" in e for e in errors)

    def test_negative_sample(self):
        v = {
            "variant_id": "v1",
            "query": "电力交易为什么要有价格上限？",
            "expected_tools": [],
            "expected_workflow": "",
            "expected_decision": "无法判断",
        }
        assert validate_variant(v, _TOOL_NAMES) == []

    def test_decision_baseline_consistent(self):
        # 800 > 760 → 不符合，决策与依据一致
        v = {
            "variant_id": "v1",
            "query": "800元/MWh是否超过价格上限？",
            "expected_tools": ["extract_table_data", "arithmetic_compare"],
            "expected_workflow": "extract_table_data(x) → arithmetic_compare(800 gt 760)",
            "expected_decision": "不符合",
            "decision_evidence": {"rule": "gt760_bad", "mwh": 800, "baseline": 760},
        }
        assert validate_variant(v, _TOOL_NAMES) == []

    def test_decision_baseline_inconsistent(self):
        # 800 > 760 却标 符合 → 校验必须拦截
        v = {
            "variant_id": "v1",
            "query": "800元/MWh是否超过价格上限？",
            "expected_tools": ["extract_table_data", "arithmetic_compare"],
            "expected_workflow": "extract_table_data(x) → arithmetic_compare(800 gt 760)",
            "expected_decision": "符合",
            "decision_evidence": {"rule": "gt760_bad", "mwh": 800, "baseline": 760},
        }
        errors = validate_variant(v, _TOOL_NAMES)
        assert any("与数值基线不一致" in e for e in errors)

    def test_decision_evidence_gte_10000(self):
        # 5000 < 10000 → 不符合
        v = {
            "variant_id": "v1",
            "query": "5000kWh是否满足要求？",
            "expected_tools": ["extract_table_data", "arithmetic_compare"],
            "expected_workflow": "extract_table_data(x) → arithmetic_compare(5000 gte 10000)",
            "expected_decision": "不符合",
            "decision_evidence": {"rule": "gte_10000_good", "mwh": 5000, "baseline": 10000},
        }
        assert validate_variant(v, _TOOL_NAMES) == []

    def test_decision_evidence_missing_skipped(self):
        # 无 decision_evidence 的旧数据不拦截（向后兼容）
        v = {
            "variant_id": "v1",
            "query": "800元/MWh是否超过价格上限？",
            "expected_tools": ["extract_table_data", "arithmetic_compare"],
            "expected_workflow": "extract_table_data(x) → arithmetic_compare(800 gt 760)",
            "expected_decision": "符合",
        }
        assert validate_variant(v, _TOOL_NAMES) == []


class TestValidateDataset:
    def test_collects_all_errors(self):
        good = {
            "variant_id": "g",
            "query": "q",
            "expected_tools": [],
            "expected_workflow": "",
            "expected_decision": "无法判断",
        }
        bad = dict(good)
        bad["variant_id"] = "b"
        bad["query"] = "{x}"
        errors, valid = validate_dataset(
            {"seeds": [good, bad]}, _TOOL_NAMES)
        assert len(errors) == 1
        assert len(valid) == 1


class TestBuildStats:
    def test_stats(self):
        seeds = [
            {"category": "compare", "expected_tools": ["arithmetic_compare"],
             "expected_decision": "不符合", "template_id": "t1", "query": "a"},
            {"category": "compare", "expected_tools": ["arithmetic_compare"],
             "expected_decision": "符合", "template_id": "t1", "query": "b"},
            {"category": "negative", "expected_tools": [],
             "expected_decision": "无法判断", "template_id": "n1", "query": "c"},
        ]
        stats = build_stats(seeds)
        assert stats["total"] == 3
        assert stats["by_tool"] == {"arithmetic_compare": 2}
        assert stats["by_decision"] == {"不符合": 1, "符合": 1, "无法判断": 1}
        assert stats["negative_ratio"] == round(1 / 3, 3)
        assert stats["template_count"] == 2
        assert stats["unique_query_count"] == 3


# ============================================================================
# 规格层校验
# ============================================================================


# ============================================================================
# 判分信号字段派生（v3）
# ============================================================================


class TestMergeSignals:
    def test_merge_static_and_derived(self):
        assignment = {
            "region": {"text": "冀北"},
            "price": {"text": "800元/MWh", "mwh": 800},
        }
        out = merge_signals(
            ["价格上限", "760"], ["省间电力现货交易规则"],
            assignment, ["region", "price"])
        assert out["expected_keywords"] == ["价格上限", "760", "冀北", "800元/MWh", "800"]
        assert out["keyword_evidence"] == {
            "static": ["价格上限", "760"], "derived": ["冀北", "800元/MWh", "800"],
        }
        assert out["expected_sources"] == ["省间电力现货交易规则"]
        assert out["source_evidence"] == {"static": ["省间电力现货交易规则"], "derived": []}

    def test_dedup_keep_first(self):
        # static 与 derived 重复 → 去重保留首现
        assignment = {"price": {"text": "760", "mwh": 760}}
        out = merge_signals(["价格上限", "760"], [], assignment, ["price"])
        assert out["expected_keywords"] == ["价格上限", "760"]

    def test_excluded_slots_not_derived(self):
        # out_region 语义是未命中→修正，强制复述会误罚 recovery
        assignment = {"out_region": {"text": "华北"}}
        out = merge_signals(["上限"], [], assignment, ["out_region"])
        assert out["expected_keywords"] == ["上限"]
        assert out["keyword_evidence"]["derived"] == []

    def test_doc_slot_derives_sources(self):
        assignment = {"doc_name": {"text": "省间电力现货交易规则"}}
        out = merge_signals([], [], assignment, ["doc_name"])
        assert out["expected_sources"] == ["省间电力现货交易规则"]
        assert out["source_evidence"]["derived"] == ["省间电力现货交易规则"]

    def test_mwh_derived_numeric(self):
        assignment = {"price": {"text": "0.8元/kWh", "mwh": 800}}
        out = merge_signals(["价格上限"], [], assignment, ["price"])
        assert out["expected_keywords"] == ["价格上限", "0.8元/kWh", "800"]

    def test_derive_keywords_direct(self):
        assert derive_keywords(
            {"region": {"text": "冀北"}, "no_such_article": {"text": "99"}},
            ["region", "no_such_article"]) == ["冀北"]

    def test_derive_sources_direct(self):
        assert derive_sources(
            {"doc_name": {"text": "省间电力现货交易规则"},
             "other_doc_name": {"text": "省内电力交易规则"},
             "region": {"text": "冀北"}},
            ["doc_name", "other_doc_name", "region"]) == \
            ["省间电力现货交易规则", "省内电力交易规则"]

    def test_source_whitelist(self):
        facts = {"docs": ["省间电力现货交易规则.pdf", "省内电力交易规则.pdf"]}
        vocab = {"doc_name": ["电力市场监管办法"], "other_doc_name": ["电力市场运营基本规则"]}
        assert source_whitelist(facts, vocab) == [
            "电力市场监管办法", "电力市场运营基本规则",
            "省内电力交易规则", "省间电力现货交易规则",
        ]


class TestValidateVariantSignals:
    WHITELIST = {"省间电力现货交易规则"}

    @staticmethod
    def _base(**over):
        v = {
            "variant_id": "v1",
            "query": "800元/MWh是否超过价格上限？",
            "expected_tools": ["extract_table_data"],
            "expected_workflow": "extract_table_data(x)",
            "expected_decision": "不符合",
            "expected_keywords": ["价格上限", "760", "800元/MWh", "800"],
            "keyword_evidence": {"static": ["价格上限", "760"], "derived": ["800元/MWh", "800"]},
            "expected_sources": ["省间电力现货交易规则"],
            "source_evidence": {"static": ["省间电力现货交易规则"], "derived": []},
        }
        v.update(over)
        return v

    def test_valid_signals(self):
        assert validate_variant(self._base(), _TOOL_NAMES, self.WHITELIST) == []

    def test_negative_with_keywords_intercepted(self):
        v = self._base(expected_tools=[], expected_workflow="", expected_decision="无法判断")
        errors = validate_variant(v, _TOOL_NAMES, self.WHITELIST)
        assert any("负样本" in e for e in errors)

    def test_non_negative_without_keywords_intercepted(self):
        v = self._base(expected_keywords=[], keyword_evidence=None)
        errors = validate_variant(v, _TOOL_NAMES, self.WHITELIST)
        assert any("不能为空" in e for e in errors)

    def test_keyword_too_long(self):
        v = self._base(expected_keywords=["价格上限规定的具体数值是多少元每兆瓦时", "760"])
        errors = validate_variant(v, _TOOL_NAMES, self.WHITELIST)
        assert any("长度超" in e for e in errors)

    def test_keyword_with_punctuation(self):
        v = self._base(expected_keywords=["价格上限，760", "760"])
        errors = validate_variant(v, _TOOL_NAMES, self.WHITELIST)
        assert any("含标点" in e for e in errors)

    def test_source_not_in_whitelist(self):
        v = self._base(expected_sources=["不存在的文档"])
        errors = validate_variant(v, _TOOL_NAMES, self.WHITELIST)
        assert any("白名单" in e for e in errors)

    def test_keyword_evidence_mismatch(self):
        v = self._base(keyword_evidence={"static": ["价格上限"], "derived": ["800元/MWh"]})
        errors = validate_variant(v, _TOOL_NAMES, self.WHITELIST)
        assert any("keyword_evidence" in e for e in errors)

    def test_source_evidence_mismatch(self):
        v = self._base(source_evidence={"static": ["省内电力交易规则"], "derived": []})
        errors = validate_variant(v, _TOOL_NAMES, self.WHITELIST)
        assert any("source_evidence" in e for e in errors)

    def test_old_data_without_signals_skipped(self):
        # v2 旧数据无信号字段（键缺失）→ 跳过信号校验（向后兼容）
        v = self._base()
        for k in ("expected_keywords", "keyword_evidence",
                  "expected_sources", "source_evidence"):
            del v[k]
        assert validate_variant(v, _TOOL_NAMES, self.WHITELIST) == []


class TestValidateSpecSignals:
    @staticmethod
    def _spec(template_extra: dict):
        tpl = {
            "template_id": "t",
            "query_template": "{region}的上限是多少？",
            "slots": ["region"],
            "expected_tools": ["extract_table_data"],
            "expected_workflow": "extract_table_data(region)",
            "decision_rule": "fixed",
            "expected_decision": "无法判断",
            "expected_keywords": ["价格上限"],
            "expected_sources": ["省间电力现货交易规则"],
        }
        tpl.update(template_extra)
        return {
            "facts": {"docs": ["省间电力现货交易规则.pdf"]},
            "slot_vocab": {"region": ["冀北"]},
            "templates": [tpl],
        }

    def test_valid_spec_signals(self):
        assert validate_spec(self._spec({})) == []

    def test_spec_missing_fields_reported(self):
        spec = self._spec({})
        del spec["templates"][0]["expected_keywords"]
        errors = validate_spec(spec)
        assert any("必须声明" in e for e in errors)

    def test_spec_non_negative_empty_keywords_no_derived_slots(self):
        # 静态空且无派生槽位 → variant 级会空，必须拦截
        spec = self._spec({
            "query_template": "电力交易的价格上限是多少？",
            "slots": [], "expected_keywords": [],
        })
        errors = validate_spec(spec)
        assert any("无派生槽位" in e for e in errors)

    def test_spec_non_negative_empty_keywords_with_derived_slots_ok(self):
        # 静态空但有派生槽位（locate 类）→ 允许，由槽位派生补足
        spec = self._spec({"expected_keywords": []})
        assert validate_spec(spec) == []

    def test_spec_source_not_in_whitelist(self):
        errors = validate_spec(self._spec({"expected_sources": ["不存在的文档"]}))
        assert any("白名单" in e for e in errors)

    def test_spec_negative_empty_signals_ok(self):
        spec = self._spec({
            "expected_tools": [], "expected_workflow": "",
            "expected_decision": "无法判断",
            "expected_keywords": [], "expected_sources": [],
        })
        assert validate_spec(spec) == []


class TestValidateSpec:
    def test_valid_spec(self):
        spec = {
            "facts": {"docs": ["省间电力现货交易规则.pdf"]},
            "slot_vocab": {"region": ["冀北"]},
            "templates": [{
                "template_id": "t",
                "query_template": "{region}的上限是多少？",
                "slots": ["region"],
                "expected_tools": ["extract_table_data"],
                "expected_workflow": "extract_table_data(region)",
                "decision_rule": "fixed",
                "expected_decision": "无法判断",
                "expected_keywords": ["价格上限"],
                "expected_sources": ["省间电力现货交易规则"],
            }],
        }
        assert validate_spec(spec) == []

    def test_unknown_slot_reference(self):
        spec = {
            "slot_vocab": {"region": ["冀北"]},
            "templates": [{
                "template_id": "t",
                "query_template": "{nope}的上限是多少？",
                "slots": ["nope"],
                "expected_tools": [],
                "expected_workflow": "",
                "decision_rule": "fixed",
                "expected_decision": "无法判断",
            }],
        }
        errors = validate_spec(spec)
        assert any("slot_vocab 中不存在" in e for e in errors)

    def test_placeholder_declared_mismatch(self):
        spec = {
            "slot_vocab": {"region": ["冀北"]},
            "templates": [{
                "template_id": "t",
                "query_template": "{region}和{price}的上限是多少？",
                "slots": ["region"],
                "expected_tools": [],
                "expected_workflow": "",
                "decision_rule": "fixed",
                "expected_decision": "无法判断",
            }],
        }
        errors = validate_spec(spec)
        assert any("query_template 槽位" in e for e in errors)

    def test_invalid_decision_rule(self):
        spec = {
            "slot_vocab": {},
            "templates": [{
                "template_id": "t",
                "query_template": "q",
                "slots": [],
                "expected_tools": [],
                "expected_workflow": "",
                "decision_rule": "bogus",
                "expected_decision": "无法判断",
            }],
        }
        errors = validate_spec(spec)
        assert any("decision_rule" in e for e in errors)


# ============================================================================
# 集成：真实 spec → 生成 → 校验
# ============================================================================


@pytest.fixture(scope="module")
def real_dataset():
    """用真实 spec 生成数据集（不写盘），返回 (dataset, spec, tool_names, whitelist)。"""
    spec_path = _PROJECT_ROOT / _SPEC_PATH
    if not spec_path.exists():
        pytest.skip("spec 文件不存在，跳过集成测试")
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    with open(_PROJECT_ROOT / "data/env_variables/tools_config.json",
              "r", encoding="utf-8") as f:
        tools_config = json.load(f)
    tool_names = set(tools_config["tools"].keys())
    whitelist = set(source_whitelist(spec.get("facts", {}), spec.get("slot_vocab", {})))

    from scripts.build_seed_dataset import DEFAULT_TARGET_COUNT, build_dataset
    variants, _stats = build_dataset(str(spec_path), DEFAULT_TARGET_COUNT)
    return {"seeds": variants}, spec, tool_names, whitelist


class TestIntegration:
    def test_spec_valid(self, real_dataset):
        _dataset, spec, _tools, _wl = real_dataset
        assert validate_spec(spec) == []

    def test_all_variants_valid(self, real_dataset):
        dataset, _spec, tool_names, whitelist = real_dataset
        errors, valid = validate_dataset(dataset, tool_names, whitelist)
        assert errors == [], f"前 5 条错误: {errors[:5]}"

    def test_count_reasonable(self, real_dataset):
        dataset, _spec, _tools, _wl = real_dataset
        assert len(dataset["seeds"]) >= 1500, "条数不足 1500"

    def test_judge_ratio_in_range(self, real_dataset):
        dataset, _spec, _tools, _wl = real_dataset
        stats = build_stats(dataset["seeds"])
        assert 0.25 <= stats["judge_ratio"] <= 0.30, \
            f"判定型占比 {stats['judge_ratio']} 超出 [25%, 30%]"

    def test_tool_coverage(self, real_dataset):
        dataset, _spec, _tools, _wl = real_dataset
        stats = build_stats(dataset["seeds"])
        assert len(stats["by_tool"]) >= 9, f"工具覆盖不足: {stats['by_tool']}"

    def test_signal_fields_consistent(self, real_dataset):
        """所有变体的 keyword/source_evidence 重算与字段值一致（同 build 的去重规则）。"""
        dataset, _spec, _tools, _wl = real_dataset
        for v in dataset["seeds"]:
            if "keyword_evidence" in v:
                recomputed = _dedup(list(v["keyword_evidence"]["static"])
                                    + list(v["keyword_evidence"]["derived"]))
                assert v["expected_keywords"] == recomputed, v["variant_id"]
            if "source_evidence" in v:
                recomputed = _dedup(list(v["source_evidence"]["static"])
                                    + list(v["source_evidence"]["derived"]))
                assert v["expected_sources"] == recomputed, v["variant_id"]
