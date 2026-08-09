"""
A 层四件套工具单元测试

覆盖 src/rule_review/tools_a_layer.py：
- locate_clause 条款定位器（条款/款切分、中文数字、doc 过滤、未命中降级）
- verify_citation 证据引用核验（文本 LCS + 元数据一致性）
- extract_numeric_fact 数值事实提取（主体+数值+单位三元组）
- detect_rule_conflict 多文档冲突检测（数值归一化比较 + 文本相似度）
- TOOL_MAP 注册与 schema 校验
"""

from __future__ import annotations

import pytest

from src.rule_review.prompts import get_system_prompt
from src.rule_review.tool_executor import ToolExecutor


def _make_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "c1",
            "text": (
                "第十二条 现货出清电价上限\n"
                "现货出清电价上限为760元/MWh。\n"
                "（一）符合规则的交易予以出清。\n"
                "（二）超过上限的申报不予出清。\n"
                "第十三条 结算规则\n"
                "结算按出清价格执行。"
            ),
            "source": "省间现货交易规则.pdf",
            "section": "第三章 价格机制",
            "page": 5,
        },
        {
            "chunk_id": "c2",
            "text": "第五条 申报要求\n申报电价不得超过上限。\n第六条 处罚规定\n违规申报处以罚款。",
            "source": "省间现货交易规则.pdf",
            "section": "第四章 申报与结算",
            "page": 6,
        },
        {
            "chunk_id": "c3",
            "text": "第五条 修订条款\n本规则修订后按新版本执行。",
            "source": "省内电力交易规则.pdf",
            "section": "第一章 总则",
            "page": 1,
        },
    ]


# ---------------------------------------------------------------------------
# locate_clause 条款定位器
# ---------------------------------------------------------------------------


class TestLocateClause:
    def test_locate_article_exact(self):
        """阿拉伯数字条款号精确定位，正确切分到下一个条款。"""
        result = ToolExecutor.execute_tool(
            "locate_clause", {"article_no": 12, "chunks": _make_chunks()}
        )
        assert result["success"] is True
        data = result["data"]
        assert data["found"] is True
        assert data["match_level"] == "article"
        assert "第十二条" in data["clause_text"]
        assert "第十三条" not in data["clause_text"]  # 不含下一条款
        assert "760元/MWh" in data["clause_text"]
        assert data["source"] == "省间现货交易规则.pdf"
        assert data["page"] == 5

    def test_locate_article_chinese_number(self):
        """中文数字条款号「第十二条」同样支持。"""
        result = ToolExecutor.execute_tool(
            "locate_clause", {"article_no": "第十二条", "chunks": _make_chunks()}
        )
        assert result["success"] is True
        assert result["data"]["found"] is True

    def test_locate_paragraph_with_bracket(self):
        """款定位：中文规则「（二）」款标识。"""
        result = ToolExecutor.execute_tool(
            "locate_clause",
            {"article_no": 12, "paragraph_no": "（二）", "chunks": _make_chunks()},
        )
        assert result["success"] is True
        data = result["data"]
        assert data["found"] is True
        assert data["match_level"] == "exact"
        assert "超过上限的申报不予出清" in data["paragraph_text"]
        assert "（一）" not in data["paragraph_text"]

    def test_article_not_found_returns_success(self):
        """条款不存在：found=false 且 success=true（定位失败不是工具错误）。"""
        result = ToolExecutor.execute_tool(
            "locate_clause", {"article_no": 99, "chunks": _make_chunks()}
        )
        assert result["success"] is True
        assert result["data"]["found"] is False
        assert "第99条" in result["data"]["message"]

    def test_paragraph_missing_keeps_article_level(self):
        """条存在款不存在：仍返回条款正文，match_level=article。"""
        result = ToolExecutor.execute_tool(
            "locate_clause",
            {"article_no": 12, "paragraph_no": 9, "chunks": _make_chunks()},
        )
        assert result["success"] is True
        assert result["data"]["found"] is True
        assert result["data"]["match_level"] == "article"
        assert result["data"]["paragraph_text"] == ""

    def test_doc_name_filter(self):
        """跨文档时按 doc_name 过滤，避免命中他文档同号条款。"""
        result = ToolExecutor.execute_tool(
            "locate_clause",
            {"article_no": 5, "doc_name": "省内", "chunks": _make_chunks()},
        )
        assert result["success"] is True
        assert result["data"]["found"] is True
        assert result["data"]["source"] == "省内电力交易规则.pdf"
        assert "修订条款" in result["data"]["clause_text"]

    def test_duplicate_article_returns_first(self):
        """同文档两个「第5条」（不同章节）：返回首个匹配。"""
        chunks = [
            {"chunk_id": "c1", "text": "第五条 A版本条款内容\n第七条 其他", "source": "d.pdf", "page": 1},
            {"chunk_id": "c2", "text": "第五章 细则\n第五条 B版本条款内容", "source": "d.pdf", "page": 3},
        ]
        result = ToolExecutor.execute_tool(
            "locate_clause", {"article_no": 5, "chunks": chunks}
        )
        assert result["success"] is True
        assert "A版本" in result["data"]["clause_text"]

    def test_invalid_article_no_schema_error(self):
        """article_no 为 bool/None 时被 schema/函数拦截。"""
        result = ToolExecutor.execute_tool("locate_clause", {"article_no": None, "chunks": []})
        assert result["success"] is False


# ---------------------------------------------------------------------------
# verify_citation 证据引用核验
# ---------------------------------------------------------------------------


class TestVerifyCitation:
    def test_verified_when_all_match(self):
        """原文一致 + 元数据吻合 → verified=true。"""
        chunks = _make_chunks()
        result = ToolExecutor.execute_tool(
            "verify_citation",
            {
                "evidence_text": "现货出清电价上限为760元/MWh。",
                "claimed_source": "省间现货交易规则.pdf",
                "claimed_section": "第三章 价格机制",
                "claimed_page": 5,
                "chunks": chunks,
            },
        )
        assert result["success"] is True
        data = result["data"]
        assert data["verified"] is True
        assert data["text_matched"] is True
        assert data["source_matched"] is True
        assert data["section_matched"] is True
        assert data["page_matched"] is True
        assert data["issues"] == []

    def test_low_text_ratio_fails(self):
        """文本匹配率低于阈值 → text_matched=false。"""
        chunks = _make_chunks()
        result = ToolExecutor.execute_tool(
            "verify_citation",
            {
                "evidence_text": "现货出清电价上限为9999元/MWh，与原文完全不同内容。",
                "claimed_source": "省间现货交易规则.pdf",
                "claimed_section": "第三章",
                "claimed_page": 5,
                "chunks": chunks,
                "lcs_threshold": 0.9,
            },
        )
        assert result["success"] is True
        assert result["data"]["text_matched"] is False
        assert result["data"]["verified"] is False
        assert any("匹配率" in issue for issue in result["data"]["issues"])

    def test_page_skipped_when_not_claimed(self):
        """claimed_page=0（未声称）→ 页码校验跳过，不产生 issue。"""
        chunks = _make_chunks()
        result = ToolExecutor.execute_tool(
            "verify_citation",
            {
                "evidence_text": "现货出清电价上限为760元/MWh。",
                "claimed_source": "省间现货交易规则.pdf",
                "claimed_page": 0,
                "chunks": chunks,
            },
        )
        assert result["success"] is True
        assert result["data"]["verified"] is True
        assert not any("页码" in i for i in result["data"]["issues"])

    def test_empty_chunks(self):
        """chunks 为空 → verified=false，issue 说明原因。"""
        result = ToolExecutor.execute_tool(
            "verify_citation",
            {"evidence_text": "任何内容", "chunks": []},
        )
        assert result["success"] is True
        assert result["data"]["verified"] is False
        assert any("无可用 chunks" in i for i in result["data"]["issues"])

    def test_source_with_book_marks(self):
        """声称来源带《》或后缀差异 → 包含匹配通过。"""
        chunks = _make_chunks()
        result = ToolExecutor.execute_tool(
            "verify_citation",
            {
                "evidence_text": "现货出清电价上限为760元/MWh。",
                "claimed_source": "《省间现货交易规则》",
                "claimed_section": "第三章",
                "claimed_page": 5,
                "chunks": chunks,
            },
        )
        assert result["success"] is True
        assert result["data"]["source_matched"] is True
        assert result["data"]["verified"] is True

    def test_empty_evidence_text(self):
        """evidence_text 为空 → 直接返回未通过。"""
        result = ToolExecutor.execute_tool(
            "verify_citation",
            {"evidence_text": "", "chunks": _make_chunks()},
        )
        assert result["success"] is True
        assert result["data"]["verified"] is False
        assert any("为空" in i for i in result["data"]["issues"])


# ---------------------------------------------------------------------------
# extract_numeric_fact 数值事实提取
# ---------------------------------------------------------------------------


class TestExtractNumericFact:
    def test_extract_with_subject(self):
        """「冀北...800元/MWh」→ 主体/数值/单位三元组。"""
        result = ToolExecutor.execute_tool(
            "extract_numeric_fact",
            {"text": "冀北地区现货出清电价上限为800元/MWh，省内为760元/MWh。"},
        )
        assert result["success"] is True
        facts = result["data"]["facts"]
        assert len(facts) == 2
        f0, f1 = facts[0], facts[1]
        assert f0["value"] == 800.0
        assert f0["unit"] == "元/MWh"
        assert "冀北" in f0["subject"]
        assert f1["value"] == 760.0
        assert "省内" in f1["subject"]

    def test_chinese_number(self):
        """中文数字「一万二千元/兆瓦时」解析为 12000。"""
        result = ToolExecutor.execute_tool(
            "extract_numeric_fact",
            {"text": "按照一万二千元/兆瓦时执行处罚标准。"},
        )
        assert result["success"] is True
        facts = result["data"]["facts"]
        assert any(f["value"] == 12000.0 for f in facts), facts

    def test_article_numbers_ignored(self):
        """条款号「第十二条」中的中文数字不应被当作数值。"""
        result = ToolExecutor.execute_tool(
            "extract_numeric_fact",
            {"text": "第十二条规定上限为760元/MWh。"},
        )
        assert result["success"] is True
        facts = result["data"]["facts"]
        assert len(facts) == 1
        assert facts[0]["value"] == 760.0

    def test_no_numbers(self):
        """无数值文本 → facts=[]，success=true。"""
        result = ToolExecutor.execute_tool(
            "extract_numeric_fact", {"text": "本规则自发布之日起施行。"}
        )
        assert result["success"] is True
        assert result["data"]["facts"] == []

    def test_max_facts_truncation(self):
        """超过 max_facts 截断。"""
        result = ToolExecutor.execute_tool(
            "extract_numeric_fact",
            {"text": "A为1元，B为2元，C为3元，D为4元，E为5元。", "max_facts": 3},
        )
        assert result["success"] is True
        assert len(result["data"]["facts"]) == 3

    def test_subject_filter(self):
        """subject 限定只返回匹配主体的 fact。"""
        result = ToolExecutor.execute_tool(
            "extract_numeric_fact",
            {
                "text": "冀北上限800元/MWh，山西上限760元/MWh。",
                "subject": "山西",
            },
        )
        assert result["success"] is True
        facts = result["data"]["facts"]
        assert all("山西" in f["subject"] for f in facts)
        assert facts[0]["value"] == 760.0

    def test_empty_text(self):
        """空文本 → facts=[]。"""
        result = ToolExecutor.execute_tool("extract_numeric_fact", {"text": ""})
        assert result["success"] is True
        assert result["data"]["facts"] == []


# ---------------------------------------------------------------------------
# detect_rule_conflict 多文档冲突检测
# ---------------------------------------------------------------------------


class TestDetectRuleConflict:
    def test_numeric_conflict(self):
        """800 vs 760（同单位）→ 数值冲突，reason 含差值。"""
        result = ToolExecutor.execute_tool(
            "detect_rule_conflict",
            {
                "pairs": [
                    {
                        "doc_a": "省间规则",
                        "doc_b": "省内规则",
                        "topic": "现货出清电价上限",
                        "rule_a_text": "上限为800元/MWh",
                        "rule_b_text": "上限为760元/MWh",
                    }
                ]
            },
        )
        assert result["success"] is True
        c = result["data"]["conflicts"][0]
        assert c["conflict"] is True
        assert c["conflict_type"] == "数值冲突"
        assert "5.3%" in c["reason"]

    def test_unit_normalized_equal(self):
        """0.76元/kWh vs 760元/MWh → 单位归一化后相等 → 无冲突。"""
        result = ToolExecutor.execute_tool(
            "detect_rule_conflict",
            {
                "pairs": [
                    {
                        "doc_a": "A",
                        "doc_b": "B",
                        "topic": "上限",
                        "rule_a_text": "上限为0.76元/kWh",
                        "rule_b_text": "上限为760元/MWh",
                    }
                ]
            },
        )
        assert result["success"] is True
        c = result["data"]["conflicts"][0]
        assert c["conflict"] is False
        assert c["conflict_type"] == "无冲突"

    def test_no_numbers_similar_text(self):
        """双方无数值且文本相似 → 无冲突。"""
        result = ToolExecutor.execute_tool(
            "detect_rule_conflict",
            {
                "pairs": [
                    {
                        "doc_a": "A",
                        "doc_b": "B",
                        "topic": "申报",
                        "rule_a_text": "申报电价不得超过价格上限",
                        "rule_b_text": "申报电价不得超过价格上限，违者处理",
                    }
                ]
            },
        )
        assert result["success"] is True
        c = result["data"]["conflicts"][0]
        assert c["conflict"] is False

    def test_no_numbers_dissimilar_text(self):
        """双方无数值且文本不相似 → 疑似冲突（待人工复核），不硬判。"""
        result = ToolExecutor.execute_tool(
            "detect_rule_conflict",
            {
                "pairs": [
                    {
                        "doc_a": "A",
                        "doc_b": "B",
                        "topic": "出清",
                        "rule_a_text": "出清按价差优先原则执行",
                        "rule_b_text": "出清按电量比例分配原则执行",
                    }
                ]
            },
        )
        assert result["success"] is True
        c = result["data"]["conflicts"][0]
        assert c["conflict"] is True
        assert c["conflict_type"] == "疑似冲突"

    def test_empty_pairs(self):
        """pairs=[] → conflicts=[]，success=true。"""
        result = ToolExecutor.execute_tool("detect_rule_conflict", {"pairs": []})
        assert result["success"] is True
        assert result["data"]["conflicts"] == []

    def test_pairs_not_list(self):
        """pairs 非数组 → success=false。"""
        result = ToolExecutor.execute_tool("detect_rule_conflict", {"pairs": "not-list"})
        assert result["success"] is False


# ---------------------------------------------------------------------------
# 注册与 schema
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_tool_map_has_four_new_tools(self):
        """四个新工具全部注册进 TOOL_MAP。"""
        for name in (
            "locate_clause",
            "verify_citation",
            "extract_numeric_fact",
            "detect_rule_conflict",
        ):
            assert name in ToolExecutor.TOOL_MAP, name

    def test_system_prompt_contains_new_tools(self):
        """V2 Prompt 动态包含四个新工具。"""
        prompt = get_system_prompt(include_tools=True)
        for name in (
            "locate_clause",
            "verify_citation",
            "extract_numeric_fact",
            "detect_rule_conflict",
        ):
            assert name in prompt, name

    def test_schema_validation_on_new_tool(self):
        """新工具缺 required 参数 → schema_error 拦截。"""
        result = ToolExecutor.execute_tool("detect_rule_conflict", {})
        assert result["success"] is False
        assert result.get("schema_error") is True
