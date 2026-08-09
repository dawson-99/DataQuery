"""
种子蒸馏脚本纯函数测试（不调 API）

覆盖 scripts/build_rule_review_seeds.py：
- 工具列表加载（含 9 个工具）
- Instruction 构建（工具列表 + 输出规范 + 工作流提示）
- 三段式输出校验（合法/非法工具/缺段/坏 JSON）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.build_rule_review_seeds import (  # noqa: E402
    build_instruction,
    format_tools_for_prompt,
    load_tools,
    validate_output,
)


class TestLoadTools:
    def test_loads_nine_tools(self):
        tools = load_tools()
        names = {t["name"] for t in tools}
        assert len(names) >= 9
        assert "extract_table_data" in names
        assert "locate_clause" in names
        assert "detect_rule_conflict" in names

    def test_prompt_format_contains_schema(self):
        text = format_tools_for_prompt()
        assert "locate_clause" in text
        assert "article_no" in text  # 参数名


class TestBuildInstruction:
    def test_contains_three_part_spec(self):
        seed = {
            "scenario_id": "compare_001",
            "category": "compare",
            "user_query": "测试问题",
            "expected_workflow": "extract → compare → 结论",
            "expected_tools": ["extract_table_data", "arithmetic_compare"],
            "expected_decision": "不符合",
        }
        instruction = build_instruction(seed)
        assert "<think>" in instruction
        assert "<tool_call>" in instruction
        assert "<response>" in instruction
        assert "extract → compare → 结论" in instruction  # 工作流提示
        assert "extract_table_data" in instruction


class TestValidateOutput:
    def test_valid_with_tools(self):
        text = (
            "<think>需要提取表格数据并比较</think>\n"
            '<tool_call>[{"tool": "extract_table_data", "args": {"table_text": "t", '
            '"filter_column": "地区", "filter_value": "冀北", "select_column": "上限"}}]</tool_call>\n'
            '<response>{"decision": "不符合", "reason": "r", "evidence": [], "confidence": 0.9}</response>'
        )
        ok, err = validate_output(text)
        assert ok, err

    def test_valid_without_tools(self):
        text = (
            "<think>无需工具</think>\n"
            '<response>{"decision": "符合", "reason": "r", "evidence": [], "confidence": 0.8}</response>'
        )
        ok, err = validate_output(text)
        assert ok, err

    def test_missing_think(self):
        text = '<response>{"decision": "符合", "reason": "r", "evidence": [], "confidence": 0.8}</response>'
        ok, err = validate_output(text)
        assert not ok
        assert "think" in err

    def test_bad_response_json(self):
        text = "<think>t</think>\n<response>not-json</response>"
        ok, err = validate_output(text)
        assert not ok
        assert "JSON" in err

    def test_unknown_tool_rejected(self):
        text = (
            "<think>t</think>\n"
            '<tool_call>[{"tool": "no_such_tool", "args": {}}]</tool_call>\n'
            '<response>{"decision": "符合", "reason": "r", "evidence": [], "confidence": 0.8}</response>'
        )
        ok, err = validate_output(text)
        assert not ok
        assert "未知工具" in err

    def test_invalid_decision_rejected(self):
        text = (
            "<think>t</think>\n"
            '<response>{"decision": "也许吧", "reason": "r", "evidence": [], "confidence": 0.8}</response>'
        )
        ok, err = validate_output(text)
        assert not ok
        assert "decision" in err
