"""
审计记录 → SFT 训练数据转换测试

覆盖 src/rule_review/training/audit_to_sft.py：
- 三字段生成（instruction/input/output 含 <think>/<tool_call>/<response>）
- 工具调用按 round 排序合并
- 低质量记录过滤（judge_skipped + 无法判断）
- weak 标签（tool_unsolved）
- 无工具调用的纯推理样本
"""

from __future__ import annotations

import json

from src.rule_review.training.audit_to_sft import (
    _is_low_quality,
    _is_weak,
    build_instruction,
    convert,
    record_to_sft,
)


def _make_record(with_tools: bool = True, tool_unsolved: bool = False, low_quality: bool = False) -> dict:
    result = {
        "decision": "不符合",
        "reason": "实际电价800元/MWh超过规则上限760元/MWh",
        "evidence": [
            {
                "source": "省间现货交易规则.pdf",
                "section": "第三章",
                "page": 5,
                "text": "现货出清电价上限为760元/MWh。",
            }
        ],
        "confidence": 0.95,
    }
    if tool_unsolved:
        result["tool_unsolved"] = True

    record = {
        "query_id": "q1",
        "timestamp": "2026-08-09T10:00:00Z",
        "original_query": "2025年3月冀北现货出清电价800元/MWh是否符合价格上限？",
        "rewritten_query": "2025-03 冀北 现货出清电价 800元/MWh 是否符合 价格上限",
        "retrieval": {"final_k": 2},
        "llm_generation": {"model": "qwen3-max", "not_found": False},
        "tool_executions": [],
        "judge_verification": {
            "verified": True,
            "hallucinated_count": 0,
            "skipped": False,
        },
        "final_result": result,
        "source_traceability": [],
        "corrective": {},
    }

    if with_tools:
        record["tool_executions"] = [
            {
                "query_id": "q1",
                "round": 1,
                "tool_name": "extract_table_data",
                "args": {"filter_value": "冀北"},
                "result": {"success": True, "data": {"value": 760}},
                "timestamp": "2026-08-09T10:00:01Z",
                "latency_ms": 5,
            },
            {
                "query_id": "q1",
                "round": 1,
                "tool_name": "arithmetic_compare",
                "args": {"actual": 800, "operator": "gt", "threshold": 760},
                "result": {"success": True, "data": {"result": True}},
                "timestamp": "2026-08-09T10:00:02Z",
                "latency_ms": 3,
            },
        ]
    if low_quality:
        record["judge_verification"] = {"skipped": True, "skipped_reason": "not_found"}
        result["decision"] = "无法判断"

    return record


class TestRecordToSFT:
    def test_instruction_contains_tools(self):
        instruction = build_instruction()
        assert "电力交易规则审查专家" in instruction
        assert "locate_clause" in instruction  # 9 个工具都应在指令中
        assert "detect_rule_conflict" in instruction
        assert "tool_call" in instruction or "工具" in instruction

    def test_with_tools_three_part_output(self):
        rec = _make_record(with_tools=True)
        sample = record_to_sft(rec, "INSTR")
        assert sample is not None
        assert sample["input"] == rec["original_query"]
        assert sample["instruction"] == "INSTR"

        output = sample["output"]
        # 三段式
        assert "<think>" in output and "超过规则上限" in output
        # 两个工具调用合并为一个 <tool_call> 块（同 round）
        assert output.count("<tool_call>") == 1
        call_block = output.split("<tool_call>")[1].split("</tool_call>")[0]
        calls = json.loads(call_block)
        assert len(calls) == 2
        assert calls[0]["tool"] == "extract_table_data"
        assert calls[1]["tool"] == "arithmetic_compare"
        # response 含决策字段
        assert "<response>" in output
        assert '"decision"' in output and '"不符合"' in output

    def test_no_tools_output(self):
        rec = _make_record(with_tools=False)
        sample = record_to_sft(rec, "INSTR")
        assert sample is not None
        output = sample["output"]
        assert "<tool_call>" not in output
        assert "<think>" in output
        assert "<response>" in output

    def test_low_quality_filtered(self):
        rec = _make_record(low_quality=True)
        assert _is_low_quality(rec)
        sample = record_to_sft(rec, "INSTR")
        assert sample is None

    def test_weak_tag(self):
        rec = _make_record(tool_unsolved=True)
        assert _is_weak(rec)
        sample = record_to_sft(rec, "INSTR")
        assert sample is not None
        assert "weak" in sample["tags"]

    def test_tags(self):
        rec = _make_record(with_tools=True)
        sample = record_to_sft(rec, "INSTR")
        assert "with_tools" in sample["tags"]
        assert "no_tools" not in sample["tags"]

    def test_empty_record(self):
        sample = record_to_sft({}, "INSTR")
        assert sample is None


class TestConvert:
    def test_convert_empty_dir(self, tmp_path):
        stats = convert(str(tmp_path), str(tmp_path / "out.jsonl"))
        assert stats["audit_records"] == 0
        assert stats["samples"] == 0

    def test_convert_with_records(self, tmp_path):
        # 构造审计目录: data/audit_logs/2026-08-09/q1.json
        date_dir = tmp_path / "2026-08-09"
        date_dir.mkdir(parents=True)
        good = _make_record(with_tools=True)
        low = _make_record(low_quality=True)
        (date_dir / "q1.json").write_text(
            json.dumps(good, ensure_ascii=False), encoding="utf-8"
        )
        (date_dir / "q2.json").write_text(
            json.dumps(low, ensure_ascii=False), encoding="utf-8"
        )

        out = tmp_path / "sft.jsonl"
        stats = convert(str(tmp_path), str(out))
        assert stats["audit_records"] == 2
        assert stats["samples"] == 1
        assert stats["dropped_low_quality"] == 1
        assert stats["with_tools"] == 1

        lines = out.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        sample = json.loads(lines[0])
        assert sample["output"].count("<tool_call>") == 1
