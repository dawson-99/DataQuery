"""审计记录 → SFT 训练数据转换。

将审计日志（data/audit_logs/）中的真实审查轨迹转换为 SFT 三字段数据：

- instruction: 系统指令（工具列表 + 输出规范，从 tools_config.json 动态生成）
- input:      用户问题
- output:     <think>推理</think> + <tool_call>工具调用</tool_call>
              + <response>最终结果 JSON</response>

格式决策（与生产一致）：
ToolRL 默认 XML 模式（{"name": ...}）与本项目声明式 JSON 模式
（tool_calls: [{tool, args}] 数组）不同——本脚本直接学习生产 JSON 模式，
外层保留 <think>/<tool_call>/<response> 包裹以保留 RL 推理能力，
训练产物可零适配替换生成 agent。

过滤规则：
- judge_skipped 且 decision=无法判断 的低质量记录丢弃
- tool_unsolved=true 的记录单独打 weak 标签（弱监督样本）

用法：
    python -m src.rule_review.training.audit_to_sft \
        --audit-dir data/audit_logs --output data/evaluation/sft_from_audit.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

TOOLS_CONFIG_PATH = "data/env_variables/tools_config.json"
DEFAULT_AUDIT_DIR = "data/audit_logs"

# 输出格式模板（与生产 JSON 模式一致）
RESPONSE_FIELDS = ("decision", "reason", "evidence", "confidence")

# 低质量判定：Judge 跳过且无法判断
def _is_low_quality(record: dict) -> bool:
    judge = record.get("judge_verification") or {}
    result = record.get("final_result") or {}
    return bool(judge.get("skipped")) and result.get("decision") == "无法判断"


def _is_weak(record: dict) -> bool:
    """弱监督样本：工具未解决降级。"""
    result = record.get("final_result") or {}
    return bool(result.get("tool_unsolved"))


def build_instruction() -> str:
    """从 tools_config.json 生成系统指令（工具列表 + 输出规范）。"""
    with open(TOOLS_CONFIG_PATH, "r", encoding="utf-8") as f:
        tools = json.load(f).get("tools", {})

    lines = [
        "你是电力交易规则审查专家。必须基于规则文档原文回答，不得编造。",
        "当需要精确计算（表格取数、数值比较、单位换算、条款定位、引用核验、"
        "数值提取、冲突检测）时，必须调用工具，不要自行估算。",
        "",
        "## 可用工具",
    ]
    for name, info in tools.items():
        params = "、".join(info.get("parameters", {}).keys()) or "无"
        lines.append(f"- {name}（{info['name']}）：{info['description']}。参数：{params}")
    lines.append("")
    lines.append("## 输出规范")
    lines.append("请严格按以下三段式输出：")
    lines.append(
        "<think>你的推理过程</think>"
    )
    lines.append(
        "<tool_call>{需要工具时：{\"tool\": \"工具名\", \"args\": {...}} 的 JSON 数组；无需工具时省略}</tool_call>"
    )
    lines.append(
        "<response>{{\"decision\": \"符合|不符合|部分符合|无法判断\", "
        "\"reason\": \"推理过程\", \"evidence\": [{{\"source\": \"文档名\", "
        "\"section\": \"章节\", \"page\": 页码, \"text\": \"原文引用\"}}], "
        "\"confidence\": 0.0-1.0}}</response>"
    )
    return "\n".join(lines)


def record_to_sft(record: dict, instruction: str) -> dict | None:
    """单条审计记录 → SFT 样本。

    Returns:
        {"instruction", "input", "output", "tags"}；低质量记录返回 None。
    """
    if _is_low_quality(record):
        return None

    original_query = record.get("original_query", "")
    result = record.get("final_result") or {}
    tool_executions = record.get("tool_executions") or []

    # --- output 组装 ---
    parts: list[str] = []

    reason = result.get("reason", "")
    if reason:
        parts.append(f"<think>{reason}</think>")

    # 工具调用（按 round 排序，同 round 合并）
    if tool_executions:
        by_round: dict[int, list[dict]] = {}
        for t in sorted(tool_executions, key=lambda x: x.get("round", 1)):
            by_round.setdefault(t.get("round", 1), []).append(t)
        call_blocks: list[str] = []
        for rnd in sorted(by_round):
            calls = [
                {"tool": t.get("tool_name", ""), "args": t.get("args", {})}
                for t in by_round[rnd]
                if t.get("tool_name")
            ]
            if calls:
                call_blocks.append(
                    "<tool_call>" + json.dumps(calls, ensure_ascii=False) + "</tool_call>"
                )
        parts.extend(call_blocks)

    # 最终结果
    response_payload = {
        k: result.get(k) for k in RESPONSE_FIELDS if k in result
    }
    if not response_payload:
        return None
    parts.append("<response>" + json.dumps(response_payload, ensure_ascii=False) + "</response>")

    output = "\n".join(parts)
    if not output:
        return None

    tags = []
    if _is_weak(record):
        tags.append("weak")
    if tool_executions:
        tags.append("with_tools")
    else:
        tags.append("no_tools")
    if record.get("corrective", {}).get("triggered"):
        tags.append("corrective")

    return {
        "instruction": instruction,
        "input": original_query,
        "output": output,
        "tags": tags,
    }


def load_audit_records(audit_dir: str, date_filter: str | None = None) -> list[dict]:
    """扫描审计目录加载全部记录（data/audit_logs/{date}/{query_id}.json）。"""
    root = Path(audit_dir)
    records: list[dict] = []
    if not root.exists():
        logger.warning("[audit_to_sft] 审计目录不存在: %s", root)
        return records

    date_dirs = [root / d for d in sorted(root.iterdir()) if d.is_dir()]
    if date_filter:
        date_dirs = [root / date_filter] if (root / date_filter).is_dir() else []

    for date_dir in date_dirs:
        for f in sorted(date_dir.glob("*.json")):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    records.append(json.load(fh))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("[audit_to_sft] 跳过损坏记录 %s: %s", f, e)
    return records


def convert(audit_dir: str, output_path: str, date_filter: str | None = None) -> dict:
    """执行转换，返回统计信息。"""
    records = load_audit_records(audit_dir, date_filter)
    instruction = build_instruction()

    samples: list[dict] = []
    dropped = 0
    weak = 0
    for rec in records:
        sample = record_to_sft(rec, instruction)
        if sample is None:
            dropped += 1
            continue
        if "weak" in sample["tags"]:
            weak += 1
        samples.append(sample)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    stats = {
        "audit_records": len(records),
        "samples": len(samples),
        "dropped_low_quality": dropped,
        "weak_samples": weak,
        "with_tools": sum(1 for s in samples if "with_tools" in s["tags"]),
        "output": str(out),
    }
    logger.info("[audit_to_sft] 转换完成: %s", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="审计记录 → SFT 训练数据")
    parser.add_argument("--audit-dir", default=DEFAULT_AUDIT_DIR, help="审计目录")
    parser.add_argument("--output", default="data/evaluation/sft_from_audit.jsonl", help="输出 JSONL 路径")
    parser.add_argument("--date", default=None, help="只转换指定日期 YYYY-MM-DD")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    stats = convert(args.audit_dir, args.output, args.date)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
