#!/usr/bin/env python3
"""
从种子规格蒸馏生成规则审查 SFT 训练数据
==========================================
输入：data/evaluation/rule_review_seed_spec.json（种子规格，25 条）
输出：data/evaluation/sft_from_seeds.json（完整 SFT 训练数据，ToolRL 三字段）

流程（复用 ToolRL-main/scripts/build_sft_from_seeds.py 骨架）：
  1. 读取种子规格（scenario_id/category/user_query/expected_workflow/expected_tools）
  2. 构建 Instruction（工具列表从 data/env_variables/tools_config.json 动态生成）
  3. 对每条种子，调用蒸馏模型（deepseek）生成 <think>/<tool_call>/<response> 三段输出
  4. 格式校验（tool_call JSON 可解析、工具名合法）+ 失败重试
  5. 保存为标准 SFT JSON 格式（instruction/input/output/tags）

使用：
  python scripts/build_rule_review_seeds.py --num_samples 25 --max_workers 4
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 自动加载 .env（含 DASHSCOPE_API_KEY 等密钥）
_ENV_PATH = _PROJECT_ROOT / ".env"
if _ENV_PATH.exists():
    with open(_ENV_PATH) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                _k, _v = _k.strip(), _v.strip()
                if _k and _v and _k not in os.environ:
                    os.environ[_k] = _v

from openai import OpenAI
from tqdm import tqdm

SEED_SPEC_PATH = "data/evaluation/rule_review_seed_spec.json"
TOOLS_CONFIG_PATH = "data/env_variables/tools_config.json"
OUTPUT_PATH = "data/evaluation/sft_from_seeds.json"

# 蒸馏模型（走 DashScope 兼容 OpenAI 接口；.env 的 DASHSCOPE_API_KEY 驱动）
DISTILL_MODEL = os.getenv("SFT_DISTILL_MODEL", "deepseek-v3")
DISTILL_BASE_URL = os.getenv(
    "SFT_DISTILL_BASE_URL",
    os.getenv("DASHSCOPE_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)
DISTILL_API_KEY = os.getenv("SFT_DISTILL_API_KEY", os.getenv("DASHSCOPE_API_KEY", ""))


# ============================================================================
# 工具列表（从项目 tools_config.json 读取）
# ============================================================================


def load_tools() -> list[dict]:
    with open(TOOLS_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    tools = []
    for name, info in config.get("tools", {}).items():
        tools.append({
            "name": name,
            "description": info.get("description", ""),
            "parameters_schema": info.get("parameters_schema", {}),
        })
    return tools


VALID_TOOL_NAMES = {t["name"] for t in load_tools()}


def format_tools_for_prompt() -> str:
    """生成给 LLM 看的工具描述文本（复用 tools_config 的 schema）。"""
    lines = []
    for i, tool in enumerate(load_tools(), 1):
        schema = tool["parameters_schema"].get("properties", {})
        required = tool["parameters_schema"].get("required", [])
        param_desc = []
        for pname, pinfo in schema.items():
            req = "必填" if pname in required else "可选"
            ptype = pinfo.get("type", "any") if isinstance(pinfo, dict) else "any"
            param_desc.append(f"    {pname} ({ptype}, {req})")
        param_text = "\n".join(param_desc) if param_desc else "    无参数"
        lines.append(
            f"{i}. {tool['name']}: {tool['description']}\n{param_text}"
        )
    return "\n".join(lines)


# ============================================================================
# Instruction 构建
# ============================================================================


def build_instruction(seed: dict) -> str:
    """构建系统指令：角色 + 工具列表 + 输出规范 + 当前任务的工作流提示。"""
    tools_text = format_tools_for_prompt()
    workflow_hint = seed.get("expected_workflow", "")
    workflow_text = (
        f"\n本问题建议的工作流（仅供参考，可调整）：{workflow_hint}"
        if workflow_hint else ""
    )
    return f"""你是电力交易规则审查专家。根据用户的问题，给出审查结论。

## 可用工具
{tools_text}

## 输出规范
请严格按三段式输出，不要输出其他内容：
<think>你的推理过程，说明需要调用哪些工具以及为什么</think>
<tool_call>[{{"tool": "工具名", "args": {{...}}}}]（需要工具时输出 JSON 数组；无需工具时省略此行）</tool_call>
<response>{{"decision": "符合|不符合|部分符合|无法判断", "reason": "推理过程", "evidence": [{{"source": "文档名", "section": "章节", "page": 页码, "text": "原文引用"}}], "confidence": 0.0-1.0}}</response>
{workflow_text}"""


# ============================================================================
# 输出格式校验
# ============================================================================


def _extract_tag(text: str, tag: str) -> str | None:
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    start = text.find(start_tag)
    if start < 0:
        return None
    start += len(start_tag)
    end = text.find(end_tag, start)
    if end < 0:
        return None
    return text[start:end].strip()


def validate_output(text: str) -> tuple[bool, str]:
    """校验三段式输出格式与工具调用合法性。"""
    think = _extract_tag(text, "think")
    if not think:
        return False, "缺少 <think> 段"

    response = _extract_tag(text, "response")
    if not response:
        return False, "缺少 <response> 段"
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        return False, "<response> 内不是合法 JSON"
    if payload.get("decision") not in ("符合", "不符合", "部分符合", "无法判断"):
        return False, f"decision 非法: {payload.get('decision')!r}"

    call = _extract_tag(text, "tool_call")
    if call:
        try:
            calls = json.loads(call)
        except json.JSONDecodeError:
            return False, "<tool_call> 内不是合法 JSON"
        if not isinstance(calls, list):
            return False, "<tool_call> 必须是 JSON 数组"
        for c in calls:
            if not isinstance(c, dict) or "tool" not in c:
                return False, "tool_call 元素缺 tool 字段"
            if c["tool"] not in VALID_TOOL_NAMES:
                return False, f"未知工具: {c['tool']}"
    return True, ""


# ============================================================================
# 蒸馏调用
# ============================================================================


def distill_one(seed: dict, instruction: str, client: OpenAI, max_retries: int = 3) -> dict | None:
    """对单条种子蒸馏，带格式校验与重试。"""
    user_content = (
        f"**规则审查请求**\n<user>{seed['user_query']}</user>\n"
        f"**期望决策**：{seed['expected_decision']}\n"
        "请按输出规范给出完整三段式回答。"
    )
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=DISTILL_MODEL,
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.6,
                max_tokens=2048,
            )
            text = resp.choices[0].message.content or ""
            ok, err = validate_output(text)
            if ok:
                return {
                    "instruction": instruction,
                    "input": seed["user_query"],
                    "output": text,
                    "tags": {
                        "scenario_id": seed["scenario_id"],
                        "category": seed["category"],
                        "expected_tools": seed.get("expected_tools", []),
                        "expected_decision": seed["expected_decision"],
                    },
                }
            if attempt < max_retries:
                time.sleep(1 + attempt)
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 + attempt)
            else:
                print(f"  [skip] {seed['scenario_id']} 调用失败: {e}")
                return None
    print(f"  [skip] {seed['scenario_id']} 格式校验失败: {err}")
    return None


# ============================================================================
# 主流程
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="规则审查种子蒸馏 SFT 数据")
    parser.add_argument("--num-samples", type=int, default=25, help="蒸馏条数")
    parser.add_argument("--max-workers", type=int, default=4, help="并发数")
    parser.add_argument("--seed-spec", default=SEED_SPEC_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()

    if not DISTILL_API_KEY:
        print("错误：缺少 API Key（设置 SFT_DISTILL_API_KEY 或 DASHSCOPE_API_KEY）")
        sys.exit(1)

    with open(args.seed_spec, "r", encoding="utf-8") as f:
        spec = json.load(f)
    seeds = spec["seeds"][: args.num_samples]

    client = OpenAI(api_key=DISTILL_API_KEY, base_url=DISTILL_BASE_URL)
    results: list[dict] = []
    errors = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(distill_one, seed, build_instruction(seed), client): seed
            for seed in seeds
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="蒸馏中"):
            sample = future.result()
            if sample is None:
                errors += 1
            else:
                results.append(sample)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n完成：{len(results)} 条 / 失败 {errors} 条 → {out_path}")
    by_category: dict[str, int] = {}
    for s in results:
        cat = s["tags"]["category"]
        by_category[cat] = by_category.get(cat, 0) + 1
    print("按类别分布:", json.dumps(by_category, ensure_ascii=False))


if __name__ == "__main__":
    main()
