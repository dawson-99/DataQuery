"""
电力规则审查系统 - 规则审查专用 Prompt

按设计文档 §7 Prompt 设计实现：
- System Prompt（动态注入工具规则 + 术语映射）
- RAG Context Prompt（检索结果上下文组装）
- 沿用现有项目模式：JSON 配置文件 → _build_xxx_section() → Prompt 模板

Phase 1: 不含 Tool 系统，System Prompt 为 Phase 1 简化版。
Phase 2: 工具规则从 tools_config.json 动态生成，嵌入 System Prompt。
"""

from __future__ import annotations

import json
from datetime import datetime

# 当前日期用于回答中参考
_current_date = datetime.now().strftime("%Y-%m-%d")

# 加载术语映射
with open("data/env_variables/rule_terms.json", "r", encoding="utf-8") as _f:
    _rule_terms = json.load(_f)


# ---------------------------------------------------------------------------
# 动态生成 Prompt 片段
# ---------------------------------------------------------------------------


def _build_terms_section() -> str:
    """从 rule_terms.json 动态生成术语映射表，嵌入 RAG Context Prompt。"""
    terms = _rule_terms.get("terms", {})
    if not terms:
        return "暂无术语映射。\n"

    lines = ["以下为电力交易领域术语的标准表达及其同义词："]
    for term, info in terms.items():
        syn_list = "、".join(info.get("synonyms", []))
        alias_list = "、".join(info.get("aliases", []))
        line = f"- {term}"
        if alias_list:
            line += f"（别名：{alias_list}）"
        if syn_list:
            line += f"（同义词：{syn_list}）"
        lines.append(line)
    return "\n".join(lines)


# 预生成的术语片段（模块加载时计算一次）
TERMS_SECTION = _build_terms_section()


def _build_tools_section() -> str:
    """从 tools_config.json 动态生成工具规则文本。

    与现有 _build_operations_section() 模式一致：
    JSON 配置文件 → 动态生成规则文本 → 嵌入 System Prompt。
    """
    try:
        with open("data/env_variables/tools_config.json", "r", encoding="utf-8") as f:
            tools_config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return "## 可用工具\n\n（工具配置未加载）\n"

    tools = tools_config.get("tools", {})
    if not tools:
        return "## 可用工具\n\n（无可用工具）\n"

    lines = ["## 可用工具\n"]
    lines.append("当遇到以下场景时，在输出的 `tool_calls` 数组中添加相应的工具调用：\n")
    for tool_name, tool_info in tools.items():
        lines.append(f"### {tool_name}（{tool_info['name']}）")
        triggers = "、".join(f'"{t}"' for t in tool_info.get("triggers", []))
        lines.append(f"- 触发词：{triggers}")
        lines.append(f"- 说明：{tool_info['description']}")
        lines.append(f"- 参数：")
        for param_name, param_desc in tool_info["parameters"].items():
            lines.append(f"  - `{param_name}`: {param_desc}")
        lines.append(f"- 返回值：{json.dumps(tool_info['output'], ensure_ascii=False)}")
        lines.append("")
    return "\n".join(lines)


# 预生成工具规则文本
TOOLS_SECTION = _build_tools_section()


# ---------------------------------------------------------------------------
# System Prompt (Phase 2)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是电力交易规则审查专家。当前日期为 {current_date}。

## 核心原则
1. 必须基于提供的规则文档原文回答，不得编造任何内容
2. 如果文档中没有相关信息，必须明确说明"文档中未找到相关规则"
3. 每条判断必须附带原文引用作为证据
4. 如果规则涉及精确数值比较或单位换算，请仔细核对，确保计算准确

## 输出格式
必须严格输出以下 JSON 格式：

{{
  "decision": "符合 | 不符合 | 部分符合 | 无法判断",
  "reason": "详细的推理过程，包括引用的规则条款编号和具体内容",
  "evidence": [
    {{
      "source": "文档名称",
      "section": "章节标题",
      "page": 页码,
      "text": "原文引用内容"
    }}
  ],
  "confidence": 0.0-1.0,
  "not_found": false
}}

## 决策指南
- "符合"：所有条件满足规则要求
- "不符合"：至少一个条件违反规则
- "部分符合"：部分条件满足但存在问题（如特例豁免、过渡期安排等）
- "无法判断"：缺少关键信息或规则不明确，证据不足以做出明确判断

## 置信度指南
- 0.9-1.0：规则条款明确，且用户问题完全匹配规则条件
- 0.7-0.9：规则适用但有轻微不确定性（如理解可能有偏差）
- 0.5-0.7：规则部分适用，但存在解释空间
- 0.0-0.5：信息不足，无法做出可靠判断

## 额外说明
- `not_found` 字段仅在文档中完全找不到相关信息时设为 true
- evidence 中的 text 必须是原文引用（尽量逐字引用），不要改写或总结
- reason 需要说明你是如何从原文推导出结论的"""


# Phase 2 System Prompt: 含 tool_calls 支持
SYSTEM_PROMPT_V2 = SYSTEM_PROMPT + """

## 工具调用规则
遇到以下场景时必须调用工具，不要自行估算：

{tools_section}

### tool_calls 数组格式
```json
"tool_calls": [
  {{
    "tool": "工具名",
    "args": {{参数对象}}
  }}
]
```

### 工具调用示例
假设用户问"冀北电价800元/MWh是否超出上限"，规则文档中的表格为：
| 地区 | 电价上限(元/MWh) |
|------|-----------------|
| 冀北 | 760             |

则应输出：
```json
{{
  "decision": "",
  "reason": "",
  "evidence": [],
  "confidence": 0.0,
  "tool_calls": [
    {{"tool": "extract_table_data", "args": {{"table_text": "| 地区 | 电价上限(元/MWh) |\\\\n|------|-----------------|\\\\n| 冀北 | 760 |", "filter_column": "地区", "filter_value": "冀北", "select_column": "电价上限(元/MWh)"}}}},
    {{"tool": "arithmetic_compare", "args": {{"actual": 800, "operator": "gt", "threshold": 760}}}}
  ]
}}
```

系统执行工具后，会将结果追加返回，请基于工具结果重新生成完整的审查结果（此时 tool_calls 为空数组）。

## 核心原则（补充）
4. 所有精确计算（数值比较、单位换算、表格提取）必须调用工具完成，不得自行估算"""


def get_system_prompt(include_tools: bool = True) -> str:
    """获取 System Prompt，Phase 2 默认包含工具规则。

    Args:
        include_tools: True 时返回含工具规则的 V2 Prompt。

    Returns:
        系统提示词字符串。
    """
    if include_tools:
        return SYSTEM_PROMPT_V2.format(
            current_date=_current_date,
            tools_section=TOOLS_SECTION,
        )
    return SYSTEM_PROMPT.format(current_date=_current_date)


# ---------------------------------------------------------------------------
# RAG Context Prompt
# ---------------------------------------------------------------------------

RAG_CONTEXT_TEMPLATE = """## 以下是从规则文档库检索到的相关内容

{context}

## 术语参考
{terms_section}

## 用户问题

{query}

请基于以上规则文档内容回答用户问题。请严格遵循输出 JSON 格式，不要输出其他内容。"""


def build_rag_context_prompt(
    query: str,
    context_chunks: list[dict],
    terms_section: str | None = None,
) -> str:
    """构建 RAG 上下文 Prompt。

    Args:
        query: 用户问题（改写后的标准化问题）。
        context_chunks: 检索到的 chunks，每个 dict 格式为：
            {{
                "text": "...",
                "source": "文档名",
                "section": "章节",
                "page": 页码,
            }}
        terms_section: 术语映射文本，None 时使用默认预生成版本。

    Returns:
        完整的 RAG Context Prompt 字符串，应作为 HumanMessage 发送给 LLM。
    """
    # 组装上下文文本
    context_parts = []
    for i, chunk in enumerate(context_chunks):
        source = chunk.get("source", "未知文档")
        section = chunk.get("section", "")
        page = chunk.get("page", "")
        text = chunk.get("text", "")

        header = f"[{i + 1}]"
        if section:
            header += f" {section}"
        if page:
            header += f"（第{page}页）"
        header += f" — 来源：{source}"

        context_parts.append(f"{header}\n{text}")

    context_text = "\n\n---\n\n".join(context_parts)
    ts = terms_section or TERMS_SECTION

    return RAG_CONTEXT_TEMPLATE.format(
        context=context_text,
        terms_section=ts,
        query=query,
    )


def build_messages(
    query: str,
    context_chunks: list[dict],
    system_prompt: str | None = None,
    tool_results: list[dict] | None = None,
) -> list[dict]:
    """构建完整 messages 列表，用于 LLM 调用。

    Args:
        query: 用户问题。
        context_chunks: 检索到的 chunks。
        system_prompt: 自定义 System Prompt，None 时使用默认。
        tool_results: 工具执行结果（Phase 2），格式为：
            [{{"tool": "xxx", "args": {{}}, "result": {{}}}}]

    Returns:
        messages 列表，可直接传给 ChatQwen / ProxyChatModel。
    """
    sys_prompt = system_prompt or SYSTEM_PROMPT

    user_content = build_rag_context_prompt(query, context_chunks)

    # 如果有工具结果，追加到 user 消息后
    if tool_results:
        tool_text = "\n\n## 工具执行结果\n"
        for tr in tool_results:
            tool_text += f"\n- **{tr.get('tool', 'unknown')}**: {json.dumps(tr.get('result', {}), ensure_ascii=False)}"
        user_content += tool_text
        user_content += "\n\n请基于以上工具结果和规则文档内容，重新生成完整的审查结果 JSON。"

    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content},
    ]
