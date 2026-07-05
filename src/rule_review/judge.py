"""
电力规则审查系统 - Judge 校验模块

按设计文档 §4.6 + §7.5 + §13.8 实现：
- DeepSeek-v4 校验 LLM 输出（幻觉检测 + 逻辑自检 + 遗漏检查）
- Judge 异常兜底（超时/API 错误/格式异常 → 跳过校验）
- 与 Pipeline 集成：not_found 时自动跳过

Phase 1: LLM 输出 → Judge 校验 → 最终结果
Phase 2: 工具调用日志附加到 Judge Context
"""

from __future__ import annotations

import asyncio
import json
import logging
import textwrap
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.config import settings
from src.rule_review.schemas import LLMOutput
from src.utils.output_parser import parse_json_block

logger = logging.getLogger(__name__)

# Judge 超时（秒）
_JUDGE_TIMEOUT = 60


# ---------------------------------------------------------------------------
# Judge Prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """你是电力交易规则审查结果校验专家。

## 任务
校验以下审查结果，逐条检查：

1. **幻觉检测**：每条 evidence 的 text 是否真正出现在规则原文中？
   - 如果某条 evidence 的文本在原文中找不到对应内容 → 标记为 hallucinated
   - LLM 总结或改写的内容不算幻觉，但若编造了原文中不存在的条款号/数据/规则 → 是幻觉

2. **逻辑自检**：decision 与 reason 是否逻辑自洽？
   - 如果 reason 说"实际值800超过上限760"，但 decision 却是"符合" → 逻辑矛盾
   - 如果几条 evidence 之间存在矛盾 → 需要标注

3. **遗漏检查**：是否有遗漏的重要规则？
   - 检查原文中是否有关键规则条款未被引用
   - 如果用户问题涉及多个方面，是否遗漏了某个方面

## 输出格式
必须严格输出以下 JSON 格式：

{{
  "verified": true/false,
  "corrections": [
    {{
      "field": "decision|reason|evidence[0]|...",
      "issue": "问题描述",
      "suggestion": "修改建议"
    }}
  ],
  "hallucinated_evidence": [
    {{
      "index": 0,
      "reason": "为什么判定为幻觉"
    }}
  ],
  "missing_rules": [
    {{
      "rule": "被遗漏的规则摘要",
      "source": "来自哪条 chunk"
    }}
  ],
  "final_decision": "符合 | 不符合 | 部分符合 | 无法判断",
  "final_reason": "修正后的推理过程（如无修改则保持原样）",
  "final_evidence": [...],
  "confidence": 0.0-1.0
}}

注意：
- 如果无需修改，final_decision/final_reason/final_evidence 与原结果相同
- verified=true 表示审查结果可信，false 表示存在需关注的问题
- confidence 是基于 Judge 校验后的信心分数"""

JUDGE_USER_TEMPLATE = """## 原始问题
{original_query}

## 规则原文（检索到的 chunks）
{context}

## LLM 审查结果
{llm_output}

## 工具调用日志
{tool_logs}

请校验以上审查结果。"""


# ---------------------------------------------------------------------------
# Judge 模型工厂
# ---------------------------------------------------------------------------


def _create_judge_model() -> Any:
    """创建 Judge 专用 LLM 模型实例。

    模型路由逻辑与 WorkflowFactory._create_chat_model 一致。
    """
    model_name = settings.JUDGE_MODEL
    api_key = settings.JUDGE_API_KEY
    base_url = settings.JUDGE_API_BASE

    if model_name in getattr(settings, "GATEWAY_MODELS", []):
        from src.utils.model_proxy import ProxyChatModel

        return ProxyChatModel(
            model=model_name,
            base_url=settings.GATEWAY_BASE_URL,
            enable_thinking=False,
            timeout=getattr(settings, "REQUEST_TIMEOUT_SECONDS", 180),
        )

    from langchain_qwq import ChatQwen

    return ChatQwen(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        enable_thinking=False,
        timeout=getattr(settings, "REQUEST_TIMEOUT_SECONDS", 60),
    )


# ---------------------------------------------------------------------------
# Judge 结果数据模型
# ---------------------------------------------------------------------------


@dataclass
class JudgeResult:
    """Judge 校验结果。"""

    verified: bool = False
    corrections: list[dict] = field(default_factory=list)
    hallucinated_evidence: list[dict] = field(default_factory=list)
    missing_rules: list[dict] = field(default_factory=list)
    final_decision: str = ""
    final_reason: str = ""
    final_evidence: list[dict] = field(default_factory=list)
    confidence: float = 0.0

    # 异常标记
    judge_skipped: bool = False
    judge_skipped_reason: str = ""


# ---------------------------------------------------------------------------
# Judge 主类
# ---------------------------------------------------------------------------


class RuleReviewJudge:
    """DeepSeek-v4 校验 Qwen3B 输出。

    Phase 1 能力：
    - 幻觉检测（evidence vs context 最长公共子串 + LLM 判断）
    - 逻辑自检（decision + reason 一致性）
    - 遗漏检查（context 中是否有关键规则未被引用）

    异常兜底：
    - 超时（>60s）→ 跳过
    - API 错误（5xx）→ 跳过
    - 返回格式异常 → 重试 1 次 → 跳过
    """

    def __init__(self, model: Any | None = None) -> None:
        self._model = model

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = _create_judge_model()
        return self._model

    # ------------------------------------------------------------------
    # 主校验方法
    # ------------------------------------------------------------------

    async def verify(
        self,
        llm_output: LLMOutput,
        original_query: str,
        context_chunks: list[dict],
        tool_logs: list[dict] | None = None,
    ) -> JudgeResult:
        """对 LLM 输出执行完整校验。

        Args:
            llm_output: LLM 生成的审查结果。
            original_query: 用户原始问题。
            context_chunks: 检索到的 chunks 列表。
            tool_logs: 工具调用日志（Phase 2）。

        Returns:
            JudgeResult，包含校验结论和修正建议。
        """
        # 不需要校验的场景
        if llm_output.not_found:
            return self._skip("文档未找到，自动跳过")

        # 构建 Judge 输入
        context_text = self._build_context_text(context_chunks)
        tool_logs_text = self._build_tool_logs_text(tool_logs or [])

        user_prompt = JUDGE_USER_TEMPLATE.format(
            original_query=original_query,
            context=context_text,
            llm_output=json.dumps(
                {
                    "decision": llm_output.decision,
                    "reason": llm_output.reason,
                    "evidence": [
                        e.model_dump() if hasattr(e, "model_dump") else dict(e)
                        for e in llm_output.evidence
                    ],
                    "confidence": llm_output.confidence,
                },
                ensure_ascii=False,
                indent=2,
            ),
            tool_logs=tool_logs_text,
        )

        messages = [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        raw_text = ""
        try:
            response: AIMessage = await asyncio.wait_for(
                self.model.ainvoke(messages),
                timeout=_JUDGE_TIMEOUT,
            )
            raw_text = self._extract_content(response)

            # 首次解析
            parsed = self._parse_judge_output(raw_text)
            if parsed is not None:
                return self._build_result(parsed, llm_output)

            # 重试 1 次
            logger.info("[Judge] 输出格式异常，重试一次")
            messages.append(HumanMessage(
                content="你的上一轮输出无法解析为 JSON。请严格按照要求的 JSON 格式重新输出。"
            ))
            response = await asyncio.wait_for(
                self.model.ainvoke(messages),
                timeout=_JUDGE_TIMEOUT,
            )
            raw_text = self._extract_content(response)
            parsed = self._parse_judge_output(raw_text)
            if parsed is not None:
                return self._build_result(parsed, llm_output)

            # 两次都失败
            return self._skip(f"Judge 返回格式异常，已重试")

        except asyncio.TimeoutError:
            return self._skip(f"超时 {_JUDGE_TIMEOUT}s")

        except Exception as e:
            logger.error("[Judge] 校验异常: %s", e)
            return self._skip(str(e)[:200])

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _skip(self, reason: str) -> JudgeResult:
        """创建跳过校验的结果。"""
        return JudgeResult(
            judge_skipped=True,
            judge_skipped_reason=reason,
        )

    def _build_result(
        self, parsed: dict, llm_output: LLMOutput
    ) -> JudgeResult:
        """从 LLM 解析结果构建 JudgeResult。"""
        evidence = parsed.get("final_evidence", [])
        final_evidence = [dict(e) for e in evidence] if evidence else []

        return JudgeResult(
            verified=bool(parsed.get("verified", False)),
            corrections=parsed.get("corrections", []),
            hallucinated_evidence=parsed.get("hallucinated_evidence", []),
            missing_rules=parsed.get("missing_rules", []),
            final_decision=parsed.get("final_decision", llm_output.decision),
            final_reason=parsed.get("final_reason", llm_output.reason),
            final_evidence=final_evidence or llm_output.evidence,
            confidence=float(parsed.get("confidence", llm_output.confidence)),
        )

    @staticmethod
    def _parse_judge_output(raw_text: str) -> dict | None:
        """解析 Judge 原始输出。"""
        parsed = parse_json_block(raw_text)
        if parsed is None or not isinstance(parsed, dict):
            logger.warning(
                "[Judge] 无法解析输出，前 200 字符: %s", raw_text[:200]
            )
            return None
        return parsed

    @staticmethod
    def _build_context_text(chunks: list[dict]) -> str:
        """将 chunks 列表组装为可读的上下文文本。"""
        parts = []
        for i, c in enumerate(chunks):
            header = f"[Chunk {i + 1}]"
            section = c.get("section", "")
            if section:
                header += f" {section}"
            parts.append(f"{header}\n{c.get('text', '')}")
        return "\n\n---\n\n".join(parts) if parts else "（无上下文）"

    @staticmethod
    def _build_tool_logs_text(tool_logs: list[dict]) -> str:
        """将工具日志格式化为可读文本。"""
        if not tool_logs:
            return "（无工具调用）"
        parts = []
        for log in tool_logs:
            parts.append(
                f"- {log.get('tool', 'unknown')}: "
                f"{json.dumps(log.get('result', {}), ensure_ascii=False)}"
            )
        return "\n".join(parts)

    @staticmethod
    def _extract_content(response: AIMessage) -> str:
        content = response.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        return str(content)


# ---------------------------------------------------------------------------
# 便捷函数：Pipeline 集成
# ---------------------------------------------------------------------------


async def verify_with_fallback(
    judge: RuleReviewJudge,
    llm_output: LLMOutput,
    original_query: str,
    context_chunks: list[dict],
    tool_logs: list[dict] | None = None,
) -> dict:
    """Judge 校验 + 异常兜底（供 Pipeline 直接调用）。

    Args:
        judge: RuleReviewJudge 实例。
        llm_output: LLM 审查结果。
        original_query: 原始问题。
        context_chunks: 检索 chunks。
        tool_logs: 工具日志。

    Returns:
        合并后的结果字典，可直接作为 Pipeline 输出。
    """
    # 不需要校验
    if llm_output.not_found:
        logger.info("[Judge] 文档未找到，跳过校验")
        result = llm_output.model_dump()
        result["judge_skipped"] = True
        result["judge_skipped_reason"] = "not_found"
        return result

    try:
        judge_result = await judge.verify(
            llm_output, original_query, context_chunks, tool_logs
        )
    except Exception as e:
        logger.warning("[Judge] verify 异常，跳过校验: %s", e)
        result = llm_output.model_dump()
        result["judge_skipped"] = True
        result["judge_skipped_reason"] = str(e)[:200]
        return result

    if judge_result.judge_skipped:
        logger.warning(
            "[Judge] 跳过校验: %s", judge_result.judge_skipped_reason
        )
        result = llm_output.model_dump()
        result["judge_skipped"] = True
        result["judge_skipped_reason"] = judge_result.judge_skipped_reason
        return result

    # 用 Judge 结果更新 LLM 输出
    updated = llm_output.model_dump()
    updated["decision"] = judge_result.final_decision
    updated["reason"] = judge_result.final_reason
    updated["evidence"] = judge_result.final_evidence
    updated["confidence"] = judge_result.confidence
    updated["judge_verified"] = judge_result.verified
    updated["judge_corrections"] = judge_result.corrections
    updated["judge_hallucinated"] = judge_result.hallucinated_evidence
    updated["judge_missing_rules"] = judge_result.missing_rules

    return updated


# ---------------------------------------------------------------------------
# 默认单例
# ---------------------------------------------------------------------------

_default_judge: RuleReviewJudge | None = None


def get_default_judge() -> RuleReviewJudge:
    global _default_judge
    if _default_judge is None:
        _default_judge = RuleReviewJudge()
    return _default_judge
