"""
电力规则审查系统 - LLM 推理模块

按设计文档 Phase 1 步骤 1.5 实现：
- LLM 推理：复用 ProxyChatModel / ChatQwen
- 流式生成（SSE 管道输出）
- 非流式生成（同步模式 + Judge 校验链）
- 输出解析：JSON 块提取 + LLMOutput 校验
- 模型工厂：复用 WorkflowFactory 的 _create_chat_model 模式

遵循现有项目模式：
- 模型实例单例创建（不在每次调用时重新创建）
- 通过 settings.RULE_REVIEW_MODEL 等配置控制模型选择
"""

from __future__ import annotations

import asyncio
import json
import logging
import textwrap
from typing import Any, AsyncGenerator

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage

from src.config import settings
from src.rule_review.prompts import SYSTEM_PROMPT, build_messages, build_rag_context_prompt
from src.rule_review.schemas import LLMOutput
from src.utils.output_parser import parse_json_block

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 模型工厂（复用 WorkflowFactory 模式）
# ---------------------------------------------------------------------------


def _create_rule_review_model() -> Any:
    """创建规则审查专用 LLM 模型实例。

    模型路由逻辑与 WorkflowFactory._create_chat_model 一致：
    - 模型名在 GATEWAY_MODELS 列表中 → ProxyChatModel（内网中转）
    - 否则 → ChatQwen（DashScope 直连）
    """
    model_name = settings.RULE_REVIEW_MODEL
    api_key = settings.RULE_REVIEW_API_KEY
    base_url = settings.RULE_REVIEW_API_BASE

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
# LLM 输出解析
# ---------------------------------------------------------------------------


def parse_llm_output(raw_text: str) -> LLMOutput | None:
    """从 LLM 原始输出中提取 JSON 并解析为 LLMOutput。

    容错处理：
    1. 先尝试标准 JSON 块提取（```json ... ``` 或直接 {...}）
    2. 解析失败 → 记录 warning 并返回 None
    3. 解析成功后，校验必要字段并填充默认值

    Args:
        raw_text: LLM 原始输出文本。

    Returns:
        LLMOutput 实例，解析失败返回 None。
    """
    parsed = parse_json_block(raw_text)

    if parsed is None:
        logger.warning("无法从 LLM 输出中解析 JSON 块，原始输出前 200 字符: %s",
                       raw_text[:200])
        return None

    if not isinstance(parsed, dict):
        logger.warning("LLM 输出解析为非 dict 类型: %s", type(parsed))
        return None

    try:
        return LLMOutput(
            decision=str(parsed.get("decision", "无法判断")),
            reason=str(parsed.get("reason", "")),
            evidence=parsed.get("evidence", []),
            confidence=float(parsed.get("confidence", 0.0)),
            tool_calls=parsed.get("tool_calls", []),
            not_found=bool(parsed.get("not_found", False)),
        )
    except Exception as e:
        logger.warning("LLMOutput 构造失败: %s, parsed=%s", e, textwrap.shorten(str(parsed), 200))
        return None


# ---------------------------------------------------------------------------
# 规则审查生成器
# ---------------------------------------------------------------------------


class RuleReviewGenerator:
    """规则审查 LLM 推理器。

    负责将检索到的 chunks 组装为 Prompt，调用 LLM 生成审查结果。

    Phase 1 能力：
    - 非流式生成：用于同步模式和 Judge 校验链
    - 流式生成：用于 SSE 管道输出

    Phase 2 增强：
    - Tool 调用循环（tool_calls 检测 + 重新生成）
    - 工具结果注入后的重新推理
    """

    def __init__(self, model: Any | None = None) -> None:
        """
        Args:
            model: LLM 模型实例。None 时使用默认规则审查模型。
        """
        self._model = model

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = _create_rule_review_model()
        return self._model

    def _to_langchain_messages(self, messages: list[dict]) -> list:
        """将 dict 格式的 messages 列表转换为 LangChain 消息对象。

        ProxyChatModel 接受字典或 LangChain 消息，ChatQwen 只接受 LangChain 消息。
        统一转换为 LangChain 消息以兼容两种模型。
        """
        result = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                result.append(SystemMessage(content=content))
            elif role == "assistant":
                result.append(AIMessage(content=content))
            else:
                result.append(HumanMessage(content=content))
        return result

    # ------------------------------------------------------------------
    # 非流式生成
    # ------------------------------------------------------------------

    async def generate(
        self,
        query: str,
        context_chunks: list[dict],
        system_prompt: str | None = None,
        tool_results: list[dict] | None = None,
        max_retries: int = 1,
    ) -> LLMOutput | None:
        """非流式生成审查结果。

        Args:
            query: 用户问题。
            context_chunks: 检索到的 chunks 列表。
            system_prompt: 自定义 System Prompt，None 时使用默认。
            tool_results: 工具执行结果（Phase 2）。
            max_retries: JSON 解析失败时的重试次数。

        Returns:
            LLMOutput 实例，生成或解析失败返回 None。
        """
        messages = build_messages(
            query=query,
            context_chunks=context_chunks,
            system_prompt=system_prompt,
            tool_results=tool_results,
        )

        raw_text = ""
        for attempt in range(max_retries + 1):
            try:
                lc_messages = self._to_langchain_messages(messages)
                response: AIMessage = await self.model.ainvoke(lc_messages)
                raw_text = _extract_content(response)

                result = parse_llm_output(raw_text)
                if result is not None:
                    return result

                if attempt < max_retries:
                    logger.info("JSON 解析失败，重试 %d/%d", attempt + 1, max_retries)
                    # 追加提示让 LLM 重新输出正确 JSON
                    messages.append({
                        "role": "user",
                        "content": (
                            "你的上一轮输出无法解析为有效的 JSON。"
                            "请严格按照要求的 JSON 格式重新输出，"
                            "确保 decision、reason、evidence、confidence 字段都在一个 JSON 对象内。"
                        ),
                    })
            except Exception as e:
                logger.error("LLM 调用异常 (attempt=%d): %s", attempt, e)
                if attempt >= max_retries:
                    return None

        return None

    # ------------------------------------------------------------------
    # 流式生成
    # ------------------------------------------------------------------

    async def generate_stream(
        self,
        query: str,
        context_chunks: list[dict],
        system_prompt: str | None = None,
        tool_results: list[dict] | None = None,
    ) -> AsyncGenerator[str, None]:
        """流式生成审查结果，逐 token yield 文本。

        用于 SSE 管道中实时推送生成进度。

        Args:
            query: 用户问题。
            context_chunks: 检索到的 chunks 列表。
            system_prompt: 自定义 System Prompt。
            tool_results: 工具执行结果（Phase 2）。

        Yields:
            每次 yield 一个增量文本片段。
        """
        messages = build_messages(
            query=query,
            context_chunks=context_chunks,
            system_prompt=system_prompt,
            tool_results=tool_results,
        )

        try:
            lc_messages = self._to_langchain_messages(messages)
            async for chunk in self.model.astream(lc_messages):
                text = _extract_chunk_content(chunk)
                if text:
                    yield text
        except Exception as e:
            logger.error("LLM 流式调用异常: %s", e)
            yield json.dumps(
                {
                    "error": f"LLM 调用失败: {str(e)}",
                    "decision": "无法判断",
                    "reason": "规则审查服务暂时不可用，请稍后重试。",
                    "evidence": [],
                    "confidence": 0.0,
                },
                ensure_ascii=False,
            )

    # ------------------------------------------------------------------
    # 便捷方法
    # ------------------------------------------------------------------

    async def generate_raw(
        self,
        messages: list[dict],
    ) -> str:
        """直接调用 LLM 并返回原始文本，不解析 JSON。

        用于需要自定义 Prompt 处理的场景（如问题澄清判断）。

        Args:
            messages: 完整的 messages 列表。

        Returns:
            LLM 原始输出文本。
        """
        try:
            lc_messages = self._to_langchain_messages(messages)
            response: AIMessage = await self.model.ainvoke(lc_messages)
            return _extract_content(response)
        except Exception as e:
            logger.error("LLM raw 调用异常: %s", e)
            return ""


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _extract_content(response: AIMessage) -> str:
    """从 AIMessage 中提取文本内容。"""
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # 多段内容拼接
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


def _extract_chunk_content(chunk: AIMessageChunk) -> str:
    """从 AIMessageChunk 中提取增量文本。"""
    content = chunk.content
    if isinstance(content, str):
        return content
    return ""


# ---------------------------------------------------------------------------
# 默认单例工厂
# ---------------------------------------------------------------------------

_default_generator: RuleReviewGenerator | None = None


def get_default_generator() -> RuleReviewGenerator:
    """获取默认规则审查生成器单例。"""
    global _default_generator
    if _default_generator is None:
        _default_generator = RuleReviewGenerator()
    return _default_generator
