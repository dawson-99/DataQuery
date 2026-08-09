"""
电力规则审查系统 - 编排器

按设计文档 §13 完整工作流设计实现 Phase 1 的 9 阶段 pipeline：
0. 问题改写 → 1. 澄清判断 → 2. 拆分判断 → 3. Query 优化
→ 4. RAG 检索 → 5. LLM 生成 → 6. Tool 调用 [Phase 2]
→ 7. Judge 校验 → 8. SSE 输出

Phase 1 仅走纯 LLM 推理路径（不含 Tool），覆盖全部 6 个分支场景。
Phase 2 增加 Tool 调用循环与终止条件。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

from src.config import settings
from src.rule_review.audit import AuditStore, build_source_traceability
from src.rule_review.document_store import DocumentStore
from src.rule_review.generator import RuleReviewGenerator, parse_llm_output
from src.rule_review.observability import LatencyStats, get_default_stats
from src.rule_review.prompts import DEFAULT_GENERATION_PROMPT
from src.rule_review.query_rewriter import QueryRewriter
from src.rule_review.retriever import HybridRetriever, RetrieveResult
from src.rule_review.schemas import (
    AuditRecord,
    ClarificationResponse,
    JudgeAudit,
    LLMGenerationAudit,
    LLMOutput,
    NotFoundResponse,
    RetrievalAudit,
    RuleReviewRequest,
    RuleReviewResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SSE 负载模型
# ---------------------------------------------------------------------------


@dataclass
class SSEProgress:
    """SSE 阶段的进度标签。"""

    text: str
    stage: str = ""


@dataclass
class CorrectiveLoopResult:
    """Corrective-RAG 回环执行结果。

    Judge 检出幻觉/遗漏后触发的"扩大检索 → 合并证据 → 重新生成 → 二次校验"
    单轮回环的状态载体，流式/非流式共用。
    """

    triggered: bool = False
    trigger_reason: str = ""  # missing_rules | hallucinated | both
    final_result: dict = field(default_factory=dict)  # 最终输出（可能降级为首轮结果）
    merged_chunks: list[dict] = field(default_factory=list)  # 合并后的 prompt chunks
    merged_retrieve_result: "RetrieveResult | None" = None
    second_llm_output: "LLMOutput | None" = None
    second_judged: dict | None = None
    corrective_query: str = ""
    second_top_k: int = 0
    retrieval_rounds: int = 1
    generation_rounds: int = 1
    judge_rounds: int = 1
    terminated_reason: str = ""  # "" | second_retrieve_empty/error | second_generate_none | second_not_found | second_judge_skipped
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    judge_ms: float = 0.0
    progress_labels: list = field(default_factory=list)  # [(label_text, stage)]


# ---------------------------------------------------------------------------
# 澄清判断
# ---------------------------------------------------------------------------

# 日期相关正则
_TIME_PATTERNS = [
    re.compile(r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日号]?"),
    re.compile(r"(今天|昨天|明天|本月|上月|本周|上周|下周|今年|去年|明年|今日|昨日|明日)"),
    re.compile(r"(元旦|春节|清明|劳动节|端午|中秋|国庆|五一|十一)"),
    re.compile(r"\d{1,2}月\d{1,2}[日号]"),
]

# 比较意图关键词
_COMPARISON_KEYWORDS = [
    "是否", "是不是", "有没有", "符合", "违反", "违反",
    "超过", "超出", "低于", "达到", "满足", "不满足",
    "大于", "小于", "等于", "介于", "处于", "合规",
]


def _has_time_info(query: str) -> bool:
    """检查 query 中是否包含时间信息。"""
    for pat in _TIME_PATTERNS:
        if pat.search(query):
            return True
    return False


# 常见电力交易地域名称（无后缀也能识别）
_KNOWN_REGION_NAMES = [
    "冀北", "山西", "蒙东", "蒙西", "四川", "华东", "华北", "华中",
    "东北", "西北", "华南", "京津唐", "浙江", "江苏", "上海", "北京",
    "山东", "河北", "河南", "湖北", "湖南", "安徽", "福建", "江西",
    "陕西", "宁夏", "新疆", "青海", "西藏", "辽宁", "吉林", "黑龙江",
    "广东", "广西", "云南", "贵州", "海南", "重庆",
]


def _has_entity_info(query: str) -> bool:
    """检查 query 中是否包含实体信息（地名等）。"""
    # 1. 检查已知地域名称
    for name in _KNOWN_REGION_NAMES:
        if name in query:
            return True

    # 2. 含"省/市/区/东/南/西/北/网"等地域特征后缀
    if re.search(r"[一-鿿]{2,}(?:省|市|区|分部|直调|电网|主网|东部|南部|西部|北部|中部)", query):
        return True

    return False


def _is_query_subsumed(
    candidate: str, query: str, parts: list[str]
) -> bool:
    """判断补充关键词是否被已有 query/补充项包含（互相包含则跳过）。

    Corrective-RAG 回环构造二次检索 query 时用于去重：
    与 rewritten_query 或已选补充片段重叠的内容不重复加入。
    """
    if not candidate:
        return True
    if candidate in query or query in candidate:
        return True
    for p in parts:
        if candidate in p or p in candidate:
            return True
    return False


def _build_corrective_audit(
    loop_result: CorrectiveLoopResult | None,
) -> dict | None:
    """构造 Corrective-RAG 回环的审计详情 dict（未触发时返回 None）。

    供 execute_stream / execute 的 AuditRecord 使用，字段全带默认值，
    旧审计记录 load 时保持向后兼容。
    """
    if loop_result is None or not loop_result.triggered:
        return None
    second_judged = loop_result.second_judged or {}
    second_llm = loop_result.second_llm_output
    return {
        "triggered": True,
        "reason": loop_result.trigger_reason,
        "corrective_query": loop_result.corrective_query,
        "second_top_k": loop_result.second_top_k,
        "merged_chunk_count": len(loop_result.merged_chunks),
        "merged_chunk_ids": [c.get("chunk_id", "") for c in loop_result.merged_chunks],
        "round2_verified": bool(second_judged.get("judge_verified", False)),
        "round2_skipped": bool(second_judged.get("judge_skipped", False)),
        "round2_tok_input": second_llm.tok_input if second_llm is not None else 0,
        "round2_tok_output": second_llm.tok_output if second_llm is not None else 0,
        "terminated_reason": loop_result.terminated_reason,
    }


def _has_comparison_intent(query: str) -> bool:
    """检查 query 中是否有规则审查的比较意图。"""
    for kw in _COMPARISON_KEYWORDS:
        if kw in query:
            return True
    return False


def check_clarification_needed(rewritten_query: str) -> ClarificationResponse:
    """判断问题是否足够明确，不明确的返回追问建议。

    纯规则判断，不调 LLM。

    Args:
        rewritten_query: 改写后的标准化问题。

    Returns:
        ClarificationResponse，needs_clarification=False 时继续后续流程。
    """
    missing: list[str] = []

    if not _has_time_info(rewritten_query):
        missing.append("时间范围（如 2025年3月15日）")

    if not _has_entity_info(rewritten_query):
        missing.append("查询主体（如省份、节点名称）")

    if not _has_comparison_intent(rewritten_query):
        missing.append("具体审查问题（如是否符合价格上限）")

    if not missing:
        return ClarificationResponse(needs_clarification=False)

    return ClarificationResponse(
        needs_clarification=True,
        missing=missing,
        suggestions=[
            f"请补充：{'、'.join(missing)}",
            "例如：2025年3月15日冀北的日前现货出清电价达到800元/MWh，是否符合价格上限规则？",
        ],
    )


# ---------------------------------------------------------------------------
# 多文档拆分
# ---------------------------------------------------------------------------

# 文档名提及模式：书名号包裹的文本
_DOC_NAME_RE = re.compile(r"《([^》]+)》")


def split_if_multi_document(
    query: str, document_store: DocumentStore
) -> list[dict]:
    """检测是否涉及多文档，如果是则拆分。

    Args:
        query: 用户问题。
        document_store: 文档存储实例。

    Returns:
        - 单文档: [{"sub_query": query, "doc_name": None}]
        - 多文档: [{"sub_query": "子问题", "doc_name": "规则A"}, ...]
    """
    # 从书名号提取提及的文档名
    mentioned_in_query = _DOC_NAME_RE.findall(query)

    # 从已入库文档名匹配
    docs = document_store.list_documents()
    doc_file_names = [d.file_name for d in docs]
    doc_ids = {d.file_name: d.doc_id for d in docs}

    # 匹配提及的文档名与入库文档
    matched: list[dict] = []
    for name in mentioned_in_query:
        for fname in doc_file_names:
            # 模糊匹配：文档名包含提及的名称
            if name in fname or fname in name:
                matched.append({"doc_name": fname, "doc_id": doc_ids[fname]})
                break

    if len(matched) <= 1:
        return [{"sub_query": query, "doc_name": None, "doc_id": None}]

    # 多文档拆分：为每个文档创建独立检索任务
    sub_items: list[dict] = []
    for m in matched:
        # 从 query 中移除该文档名，得到子问题
        sub_query = query
        for mn in mentioned_in_query:
            sub_query = sub_query.replace(f"《{mn}》", "").strip()
        sub_items.append({
            "sub_query": sub_query or query,
            "doc_name": m["doc_name"],
            "doc_id": m["doc_id"],
        })
    return sub_items


# ---------------------------------------------------------------------------
# 编排器核心
# ---------------------------------------------------------------------------


class RuleReviewPipeline:
    """规则审查编排器。

    按设计文档 §13 的 9 阶段工作流编排执行，覆盖所有分支场景。

    Phase 1: 阶段 0-5 + 阶段 7（Judge）+ 阶段 8（SSE 输出）
    Phase 2: 增加阶段 6（Tool 调用循环）
    """

    def __init__(
        self,
        rewriter: QueryRewriter | None = None,
        document_store: DocumentStore | None = None,
        retriever: HybridRetriever | None = None,
        generator: RuleReviewGenerator | None = None,
        judge: Any | None = None,  # RuleReviewJudge，Phase 1 可选
        audit_store: AuditStore | None = None,  # 审计存储，None 时不记录
        stats: LatencyStats | None = None,  # 延迟统计，None 时用默认单例
    ) -> None:
        """
        Args:
            rewriter: 问题改写器，None 时自动创建。
            document_store: 文档存储实例，None 时自动创建。
            retriever: 混合检索器，None 时自动创建（依赖 document_store）。
            generator: LLM 推理器，None 时自动创建。
            judge: Judge 校验器，None 时跳过校验阶段。
            audit_store: 审计存储，None 时不记录审计日志。
            stats: 阶段延迟统计（可观测性），None 时用默认单例。
        """
        self.rewriter = rewriter or QueryRewriter()
        self.doc_store = document_store or DocumentStore()
        self.retriever = retriever or HybridRetriever(document_store=self.doc_store)
        self.generator = generator or RuleReviewGenerator()
        self.judge = judge  # None → 跳过 Judge 阶段
        self.audit_store = audit_store  # None → 不记录审计
        self.stats = stats  # None → 记录时取默认单例

    # ------------------------------------------------------------------
    # 全流程：生成器方法
    # ------------------------------------------------------------------

    async def execute_stream(
        self, request: RuleReviewRequest
    ) -> AsyncGenerator[str, None]:
        """SSE 流式执行完整规则审查流程。

        按阶段顺序推进，每阶段通过 SSE progress 标签推送进度，
        最终输出审查结果 JSON。

        Args:
            request: 规则审查请求。

        Yields:
            SSE 格式的字符串（每行以 "data: " 开头）。
        """
        query_id = str(uuid.uuid4())
        stage_start = time.monotonic()

        # ---- 阶段 0：问题改写 ----
        yield self._sse_label("查询预处理中...", "rewrite")
        rewritten_query = self.rewriter.rewrite(request.question)
        logger.info("[pipeline] 改写: %s → %s", request.question[:50], rewritten_query[:50])

        # ---- 阶段 1：澄清判断 ----
        yield self._sse_label("问题分析中...", "clarification")
        clarification = check_clarification_needed(rewritten_query)
        if clarification.needs_clarification:
            logger.info("[pipeline] 问题不明确，返回澄清追问")
            yield self._sse_content(
                clarification.model_dump_json(),
                event="clarification",
            )
            yield self._sse_done(query_id)
            return

        # ---- 阶段 2：拆分判断 ----
        sub_items = split_if_multi_document(rewritten_query, self.doc_store)
        is_multi_doc = len(sub_items) > 1
        if is_multi_doc:
            yield self._sse_label(f"检测到多文档查询，拆分为 {len(sub_items)} 个子问题...", "split")

        # ---- 阶段 3 + 4：Query 优化 + RAG 检索 ----
        yield self._sse_label("检索相关知识中...", "retrieval")
        retrieval_start = time.monotonic()

        if is_multi_doc:
            # 多文档：并行检索（retrieve_with_fallback 是同步方法）
            retrieve_tasks = [
                asyncio.to_thread(
                    self.retriever.retrieve_with_fallback,
                    item["sub_query"],
                    top_k=request.top_k,
                    doc_filter=item.get("doc_id"),
                )
                for item in sub_items
            ]
            all_retrieve_results = await asyncio.gather(*retrieve_tasks)
            retrieve_result = self._merge_retrieve_results(
                all_retrieve_results, top_k=request.top_k
            )
        else:
            retrieve_result = self.retriever.retrieve_with_fallback(
                rewritten_query, top_k=request.top_k
            )

        retrieval_end = time.monotonic()
        retrieval_latency_ms = (retrieval_end - retrieval_start) * 1000

        # ---- 空检索兜底 ----
        if retrieve_result.not_found and not retrieve_result.results:
            logger.info("[pipeline] 检索无结果，返回 not_found")
            yield self._sse_content(
                NotFoundResponse().model_dump_json(),
                event="not_found",
            )
            yield self._sse_done(query_id)
            return

        if retrieve_result.search_expanded:
            yield self._sse_label("扩大检索范围中...", "retrieval_expand")

        # ---- 阶段 5：LLM 生成 ----
        yield self._sse_label("规则推理中...", "generation")
        generation_start = time.monotonic()

        # 将检索结果转为 prompt 所需的格式
        context_chunks = self._chunks_to_dict_list(retrieve_result.results)
        llm_output = await self.generator.generate(
            query=rewritten_query,
            context_chunks=context_chunks,
            system_prompt=DEFAULT_GENERATION_PROMPT,
        )

        generation_end = time.monotonic()
        generation_latency_ms = (generation_end - generation_start) * 1000

        if llm_output is None:
            # LLM 生成失败
            logger.error("[pipeline] LLM 生成失败")
            yield self._sse_content(
                json.dumps(
                    {
                        "decision": "无法判断",
                        "reason": "规则审查服务暂时不可用，请稍后重试。",
                        "evidence": [],
                        "confidence": 0.0,
                    },
                    ensure_ascii=False,
                ),
                event="error",
            )
            yield self._sse_done(query_id)
            return

        # LLM 判断文档中无相关规则 → 跳过后续步骤
        if llm_output.not_found:
            logger.info("[pipeline] LLM 返回 not_found，跳过后续步骤")
            yield self._sse_content(
                llm_output.model_dump_json(),
                event="content",
            )
            yield self._sse_done(query_id)
            return

        # ---- 阶段 6：Tool 调用 ----
        final_output = llm_output
        # 前置初始化：无 tool_calls 时为空列表，供阶段 7 Judge 的 tool_logs 使用
        tool_logs: list[dict] = []

        if llm_output.tool_calls:
            yield self._sse_label("工具调用中...", "tool")
            try:
                from src.rule_review.tool_executor import execute_with_tool_loop

                tool_result, tool_logs = await execute_with_tool_loop(
                    self.generator, rewritten_query, context_chunks
                )
                if tool_result is not None:
                    final_output = LLMOutput(**tool_result) if isinstance(tool_result, dict) else tool_result
                if tool_result and tool_result.get("tool_unsolved"):
                    yield self._sse_label("工具未完成，降级为直接推理...", "tool_fallback")
            except Exception as e:
                logger.warning("[pipeline] Tool 阶段异常: %s", e)

        # ---- 阶段 7：Judge 校验 ----
        judge_audit = JudgeAudit(skipped=True, skipped_reason="no_judge_configured")
        corrective_loop_result: CorrectiveLoopResult | None = None
        if self.judge is not None:
            yield self._sse_label("结果校验中...", "judge")
            judge_start = time.monotonic()
            try:
                from src.rule_review.judge import verify_with_fallback

                # 使用 tool 阶段后的 final_output 进行校验
                judged = await verify_with_fallback(
                    self.judge,
                    final_output if isinstance(final_output, LLMOutput) else llm_output,
                    rewritten_query, context_chunks,
                    tool_logs=tool_logs,
                )
                if judged.get("judge_skipped"):
                    yield self._sse_label(
                        f"校验跳过: {judged.get('judge_skipped_reason', '')}...",
                        "judge_skipped",
                    )
                    judge_audit = JudgeAudit(
                        skipped=True,
                        skipped_reason=judged.get("judge_skipped_reason", "unknown"),
                        latency_ms=(time.monotonic() - judge_start) * 1000,
                    )
                else:
                    judge_audit = JudgeAudit(
                        verified=judged.get("verified", False),
                        hallucinated_count=len(judged.get("hallucinated_evidence", [])),
                        skipped=False,
                        latency_ms=(time.monotonic() - judge_start) * 1000,
                    )
                final_output = judged

                # ---- Corrective-RAG 回环：Judge 检出幻觉/遗漏 → 扩大检索 ----
                # 最多 1 轮：二次检索 → 合并证据 → 带反馈重新生成 → 二次校验
                if self._should_run_corrective(final_output):
                    loop_labels: list[tuple[str, str]] = []
                    corrective_loop_result = await self._run_corrective_loop(
                        rewritten_query=rewritten_query,
                        first_retrieve_result=retrieve_result,
                        first_llm_output=(
                            final_output if isinstance(final_output, LLMOutput) else llm_output
                        ),
                        first_judged=final_output,
                        top_k=request.top_k,
                        progress=lambda t, s: loop_labels.append((t, s)),
                    )
                    # SSE 标签在回环结束后统一 flush（生成本就是非流式调用）
                    for label_text, label_stage in loop_labels:
                        yield self._sse_label(label_text, label_stage)
                    final_output = corrective_loop_result.final_result
                    # 审计与统计更新为第二轮结论
                    judge_audit = JudgeAudit(
                        verified=final_output.get("judge_verified", False),
                        hallucinated_count=len(final_output.get("judge_hallucinated", [])),
                        skipped=final_output.get("judge_skipped", False),
                        skipped_reason=final_output.get("judge_skipped_reason", ""),
                        rounds=2,
                        latency_ms=(time.monotonic() - judge_start) * 1000,
                    )
            except Exception as e:
                logger.warning("[pipeline] Judge 阶段异常: %s", e)
                yield self._sse_label("校验服务繁忙，已跳过校验...", "judge_skipped")
                judge_audit = JudgeAudit(skipped=True, skipped_reason=str(e))

        # ---- 阶段 8：SSE 最终输出 ----
        yield self._sse_label("生成结果中...", "result")

        if isinstance(final_output, dict):
            yield self._sse_content(
                json.dumps(final_output, ensure_ascii=False),
                event="content",
            )
        else:
            yield self._sse_content(
                final_output.model_dump_json(),
                event="content",
            )

        elapsed = time.monotonic() - stage_start
        logger.info(
            "[pipeline] 完成，耗时 %.2fs, query_id=%s",
            elapsed, query_id,
            extra={"stage": "total", "latency_ms": round(elapsed * 1000, 2)},
        )

        # ---- 延迟统计（可观测性）----
        self._record_latencies(
            retrieval_ms=retrieval_latency_ms
            + (corrective_loop_result.retrieval_ms if corrective_loop_result else 0.0),
            generation_ms=generation_latency_ms
            + (corrective_loop_result.generation_ms if corrective_loop_result else 0.0),
            judge_ms=judge_audit.latency_ms if judge_audit else 0.0,
            total_ms=elapsed * 1000,
        )

        # ---- 构建并保存审计记录 ----
        if self.audit_store is not None:
            try:
                # 构建溯源链（Corrective 回环后使用合并后的 chunks）
                final_dict = final_output if isinstance(final_output, dict) else final_output.model_dump()
                source_chunks = (
                    corrective_loop_result.merged_chunks
                    if corrective_loop_result and corrective_loop_result.merged_chunks
                    else context_chunks
                )
                source_traces = build_source_traceability(final_dict, source_chunks)

                audit_record = AuditRecord(
                    query_id=query_id,
                    timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    original_query=request.question,
                    rewritten_query=rewritten_query,
                    retrieval=RetrievalAudit(
                        bm25_k=retrieve_result.bm25_hits,
                        vector_k=retrieve_result.vector_hits,
                        sparse_k=0,  # SparseRetriever 未接入运行路径，如实记录
                        final_k=len(retrieve_result.results),
                        search_expanded=retrieve_result.search_expanded,
                        retrieval_latency_ms=round(retrieval_latency_ms, 2),
                        retrieval_rounds=(
                            corrective_loop_result.retrieval_rounds
                            if corrective_loop_result else 1
                        ),
                        corrective_expanded=(
                            corrective_loop_result.triggered
                            if corrective_loop_result else False
                        ),
                    ),
                    llm_generation=LLMGenerationAudit(
                        model=settings.RULE_REVIEW_MODEL,
                        tok_input=llm_output.tok_input,
                        tok_output=llm_output.tok_output,
                        not_found=llm_output.not_found,
                        latency_ms=round(generation_latency_ms, 2),
                        rounds=(
                            corrective_loop_result.generation_rounds
                            if corrective_loop_result else 1
                        ),
                    ),
                    judge_verification=judge_audit,
                    final_result=final_dict,
                    source_traceability=source_traces,
                    corrective=_build_corrective_audit(corrective_loop_result),
                )
                self.audit_store.save(audit_record)
            except Exception as e:
                logger.warning("[pipeline] 审计记录保存失败: %s", e)

        yield self._sse_done(query_id)

    async def execute(
        self, request: RuleReviewRequest
    ) -> dict[str, Any]:
        """非流式执行完整规则审查流程，返回最终结果字典。

        Args:
            request: 规则审查请求。

        Returns:
            包含最终结果和所有阶段信息的字典。
        """
        query_id = str(uuid.uuid4())
        stage_start = time.monotonic()
        stages_log: list[dict] = []

        # 阶段 0
        rewritten_query = self.rewriter.rewrite(request.question)
        stages_log.append({"stage": "rewrite", "input": request.question[:100], "output": rewritten_query[:100]})

        # 阶段 1
        clarification = check_clarification_needed(rewritten_query)
        if clarification.needs_clarification:
            return {
                "query_id": query_id,
                "clarification": clarification.model_dump(),
                "stages": stages_log,
            }
        stages_log.append({"stage": "clarification", "needs_clarification": False})

        # 阶段 2
        sub_items = split_if_multi_document(rewritten_query, self.doc_store)
        is_multi_doc = len(sub_items) > 1
        stages_log.append({"stage": "split", "is_multi_doc": is_multi_doc, "sub_items": len(sub_items)})

        # 阶段 3+4: 检索
        retrieval_start = time.monotonic()
        if is_multi_doc:
            retrieve_tasks = [
                asyncio.to_thread(
                    self.retriever.retrieve_with_fallback,
                    item["sub_query"],
                    top_k=request.top_k,
                    doc_filter=item.get("doc_id"),
                )
                for item in sub_items
            ]
            all_retrieve_results = await asyncio.gather(*retrieve_tasks)
            retrieve_result = self._merge_retrieve_results(all_retrieve_results, top_k=request.top_k)
        else:
            retrieve_result = self.retriever.retrieve_with_fallback(
                rewritten_query, top_k=request.top_k
            )
        retrieval_latency_ms = (time.monotonic() - retrieval_start) * 1000

        stages_log.append({
            "stage": "retrieval",
            "not_found": retrieve_result.not_found,
            "result_count": len(retrieve_result.results),
            "search_expanded": retrieve_result.search_expanded,
            # 检索到的 chunk 明细（文本截断），供离线评估做幻觉检测与 recall@k/MRR 计算
            "retrieved_chunks": [
                {
                    "chunk_id": getattr(getattr(r, "chunk", r), "chunk_id", ""),
                    "source": getattr(getattr(r, "chunk", r), "source", "未知文档"),
                    "text": getattr(getattr(r, "chunk", r), "text", "")[:200],
                }
                for r in retrieve_result.results
            ],
            # 完整 chunk 文本（不截断），供离线评估做 RAGAS 风格 LLM-judge 指标判分
            "retrieved_chunks_full": [
                {
                    "chunk_id": getattr(c, "chunk_id", ""),
                    "source": getattr(c, "source", "未知文档"),
                    "section": getattr(c, "section", ""),
                    "page": getattr(c, "page", 0),
                    "text": getattr(c, "text", ""),
                }
                for r in retrieve_result.results
                for c in [getattr(r, "chunk", r)]
            ],
        })

        # 空检索
        if retrieve_result.not_found and not retrieve_result.results:
            return {
                "query_id": query_id,
                "result": NotFoundResponse().model_dump(),
                "stages": stages_log,
            }

        # 阶段 5: LLM 生成
        generation_start = time.monotonic()
        context_chunks = self._chunks_to_dict_list(retrieve_result.results)
        llm_output = await self.generator.generate(
            query=rewritten_query,
            context_chunks=context_chunks,
            system_prompt=DEFAULT_GENERATION_PROMPT,
        )
        generation_latency_ms = (time.monotonic() - generation_start) * 1000

        if llm_output is None:
            return {
                "query_id": query_id,
                "result": {
                    "decision": "无法判断",
                    "reason": "规则审查服务暂时不可用，请稍后重试。",
                    "evidence": [],
                    "confidence": 0.0,
                },
                "stages": stages_log,
            }

        stages_log.append({
            "stage": "generation",
            "not_found": llm_output.not_found,
            "decision": llm_output.decision,
            "confidence": llm_output.confidence,
        })

        # ---- 阶段 6：Tool 调用 ----
        final_llm_output = llm_output
        tool_logs = []
        if llm_output.tool_calls:
            from src.rule_review.tool_executor import execute_with_tool_loop

            tool_result, tool_logs = await execute_with_tool_loop(
                self.generator, rewritten_query, context_chunks
            )
            if tool_result is not None:
                final_llm_output = LLMOutput(**tool_result) if isinstance(tool_result, dict) else tool_result
            stages_log.append({
                "stage": "tool",
                "rounds": len(set(t.get("round", 0) for t in tool_logs)),
                "tool_calls": len(tool_logs),
                "tool_unsolved": tool_result.get("tool_unsolved", False) if tool_result else True,
            })

        # ---- 阶段 7：Judge 校验 ----
        judge_audit = JudgeAudit(skipped=True, skipped_reason="no_judge_configured")
        corrective_loop_result: CorrectiveLoopResult | None = None
        final_result = final_llm_output.model_dump()
        if self.judge is not None:
            judge_start = time.monotonic()
            try:
                from src.rule_review.judge import verify_with_fallback

                judged = await verify_with_fallback(
                    self.judge, final_llm_output, rewritten_query, context_chunks,
                    tool_logs=tool_logs,
                )
                final_result = judged
                if judged.get("judge_skipped"):
                    judge_audit = JudgeAudit(
                        skipped=True,
                        skipped_reason=judged.get("judge_skipped_reason", "unknown"),
                        latency_ms=(time.monotonic() - judge_start) * 1000,
                    )
                else:
                    judge_audit = JudgeAudit(
                        verified=judged.get("verified", False),
                        hallucinated_count=len(judged.get("hallucinated_evidence", [])),
                        skipped=False,
                        latency_ms=(time.monotonic() - judge_start) * 1000,
                    )

                # ---- Corrective-RAG 回环：Judge 检出幻觉/遗漏 → 扩大检索 ----
                if self._should_run_corrective(final_result):
                    corrective_loop_result = await self._run_corrective_loop(
                        rewritten_query=rewritten_query,
                        first_retrieve_result=retrieve_result,
                        first_llm_output=(
                            final_llm_output
                            if isinstance(final_llm_output, LLMOutput)
                            else llm_output
                        ),
                        first_judged=final_result,
                        top_k=request.top_k,
                    )
                    final_result = corrective_loop_result.final_result
                    # 审计与统计更新为第二轮结论
                    judge_audit = JudgeAudit(
                        verified=final_result.get("judge_verified", False),
                        hallucinated_count=len(final_result.get("judge_hallucinated", [])),
                        skipped=final_result.get("judge_skipped", False),
                        skipped_reason=final_result.get("judge_skipped_reason", ""),
                        rounds=2,
                        latency_ms=(time.monotonic() - judge_start) * 1000,
                    )
            except Exception as e:
                logger.warning("[pipeline] Judge 阶段异常: %s", e)
                judge_audit = JudgeAudit(skipped=True, skipped_reason=str(e))
            stages_log.append({
                "stage": "judge",
                "skipped": judged.get("judge_skipped", False),
                "verified": judged.get("judge_verified", False),
            })

            # Corrective 回环阶段日志（格式与首轮 retrieval 一致，供离线评估取合并后证据）
            if corrective_loop_result is not None:
                merged_rr = corrective_loop_result.merged_retrieve_result
                stages_log.append({
                    "stage": "re_retrieval",
                    "triggered_by": corrective_loop_result.trigger_reason,
                    "corrective_query": corrective_loop_result.corrective_query,
                    "second_top_k": corrective_loop_result.second_top_k,
                    "result_count": len(merged_rr.results) if merged_rr else 0,
                    "terminated_reason": corrective_loop_result.terminated_reason,
                    "retrieved_chunks": [
                        {
                            "chunk_id": getattr(getattr(r, "chunk", r), "chunk_id", ""),
                            "source": getattr(getattr(r, "chunk", r), "source", "未知文档"),
                            "text": getattr(getattr(r, "chunk", r), "text", "")[:200],
                        }
                        for r in (merged_rr.results if merged_rr else [])
                    ],
                    "retrieved_chunks_full": [
                        {
                            "chunk_id": getattr(c, "chunk_id", ""),
                            "source": getattr(c, "source", "未知文档"),
                            "section": getattr(c, "section", ""),
                            "page": getattr(c, "page", 0),
                            "text": getattr(c, "text", ""),
                        }
                        for r in (merged_rr.results if merged_rr else [])
                        for c in [getattr(r, "chunk", r)]
                    ],
                })
                if corrective_loop_result.second_llm_output is not None:
                    stages_log.append({
                        "stage": "generation",
                        "round": 2,
                        "not_found": corrective_loop_result.second_llm_output.not_found,
                        "decision": corrective_loop_result.second_llm_output.decision,
                        "confidence": corrective_loop_result.second_llm_output.confidence,
                    })
                if corrective_loop_result.second_judged is not None:
                    stages_log.append({
                        "stage": "judge",
                        "round": 2,
                        "skipped": corrective_loop_result.second_judged.get("judge_skipped", False),
                        "verified": corrective_loop_result.second_judged.get("judge_verified", False),
                    })

        # ---- 延迟统计（可观测性）----
        self._record_latencies(
            retrieval_ms=retrieval_latency_ms
            + (corrective_loop_result.retrieval_ms if corrective_loop_result else 0.0),
            generation_ms=generation_latency_ms
            + (corrective_loop_result.generation_ms if corrective_loop_result else 0.0),
            judge_ms=judge_audit.latency_ms if judge_audit else 0.0,
            total_ms=(time.monotonic() - stage_start) * 1000,
        )

        # ---- 构建并保存审计记录 ----
        if self.audit_store is not None:
            try:
                # Corrective 回环后使用合并后的 chunks 构建溯源链
                source_chunks = (
                    corrective_loop_result.merged_chunks
                    if corrective_loop_result and corrective_loop_result.merged_chunks
                    else self._chunks_to_dict_list(retrieve_result.results)
                )
                source_traces = build_source_traceability(final_result, source_chunks)

                audit_record = AuditRecord(
                    query_id=query_id,
                    timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    original_query=request.question,
                    rewritten_query=rewritten_query,
                    retrieval=RetrievalAudit(
                        bm25_k=retrieve_result.bm25_hits,
                        vector_k=retrieve_result.vector_hits,
                        sparse_k=0,  # SparseRetriever 未接入运行路径，如实记录
                        final_k=len(retrieve_result.results),
                        search_expanded=retrieve_result.search_expanded,
                        retrieval_latency_ms=round(retrieval_latency_ms, 2),
                        retrieval_rounds=(
                            corrective_loop_result.retrieval_rounds
                            if corrective_loop_result else 1
                        ),
                        corrective_expanded=(
                            corrective_loop_result.triggered
                            if corrective_loop_result else False
                        ),
                    ),
                    llm_generation=LLMGenerationAudit(
                        model=settings.RULE_REVIEW_MODEL,
                        tok_input=llm_output.tok_input,
                        tok_output=llm_output.tok_output,
                        not_found=llm_output.not_found,
                        latency_ms=round(generation_latency_ms, 2),
                        rounds=(
                            corrective_loop_result.generation_rounds
                            if corrective_loop_result else 1
                        ),
                    ),
                    tool_executions=[
                        ToolCallLog(
                            query_id=query_id,
                            round=t.get("round", 1),
                            tool_name=t.get("tool_name", t.get("tool", "")),
                            args=t.get("args", {}),
                            result=t.get("result", {}),
                            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                            latency_ms=t.get("latency_ms", 0),
                        )
                        for t in tool_logs
                    ],
                    judge_verification=judge_audit,
                    final_result=final_result,
                    source_traceability=source_traces,
                    corrective=_build_corrective_audit(corrective_loop_result),
                )
                self.audit_store.save(audit_record)
            except Exception as e:
                logger.warning("[pipeline] 审计记录保存失败: %s", e)

        return {
            "query_id": query_id,
            "rewritten_query": rewritten_query,
            "result": final_result,
            "stages": stages_log,
        }

    # ------------------------------------------------------------------
    # 便捷方法：各阶段独立调用
    # ------------------------------------------------------------------

    async def run_rewrite_and_clarify(
        self, question: str
    ) -> tuple[str, ClarificationResponse]:
        """执行阶段 0-1：改写 + 澄清判断。"""
        rewritten = self.rewriter.rewrite(question)
        clarification = check_clarification_needed(rewritten)
        return rewritten, clarification

    async def run_retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> RetrieveResult:
        """执行阶段 3-4：检索（含空检索兜底）。

        注意：retrieve_with_fallback 是同步方法，此处直接返回其结果，
        由调用方决定是否 to_thread 包装（async 包装同步调用，await 可取值）。
        """
        return self.retriever.retrieve_with_fallback(query, top_k=top_k)

    async def run_generate(
        self,
        query: str,
        retrieve_result: RetrieveResult,
    ) -> LLMOutput | None:
        """执行阶段 5：LLM 生成。"""
        context_chunks = self._chunks_to_dict_list(retrieve_result.results)
        return await self.generator.generate(query=query, context_chunks=context_chunks)

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _record_latencies(
        self,
        retrieval_ms: float,
        generation_ms: float,
        judge_ms: float,
        total_ms: float,
    ) -> None:
        """记录各阶段耗时到延迟统计（可观测性）。

        统计失败仅记 debug 日志，绝不阻断主流程。
        """
        try:
            stats = self.stats or get_default_stats()
            stats.record("retrieval", retrieval_ms)
            stats.record("generation", generation_ms)
            stats.record("judge", judge_ms)
            stats.record("total", total_ms)
        except Exception:
            logger.debug("[pipeline] 延迟统计记录失败", exc_info=True)

    @staticmethod
    def _chunks_to_dict_list(
        results: list[Any],
    ) -> list[dict]:
        """将 HybridSearchResult 列表转为 prompts 所需格式。"""
        from src.rule_review.retriever import HybridSearchResult

        output = []
        for r in results:
            if isinstance(r, HybridSearchResult):
                chunk = r.chunk
            else:
                chunk = r

            output.append({
                "text": chunk.text,
                "source": getattr(chunk, "source", "未知文档"),
                "section": getattr(chunk, "section", ""),
                "page": getattr(chunk, "page", 0),
                "chunk_id": getattr(chunk, "chunk_id", ""),
            })
        return output

    @staticmethod
    def _merge_retrieve_results(
        results: list[RetrieveResult],
        top_k: int = 10,
    ) -> RetrieveResult:
        """合并多文档检索结果，去重后取 top_k。

        Args:
            results: 各子问题的检索结果列表。
            top_k: 最终保留数量。

        Returns:
            合并后的 RetrieveResult。
        """
        seen_ids: set[str] = set()
        merged: list[Any] = []
        not_found_all = True
        search_expanded = False
        total_bm25 = 0
        total_vector = 0

        for rr in results:
            if not rr.not_found:
                not_found_all = False
            if rr.search_expanded:
                search_expanded = True
            total_bm25 += rr.bm25_hits
            total_vector += rr.vector_hits
            for r in rr.results:
                cid = r.chunk.chunk_id if hasattr(r, "chunk") else str(id(r))
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    merged.append(r)

        # 按分数排序
        merged.sort(key=lambda x: x.score if hasattr(x, "score") else 0, reverse=True)
        merged = merged[:top_k]

        return RetrieveResult(
            results=merged,
            not_found=len(merged) == 0,
            search_expanded=search_expanded,
            bm25_hits=total_bm25,
            vector_hits=total_vector,
            fused_hits=len(merged),
        )

    # ------------------------------------------------------------------
    # Corrective-RAG 回环（设计文档 §13.x）
    # ------------------------------------------------------------------

    @staticmethod
    def _should_run_corrective(judged: dict) -> bool:
        """判断是否触发 Corrective-RAG 回环。

        Args:
            judged: verify_with_fallback 返回的结果 dict。

        Returns:
            True 表示需要触发二次检索：开关开启 + Judge 未跳过
            + 检出幻觉证据或遗漏规则（任一）。
        """
        if not settings.RULE_REVIEW_CORRECTIVE_ENABLED:
            return False
        if judged.get("judge_skipped"):
            return False
        return bool(judged.get("judge_hallucinated")) or bool(
            judged.get("judge_missing_rules")
        )

    @staticmethod
    def _build_corrective_query(
        rewritten_query: str,
        judged: dict,
        first_llm_output: LLMOutput | None,
    ) -> str:
        """构造二次检索 query。

        规则：
        1. 核心补充 = missing_rules 的 rule 文本（"漏了"的东西才是补检重点），
           去空白、截断 ≤50 字符、最多 3 条；
        2. 幻觉关键词仅当无 missing_rules 时使用（幻觉证据本身是错的，不重查；
           但幻觉意味着上下文缺支撑，用证据的 section 标题做关键词），
           index 越界时回退 reason 前 30 字符，最多 3 条；
        3. 与 rewritten_query 或已选补充互相包含的跳过；总长 ≤200；
        4. 无有效补充时原样返回 rewritten_query（仅靠扩大 top_k）。
        """
        parts: list[str] = []
        missing_rules = judged.get("judge_missing_rules") or []
        hallucinated = judged.get("judge_hallucinated") or []

        for mr in missing_rules[:3]:
            rule = (mr.get("rule", "") if isinstance(mr, dict) else "") or ""
            rule = re.sub(r"\s+", " ", rule).strip()[:50]
            if rule and not _is_query_subsumed(rule, rewritten_query, parts):
                parts.append(rule)

        if not parts:
            for h in hallucinated[:3]:
                kw = ""
                if isinstance(h, dict):
                    idx = h.get("index")
                    if (
                        isinstance(idx, int)
                        and first_llm_output is not None
                        and 0 <= idx < len(first_llm_output.evidence)
                    ):
                        kw = first_llm_output.evidence[idx].section
                    if not kw:
                        kw = str(h.get("reason", ""))[:30]
                kw = re.sub(r"\s+", " ", str(kw)).strip()
                if kw and not _is_query_subsumed(kw, rewritten_query, parts):
                    parts.append(kw)

        if not parts:
            return rewritten_query
        merged = f"{rewritten_query} {' '.join(parts)}"
        return merged[:200]

    async def _run_corrective_loop(
        self,
        *,
        rewritten_query: str,
        first_retrieve_result: RetrieveResult,
        first_llm_output: LLMOutput | None,
        first_judged: dict,
        top_k: int,
        progress: "Any | None" = None,
    ) -> CorrectiveLoopResult:
        """执行一轮 Corrective-RAG 回环（最大 1 轮，不循环）。

        时序：构造补充 query → 二次检索（top_k×2）→ 与首轮合并去重
        → 带 Judge 反馈重新生成 → 二次校验（针对 rewritten_query）。

        Args:
            rewritten_query: 改写后的用户问题（二次校验的校验基准）。
            first_retrieve_result: 首轮检索结果（合并去重的底）。
            first_llm_output: 首轮（Tool 后）LLM 输出，用于幻觉关键词反查。
            first_judged: 首轮 Judge 结果 dict（含 judge_hallucinated/missing_rules）。
            top_k: 请求的 top_k，二次检索与合并取 min(top_k*2, 50)。
            progress: 可选回调 (label_text, stage)，用于收集 SSE 进度标签。

        Returns:
            CorrectiveLoopResult；final_result 为最终输出，
            第二轮结果一律为终判（可能降级回首轮）。
        """
        result = CorrectiveLoopResult(triggered=True)
        has_missing = bool(first_judged.get("judge_missing_rules"))
        has_hallucinated = bool(first_judged.get("judge_hallucinated"))
        result.trigger_reason = (
            "both"
            if (has_missing and has_hallucinated)
            else ("missing_rules" if has_missing else "hallucinated")
        )
        if progress:
            result.progress_labels.append(("补充检索中...", "re_retrieval"))
            progress("补充检索中...", "re_retrieval")

        # 1. 构造补充 query + 扩大 top_k（请求上限 50）
        result.corrective_query = self._build_corrective_query(
            rewritten_query, first_judged, first_llm_output
        )
        second_top_k = min(top_k * 2, 50)
        result.second_top_k = second_top_k

        # 2. 二次检索（复用 run_retrieve，内部 async 包装同步检索）
        retrieve_start = time.monotonic()
        try:
            second_retrieve = await self.run_retrieve(
                result.corrective_query, top_k=second_top_k
            )
            result.retrieval_rounds = 2
        except Exception as e:
            logger.warning("[pipeline] Corrective 二次检索异常: %s", e)
            result.retrieval_ms = (time.monotonic() - retrieve_start) * 1000
            result.final_result = first_judged
            result.merged_chunks = self._chunks_to_dict_list(
                first_retrieve_result.results
            )
            result.terminated_reason = "second_retrieve_error"
            return result
        result.retrieval_ms = (time.monotonic() - retrieve_start) * 1000

        # 3. 次轮无结果 → 保留首轮结果，不掩盖
        if second_retrieve.not_found and not second_retrieve.results:
            logger.info("[pipeline] Corrective 二次检索无结果，保留首轮结果")
            result.merged_retrieve_result = second_retrieve
            result.final_result = first_judged
            result.merged_chunks = self._chunks_to_dict_list(
                first_retrieve_result.results
            )
            result.terminated_reason = "second_retrieve_empty"
            return result

        # 4. 与首轮结果合并去重（复用 _merge_retrieve_results）
        merged = self._merge_retrieve_results(
            [first_retrieve_result, second_retrieve], top_k=second_top_k
        )
        result.merged_retrieve_result = merged
        result.merged_chunks = self._chunks_to_dict_list(merged.results)

        # 5. 带 Judge 反馈重新生成（generator 幂等，可安全二次调用）
        if progress:
            result.progress_labels.append(("重新推理中...", "re_generation"))
            progress("重新推理中...", "re_generation")
        feedback = {
            "hallucinated_evidence": first_judged.get("judge_hallucinated", []),
            "missing_rules": first_judged.get("judge_missing_rules", []),
        }
        gen_start = time.monotonic()
        try:
            second_llm = await self.generator.generate(
                query=result.corrective_query,
                context_chunks=result.merged_chunks,
                judge_feedback=feedback,
                system_prompt=DEFAULT_GENERATION_PROMPT,
            )
        except Exception as e:
            logger.warning("[pipeline] Corrective 重新生成异常: %s", e)
            second_llm = None
        result.generation_ms = (time.monotonic() - gen_start) * 1000
        result.generation_rounds = 2

        # 6. 终止矩阵：第二轮结果一律为最终输出，不再循环
        if second_llm is None:
            logger.warning("[pipeline] Corrective 重新生成失败，降级输出首轮结果")
            result.final_result = first_judged
            result.terminated_reason = "second_generate_none"
            return result

        if second_llm.not_found:
            logger.info("[pipeline] Corrective 第二轮 LLM 判定无相关规则")
            final = second_llm.model_dump()
            final["judge_skipped"] = True
            final["judge_skipped_reason"] = "not_found"
            result.second_llm_output = second_llm
            result.final_result = final
            result.terminated_reason = "second_not_found"
            return result

        # 1 轮预算内不重跑 Tool：清空 tool_calls，直接送 Judge
        if second_llm.tool_calls:
            logger.info(
                "[pipeline] Corrective 第二轮带 tool_calls，预算内不重跑 Tool，直接送 Judge"
            )
            second_llm.tool_calls = []

        # 7. 二次校验（传 rewritten_query，Judge 必须针对用户原问题）
        if progress:
            result.progress_labels.append(("二次校验中...", "re_judge"))
            progress("二次校验中...", "re_judge")
        judge_start = time.monotonic()
        try:
            if self.judge is not None:
                from src.rule_review.judge import verify_with_fallback

                second_judged = await verify_with_fallback(
                    self.judge, second_llm, rewritten_query, result.merged_chunks
                )
            else:
                second_judged = second_llm.model_dump()
                second_judged["judge_skipped"] = True
                second_judged["judge_skipped_reason"] = "no_judge_configured"
        except Exception as e:
            logger.warning("[pipeline] Corrective 二次校验异常: %s", e)
            second_judged = second_llm.model_dump()
            second_judged["judge_skipped"] = True
            second_judged["judge_skipped_reason"] = str(e)[:200]
        result.judge_ms = (time.monotonic() - judge_start) * 1000
        result.judge_rounds = 2
        result.second_judged = second_judged
        result.final_result = second_judged
        if second_judged.get("judge_skipped"):
            result.terminated_reason = "second_judge_skipped"
        return result

    # ------------------------------------------------------------------
    # SSE 格式化
    # ------------------------------------------------------------------

    @staticmethod
    def _sse_label(text: str, stage: str = "") -> str:
        """生成 SSE 进度标签消息。"""
        payload = {
            "type": "messageLabel",
            "answer": f"- <span>{text}</span>",
        }
        if stage:
            payload["stage"] = stage
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    @staticmethod
    def _sse_content(content: str, event: str = "content") -> str:
        """生成 SSE 内容消息。"""
        payload = {"answer": content, "type": event}
        return f"event: message\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    @staticmethod
    def _sse_done(query_id: str = "") -> str:
        """生成 SSE 完成消息。"""
        payload = {"done": True}
        if query_id:
            payload["query_id"] = query_id
        return f"event: done\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# 默认工厂
# ---------------------------------------------------------------------------

_default_pipeline: RuleReviewPipeline | None = None


def get_default_pipeline() -> RuleReviewPipeline:
    """获取默认规则审查编排器单例。"""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = RuleReviewPipeline()
    return _default_pipeline
