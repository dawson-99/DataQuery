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

from src.rule_review.audit import AuditStore, build_source_traceability
from src.rule_review.document_store import DocumentStore
from src.rule_review.generator import RuleReviewGenerator, parse_llm_output
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
    ) -> None:
        """
        Args:
            rewriter: 问题改写器，None 时自动创建。
            document_store: 文档存储实例，None 时自动创建。
            retriever: 混合检索器，None 时自动创建（依赖 document_store）。
            generator: LLM 推理器，None 时自动创建。
            judge: Judge 校验器，None 时跳过校验阶段。
            audit_store: 审计存储，None 时不记录审计日志。
        """
        self.rewriter = rewriter or QueryRewriter()
        self.doc_store = document_store or DocumentStore()
        self.retriever = retriever or HybridRetriever(document_store=self.doc_store)
        self.generator = generator or RuleReviewGenerator()
        self.judge = judge  # None → 跳过 Judge 阶段
        self.audit_store = audit_store  # None → 不记录审计

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
        logger.info("[pipeline] 完成，耗时 %.2fs, query_id=%s", elapsed, query_id)

        # ---- 构建并保存审计记录 ----
        if self.audit_store is not None:
            try:
                # 构建溯源链
                final_dict = final_output if isinstance(final_output, dict) else final_output.model_dump()
                source_traces = build_source_traceability(final_dict, context_chunks)

                audit_record = AuditRecord(
                    query_id=query_id,
                    timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    original_query=request.question,
                    rewritten_query=rewritten_query,
                    retrieval=RetrievalAudit(
                        bm25_k=retrieve_result.bm25_hits,
                        vector_k=retrieve_result.vector_hits,
                        final_k=len(retrieve_result.results),
                        search_expanded=retrieve_result.search_expanded,
                        retrieval_latency_ms=round(retrieval_latency_ms, 2),
                    ),
                    llm_generation=LLMGenerationAudit(
                        not_found=llm_output.not_found,
                        latency_ms=round(generation_latency_ms, 2),
                    ),
                    judge_verification=judge_audit,
                    final_result=final_dict,
                    source_traceability=source_traces,
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

        stages_log.append({
            "stage": "retrieval",
            "not_found": retrieve_result.not_found,
            "result_count": len(retrieve_result.results),
            "search_expanded": retrieve_result.search_expanded,
        })

        # 空检索
        if retrieve_result.not_found and not retrieve_result.results:
            return {
                "query_id": query_id,
                "result": NotFoundResponse().model_dump(),
                "stages": stages_log,
            }

        # 阶段 5: LLM 生成
        context_chunks = self._chunks_to_dict_list(retrieve_result.results)
        llm_output = await self.generator.generate(
            query=rewritten_query,
            context_chunks=context_chunks,
        )

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
        final_result = final_llm_output.model_dump()
        if self.judge is not None:
            from src.rule_review.judge import verify_with_fallback

            judged = await verify_with_fallback(
                self.judge, final_llm_output, rewritten_query, context_chunks
            )
            final_result = judged
            stages_log.append({
                "stage": "judge",
                "skipped": judged.get("judge_skipped", False),
                "verified": judged.get("judge_verified", False),
            })

        # ---- 构建并保存审计记录 ----
        if self.audit_store is not None:
            try:
                context_chunks = self._chunks_to_dict_list(retrieve_result.results)
                source_traces = build_source_traceability(final_result, context_chunks)

                audit_record = AuditRecord(
                    query_id=query_id,
                    timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    original_query=request.question,
                    rewritten_query=rewritten_query,
                    retrieval=RetrievalAudit(
                        bm25_k=retrieve_result.bm25_hits,
                        vector_k=retrieve_result.vector_hits,
                        final_k=len(retrieve_result.results),
                        search_expanded=retrieve_result.search_expanded,
                    ),
                    llm_generation=LLMGenerationAudit(
                        not_found=llm_output.not_found,
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
                    final_result=final_result,
                    source_traceability=source_traces,
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
        """执行阶段 3-4：检索（含空检索兜底）。"""
        return await self.retriever.retrieve_with_fallback(query, top_k=top_k)

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
