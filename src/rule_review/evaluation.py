"""
电力规则审查系统 - 评估体系（Phase 3）

提供规则审查系统的离线评估能力：
- 测试集管理（JSON 格式，含标准答案）
- 多维度评估指标（decision 准确率、evidence 召回率、幻觉率）
- 批量评估运行器
- 评估报告生成

测试集格式 (data/evaluation/test_cases.json):
[
  {
    "id": "tc-001",
    "question": "2025年3月15日冀北的日前现货出清电价800元/MWh是否符合价格上限？",
    "expected_decision": "不符合",
    "expected_keywords": ["760", "价格上限"],
    "expected_evidence_sources": ["省间电力现货交易规则"],
    "documents": ["省间电力现货交易规则.pdf"],
    "tags": ["价格上限", "冀北", "日前现货"],
    "difficulty": "easy"
  }
]
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.rule_review.llm_judge_metrics import RagasMetrics, compute_ragas_metrics

logger = logging.getLogger(__name__)

DEFAULT_TEST_CASES_PATH = "data/evaluation/test_cases.json"
DEFAULT_REPORTS_DIR = "data/evaluation/reports"


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


@dataclass
class TestCase:
    """单个测试用例。"""

    id: str
    question: str
    expected_decision: str = ""
    expected_keywords: list[str] = field(default_factory=list)
    expected_evidence_sources: list[str] = field(default_factory=list)
    documents: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    difficulty: str = "medium"  # easy | medium | hard
    # 期望命中的 chunk_id（内容哈希后可精确标注；当前检索指标用 expected_keywords 代理）
    expected_chunk_ids: list[str] = field(default_factory=list)
    # RAGAS 指标参考答案（Context Precision/Recall 的判定基准；缺省时两指标降级为 None）
    reference_answer: str = ""


@dataclass
class EvalMetrics:
    """单条评估指标。"""

    case_id: str
    decision_match: bool = False
    decision_actual: str = ""
    decision_expected: str = ""

    keyword_recall: float = 0.0  # 关键词召回率 0-1
    keywords_found: list[str] = field(default_factory=list)
    keywords_missed: list[str] = field(default_factory=list)

    evidence_source_recall: float = 0.0
    sources_found: list[str] = field(default_factory=list)
    sources_missed: list[str] = field(default_factory=list)

    # 检索层指标：recall@k 与 MRR（相关性代理：chunk 文本含任一期望关键词）
    recall_at_k: float = 0.0
    mrr: float = 0.0

    has_hallucination: bool = False  # 是否检测到幻觉
    hallucinated_text: list[str] = field(default_factory=list)
    hallucination_check_skipped: bool = False  # 无检索文本时跳过幻觉检测

    confidence: float = 0.0
    latency_ms: float = 0.0

    not_found: bool = False  # 是否判定为未找到
    judge_skipped: bool = False

    # RAGAS 风格 LLM-as-judge 指标（自研，见 llm_judge_metrics.py）
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    ragas_skipped: bool = False  # 整体跳过（如回答为空）
    ragas_skip_reason: str = ""
    ragas_details: dict = field(default_factory=dict)  # 各指标明细 + judge_latency_ms

    error: str = ""  # 执行中的异常


@dataclass
class EvalReport:
    """批量评估报告。"""

    timestamp: str = ""
    total_cases: int = 0
    success_cases: int = 0
    error_cases: int = 0

    # 决策准确率
    decision_accuracy: float = 0.0

    # 关键词召回（平均值）
    avg_keyword_recall: float = 0.0

    # 证据来源召回
    avg_source_recall: float = 0.0

    # 幻觉率
    hallucination_rate: float = 0.0

    # 检索层指标（平均值）
    avg_recall_at_k: float = 0.0
    avg_mrr: float = 0.0

    # 平均置信度
    avg_confidence: float = 0.0

    # 平均延迟
    avg_latency_ms: float = 0.0

    # RAGAS 指标均值（LLM-as-judge；全为 None 时返回 None，即未启用/全部降级）
    avg_faithfulness: float | None = None
    avg_answer_relevancy: float | None = None
    avg_context_precision: float | None = None
    avg_context_recall: float | None = None
    ragas_metrics_count: int = 0  # 至少算出一个 RAGAS 指标的用例数

    # 按难度分组
    by_difficulty: dict[str, dict] = field(default_factory=dict)

    # 按标签分组
    by_tag: dict[str, dict] = field(default_factory=dict)

    # 逐条详情
    details: list[EvalMetrics] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 测试集管理
# ---------------------------------------------------------------------------


class TestCaseManager:
    """测试集管理：加载、验证、统计。"""

    def __init__(self, path: str | None = None):
        self.path = Path(path or DEFAULT_TEST_CASES_PATH)
        self._cases: list[TestCase] = []

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    def load(self) -> list[TestCase]:
        """从 JSON 文件加载测试用例。"""
        if not self.path.exists():
            logger.warning("[Eval] 测试集文件不存在: %s", self.path)
            return []

        with open(self.path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        cases = []
        for item in raw:
            try:
                cases.append(TestCase(
                    id=item.get("id", ""),
                    question=item["question"],
                    expected_decision=item.get("expected_decision", ""),
                    expected_keywords=item.get("expected_keywords", []),
                    expected_evidence_sources=item.get("expected_evidence_sources", []),
                    documents=item.get("documents", []),
                    tags=item.get("tags", []),
                    difficulty=item.get("difficulty", "medium"),
                    expected_chunk_ids=item.get("expected_chunk_ids", []),
                    reference_answer=item.get("reference_answer", ""),
                ))
            except KeyError as e:
                logger.warning("[Eval] 跳过无效用例 %s: 缺少字段 %s", item.get("id", "?"), e)

        self._cases = cases
        logger.info("[Eval] 加载 %d 个测试用例", len(cases))
        return cases

    def save(self, cases: list[TestCase] | None = None) -> None:
        """保存测试用例到 JSON 文件。"""
        data = []
        for tc in (cases or self._cases):
            data.append({
                "id": tc.id,
                "question": tc.question,
                "expected_decision": tc.expected_decision,
                "expected_keywords": tc.expected_keywords,
                "expected_evidence_sources": tc.expected_evidence_sources,
                "documents": tc.documents,
                "tags": tc.tags,
                "difficulty": tc.difficulty,
                "expected_chunk_ids": tc.expected_chunk_ids,
                "reference_answer": tc.reference_answer,
            })

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("[Eval] 保存 %d 个测试用例", len(data))

    def add_case(self, case: TestCase) -> None:
        """添加单个测试用例。"""
        self._cases.append(case)
        self.save()

    @property
    def cases(self) -> list[TestCase]:
        if not self._cases:
            self.load()
        return self._cases

    def stats(self) -> dict:
        """测试集统计。"""
        cases = self.cases
        by_diff: dict[str, int] = {}
        by_tag: dict[str, int] = {}
        for tc in cases:
            by_diff[tc.difficulty] = by_diff.get(tc.difficulty, 0) + 1
            for tag in tc.tags:
                by_tag[tag] = by_tag.get(tag, 0) + 1

        return {
            "total_cases": len(cases),
            "by_difficulty": by_diff,
            "by_tag": by_tag,
            "has_expected_decision": sum(1 for tc in cases if tc.expected_decision),
            "has_expected_keywords": sum(1 for tc in cases if tc.expected_keywords),
        }


# ---------------------------------------------------------------------------
# 评估指标计算
# ---------------------------------------------------------------------------


def compute_decision_accuracy(predicted: str, expected: str) -> bool:
    """判断决策是否匹配。

    支持模糊匹配：忽略标点符号和前后空格。
    """
    if not expected:
        return True  # 无期望值时不算错

    p = predicted.strip().replace(" ", "")
    e = expected.strip().replace(" ", "")
    return p == e


def compute_keyword_recall(
    reason_text: str,
    evidence_texts: list[str],
    expected_keywords: list[str],
) -> tuple[float, list[str], list[str]]:
    """计算关键词召回率。

    在 reason + evidence 中搜索期望关键词。

    Returns:
        (recall, found_keywords, missed_keywords)
    """
    if not expected_keywords:
        return 1.0, [], []

    combined = reason_text + " " + " ".join(evidence_texts)
    found = [kw for kw in expected_keywords if kw in combined]
    missed = [kw for kw in expected_keywords if kw not in combined]
    recall = len(found) / len(expected_keywords) if expected_keywords else 1.0
    return recall, found, missed


def compute_source_recall(
    evidence_sources: list[str],
    expected_sources: list[str],
) -> tuple[float, list[str], list[str]]:
    """计算证据来源召回率。

    模糊匹配：来源包含期望来源名或期望来源名包含来源。

    Returns:
        (recall, found_sources, missed_sources)
    """
    if not expected_sources:
        return 1.0, [], []

    found = []
    for es in expected_sources:
        for s in evidence_sources:
            if es in s or s in es:
                found.append(es)
                break

    missed = [es for es in expected_sources if es not in found]
    recall = len(found) / len(expected_sources) if expected_sources else 1.0
    return recall, found, missed


def compute_recall_at_k(
    retrieved_chunks: list[dict],
    expected_keywords: list[str],
    k: int | None = None,
) -> float:
    """计算检索层 recall@k（单 query 场景为 0/1，报告层取平均即为召回率）。

    相关性代理：chunk 文本包含任一期望关键词即视为命中。
    当前 chunk_id 为随机生成无法预标注，故用关键词代理；
    内容哈希 chunk 后可改用 expected_chunk_ids 精确标注。

    Args:
        retrieved_chunks: [{"text": ..., ...}, ...] 检索结果（按相关性排序）。
        expected_keywords: 期望命中的关键词列表。
        k: 只考虑前 k 个结果；None 时取全部。

    Returns:
        1.0（前 k 个中至少一个命中）或 0.0。
    """
    if not expected_keywords or not retrieved_chunks:
        return 0.0

    top = retrieved_chunks if k is None else retrieved_chunks[:k]
    for chunk in top:
        if any(kw in chunk.get("text", "") for kw in expected_keywords):
            return 1.0
    return 0.0


def compute_mrr(
    retrieved_chunks: list[dict],
    expected_keywords: list[str],
    k: int | None = None,
) -> float:
    """计算检索层 MRR（Mean Reciprocal Rank）。

    取第一个命中 chunk 的倒数排名；全部未命中返回 0。

    Args:
        retrieved_chunks: [{"text": ..., ...}, ...] 检索结果（按相关性排序）。
        expected_keywords: 期望命中的关键词列表。
        k: 只考虑前 k 个结果；None 时取全部。

    Returns:
        mrr 值（0.0 ~ 1.0）。
    """
    if not expected_keywords:
        return 0.0

    top = retrieved_chunks if k is None else retrieved_chunks[:k]
    for rank, chunk in enumerate(top, start=1):
        if any(kw in chunk.get("text", "") for kw in expected_keywords):
            return 1.0 / rank
    return 0.0


def detect_hallucination(
    evidence_texts: list[str],
    retrieved_texts: list[str],
    threshold: float = 0.3,
) -> tuple[bool, list[str]]:
    """检测幻觉：evidence 中的文本在检索结果中找不到匹配。

    使用最长公共子串近似匹配。

    Returns:
        (has_hallucination, hallucinated_texts)
    """
    hallucinated = []
    for ev_text in evidence_texts:
        if not ev_text:
            continue
        best_ratio = 0.0
        for rt in retrieved_texts:
            if not rt:
                continue
            lcs = _lcs_length(ev_text, rt)
            ratio = lcs / max(len(ev_text), 1)
            if ratio > best_ratio:
                best_ratio = ratio

        if best_ratio < threshold:
            hallucinated.append(ev_text[:200])

    return len(hallucinated) > 0, hallucinated


def _lcs_length(s1: str, s2: str) -> int:
    """最长公共子串长度（动态规划，空间优化）。"""
    if not s1 or not s2:
        return 0
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    m, n = len(s1), len(s2)
    prev = [0] * (m + 1)
    max_len = 0
    for j in range(1, n + 1):
        curr = [0] * (m + 1)
        for i in range(1, m + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[i] = prev[i - 1] + 1
                if curr[i] > max_len:
                    max_len = curr[i]
        prev = curr
    return max_len


# ---------------------------------------------------------------------------
# 批量评估运行器
# ---------------------------------------------------------------------------


class EvalRunner:
    """批量评估运行器。

    对测试集中每条用例执行 pipeline，收集指标并生成报告。
    支持 RAGAS 风格 LLM-as-judge 指标（llm_judge_metrics.py）：
    - judge_model 可注入 mock/指定模型；None 时惰性使用默认评测模型
    - LLM 不可用/输入缺失时指标置 None 并标记跳过，绝不阻断评估主流程
    """

    def __init__(
        self,
        pipeline=None,
        judge_model: Any | None = None,
        ragas_enabled: bool = True,
    ):
        """
        Args:
            pipeline: RuleReviewPipeline 实例，None 时需要后续注入。
            judge_model: RAGAS 指标使用的 LLM 模型实例；None 时使用默认评测模型。
            ragas_enabled: 是否启用 LLM-as-judge 指标（关闭后行为与旧版一致）。
        """
        self._pipeline = pipeline
        self._judge_model = judge_model
        self._ragas_enabled = ragas_enabled

    def set_pipeline(self, pipeline) -> None:
        self._pipeline = pipeline

    async def run(
        self,
        test_cases: list[TestCase],
        top_k: int = 10,
    ) -> EvalReport:
        """批量运行评估。

        Args:
            test_cases: 测试用例列表。
            top_k: 送入 LLM 的 chunk 数。

        Returns:
            EvalReport 包含所有指标和逐条详情。
        """
        from datetime import datetime, timezone

        if self._pipeline is None:
            raise RuntimeError("pipeline 未设置，请先调用 set_pipeline()")

        from src.rule_review.schemas import RuleReviewRequest

        metrics_list: list[EvalMetrics] = []
        success_count = 0
        error_count = 0

        for tc in test_cases:
            start = time.monotonic()
            try:
                req = RuleReviewRequest(
                    question=tc.question,
                    stream=False,
                    top_k=top_k,
                )
                result = await self._pipeline.execute(req)
                elapsed = (time.monotonic() - start) * 1000

                # 提取实际输出
                actual = result.get("result", {})
                stages = result.get("stages", [])

                # 计算各项指标
                decision = actual.get("decision", "")
                evidence = actual.get("evidence", [])
                evidence_texts = [e.get("text", "") for e in evidence]
                evidence_sources = [e.get("source", "") for e in evidence]
                reason = actual.get("reason", "")

                # 检索到的文本（用于幻觉检测与 recall@k/MRR 计算）
                retrieved_chunks = []
                for s in stages:
                    if s.get("stage") == "retrieval":
                        retrieved_chunks = s.get("retrieved_chunks", []) or []
                        break
                retrieved_texts = [c.get("text", "") for c in retrieved_chunks]

                kw_recall, kw_found, kw_missed = compute_keyword_recall(
                    reason, evidence_texts, tc.expected_keywords
                )
                src_recall, src_found, src_missed = compute_source_recall(
                    evidence_sources, tc.expected_evidence_sources
                )

                # 检索结果为空（旧版 stages 无 retrieved_chunks 或检索失败）时跳过
                # 幻觉检测——否则空集合会把所有 evidence 判为幻觉，指标失真
                hallucination_check_skipped = not retrieved_texts
                if hallucination_check_skipped:
                    has_hallu, hallu_texts = False, []
                else:
                    has_hallu, hallu_texts = detect_hallucination(
                        evidence_texts, retrieved_texts
                    )

                recall_at_k = compute_recall_at_k(
                    retrieved_chunks, tc.expected_keywords, k=top_k
                )
                mrr = compute_mrr(retrieved_chunks, tc.expected_keywords, k=top_k)

                # RAGAS 风格 LLM-as-judge 指标（完整文本优先，缺失时回退截断版）
                ragas = RagasMetrics(skipped=True, skip_reason="RAGAS 指标已关闭")
                if self._ragas_enabled:
                    full_chunks = []
                    for s in stages:
                        if s.get("stage") == "retrieval":
                            full_chunks = (
                                s.get("retrieved_chunks_full")
                                or s.get("retrieved_chunks")
                                or []
                            )
                            break
                    answer_text = reason + " " + " ".join(evidence_texts)
                    try:
                        ragas = await compute_ragas_metrics(
                            question=tc.question,
                            answer_text=answer_text,
                            context_chunks=full_chunks,
                            reference_answer=tc.reference_answer,
                            model=self._judge_model,
                        )
                    except Exception as e:
                        logger.error("[Eval] %s RAGAS 指标计算异常: %s", tc.id, e)
                        ragas = RagasMetrics(
                            skipped=True,
                            skip_reason=f"RAGAS 指标计算异常: {e}",
                            errors=[str(e)],
                        )

                metrics = EvalMetrics(
                    case_id=tc.id,
                    decision_match=compute_decision_accuracy(
                        decision, tc.expected_decision
                    ),
                    decision_actual=decision,
                    decision_expected=tc.expected_decision,
                    keyword_recall=round(kw_recall, 4),
                    keywords_found=kw_found,
                    keywords_missed=kw_missed,
                    evidence_source_recall=round(src_recall, 4),
                    sources_found=src_found,
                    sources_missed=src_missed,
                    recall_at_k=recall_at_k,
                    mrr=round(mrr, 4),
                    has_hallucination=has_hallu,
                    hallucinated_text=hallu_texts,
                    hallucination_check_skipped=hallucination_check_skipped,
                    confidence=actual.get("confidence", 0),
                    latency_ms=round(elapsed, 2),
                    not_found=actual.get("not_found", False),
                    judge_skipped=actual.get("judge_skipped", False),
                    faithfulness=ragas.faithfulness,
                    answer_relevancy=ragas.answer_relevancy,
                    context_precision=ragas.context_precision,
                    context_recall=ragas.context_recall,
                    ragas_skipped=ragas.skipped,
                    ragas_skip_reason=ragas.skip_reason,
                    ragas_details=ragas.details,
                )
                metrics_list.append(metrics)
                success_count += 1
                logger.info(
                    "[Eval] %s: decision=%s (expected=%s), kw_recall=%.2f, %dms",
                    tc.id, decision, tc.expected_decision, kw_recall, int(elapsed),
                )

            except Exception as e:
                logger.error("[Eval] %s 执行失败: %s", tc.id, e)
                metrics_list.append(EvalMetrics(
                    case_id=tc.id,
                    error=str(e),
                ))
                error_count += 1

        # 汇总报告
        report = self._build_report(
            metrics_list, test_cases, success_count, error_count
        )
        return report

    @staticmethod
    def _avg_optional(values: list[float | None]) -> float | None:
        """对可选值求平均（None 不计入；全为 None 返回 None）。"""
        present = [v for v in values if v is not None]
        if not present:
            return None
        return round(sum(present) / len(present), 4)

    def _build_report(
        self,
        metrics: list[EvalMetrics],
        cases: list[TestCase],
        success: int,
        errors: int,
    ) -> EvalReport:
        """从逐条指标构建汇总报告。"""
        from datetime import datetime, timezone

        valid = [m for m in metrics if not m.error]
        if not valid:
            return EvalReport(
                timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                total_cases=len(cases),
                success_cases=success,
                error_cases=errors,
                details=metrics,
            )

        n = len(valid)

        # 决策准确率
        dec_acc = sum(1 for m in valid if m.decision_match) / n

        # 平均关键词召回
        avg_kw = sum(m.keyword_recall for m in valid) / n

        # 平均来源召回
        avg_src = sum(m.evidence_source_recall for m in valid) / n

        # 幻觉率
        hallu_count = sum(1 for m in valid if m.has_hallucination)
        hallu_rate = hallu_count / n

        # 检索层指标（平均值）
        avg_recall = sum(m.recall_at_k for m in valid) / n
        avg_mrr = sum(m.mrr for m in valid) / n

        # 平均置信度
        avg_conf = sum(m.confidence for m in valid) / n

        # 平均延迟
        avg_lat = sum(m.latency_ms for m in valid) / n

        # RAGAS 指标均值（None 不计入；全部为 None 时返回 None）
        ragas_faith = self._avg_optional([m.faithfulness for m in valid])
        ragas_rel = self._avg_optional([m.answer_relevancy for m in valid])
        ragas_prec = self._avg_optional([m.context_precision for m in valid])
        ragas_recall = self._avg_optional([m.context_recall for m in valid])
        ragas_count = sum(
            1 for m in valid
            if m.faithfulness is not None or m.answer_relevancy is not None
            or m.context_precision is not None or m.context_recall is not None
        )

        # 按难度分组
        by_diff: dict[str, dict] = {}
        case_map = {c.id: c for c in cases}
        for m in valid:
            tc = case_map.get(m.case_id)
            diff = tc.difficulty if tc else "unknown"
            if diff not in by_diff:
                by_diff[diff] = {"count": 0, "decision_accuracy": 0.0, "matches": 0}
            by_diff[diff]["count"] += 1
            if m.decision_match:
                by_diff[diff]["matches"] += 1
        for d, v in by_diff.items():
            v["decision_accuracy"] = round(v["matches"] / v["count"], 4) if v["count"] > 0 else 0.0

        # 按标签分组
        by_tag: dict[str, dict] = {}
        for m in valid:
            tc = case_map.get(m.case_id)
            if tc:
                for tag in tc.tags:
                    if tag not in by_tag:
                        by_tag[tag] = {"count": 0, "decision_accuracy": 0.0, "matches": 0}
                    by_tag[tag]["count"] += 1
                    if m.decision_match:
                        by_tag[tag]["matches"] += 1
        for t, v in by_tag.items():
            v["decision_accuracy"] = round(v["matches"] / v["count"], 4) if v["count"] > 0 else 0.0

        return EvalReport(
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            total_cases=len(cases),
            success_cases=success,
            error_cases=errors,
            decision_accuracy=round(dec_acc, 4),
            avg_keyword_recall=round(avg_kw, 4),
            avg_source_recall=round(avg_src, 4),
            hallucination_rate=round(hallu_rate, 4),
            avg_recall_at_k=round(avg_recall, 4),
            avg_mrr=round(avg_mrr, 4),
            avg_confidence=round(avg_conf, 4),
            avg_latency_ms=round(avg_lat, 2),
            avg_faithfulness=ragas_faith,
            avg_answer_relevancy=ragas_rel,
            avg_context_precision=ragas_prec,
            avg_context_recall=ragas_recall,
            ragas_metrics_count=ragas_count,
            by_difficulty=by_diff,
            by_tag=by_tag,
            details=metrics,
        )

    # ------------------------------------------------------------------
    # 报告持久化
    # ------------------------------------------------------------------

    @staticmethod
    def save_report(report: EvalReport, path: str | None = None) -> str:
        """保存评估报告为 JSON 文件。"""
        from dataclasses import asdict

        if path is None:
            ts = report.timestamp.replace(":", "-")[:19]
            path = f"{DEFAULT_REPORTS_DIR}/eval_{ts}.json"

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        with open(p, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, ensure_ascii=False, indent=2)

        logger.info("[Eval] 报告已保存: %s", p)
        return str(p)

    @staticmethod
    def load_report(path: str) -> EvalReport:
        """加载评估报告。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        details = [EvalMetrics(**d) for d in data.get("details", [])]
        return EvalReport(
            timestamp=data.get("timestamp", ""),
            total_cases=data.get("total_cases", 0),
            success_cases=data.get("success_cases", 0),
            error_cases=data.get("error_cases", 0),
            decision_accuracy=data.get("decision_accuracy", 0),
            avg_keyword_recall=data.get("avg_keyword_recall", 0),
            avg_source_recall=data.get("avg_source_recall", 0),
            hallucination_rate=data.get("hallucination_rate", 0),
            avg_recall_at_k=data.get("avg_recall_at_k", 0),
            avg_mrr=data.get("avg_mrr", 0),
            avg_confidence=data.get("avg_confidence", 0),
            avg_latency_ms=data.get("avg_latency_ms", 0),
            avg_faithfulness=data.get("avg_faithfulness"),
            avg_answer_relevancy=data.get("avg_answer_relevancy"),
            avg_context_precision=data.get("avg_context_precision"),
            avg_context_recall=data.get("avg_context_recall"),
            ragas_metrics_count=data.get("ragas_metrics_count", 0),
            by_difficulty=data.get("by_difficulty", {}),
            by_tag=data.get("by_tag", {}),
            details=details,
        )


# ---------------------------------------------------------------------------
# 默认工厂
# ---------------------------------------------------------------------------


_default_eval_runner: EvalRunner | None = None


def get_default_eval_runner() -> EvalRunner:
    global _default_eval_runner
    if _default_eval_runner is None:
        _default_eval_runner = EvalRunner()
    return _default_eval_runner


# ---------------------------------------------------------------------------
# CLI 入口：一键离线评估
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """离线评估 CLI。

    用法:
        python -m src.rule_review.evaluation [--cases PATH] [--top-k N]
                                              [--report PATH] [--no-ragas]

    RAGAS 指标默认开启；--no-ragas 关闭 LLM-as-judge 指标（行为与旧版一致）。
    LLM 不可用时指标自动降级为 None 并统计跳过，不影响报告生成。
    """
    import argparse
    import asyncio
    import sys

    parser = argparse.ArgumentParser(
        description="规则审查系统离线评估（含 RAGAS 风格 LLM-as-judge 指标）"
    )
    parser.add_argument("--cases", default=DEFAULT_TEST_CASES_PATH, help="测试集 JSON 路径")
    parser.add_argument("--top-k", type=int, default=10, help="送入 LLM 的 chunk 数")
    parser.add_argument("--report", default=None, help="报告输出路径（默认 data/evaluation/reports/）")
    parser.add_argument("--no-ragas", action="store_true", help="关闭 RAGAS LLM-as-judge 指标")
    args = parser.parse_args(argv)

    from src.rule_review.pipeline import get_default_pipeline

    manager = TestCaseManager(args.cases)
    cases = manager.load()
    if not cases:
        print("[Eval] 测试集为空或加载失败: %s", args.cases)
        return 1

    runner = EvalRunner(
        pipeline=get_default_pipeline(),
        ragas_enabled=not args.no_ragas,
    )
    report = asyncio.run(runner.run(cases, top_k=args.top_k))

    path = EvalRunner.save_report(report, args.report)
    print(f"\n[Eval] 报告已保存: {path}")
    print(f"[Eval] 总用例 {report.total_cases} | 成功 {report.success_cases} | 失败 {report.error_cases}")
    print(f"[Eval] 决策准确率 {report.decision_accuracy:.2%} | 关键词召回 {report.avg_keyword_recall:.2%}")
    print(f"[Eval] 来源召回 {report.avg_source_recall:.2%} | 幻觉率 {report.hallucination_rate:.2%}")
    print(f"[Eval] recall@k {report.avg_recall_at_k:.4f} | MRR {report.avg_mrr:.4f} | 延迟 {report.avg_latency_ms:.0f}ms")
    if args.no_ragas:
        print("[Eval] RAGAS 指标已关闭（--no-ragas）")
    else:
        print("[Eval] RAGAS 指标（LLM-as-judge，None=降级跳过）:")
        print(f"  faithfulness      = {report.avg_faithfulness}")
        print(f"  answer_relevancy  = {report.avg_answer_relevancy}")
        print(f"  context_precision = {report.avg_context_precision}")
        print(f"  context_recall    = {report.avg_context_recall}")
        print(f"  (有效用例 {report.ragas_metrics_count}/{report.success_cases})")
        skipped = [m for m in report.details if m.ragas_skipped and not m.error]
        if skipped:
            print(f"[Eval] 注意: {len(skipped)} 条用例 RAGAS 指标整体跳过（如回答为空）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

