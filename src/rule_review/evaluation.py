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

    has_hallucination: bool = False  # 是否检测到幻觉
    hallucinated_text: list[str] = field(default_factory=list)

    confidence: float = 0.0
    latency_ms: float = 0.0

    not_found: bool = False  # 是否判定为未找到
    judge_skipped: bool = False

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

    # 平均置信度
    avg_confidence: float = 0.0

    # 平均延迟
    avg_latency_ms: float = 0.0

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
    """

    def __init__(self, pipeline=None):
        """
        Args:
            pipeline: RuleReviewPipeline 实例，None 时需要后续注入。
        """
        self._pipeline = pipeline

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

                # 检索到的文本（用于幻觉检测）
                retrieved_texts = []
                for s in stages:
                    if s.get("stage") == "retrieval":
                        # stages 中没有 chunks 详情，从 pipeline 输出推断
                        pass

                kw_recall, kw_found, kw_missed = compute_keyword_recall(
                    reason, evidence_texts, tc.expected_keywords
                )
                src_recall, src_found, src_missed = compute_source_recall(
                    evidence_sources, tc.expected_evidence_sources
                )
                has_hallu, hallu_texts = detect_hallucination(
                    evidence_texts, retrieved_texts
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
                    has_hallucination=has_hallu,
                    hallucinated_text=hallu_texts,
                    confidence=actual.get("confidence", 0),
                    latency_ms=round(elapsed, 2),
                    not_found=actual.get("not_found", False),
                    judge_skipped=actual.get("judge_skipped", False),
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

        # 平均置信度
        avg_conf = sum(m.confidence for m in valid) / n

        # 平均延迟
        avg_lat = sum(m.latency_ms for m in valid) / n

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
            avg_confidence=round(avg_conf, 4),
            avg_latency_ms=round(avg_lat, 2),
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
            avg_confidence=data.get("avg_confidence", 0),
            avg_latency_ms=data.get("avg_latency_ms", 0),
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
