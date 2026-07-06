"""
电力规则审查系统 - 审计追溯模块

按设计文档 §4.7 实现：
- AuditStore: 审计日志存储（JSON 文件存储）
- build_source_traceability(): 答案→原文溯源链构建
- 最长公共子串匹配 + 模糊匹配 + LLM 提取标记
"""

from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.rule_review.schemas import (
    AuditRecord,
    JudgeAudit,
    LLMGenerationAudit,
    RetrievalAudit,
    SourceTrace,
    ToolCallLog,
)

logger = logging.getLogger(__name__)

# 默认存储目录
_DEFAULT_AUDIT_DIR = "data/audit_logs"


# ---------------------------------------------------------------------------
# 最长公共子串
# ---------------------------------------------------------------------------


def _longest_common_substring(s1: str, s2: str) -> int:
    """返回两个字符串的最长公共子串长度。

    使用动态规划，O(m*n) 时间，O(min(m,n)) 空间。

    Args:
        s1: 第一个字符串。
        s2: 第二个字符串。

    Returns:
        最长公共连续子串的字符数。
    """
    if not s1 or not s2:
        return 0

    # 确保 s1 是较短的字符串以节省空间
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
            else:
                curr[i] = 0
        prev = curr

    return max_len


# ---------------------------------------------------------------------------
# 溯源链构建
# ---------------------------------------------------------------------------


def build_source_traceability(
    final_result: dict,
    retrieved_chunks: list[dict],
) -> list[SourceTrace]:
    """为每条 evidence 构建溯源链。

    方法:
    1. 从最终结果的 evidence 列表中逐条取 text
    2. 在 retrieved_chunks 中找最长公共子串匹配
    3. 记录匹配到的 chunk 来源信息
    4. 无法精确匹配的标记为 "llm_extracted"

    这样合规人员可以看到:
    - "这句话来自《省间电力现货交易规则》第12条 第23页"

    Args:
        final_result: 最终审查结果（包含 evidence 列表）。
        retrieved_chunks: 检索到的 chunks，每个 dict 包含：
            {"text": "...", "source": "文档名", "section": "...", "page": N, "chunk_id": "..."}

    Returns:
        溯源信息列表，与 evidence 数组一一对应。
    """
    traces: list[SourceTrace] = []

    evidence_list = final_result.get("evidence", [])
    if not evidence_list:
        return traces

    for i, evidence in enumerate(evidence_list):
        evidence_text = evidence.get("text", "")
        if not evidence_text:
            traces.append(SourceTrace(
                result_field=f"evidence[{i}]",
                result_excerpt="",
                match_type="llm_extracted",
            ))
            continue

        best_match: dict | None = None
        best_score = 0

        for chunk in retrieved_chunks:
            chunk_text = chunk.get("text", "")
            if not chunk_text:
                continue

            lcs_len = _longest_common_substring(evidence_text, chunk_text)
            if lcs_len > best_score:
                best_score = lcs_len
                # 阈值: 至少 30% 匹配才算可追溯
                match_ratio = lcs_len / max(len(evidence_text), 1)
                if match_ratio > 0.3:
                    best_match = chunk

        if best_match and best_score > 0:
            match_ratio = best_score / max(len(evidence_text), 1)
            match_type = "exact" if match_ratio > 0.8 else "fuzzy"

            traces.append(SourceTrace(
                result_field=f"evidence[{i}]",
                result_excerpt=evidence_text[:200],
                source_doc=best_match.get("source", best_match.get("doc_name", "")),
                source_section=best_match.get("section", ""),
                source_page=best_match.get("page", 0),
                source_text=best_match.get("text", "")[:300],
                source_chunk_id=best_match.get("chunk_id", ""),
                match_type=match_type,
            ))
        else:
            traces.append(SourceTrace(
                result_field=f"evidence[{i}]",
                result_excerpt=evidence_text[:200],
                source_doc="",
                source_section="",
                source_page=0,
                source_text="",
                source_chunk_id="",
                match_type="llm_extracted",
            ))

    return traces


# ---------------------------------------------------------------------------
# 审计存储
# ---------------------------------------------------------------------------


class AuditStore:
    """审计日志存储。

    Phase 1: JSON 文件存储（data/audit_logs/{date}/{query_id}.json）
    Phase 2: 数据库存储（PostgreSQL / MongoDB）

    每个审查请求自动生成一条审计记录，包含完整的溯源链。
    """

    def __init__(self, storage_dir: str | None = None):
        """
        Args:
            storage_dir: 审计日志存储目录，None 时使用默认路径。
        """
        self.storage_dir = storage_dir or _DEFAULT_AUDIT_DIR

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def save(self, record: AuditRecord) -> str:
        """保存审计记录到 JSON 文件。

        Args:
            record: 审计记录。

        Returns:
            保存的文件路径。
        """
        # 使用记录中的时间戳或当前时间
        ts = record.timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        date_str = ts[:10]  # YYYY-MM-DD

        dir_path = Path(self.storage_dir) / date_str
        dir_path.mkdir(parents=True, exist_ok=True)

        file_path = dir_path / f"{record.query_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(record.model_dump(), f, ensure_ascii=False, indent=2)

        logger.info("[AuditStore] 审计记录已保存: %s", file_path)
        return str(file_path)

    def load(self, query_id: str, date: str | None = None) -> AuditRecord | None:
        """加载审计记录。

        Args:
            query_id: 查询 ID。
            date: 日期字符串 YYYY-MM-DD，None 时搜索所有日期目录。

        Returns:
            AuditRecord 或 None（未找到时）。
        """
        if date:
            file_path = Path(self.storage_dir) / date / f"{query_id}.json"
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return AuditRecord(**data)
            return None

        # 搜索所有日期目录
        base = Path(self.storage_dir)
        if not base.exists():
            return None

        for date_dir in sorted(base.iterdir(), reverse=True):
            if not date_dir.is_dir():
                continue
            file_path = date_dir / f"{query_id}.json"
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return AuditRecord(**data)

        return None

    def list_by_date(self, date: str) -> list[str]:
        """列出某天的所有审计记录 query_id。

        Args:
            date: 日期字符串 YYYY-MM-DD。

        Returns:
            query_id 列表。
        """
        dir_path = Path(self.storage_dir) / date
        if not dir_path.exists():
            return []

        return sorted([
            f.stem for f in dir_path.glob("*.json") if f.is_file()
        ])

    def sample_for_review(
        self,
        date: str,
        count: int = 10,
    ) -> list[AuditRecord]:
        """抽样用于质检——合规部门每日随机抽样核查。

        Args:
            date: 日期字符串 YYYY-MM-DD。
            count: 抽样数量。

        Returns:
            随机抽取的审计记录列表。
        """
        query_ids = self.list_by_date(date)
        if not query_ids:
            return []

        sample_size = min(count, len(query_ids))
        sampled_ids = random.sample(query_ids, sample_size)

        records: list[AuditRecord] = []
        for qid in sampled_ids:
            record = self.load(qid, date=date)
            if record is not None:
                records.append(record)

        return records

    def get_stats(
        self,
        start_date: str,
        end_date: str,
    ) -> dict:
        """获取审计统计信息。

        Args:
            start_date: 起始日期 YYYY-MM-DD。
            end_date: 结束日期 YYYY-MM-DD。

        Returns:
            统计数据字典。
        """
        base = Path(self.storage_dir)
        if not base.exists():
            return {
                "total_reviews": 0,
                "date_range": {"start": start_date, "end": end_date},
                "hallucination_rate": 0.0,
                "judge_skip_rate": 0.0,
                "not_found_rate": 0.0,
                "avg_confidence": 0.0,
                "avg_latency_ms": 0.0,
            }

        total = 0
        hallucinated_total = 0
        judge_skipped = 0
        not_found_count = 0
        confidences: list[float] = []
        latencies: list[float] = []

        for date_dir in base.iterdir():
            if not date_dir.is_dir():
                continue
            dir_date = date_dir.name
            if dir_date < start_date or dir_date > end_date:
                continue

            for f in date_dir.glob("*.json"):
                try:
                    with open(f, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                except (json.JSONDecodeError, OSError):
                    continue

                total += 1

                # 幻觉统计
                judge = data.get("judge_verification") or {}
                if judge.get("hallucinated_count", 0) > 0:
                    hallucinated_total += 1
                if judge.get("skipped", False):
                    judge_skipped += 1

                # not_found 统计
                llm_gen = data.get("llm_generation") or {}
                if llm_gen.get("not_found", False):
                    not_found_count += 1

                # 置信度
                final = data.get("final_result") or {}
                conf = final.get("confidence", 0)
                if isinstance(conf, (int, float)):
                    confidences.append(float(conf))

                # 总延迟
                retrieval = data.get("retrieval") or {}
                r_lat = retrieval.get("retrieval_latency_ms", 0)
                l_lat = llm_gen.get("latency_ms", 0)
                j_lat = judge.get("latency_ms", 0)
                latencies.append(r_lat + l_lat + j_lat)

        return {
            "total_reviews": total,
            "date_range": {"start": start_date, "end": end_date},
            "hallucination_rate": (
                round(hallucinated_total / total, 4) if total > 0 else 0.0
            ),
            "judge_skip_rate": (
                round(judge_skipped / total, 4) if total > 0 else 0.0
            ),
            "not_found_rate": (
                round(not_found_count / total, 4) if total > 0 else 0.0
            ),
            "avg_confidence": (
                round(sum(confidences) / len(confidences), 4)
                if confidences else 0.0
            ),
            "avg_latency_ms": (
                round(sum(latencies) / len(latencies), 2)
                if latencies else 0.0
            ),
        }

    def delete(self, query_id: str, date: str) -> bool:
        """删除指定审计记录。

        Args:
            query_id: 查询 ID。
            date: 日期字符串 YYYY-MM-DD。

        Returns:
            是否删除成功。
        """
        file_path = Path(self.storage_dir) / date / f"{query_id}.json"
        if file_path.exists():
            file_path.unlink()
            logger.info("[AuditStore] 审计记录已删除: %s", file_path)
            return True
        return False

    def date_range(self) -> tuple[str, str]:
        """获取审计记录的日期范围。

        Returns:
            (最早日期, 最晚日期)，无记录时返回空字符串对。
        """
        base = Path(self.storage_dir)
        if not base.exists():
            return ("", "")

        dates = sorted([
            d.name for d in base.iterdir()
            if d.is_dir() and len(d.name) == 10  # YYYY-MM-DD
        ])
        if not dates:
            return ("", "")

        return (dates[0], dates[-1])


# ---------------------------------------------------------------------------
# 默认工厂
# ---------------------------------------------------------------------------


_default_audit_store: AuditStore | None = None


def get_default_audit_store() -> AuditStore:
    """获取默认审计存储单例。"""
    global _default_audit_store
    if _default_audit_store is None:
        _default_audit_store = AuditStore()
    return _default_audit_store
