"""
规则审查系统 - 审计追溯模块单元测试

覆盖：
- src/rule_review/audit.py：_longest_common_substring、build_source_traceability、AuditStore CRUD
- src/rule_review/schemas.py：AuditRecord、RetrievalAudit、LLMGenerationAudit、JudgeAudit、SourceTrace
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.rule_review.audit import (
    AuditStore,
    _longest_common_substring,
    build_source_traceability,
    get_default_audit_store,
)
from src.rule_review.schemas import (
    AuditRecord,
    JudgeAudit,
    LLMGenerationAudit,
    RetrievalAudit,
    SourceTrace,
    ToolCallLog,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_audit_dir():
    """临时审计目录。"""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def audit_store(temp_audit_dir):
    """注入临时目录的 AuditStore。"""
    return AuditStore(storage_dir=temp_audit_dir)


@pytest.fixture
def sample_chunks():
    """模拟检索返回的 chunks。"""
    return [
        {
            "text": "省间日前现货出清电价上限为760元/MWh，各省按此标准执行。",
            "source": "测试规则.pdf",
            "section": "第2条 价格上限",
            "page": 2,
            "chunk_id": "chunk-001",
        },
        {
            "text": "违反价格上限规则的交易主体将被处以违规电量差额罚款。",
            "source": "测试规则.pdf",
            "section": "第4条 违规处罚",
            "page": 4,
            "chunk_id": "chunk-002",
        },
        {
            "text": "蒙西地区日前现货出清电价上限为360元/MWh。",
            "source": "区域规则.pdf",
            "section": "第8条 区域限价",
            "page": 8,
            "chunk_id": "chunk-003",
        },
    ]


@pytest.fixture
def sample_final_result():
    """模拟最终审查结果。"""
    return {
        "decision": "不符合",
        "reason": "冀北日前现货出清电价800元/MWh超出了上限760元/MWh。",
        "evidence": [
            {
                "source": "测试规则.pdf",
                "section": "第2条 价格上限",
                "page": 2,
                "text": "省间日前现货出清电价上限为760元/MWh，各省按此标准执行。",
                "chunk_id": "chunk-001",
            },
            {
                "source": "测试规则.pdf",
                "section": "第4条 违规处罚",
                "page": 4,
                "text": "违反价格上限规则的交易主体将被处罚。",
                "chunk_id": "chunk-002",
            },
        ],
        "confidence": 0.95,
        "not_found": False,
    }


@pytest.fixture
def sample_audit_record(sample_chunks, sample_final_result):
    """完整的审计记录。"""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    traces = build_source_traceability(sample_final_result, sample_chunks)

    return AuditRecord(
        query_id="q-test-001",
        timestamp=now,
        original_query="2025年3月15日冀北的日前现货出清电价800元/MWh是否符合价格上限？",
        rewritten_query="2025-03-15 冀北 日前现货出清电价 800元/MWh 符合 价格上限",
        retrieval=RetrievalAudit(
            bm25_k=30,
            vector_k=30,
            final_k=10,
            search_expanded=False,
            retrieval_latency_ms=45.2,
        ),
        llm_generation=LLMGenerationAudit(
            model="qwen3-max",
            tok_input=1200,
            tok_output=350,
            latency_ms=2500.0,
            not_found=False,
        ),
        tool_executions=[
            ToolCallLog(
                query_id="q-test-001",
                round=1,
                tool_name="arithmetic_compare",
                args={"actual": 800, "operator": "gt", "threshold": 760},
                result={"success": True, "data": {"result": True}},
                timestamp=now,
                latency_ms=12.0,
            )
        ],
        judge_verification=JudgeAudit(
            model="deepseek-v4",
            verified=True,
            hallucinated_count=0,
            skipped=False,
            latency_ms=1800.0,
        ),
        final_result=sample_final_result,
        source_traceability=traces,
    )


# ---------------------------------------------------------------------------
# _longest_common_substring 测试
# ---------------------------------------------------------------------------


class TestLongestCommonSubstring:
    def test_identical_strings(self):
        assert _longest_common_substring("abc", "abc") == 3

    def test_partial_overlap(self):
        assert _longest_common_substring("abcdef", "xbcdey") == 4  # "bcde"

    def test_no_overlap(self):
        assert _longest_common_substring("abc", "xyz") == 0

    def test_empty_strings(self):
        assert _longest_common_substring("", "abc") == 0
        assert _longest_common_substring("abc", "") == 0
        assert _longest_common_substring("", "") == 0

    def test_chinese_characters(self):
        assert _longest_common_substring("电价上限为760元", "出清电价上限为760元每兆瓦时") == 9  # "电价上限为760元"

    def test_single_char(self):
        assert _longest_common_substring("a", "a") == 1
        assert _longest_common_substring("a", "b") == 0

    def test_contained_string(self):
        assert _longest_common_substring("hello", "hello world") == 5

    def test_multiple_matches_longest_wins(self):
        # "abc" appears in both but " defg " is longer (with spaces)
        assert _longest_common_substring("abc defg", "xyz defg abc") == 5  # " defg"


# ---------------------------------------------------------------------------
# build_source_traceability 测试
# ---------------------------------------------------------------------------


class TestBuildSourceTraceability:
    def test_exact_match(self, sample_chunks, sample_final_result):
        traces = build_source_traceability(sample_final_result, sample_chunks)
        assert len(traces) == 2
        # 第一条 evidence 应该精确匹配
        assert traces[0].match_type == "exact"
        assert traces[0].source_doc == "测试规则.pdf"
        assert traces[0].source_section == "第2条 价格上限"
        assert traces[0].source_page == 2
        assert traces[0].source_chunk_id == "chunk-001"

    def test_fuzzy_match(self):
        """部分匹配应标记为 fuzzy。"""
        result = {
            "decision": "不符合",
            "evidence": [
                {
                    "text": "冀北地区的电价上限为760元每兆瓦时",
                    "source": "doc.pdf",
                }
            ],
        }
        chunks = [
            {
                "text": "省间日前现货出清电价上限为760元/MWh",
                "source": "doc.pdf",
                "section": "第2条",
                "page": 2,
                "chunk_id": "ck-1",
            }
        ]
        traces = build_source_traceability(result, chunks)
        assert len(traces) == 1
        # 有部分重叠（"760元" + "电价上限"相关）
        assert traces[0].match_type in ("fuzzy", "llm_extracted")

    def test_no_match_marks_llm_extracted(self):
        """完全无法匹配的 evidence 标记为 llm_extracted。"""
        result = {
            "decision": "无法判断",
            "evidence": [
                {
                    "text": "这是一段LLM完全编造的内容，不在任何文档中。",
                    "source": "unknown.pdf",
                }
            ],
        }
        chunks = [
            {
                "text": "完全不同的规则内容abc123",
                "source": "doc.pdf",
                "section": "第1条",
                "page": 1,
                "chunk_id": "ck-x",
            }
        ]
        traces = build_source_traceability(result, chunks)
        assert len(traces) == 1
        assert traces[0].match_type == "llm_extracted"
        assert traces[0].source_doc == ""
        assert traces[0].source_chunk_id == ""

    def test_empty_evidence(self):
        """空 evidence 列表返回空 traces。"""
        result = {"decision": "符合", "evidence": []}
        traces = build_source_traceability(result, [])
        assert traces == []

    def test_empty_evidence_text(self):
        """evidence text 为空时仍生成 trace 但标记为 llm_extracted。"""
        result = {
            "decision": "符合",
            "evidence": [{"text": "", "source": "doc.pdf"}],
        }
        traces = build_source_traceability(result, [])
        assert len(traces) == 1
        assert traces[0].match_type == "llm_extracted"
        assert traces[0].result_excerpt == ""

    def test_result_excerpt_truncated(self):
        """result_excerpt 截断到 200 字符。"""
        long_text = "长文本" * 80  # 240 字符
        result = {
            "decision": "符合",
            "evidence": [{"text": long_text}],
        }
        chunks = [{"text": long_text, "source": "d.pdf"}]
        traces = build_source_traceability(result, chunks)
        assert len(traces[0].result_excerpt) <= 200

    def test_source_text_truncated(self):
        """source_text 截断到 300 字符。"""
        result = {
            "decision": "符合",
            "evidence": [{"text": "电价上限为760元"}],
        }
        long_chunk_text = "规则内容" * 100  # 400 字符
        chunks = [{"text": long_chunk_text, "source": "d.pdf"}]
        traces = build_source_traceability(result, chunks)
        assert len(traces[0].source_text) <= 300

    def test_multiple_chunks_best_match_wins(self):
        """在多个 chunk 中找最匹配的那个。"""
        result = {
            "evidence": [
                {"text": "蒙西地区日前现货出清电价上限为360元/MWh。"}
            ],
        }
        chunks = [
            {
                "text": "冀北电价上限为760元/MWh。",
                "source": "doc-a.pdf",
                "section": "第1条",
                "page": 1,
                "chunk_id": "a-1",
            },
            {
                "text": "蒙西地区日前现货出清电价上限为360元/MWh。",
                "source": "doc-b.pdf",
                "section": "第8条 区域限价",
                "page": 8,
                "chunk_id": "b-8",
            },
        ]
        traces = build_source_traceability(result, chunks)
        assert len(traces) == 1
        assert traces[0].match_type == "exact"
        assert traces[0].source_doc == "doc-b.pdf"
        assert traces[0].source_section == "第8条 区域限价"


# ---------------------------------------------------------------------------
# AuditStore CRUD 测试
# ---------------------------------------------------------------------------


class TestAuditStoreSaveAndLoad:
    def test_save_and_load(self, audit_store, sample_audit_record):
        audit_store.save(sample_audit_record)

        loaded = audit_store.load(sample_audit_record.query_id)
        assert loaded is not None
        assert loaded.query_id == sample_audit_record.query_id
        assert loaded.original_query == sample_audit_record.original_query
        assert loaded.rewritten_query == sample_audit_record.rewritten_query
        assert loaded.retrieval.bm25_k == 30
        assert loaded.llm_generation.not_found is False
        assert loaded.judge_verification is not None
        assert loaded.judge_verification.verified is True
        assert len(loaded.source_traceability) == 2

    def test_load_with_date(self, audit_store, sample_audit_record):
        audit_store.save(sample_audit_record)
        date_str = sample_audit_record.timestamp[:10]

        loaded = audit_store.load(sample_audit_record.query_id, date=date_str)
        assert loaded is not None
        assert loaded.query_id == sample_audit_record.query_id

    def test_load_wrong_date_returns_none(self, audit_store, sample_audit_record):
        audit_store.save(sample_audit_record)

        loaded = audit_store.load(sample_audit_record.query_id, date="2000-01-01")
        assert loaded is None

    def test_load_nonexistent(self, audit_store):
        assert audit_store.load("nonexistent-id") is None

    def test_load_without_date_searches_all(self, audit_store, sample_audit_record):
        audit_store.save(sample_audit_record)

        loaded = audit_store.load(sample_audit_record.query_id)
        assert loaded is not None

    def test_save_creates_directory(self, temp_audit_dir):
        store = AuditStore(storage_dir=os.path.join(temp_audit_dir, "nested", "audit"))
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        record = AuditRecord(
            query_id="q-dir-test",
            timestamp=now,
            original_query="test",
        )
        store.save(record)
        assert os.path.exists(store.storage_dir)

    def test_save_returns_path(self, audit_store, sample_audit_record):
        path = audit_store.save(sample_audit_record)
        assert path.endswith(f"{sample_audit_record.query_id}.json")
        assert os.path.exists(path)


class TestAuditStoreList:
    def test_list_by_date(self, audit_store, sample_audit_record):
        audit_store.save(sample_audit_record)
        date_str = sample_audit_record.timestamp[:10]

        ids = audit_store.list_by_date(date_str)
        assert sample_audit_record.query_id in ids

    def test_list_empty_date(self, audit_store):
        ids = audit_store.list_by_date("2000-01-01")
        assert ids == []

    def test_list_nonexistent_dir(self, temp_audit_dir):
        store = AuditStore(storage_dir=os.path.join(temp_audit_dir, "nonexistent"))
        ids = store.list_by_date("2000-01-01")
        assert ids == []


class TestAuditStoreSample:
    def test_sample_for_review(self, audit_store):
        """抽样返回不超过 count 的记录。"""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        date_str = now[:10]

        for i in range(20):
            record = AuditRecord(
                query_id=f"q-sample-{i:03d}",
                timestamp=now,
                original_query=f"测试问题 {i}",
            )
            audit_store.save(record)

        sampled = audit_store.sample_for_review(date_str, count=5)
        assert len(sampled) == 5

        for r in sampled:
            assert r.query_id.startswith("q-sample-")

    def test_sample_empty_date(self, audit_store):
        sampled = audit_store.sample_for_review("2000-01-01", count=10)
        assert sampled == []

    def test_sample_count_exceeds_available(self, audit_store):
        """抽样数量超过实际记录数时返回全部。"""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        date_str = now[:10]

        for i in range(3):
            record = AuditRecord(
                query_id=f"q-small-{i}",
                timestamp=now,
            )
            audit_store.save(record)

        sampled = audit_store.sample_for_review(date_str, count=10)
        assert len(sampled) == 3


class TestAuditStoreStats:
    def test_get_stats_empty(self, audit_store):
        stats = audit_store.get_stats("2000-01-01", "2000-12-31")
        assert stats["total_reviews"] == 0
        assert stats["hallucination_rate"] == 0.0
        assert stats["judge_skip_rate"] == 0.0

    def test_get_stats_with_data(self, audit_store, sample_audit_record):
        audit_store.save(sample_audit_record)

        date_str = sample_audit_record.timestamp[:10]
        stats = audit_store.get_stats(date_str, date_str)
        assert stats["total_reviews"] == 1
        assert stats["hallucination_rate"] == 0.0  # hallucinated_count=0
        assert stats["judge_skip_rate"] == 0.0  # skipped=False
        assert stats["not_found_rate"] == 0.0
        assert stats["avg_confidence"] == 0.95

    def test_get_stats_hallucination(self, audit_store):
        """有幻觉记录时正确统计。"""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        date_str = now[:10]

        record = AuditRecord(
            query_id="q-halluc",
            timestamp=now,
            judge_verification=JudgeAudit(
                verified=False,
                hallucinated_count=3,
                skipped=False,
            ),
            final_result={"confidence": 0.5},
        )
        audit_store.save(record)

        stats = audit_store.get_stats(date_str, date_str)
        assert stats["total_reviews"] == 1
        assert stats["hallucination_rate"] == 1.0  # 3 hallucinations > 0

    def test_get_stats_judge_skipped(self, audit_store):
        """Judge 跳过的记录正确统计。"""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        date_str = now[:10]

        record = AuditRecord(
            query_id="q-skip",
            timestamp=now,
            judge_verification=JudgeAudit(
                skipped=True,
                skipped_reason="timeout",
            ),
            final_result={"confidence": 0.8},
        )
        audit_store.save(record)

        stats = audit_store.get_stats(date_str, date_str)
        assert stats["judge_skip_rate"] == 1.0

    def test_get_stats_not_found(self, audit_store):
        """not_found 的记录正确统计。"""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        date_str = now[:10]

        record = AuditRecord(
            query_id="q-nf",
            timestamp=now,
            llm_generation=LLMGenerationAudit(not_found=True),
            final_result={"confidence": 0.0},
        )
        audit_store.save(record)

        stats = audit_store.get_stats(date_str, date_str)
        assert stats["not_found_rate"] == 1.0

    def test_get_stats_date_range(self, audit_store):
        """只统计日期范围内的记录。"""
        record_old = AuditRecord(
            query_id="q-old",
            timestamp="2020-01-15T10:00:00Z",
            final_result={"confidence": 0.5},
        )
        record_new = AuditRecord(
            query_id="q-new",
            timestamp="2020-06-15T10:00:00Z",
            final_result={"confidence": 0.8},
        )
        audit_store.save(record_old)
        audit_store.save(record_new)

        # 只包含 q-new
        stats = audit_store.get_stats("2020-06-01", "2020-06-30")
        assert stats["total_reviews"] == 1

        # 包含两个
        stats = audit_store.get_stats("2020-01-01", "2020-12-31")
        assert stats["total_reviews"] == 2


class TestAuditStoreDelete:
    def test_delete_existing(self, audit_store, sample_audit_record):
        audit_store.save(sample_audit_record)
        date_str = sample_audit_record.timestamp[:10]

        success = audit_store.delete(sample_audit_record.query_id, date_str)
        assert success is True
        assert audit_store.load(sample_audit_record.query_id, date=date_str) is None

    def test_delete_nonexistent(self, audit_store):
        success = audit_store.delete("nonexistent", "2000-01-01")
        assert success is False

    def test_delete_then_reload(self, audit_store, sample_audit_record):
        audit_store.save(sample_audit_record)
        date_str = sample_audit_record.timestamp[:10]

        audit_store.delete(sample_audit_record.query_id, date_str)

        loaded = audit_store.load(sample_audit_record.query_id, date=date_str)
        assert loaded is None


class TestAuditStoreDateRange:
    def test_date_range(self, audit_store):
        assert audit_store.date_range() == ("", "")

    def test_date_range_with_data(self, audit_store):
        record1 = AuditRecord(query_id="q-1", timestamp="2020-01-15T10:00:00Z")
        record2 = AuditRecord(query_id="q-2", timestamp="2020-06-15T10:00:00Z")
        audit_store.save(record1)
        audit_store.save(record2)

        start, end = audit_store.date_range()
        assert start == "2020-01-15"
        assert end == "2020-06-15"


# ---------------------------------------------------------------------------
# AuditRecord 模型测试
# ---------------------------------------------------------------------------


class TestAuditRecordModel:
    def test_minimal_audit_record(self):
        """最简审计记录可正常创建。"""
        record = AuditRecord(
            query_id="q-min",
            timestamp="2026-01-01T00:00:00Z",
        )
        assert record.query_id == "q-min"
        assert record.original_query == ""
        assert record.retrieval.bm25_k == 0
        assert record.llm_generation.not_found is False
        assert record.source_traceability == []

    def test_audit_record_serialization(self, sample_audit_record):
        data = sample_audit_record.model_dump()
        assert data["query_id"] == "q-test-001"
        assert data["retrieval"]["bm25_k"] == 30
        assert len(data["tool_executions"]) == 1

    def test_audit_record_deserialization(self, sample_audit_record):
        data = sample_audit_record.model_dump()
        restored = AuditRecord(**data)
        assert restored.query_id == sample_audit_record.query_id
        assert restored.retrieval.bm25_k == sample_audit_record.retrieval.bm25_k

    def test_judge_verification_none(self):
        """judge_verification 可以为 None。"""
        record = AuditRecord(
            query_id="q-no-judge",
            timestamp="2026-01-01T00:00:00Z",
            judge_verification=None,
        )
        assert record.judge_verification is None
        data = record.model_dump()
        assert data["judge_verification"] is None


# ---------------------------------------------------------------------------
# RetrievalAudit 模型测试
# ---------------------------------------------------------------------------


class TestRetrievalAudit:
    def test_default_values(self):
        r = RetrievalAudit()
        assert r.bm25_k == 0
        assert r.vector_k == 0
        assert r.sparse_k == 0
        assert r.fusion_method == "RRF"
        assert r.final_k == 0
        assert r.search_expanded is False
        assert r.retrieval_latency_ms == 0.0

    def test_custom_values(self):
        r = RetrievalAudit(
            bm25_k=30,
            vector_k=30,
            sparse_k=10,
            fusion_method="RRF",
            final_k=10,
            search_expanded=True,
            retrieval_latency_ms=85.5,
        )
        assert r.bm25_k == 30
        assert r.search_expanded is True
        assert r.retrieval_latency_ms == 85.5


# ---------------------------------------------------------------------------
# LLMGenerationAudit 模型测试
# ---------------------------------------------------------------------------


class TestLLMGenerationAudit:
    def test_default_values(self):
        g = LLMGenerationAudit()
        assert g.model == ""
        assert g.tok_input == 0
        assert g.not_found is False

    def test_not_found_flag(self):
        g = LLMGenerationAudit(not_found=True)
        assert g.not_found is True


# ---------------------------------------------------------------------------
# JudgeAudit 模型测试
# ---------------------------------------------------------------------------


class TestJudgeAudit:
    def test_default_values(self):
        j = JudgeAudit()
        assert j.verified is False
        assert j.hallucinated_count == 0
        assert j.skipped is False
        assert j.skipped_reason == ""

    def test_skipped_with_reason(self):
        j = JudgeAudit(skipped=True, skipped_reason="timeout")
        assert j.skipped is True
        assert j.skipped_reason == "timeout"

    def test_hallucinated(self):
        j = JudgeAudit(verified=False, hallucinated_count=5)
        assert j.verified is False
        assert j.hallucinated_count == 5


# ---------------------------------------------------------------------------
# SourceTrace 模型测试
# ---------------------------------------------------------------------------


class TestSourceTrace:
    def test_exact_match_trace(self):
        trace = SourceTrace(
            result_field="evidence[0]",
            result_excerpt="电价上限为760元/MWh",
            source_doc="规则.pdf",
            source_section="第2条",
            source_page=2,
            source_text="省间日前现货出清电价上限为760元/MWh。",
            source_chunk_id="chunk-001",
            match_type="exact",
        )
        assert trace.match_type == "exact"
        assert trace.source_page == 2

    def test_llm_extracted_trace(self):
        trace = SourceTrace(
            result_field="evidence[0]",
            result_excerpt="编造的内容",
            match_type="llm_extracted",
        )
        assert trace.match_type == "llm_extracted"
        assert trace.source_doc == ""


# ---------------------------------------------------------------------------
# 默认工厂
# ---------------------------------------------------------------------------


class TestDefaultAuditStore:
    def test_singleton(self):
        s1 = get_default_audit_store()
        s2 = get_default_audit_store()
        assert s1 is s2

    def test_instance_type(self):
        store = get_default_audit_store()
        assert isinstance(store, AuditStore)


# ---------------------------------------------------------------------------
# 审计记录文件格式验证
# ---------------------------------------------------------------------------


class TestAuditRecordFileFormat:
    def test_saved_json_is_valid(self, audit_store, sample_audit_record):
        path = audit_store.save(sample_audit_record)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 验证关键字段存在
        assert "query_id" in data
        assert "timestamp" in data
        assert "original_query" in data
        assert "rewritten_query" in data
        assert "retrieval" in data
        assert "llm_generation" in data
        assert "tool_executions" in data
        assert "judge_verification" in data
        assert "final_result" in data
        assert "source_traceability" in data

    def test_json_is_pretty_printed(self, audit_store, sample_audit_record):
        path = audit_store.save(sample_audit_record)

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # 验证有缩进（JSON 美化）
        assert "  " in content
        assert "\n" in content
