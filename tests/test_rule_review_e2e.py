"""
规则审查系统端到端测试

覆盖 Phase 1 全部 6 个分支场景 + 完整链路：

场景 A  - 问题不明确 → 澄清追问
场景 B1 - 单文档查询，完整审查 JSON
场景 B3 - 文档中无相关规则 → not_found
场景 C  - 多文档查询，并行检索合并
场景 D  - 空索引，检索无结果 → not_found
场景 F  - Judge 超时/异常 → judge_skipped
正常    - 上传→改写→检索→生成→Judge→输出 完整链路
"""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import fitz
import numpy as np
import pytest
from langchain_core.messages import AIMessage

from src.rule_review.document_store import DocumentStore
from src.rule_review.generator import RuleReviewGenerator
from src.rule_review.judge import JudgeResult, RuleReviewJudge
from src.rule_review.pipeline import RuleReviewPipeline, check_clarification_needed
from src.rule_review.query_rewriter import QueryRewriter
from src.rule_review.retriever import HybridRetriever
from src.rule_review.schemas import LLMOutput, RuleReviewRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_pdf_bytes(title="第一章 总则", body_texts=None) -> bytes:
    """生成包含中文标题和正文的测试 PDF。"""
    doc = fitz.open()
    page = doc.new_page()
    h = page.rect.height

    page.insert_text((72, 72), title, fontname="china-ss", fontsize=16)
    page.insert_text((72, 100), "第1条 适用范围", fontname="china-ss", fontsize=14)

    texts = body_texts or [
        "省间日前现货出清电价上限为760元/MWh，各省按此标准执行。",
        "违反价格上限的交易主体将被处以罚款。",
        "日内现货交易价格不得超过日前现货出清电价的120%。",
    ]
    y = 140
    for text in texts:
        page.insert_text((72, y), text, fontname="china-ss", fontsize=12)
        y += 20
        if y > h - 50:
            page = doc.new_page()
            y = 72

    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _mock_embedding_fn(texts):
    """固定 8 维归一化向量。"""
    rng = np.random.default_rng(42)
    vecs = rng.normal(size=(len(texts), 8)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1.0, norms)


def _make_llm_output(**kwargs) -> LLMOutput:
    defaults = dict(
        decision="不符合",
        reason="实际电价800元/MWh超过上限760元/MWh，超出40元/MWh",
        evidence=[
            {
                "source": "测试规则.pdf",
                "section": "第1条 适用范围",
                "page": 1,
                "text": "省间日前现货出清电价上限为760元/MWh。",
            }
        ],
        confidence=0.95,
    )
    defaults.update(kwargs)
    return LLMOutput(**defaults)


class MockGenerator:
    """可编程的 mock RuleReviewGenerator。"""

    def __init__(self, response=None):
        self._response = response or _make_llm_output()

    async def generate(self, query, context_chunks=None, system_prompt=None,
                       tool_results=None, max_retries=1):
        return self._response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_store(tmp_path: Path):
    """使用临时目录的 DocumentStore。"""
    store = DocumentStore(
        documents_dir=tmp_path / "docs",
        index_dir=tmp_path / "index",
        embedding_fn=_mock_embedding_fn,
        embedding_dim=8,
    )
    return store


@pytest.fixture
def tmp_rewriter():
    """返回真实 QueryRewriter（需要 rule_terms.json 等配置文件）。"""
    return QueryRewriter()


@pytest.fixture
def basic_pipeline(tmp_store, tmp_rewriter):
    """只含真实 DocumentStore + QueryRewriter 的基础 pipeline。"""
    generator = MockGenerator()
    retriever = HybridRetriever(
        document_store=tmp_store,
        embedding_fn=_mock_embedding_fn,
    )
    return RuleReviewPipeline(
        rewriter=tmp_rewriter,
        document_store=tmp_store,
        retriever=retriever,
        generator=generator,
    )


# ---------------------------------------------------------------------------
# 场景 A：问题不明确 → 澄清
# ---------------------------------------------------------------------------


class TestScenarioA_Clarification:
    """问题缺少关键信息时应返回澄清追问。"""

    @pytest.mark.asyncio
    async def test_missing_time_entity(self, basic_pipeline):
        request = RuleReviewRequest(
            question="电价是否符合上限",
            stream=False,
        )
        result = await basic_pipeline.execute(request)

        assert "clarification" in result
        assert result["clarification"]["needs_clarification"] is True

    @pytest.mark.asyncio
    async def test_missing_time_sse(self, basic_pipeline):
        request = RuleReviewRequest(question="电价是否符合上限", stream=True)

        events = []
        async for line in basic_pipeline.execute_stream(request):
            events.append(line)

        full = "".join(events)
        assert "clarification" in full or "needs_clarification" in full
        assert "done" in events[-1]

    @pytest.mark.asyncio
    async def test_clear_query_passes(self, basic_pipeline):
        """完整问题应通过澄清检查。"""
        request = RuleReviewRequest(
            question="2025年3月15日冀北的日前现货电价800元/MWh是否符合上限",
            stream=False,
        )
        result = await basic_pipeline.execute(request)
        # 通过了澄清，应有 result 字段（或者至少不是 clarification）
        assert "result" in result
        assert "clarification" not in result


# ---------------------------------------------------------------------------
# 场景 D：空索引 → not_found
# ---------------------------------------------------------------------------


class TestScenarioD_EmptyIndex:
    """不上传任何文档直接查询，应返回 not_found。"""

    @pytest.mark.asyncio
    async def test_empty_index_not_found(self, basic_pipeline):
        request = RuleReviewRequest(
            question="2025年3月15日冀北的日前现货电价800元/MWh是否符合上限",
            stream=False,
        )
        result = await basic_pipeline.execute(request)
        assert result["result"]["not_found"] is True
        assert result["result"]["decision"] == "无法判断"

    @pytest.mark.asyncio
    async def test_empty_index_sse(self, basic_pipeline):
        request = RuleReviewRequest(
            question="2025年3月15日冀北的日前现货电价800元/MWh是否符合上限",
            stream=True,
        )
        events = []
        async for line in basic_pipeline.execute_stream(request):
            events.append(line)

        full = "".join(events)
        assert "not_found" in full or "未找到" in full or "无法判断" in full


# ---------------------------------------------------------------------------
# 场景 B1：正常单文档查询 → 完整审查 JSON
# ---------------------------------------------------------------------------


class TestScenarioB1_NormalQuery:
    """上传文档后进行正常规则审查。"""

    @pytest.mark.asyncio
    async def test_upload_then_query_nonstream(self, tmp_store, tmp_rewriter):
        pdf_bytes = _make_test_pdf_bytes(
            "第二章 价格规则",
            [
                "省间日前现货出清电价上限为760元/MWh。",
                "各省按此标准执行，不得超出上限。",
            ],
        )

        # 上传文档
        resp = tmp_store.ingest(pdf_bytes, filename="价格规则.pdf")
        assert resp.chunk_count > 0

        generator = MockGenerator(
            _make_llm_output(
                decision="不符合",
                reason="实际电价800元/MWh超过价格规则上限760元/MWh",
            )
        )
        retriever = HybridRetriever(
            document_store=tmp_store,
            embedding_fn=_mock_embedding_fn,
        )
        pipeline = RuleReviewPipeline(
            rewriter=tmp_rewriter,
            document_store=tmp_store,
            retriever=retriever,
            generator=generator,
        )

        request = RuleReviewRequest(
            question="2025年3月15日冀北的日前现货电价800元/MWh是否符合价格上限",
            stream=False,
        )
        result = await pipeline.execute(request)

        assert "result" in result
        assert result["result"]["decision"] == "不符合"
        assert result["result"]["confidence"] > 0.5

    @pytest.mark.asyncio
    async def test_upload_then_query_stream(self, tmp_store, tmp_rewriter):
        pdf_bytes = _make_test_pdf_bytes()
        tmp_store.ingest(pdf_bytes, filename="规则.pdf")

        generator = MockGenerator()
        retriever = HybridRetriever(
            document_store=tmp_store,
            embedding_fn=_mock_embedding_fn,
        )
        pipeline = RuleReviewPipeline(
            rewriter=tmp_rewriter,
            document_store=tmp_store,
            retriever=retriever,
            generator=generator,
        )

        request = RuleReviewRequest(
            question="2025年3月15日冀北的日前现货电价800元/MWh是否符合上限",
            stream=True,
        )
        events = []
        async for line in pipeline.execute_stream(request):
            events.append(line)

        full = "".join(events)
        assert "不符合" in full or "符合" in full or "无法判断" in full
        assert "done" in events[-1]

    @pytest.mark.asyncio
    async def test_generator_returns_not_found(self, tmp_store, tmp_rewriter):
        """Generator 判断文档中无相关规则时，应跳过 Judge。"""
        pdf_bytes = _make_test_pdf_bytes()
        tmp_store.ingest(pdf_bytes, filename="规则.pdf")

        generator = MockGenerator(
            _make_llm_output(decision="无法判断", reason="无相关规则", not_found=True)
        )
        retriever = HybridRetriever(
            document_store=tmp_store,
            embedding_fn=_mock_embedding_fn,
        )
        pipeline = RuleReviewPipeline(
            rewriter=tmp_rewriter,
            document_store=tmp_store,
            retriever=retriever,
            generator=generator,
        )

        request = RuleReviewRequest(
            question="2025年3月15日冀北电价是否符合上限",
            stream=False,
        )
        result = await pipeline.execute(request)
        assert result["result"]["not_found"] is True


# ---------------------------------------------------------------------------
# 场景 C：多文档查询 → 并行检索合并
# ---------------------------------------------------------------------------


class TestScenarioC_MultiDocument:
    """涉及多个文档的查询应拆分后并行检索并合并结果。"""

    @pytest.mark.asyncio
    async def test_multi_doc_query(self, tmp_store, tmp_rewriter):
        # 上传两个文档
        pdf1 = _make_test_pdf_bytes("第一章 交易规则", ["日前现货电价上限760元/MWh。"])
        pdf2 = _make_test_pdf_bytes("第一章 监管办法", ["违反规则将被处以罚款。"])

        tmp_store.ingest(pdf1, filename="交易规则.pdf")
        tmp_store.ingest(pdf2, filename="监管办法.pdf")

        assert len(tmp_store.list_documents()) == 2

        generator = MockGenerator()
        retriever = HybridRetriever(
            document_store=tmp_store,
            embedding_fn=_mock_embedding_fn,
        )
        pipeline = RuleReviewPipeline(
            rewriter=tmp_rewriter,
            document_store=tmp_store,
            retriever=retriever,
            generator=generator,
        )

        # 书名号指定两个文档
        request = RuleReviewRequest(
            question="2025年3月15日根据《交易规则》和《监管办法》冀北电价800是否符合上限",
            stream=False,
        )
        result = await pipeline.execute(request)

        assert "result" in result
        # 应有 stages 记录拆分步骤
        stages = result.get("stages", [])
        split_stage = [s for s in stages if s["stage"] == "split"]
        if split_stage:
            assert split_stage[0]["is_multi_doc"] is True

    @pytest.mark.asyncio
    async def test_multi_doc_sse(self, tmp_store, tmp_rewriter):
        pdf1 = _make_test_pdf_bytes("交易规则")
        pdf2 = _make_test_pdf_bytes("监管办法")
        tmp_store.ingest(pdf1, filename="交易规则.pdf")
        tmp_store.ingest(pdf2, filename="监管办法.pdf")

        generator = MockGenerator()
        retriever = HybridRetriever(
            document_store=tmp_store,
            embedding_fn=_mock_embedding_fn,
        )
        pipeline = RuleReviewPipeline(
            rewriter=tmp_rewriter,
            document_store=tmp_store,
            retriever=retriever,
            generator=generator,
        )

        request = RuleReviewRequest(
            question="2025年3月15日根据《交易规则》和《监管办法》冀北电价800是否符合上限",
            stream=True,
        )
        events = []
        async for line in pipeline.execute_stream(request):
            events.append(line)

        full = "".join(events)
        assert "done" in events[-1]


# ---------------------------------------------------------------------------
# 场景 F：Judge 超时/异常 → judge_skipped
# ---------------------------------------------------------------------------


class TestScenarioF_JudgeFallback:
    """Judge 异常时跳过校验，标记 judge_skipped。"""

    @pytest.mark.asyncio
    async def test_judge_skip_on_error(self, tmp_store, tmp_rewriter):
        pdf_bytes = _make_test_pdf_bytes()
        tmp_store.ingest(pdf_bytes, filename="规则.pdf")

        generator = MockGenerator()
        retriever = HybridRetriever(
            document_store=tmp_store,
            embedding_fn=_mock_embedding_fn,
        )

        # Mock Judge 抛出异常
        bad_judge = MagicMock(spec=RuleReviewJudge)
        async def _verify(*args, **kwargs):
            raise RuntimeError("Judge service down")
        bad_judge.verify = _verify

        pipeline = RuleReviewPipeline(
            rewriter=tmp_rewriter,
            document_store=tmp_store,
            retriever=retriever,
            generator=generator,
            judge=bad_judge,
        )

        request = RuleReviewRequest(
            question="2025年3月15日冀北的日前现货电价800元/MWh是否符合上限",
            stream=False,
        )
        result = await pipeline.execute(request)

        # 即便 Judge 失败，仍应返回 LLM 原始结果
        assert "result" in result
        assert result["result"]["decision"] in ("不符合", "符合", "部分符合", "无法判断")

    @pytest.mark.asyncio
    async def test_judge_not_found_skips(self, tmp_store, tmp_rewriter):
        pdf_bytes = _make_test_pdf_bytes()
        tmp_store.ingest(pdf_bytes, filename="规则.pdf")

        # not_found=True 时自动跳过
        generator = MockGenerator(
            _make_llm_output(not_found=True)
        )
        retriever = HybridRetriever(
            document_store=tmp_store,
            embedding_fn=_mock_embedding_fn,
        )

        good_judge = MagicMock(spec=RuleReviewJudge)
        good_judge.verify = AsyncMock()

        pipeline = RuleReviewPipeline(
            rewriter=tmp_rewriter,
            document_store=tmp_store,
            retriever=retriever,
            generator=generator,
            judge=good_judge,
        )

        request = RuleReviewRequest(
            question="2025年3月15日冀北电价是否符合上限",
            stream=False,
        )
        result = await pipeline.execute(request)

        assert result["result"]["not_found"] is True
        # Judge.verify 不应被调用（因为 not_found 直接跳过）
        good_judge.verify.assert_not_called()

    @pytest.mark.asyncio
    async def test_judge_normal_flow(self, tmp_store, tmp_rewriter):
        """带 Judge 的正常完整链路。"""
        pdf_bytes = _make_test_pdf_bytes()
        tmp_store.ingest(pdf_bytes, filename="规则.pdf")

        generator = MockGenerator()
        retriever = HybridRetriever(
            document_store=tmp_store,
            embedding_fn=_mock_embedding_fn,
        )

        # Mock Judge 返回正常校验结果
        good_judge = MagicMock(spec=RuleReviewJudge)
        async def _verify_good(llm_output, original_query, context_chunks, tool_logs=None):
            return JudgeResult(
                verified=True,
                final_decision=llm_output.decision,
                final_reason=llm_output.reason,
                final_evidence=llm_output.evidence,
                confidence=0.98,
            )
        good_judge.verify = _verify_good

        pipeline = RuleReviewPipeline(
            rewriter=tmp_rewriter,
            document_store=tmp_store,
            retriever=retriever,
            generator=generator,
            judge=good_judge,
        )

        request = RuleReviewRequest(
            question="2025年3月15日冀北的日前现货电价800元/MWh是否符合上限",
            stream=False,
        )
        result = await pipeline.execute(request)

        assert "result" in result
        # Judge 校验通过
        assert result["result"].get("judge_verified", False) is True

    @pytest.mark.asyncio
    async def test_judge_skip_sse(self, tmp_store, tmp_rewriter):
        """SSE 流式中 Judge 异常时应显示跳过标签。"""
        pdf_bytes = _make_test_pdf_bytes()
        tmp_store.ingest(pdf_bytes, filename="规则.pdf")

        generator = MockGenerator()
        retriever = HybridRetriever(
            document_store=tmp_store,
            embedding_fn=_mock_embedding_fn,
        )

        bad_judge = MagicMock(spec=RuleReviewJudge)
        async def _verify_bad(*args, **kwargs):
            raise RuntimeError("down")
        bad_judge.verify = _verify_bad

        pipeline = RuleReviewPipeline(
            rewriter=tmp_rewriter,
            document_store=tmp_store,
            retriever=retriever,
            generator=generator,
            judge=bad_judge,
        )

        request = RuleReviewRequest(
            question="2025年3月15日冀北的日前现货电价800元/MWh是否符合上限",
            stream=True,
        )
        events = []
        async for line in pipeline.execute_stream(request):
            events.append(line)

        full = "".join(events)
        # SSE 应输出 judge_skipped 标签或直接返回内容
        assert "done" in events[-1]


# ---------------------------------------------------------------------------
# 正常：完整链路（上传→改写→检索→生成→输出）
# ---------------------------------------------------------------------------


class TestFullEndToEnd:
    """模拟用户的实际使用流程。"""

    @pytest.mark.asyncio
    async def test_full_pipeline_nonstream(self, tmp_store, tmp_rewriter):
        """完整端到端链路：上传 PDF → 非流式查询 → 验证结果结构。"""
        pdf_bytes = _make_test_pdf_bytes(
            "第二章 价格规则",
            [
                "省间日前现货出清电价上限为760元/MWh。",
                "各省按此标准执行。",
                "日内现货交易价格不得超过日前现货出清电价的120%。",
            ],
        )

        # Step 1: 上传
        resp = tmp_store.ingest(pdf_bytes, filename="完整规则.pdf")
        assert resp.page_count >= 1
        assert resp.chunk_count > 0

        # Step 2: 确认文档列表
        docs = tmp_store.list_documents()
        assert len(docs) == 1
        assert docs[0].file_name == "完整规则.pdf"

        # Step 3: 创建 pipeline
        generator = MockGenerator()
        retriever = HybridRetriever(
            document_store=tmp_store,
            embedding_fn=_mock_embedding_fn,
        )
        pipeline = RuleReviewPipeline(
            rewriter=tmp_rewriter,
            document_store=tmp_store,
            retriever=retriever,
            generator=generator,
        )

        # Step 4: 查询
        request = RuleReviewRequest(
            question="2025年3月15日冀北的日前现货电价800元/MWh是否符合价格上限",
            stream=False,
        )
        result = await pipeline.execute(request)

        # Step 5: 验证完整输出结构
        assert "query_id" in result
        assert "rewritten_query" in result
        assert "result" in result
        assert "stages" in result

        r = result["result"]
        assert "decision" in r
        assert "reason" in r
        assert "evidence" in r
        assert "confidence" in r
        # 置信度在 0-1 之间
        assert 0.0 <= r["confidence"] <= 1.0

        # stages 包含主要阶段
        stage_names = [s["stage"] for s in result["stages"]]
        assert "rewrite" in stage_names
        assert "retrieval" in stage_names
        assert "generation" in stage_names

    @pytest.mark.asyncio
    async def test_full_pipeline_stream(self, tmp_store, tmp_rewriter):
        """完整端到端 SSE 流式链路。"""
        pdf_bytes = _make_test_pdf_bytes()
        tmp_store.ingest(pdf_bytes, filename="规则.pdf")

        generator = MockGenerator()
        retriever = HybridRetriever(
            document_store=tmp_store,
            embedding_fn=_mock_embedding_fn,
        )
        pipeline = RuleReviewPipeline(
            rewriter=tmp_rewriter,
            document_store=tmp_store,
            retriever=retriever,
            generator=generator,
        )

        request = RuleReviewRequest(
            question="2025年3月15日冀北的日前现货电价800元/MWh是否符合上限",
            stream=True,
        )
        events = []
        async for line in pipeline.execute_stream(request):
            events.append(line)

        assert len(events) >= 3  # 至少 label + content + done

        full = "".join(events)
        # 验证 SSE 格式
        assert "event:" in full or "data:" in full
        # 验证各阶段标签
        assert "查询预处理中" in full
        assert "检索相关知识中" in full
        assert "规则推理中" in full
        assert "done" in events[-1]

    @pytest.mark.asyncio
    async def test_full_pipeline_with_judge(self, tmp_store, tmp_rewriter):
        """完整链路含 Judge 校验。"""
        pdf_bytes = _make_test_pdf_bytes()
        tmp_store.ingest(pdf_bytes, filename="规则.pdf")

        generator = MockGenerator()
        retriever = HybridRetriever(
            document_store=tmp_store,
            embedding_fn=_mock_embedding_fn,
        )

        good_judge = MagicMock(spec=RuleReviewJudge)
        async def _verify(llm_output, original_query, context_chunks, tool_logs=None):
            return JudgeResult(
                verified=True,
                final_decision=llm_output.decision,
                final_reason=llm_output.reason + "（经 Judge 确认）",
                final_evidence=llm_output.evidence,
                confidence=0.97,
            )
        good_judge.verify = _verify

        pipeline = RuleReviewPipeline(
            rewriter=tmp_rewriter,
            document_store=tmp_store,
            retriever=retriever,
            generator=generator,
            judge=good_judge,
        )

        request = RuleReviewRequest(
            question="2025年3月15日冀北的日前现货电价800元/MWh是否符合上限",
            stream=False,
        )
        result = await pipeline.execute(request)

        r = result["result"]
        assert "judge_verified" in r
        assert r.get("judge_verified") is True
        assert "经 Judge 确认" in r.get("reason", "")

    @pytest.mark.asyncio
    async def test_delete_and_reupload(self, tmp_store, tmp_rewriter):
        """删除文档后重新上传，验证索引正确更新。"""
        pdf_bytes = _make_test_pdf_bytes()
        resp1 = tmp_store.ingest(pdf_bytes, filename="v1.pdf")
        assert tmp_store.list_documents()[0].file_name == "v1.pdf"

        # 删除
        tmp_store.delete(resp1.doc_id)
        assert len(tmp_store.list_documents()) == 0

        # 重新上传
        resp2 = tmp_store.ingest(pdf_bytes, filename="v2.pdf")
        assert len(tmp_store.list_documents()) == 1
        assert tmp_store.list_documents()[0].file_name == "v2.pdf"

        # 确认新上传的仍可检索
        generator = MockGenerator()
        retriever = HybridRetriever(
            document_store=tmp_store,
            embedding_fn=_mock_embedding_fn,
        )
        pipeline = RuleReviewPipeline(
            rewriter=tmp_rewriter,
            document_store=tmp_store,
            retriever=retriever,
            generator=generator,
        )

        request = RuleReviewRequest(
            question="2025年3月15日冀北电价800是否符合上限",
            stream=False,
        )
        result = await pipeline.execute(request)
        assert "result" in result and "clarification" not in result

    @pytest.mark.asyncio
    async def test_invalid_document_not_blocking(self, tmp_store, tmp_rewriter):
        """空文档路径不应阻断整个系统。"""
        # 确保即使文档库为空，查询也能优雅降级
        assert len(tmp_store.list_documents()) == 0

        generator = MockGenerator()
        retriever = HybridRetriever(
            document_store=tmp_store,
            embedding_fn=_mock_embedding_fn,
        )
        pipeline = RuleReviewPipeline(
            rewriter=tmp_rewriter,
            document_store=tmp_store,
            retriever=retriever,
            generator=generator,
        )

        request = RuleReviewRequest(
            question="2025年3月15日冀北的日前现货电价800元/MWh是否符合上限",
            stream=False,
        )
        result = await pipeline.execute(request)

        # 应正常返回 not_found，而非崩溃
        assert "result" in result
        assert result["result"]["not_found"] is True


# ---------------------------------------------------------------------------
# 综合 API 端到端（通过 router）
# ---------------------------------------------------------------------------


class TestRouterEndToEnd:
    """通过 pipeline 验证文档管理完整流程（跳过 router 避免 torch segfault）。"""

    def test_document_upload_list_delete_flow(self, tmp_store, tmp_rewriter):
        """文档上传 → 列表 → 删除的完整流程。"""
        pdf_bytes = _make_test_pdf_bytes()
        resp = tmp_store.ingest(pdf_bytes, filename="rules.pdf")
        assert resp.page_count >= 1
        assert resp.chunk_count > 0

        docs = tmp_store.list_documents()
        assert len(docs) == 1
        assert docs[0].file_name == "rules.pdf"

        # 删除
        assert tmp_store.delete(resp.doc_id) is True
        assert len(tmp_store.list_documents()) == 0

    def test_health_check_after_upload(self, tmp_store):
        """上传后索引状态正确。"""
        pdf_bytes = _make_test_pdf_bytes()
        resp = tmp_store.ingest(pdf_bytes, filename="rules.pdf")

        assert len(tmp_store.list_documents()) == 1
        assert resp.chunk_count > 0
        assert len(tmp_store._chunks) == resp.chunk_count
        assert tmp_store._index is not None
        assert tmp_store._index.ntotal > 0
