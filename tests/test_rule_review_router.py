"""
规则审查系统 API 路由单元测试

覆盖 src/rule_review/router.py 的全部端点：
- POST /v1/rule-review（流式/非流式）
- POST /v1/rule-review/documents（上传，含解析分层 importance/parse_mode）
- POST /v1/rule-review/documents/manual（手动入库，解析分层新增）
- GET  /v1/rule-review/documents（列表）
- DELETE /v1/rule-review/documents/{doc_id}（删除）
- GET  /v1/rule-review/health（健康检查）
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.rule_review.document_store import DocumentInfo, DocumentStore
from src.rule_review.pipeline import RuleReviewPipeline
from src.rule_review.retriever import HybridRetriever, RetrieveResult
from src.rule_review.router import _get_document_store, _get_pipeline, router
from src.rule_review.schemas import DocumentUploadResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_pdf_bytes() -> bytes:
    """生成最小的合法 PDF 字节。"""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "测试规则文档", fontname="china-ss", fontsize=12)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _mock_pipeline():
    """创建 mock RuleReviewPipeline。"""
    pipeline = MagicMock(spec=RuleReviewPipeline)

    async def _mock_execute(request):
        return {
            "query_id": "test-id",
            "result": {
                "decision": "不符合",
                "reason": "测试推理",
                "evidence": [],
                "confidence": 0.95,
            },
            "stages": [],
        }

    async def _mock_stream(request):
        yield 'data: {"type":"messageLabel","answer":"- <span>查询预处理中...</span>"}\n\n'
        yield 'data: {"type":"messageLabel","answer":"- <span>规则推理中...</span>"}\n\n'
        yield 'event: message\ndata: {"answer":"{\\"decision\\":\\"不符合\\"}","type":"content"}\n\n'
        yield 'event: done\ndata: {"done":true}\n\n'

    pipeline.execute = _mock_execute
    pipeline.execute_stream = _mock_stream
    pipeline.retriever = MagicMock(spec=HybridRetriever)
    pipeline.retriever.refresh_bm25 = MagicMock()
    return pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singletons():
    """每个测试前重置全局单例，避免测试间相互污染。"""
    import src.rule_review.router as rmod

    old_pipeline = rmod._pipeline
    old_store = rmod._document_store
    rmod._pipeline = None
    rmod._document_store = None
    yield
    rmod._pipeline = old_pipeline
    rmod._document_store = old_store


@pytest.fixture
def client(tmp_path: Path):
    """创建带临时目录的 TestClient。"""
    # 覆盖默认 DocumentStore 的目录
    import src.rule_review.router as rmod

    rmod._document_store = DocumentStore(
        documents_dir=tmp_path / "docs",
        index_dir=tmp_path / "index",
        embedding_fn=lambda texts: __import__("numpy").random.default_rng(42).normal(
            size=(len(texts), 8)
        ).astype("float32"),
        embedding_dim=8,
    )

    app = __import__("fastapi").FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def stub_pipeline(monkeypatch):
    """用 mock pipeline 替换 router._get_pipeline。

    真实 pipeline 构造会加载 bge-m3 embedding 模型并访问 HuggingFace
    （离线/沙箱环境会挂起），文档管理端点的 BM25 刷新只需 mock 即可。
    """
    monkeypatch.setattr(
        "src.rule_review.router._get_pipeline", lambda: _mock_pipeline()
    )


@pytest.fixture
def mock_client():
    """使用 mock pipeline 的 TestClient。"""
    import src.rule_review.router as rmod

    rmod._pipeline = _mock_pipeline()
    rmod._document_store = MagicMock(spec=DocumentStore)
    rmod._document_store.list_documents.return_value = []
    rmod._document_store._chunks = {}
    rmod._document_store._index = None

    app = __import__("fastapi").FastAPI()
    app.include_router(router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------------------


class TestHealthCheck:
    def test_health_returns_200(self, client):
        response = client.get("/v1/rule-review/health")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "rule_review"
        assert data["status"] == "healthy"
        assert "components" in data


# ---------------------------------------------------------------------------
# 文档上传
# ---------------------------------------------------------------------------


class TestDocumentUpload:
    def test_upload_valid_pdf(self, client):
        pdf_bytes = _make_test_pdf_bytes()
        response = client.post(
            "/v1/rule-review/documents",
            files={"file": ("test_rule.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "success"
        assert data["data"]["file_name"] == "test_rule.pdf"
        assert data["data"]["chunk_count"] > 0

    def test_upload_wrong_extension(self, client):
        response = client.post(
            "/v1/rule-review/documents",
            files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        assert response.status_code == 400
        assert "不支持" in response.json()["detail"]

    def test_upload_no_filename(self, client):
        response = client.post(
            "/v1/rule-review/documents",
            files={"file": (None, io.BytesIO(b""), "application/pdf")},
        )
        assert response.status_code == 400 or response.status_code == 422


# ---------------------------------------------------------------------------
# 文档上传（解析分层：importance / parse_mode）
# ---------------------------------------------------------------------------


class TestDocumentUploadTiers:
    """解析分层参数：importance / parse_mode 透传与校验。

    使用 stub_pipeline 避免真实 pipeline 的模型加载/网络依赖。
    """

    def test_upload_with_importance_and_parse_mode(self, client, stub_pipeline):
        pdf_bytes = _make_test_pdf_bytes()
        response = client.post(
            "/v1/rule-review/documents",
            files={"file": ("rule.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
            data={"importance": "high", "parse_mode": "pymupdf"},
        )
        assert response.status_code == 201
        data = response.json()["data"]
        assert data["importance"] == "high"
        assert data["parse_mode"] == "pymupdf"

    def test_upload_default_tier_fields(self, client, stub_pipeline):
        pdf_bytes = _make_test_pdf_bytes()
        response = client.post(
            "/v1/rule-review/documents",
            files={"file": ("rule.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        assert response.status_code == 201
        data = response.json()["data"]
        assert data["importance"] == "low"
        assert data["parse_mode"] == "pymupdf"  # 未安装 MinerU 时 auto 降级

    def test_upload_invalid_importance(self, client, stub_pipeline):
        pdf_bytes = _make_test_pdf_bytes()
        response = client.post(
            "/v1/rule-review/documents",
            files={"file": ("rule.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
            data={"importance": "weird"},
        )
        assert response.status_code == 400
        assert "importance" in response.json()["detail"]

    def test_upload_invalid_parse_mode(self, client, stub_pipeline):
        pdf_bytes = _make_test_pdf_bytes()
        response = client.post(
            "/v1/rule-review/documents",
            files={"file": ("rule.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
            data={"parse_mode": "bogus"},
        )
        assert response.status_code == 400
        assert "parse_mode" in response.json()["detail"]


# ---------------------------------------------------------------------------
# 手动入库（重要政策问答表格，POST /documents/manual）
# ---------------------------------------------------------------------------


class TestManualUpload:
    """手动入库端点：重要政策问答表格以 Markdown 直接入库。"""

    MANUAL_MARKDOWN = (
        "## 第2条 价格上限\n\n下表为各省日前现货出清电价上限：\n\n"
        "| 省份 | 电价上限(元/MWh) |\n|---|---|\n| 冀北 | 760 |\n| 山西 | 780 |"
    )

    def test_manual_upload_success(self, client, stub_pipeline):
        response = client.post(
            "/v1/rule-review/documents/manual",
            json={
                "markdown": self.MANUAL_MARKDOWN,
                "filename": "价格上限表.md",
                "importance": "high",
                "source": "政策问答表格人工整理",
            },
        )
        assert response.status_code == 201
        data = response.json()["data"]
        assert data["parse_mode"] == "manual"
        assert data["importance"] == "high"
        assert data["chunk_count"] > 0

        # 列表可见
        list_resp = client.get("/v1/rule-review/documents")
        docs = list_resp.json()["data"]
        assert any(
            d["doc_id"] == data["doc_id"] and d["parse_mode"] == "manual"
            for d in docs
        )

    def test_manual_upload_default_importance_high(self, client, stub_pipeline):
        response = client.post(
            "/v1/rule-review/documents/manual",
            json={"markdown": "# 第1条 测试\n\n正文内容"},
        )
        assert response.status_code == 201
        assert response.json()["data"]["importance"] == "high"

    def test_manual_upload_empty_markdown(self, client, stub_pipeline):
        response = client.post(
            "/v1/rule-review/documents/manual",
            json={"markdown": "   "},
        )
        assert response.status_code == 400
        assert "markdown" in response.json()["detail"]

    def test_manual_upload_invalid_importance(self, client, stub_pipeline):
        response = client.post(
            "/v1/rule-review/documents/manual",
            json={"markdown": "正文", "importance": "weird"},
        )
        assert response.status_code == 400
        assert "importance" in response.json()["detail"]

    def test_manual_upload_triggers_bm25_refresh(self, client, monkeypatch):
        """手动入库后必须刷新 BM25（与 PDF 上传端点一致）。"""
        import src.rule_review.router as rmod

        refresh_mock = MagicMock()
        pipeline = _mock_pipeline()
        pipeline.retriever.refresh_bm25 = refresh_mock
        monkeypatch.setattr(rmod, "_get_pipeline", lambda: pipeline)

        response = client.post(
            "/v1/rule-review/documents/manual",
            json={"markdown": self.MANUAL_MARKDOWN},
        )
        assert response.status_code == 201
        refresh_mock.assert_called_once()


# ---------------------------------------------------------------------------
# 文档列表
# ---------------------------------------------------------------------------


class TestDocumentList:
    def test_list_empty(self, client):
        response = client.get("/v1/rule-review/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["data"] == []

    def test_list_after_upload(self, client):
        pdf_bytes = _make_test_pdf_bytes()
        client.post(
            "/v1/rule-review/documents",
            files={"file": ("rule.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        response = client.get("/v1/rule-review/documents")
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["file_name"] == "rule.pdf"


# ---------------------------------------------------------------------------
# 文档删除
# ---------------------------------------------------------------------------


class TestDocumentDelete:
    def test_delete_existing(self, client):
        pdf_bytes = _make_test_pdf_bytes()
        upload_resp = client.post(
            "/v1/rule-review/documents",
            files={"file": ("rule.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        doc_id = upload_resp.json()["data"]["doc_id"]

        response = client.delete(f"/v1/rule-review/documents/{doc_id}")
        assert response.status_code == 200
        assert response.json()["deleted"] is True

        # 再次列表应为空
        list_resp = client.get("/v1/rule-review/documents")
        assert len(list_resp.json()["data"]) == 0

    def test_delete_nonexistent(self, client):
        response = client.delete("/v1/rule-review/documents/nonexistent_id")
        assert response.status_code == 404

    def test_delete_nonexistent_force(self, client):
        response = client.delete("/v1/rule-review/documents/nonexistent_id?force=true")
        assert response.status_code == 200
        assert response.json()["deleted"] is False


# ---------------------------------------------------------------------------
# 规则审查主接口
# ---------------------------------------------------------------------------


class TestRuleReviewEndpoint:
    def test_non_streaming_returns_json(self, mock_client):
        response = mock_client.post(
            "/v1/rule-review",
            json={
                "question": "2025年3月15日冀北的日前现货电价800元/MWh是否符合上限",
                "stream": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert data["result"]["decision"] == "不符合"

    def test_streaming_returns_sse(self, mock_client):
        response = mock_client.post(
            "/v1/rule-review",
            json={
                "question": "测试问题",
                "stream": True,
            },
        )
        assert response.status_code == 200
        body = response.text
        assert "event:" in body or "data:" in body
        assert "done" in body

    def test_missing_question(self, mock_client):
        response = mock_client.post(
            "/v1/rule-review",
            json={"stream": False},
        )
        assert response.status_code == 422

    def test_invalid_top_k(self, mock_client):
        response = mock_client.post(
            "/v1/rule-review",
            json={
                "question": "测试问题",
                "top_k": 100,
                "stream": False,
            },
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# 完整集成测试（上传→审查→清理）
# ---------------------------------------------------------------------------


class TestIntegrationFlow:
    def test_upload_then_health(self, client):
        """上传文档后健康检查应反映更新。"""
        pdf_bytes = _make_test_pdf_bytes()
        client.post(
            "/v1/rule-review/documents",
            files={"file": ("rules.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )

        health = client.get("/v1/rule-review/health")
        data = health.json()
        assert data["components"]["document_store"]["documents"] == 1
        assert data["components"]["document_store"]["chunks"] > 0
