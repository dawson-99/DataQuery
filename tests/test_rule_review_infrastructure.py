"""
规则审查系统基础设施单元测试

覆盖范围：
- src/config.py 中新增的规则审查配置项
- src/rule_review/schemas.py 中的 Pydantic 模型
- app.py 正确注册 /v1/rule-review 路由
- src/rule_review/__init__.py 自动创建数据目录
"""

import os
from datetime import datetime

import pytest
from pydantic import ValidationError

from src.config import settings
from src.rule_review import __version__
from src.rule_review.schemas import (
    ClarificationResponse,
    DocumentUploadResponse,
    EvidenceItem,
    LLMOutput,
    NotFoundResponse,
    RuleReviewRequest,
    RuleReviewResult,
    ToolCallLog,
)


class TestRuleReviewSettings:
    """验证 config.py 中为规则审查系统新增的配置项。"""

    def test_rule_review_model_defaults(self):
        assert hasattr(settings, "RULE_REVIEW_MODEL")
        assert settings.RULE_REVIEW_MODEL == "qwen3-max"

    def test_rule_review_api_defaults(self):
        assert hasattr(settings, "RULE_REVIEW_API_KEY")
        assert hasattr(settings, "RULE_REVIEW_API_BASE")
        # 默认应回退到 DashScope 配置
        assert settings.RULE_REVIEW_API_KEY == settings.DASHSCOPE_API_KEY
        assert settings.RULE_REVIEW_API_BASE == settings.DASHSCOPE_API_BASE

    def test_judge_model_defaults(self):
        assert hasattr(settings, "JUDGE_MODEL")
        assert hasattr(settings, "JUDGE_API_KEY")
        assert hasattr(settings, "JUDGE_API_BASE")
        assert settings.JUDGE_MODEL == "deepseek-v4"

    def test_embedding_model_defaults(self):
        assert hasattr(settings, "EMBEDDING_MODEL")
        assert settings.EMBEDDING_MODEL == "BAAI/bge-m3"

    def test_rule_directories_defaults(self):
        assert hasattr(settings, "RULE_DOCUMENTS_DIR")
        assert hasattr(settings, "RULE_INDEX_DIR")
        assert settings.RULE_DOCUMENTS_DIR == "data/rule_documents"
        assert settings.RULE_INDEX_DIR == "data/rule_index"


class TestRuleReviewSchemas:
    """验证规则审查相关的 Pydantic 模型。"""

    def test_rule_review_request_defaults(self):
        req = RuleReviewRequest(question="测试问题")
        assert req.question == "测试问题"
        assert req.stream is True
        assert req.sessionId == ""
        assert req.userInfo is None
        assert req.top_k == 10

    def test_rule_review_request_top_k_validation(self):
        with pytest.raises(ValidationError):
            RuleReviewRequest(question="测试", top_k=0)
        with pytest.raises(ValidationError):
            RuleReviewRequest(question="测试", top_k=100)

    def test_evidence_item(self):
        ev = EvidenceItem(
            source="省间电力现货交易规则",
            section="第三章 第12条",
            page=23,
            text="冀北日前现货出清电价上限为 760 元/MWh。",
            chunk_id="chunk-001",
        )
        assert ev.page == 23

    def test_rule_review_result(self):
        result = RuleReviewResult(
            decision="不符合",
            reason="实际电价超过上限。",
            evidence=[
                EvidenceItem(
                    source="规则",
                    section="第三章",
                    page=1,
                    text="上限 760 元/MWh",
                    chunk_id="c1",
                )
            ],
            confidence=0.95,
        )
        assert result.decision == "不符合"
        assert len(result.evidence) == 1

    def test_llm_output_defaults(self):
        out = LLMOutput()
        assert out.tool_calls == []
        assert out.evidence == []
        assert out.not_found is False
        assert out.confidence == 0.0

    def test_clarification_response(self):
        resp = ClarificationResponse(
            needs_clarification=True,
            missing=["时间范围"],
            suggestions=["请补充时间范围。"],
        )
        assert resp.needs_clarification is True

    def test_not_found_response(self):
        resp = NotFoundResponse()
        assert resp.not_found is True
        assert resp.decision == "无法判断"
        assert resp.confidence == 0.0

    def test_tool_call_log(self):
        log = ToolCallLog(
            query_id="q-001",
            round=1,
            tool_name="arithmetic_compare",
            args={"actual": 800, "operator": "gt", "threshold": 760},
            result={"success": True, "data": {"result": True}},
            timestamp=datetime.now().isoformat(),
            latency_ms=12.5,
        )
        assert log.round == 1

    def test_document_upload_response(self):
        resp = DocumentUploadResponse(
            doc_id="doc-001",
            file_name="规则.pdf",
            page_count=45,
            chunk_count=120,
            uploaded_at=datetime.now().isoformat(),
        )
        assert resp.page_count == 45


class TestRuleReviewAppRegistration:
    """验证 app.py 正确注册了规则审查路由。"""

    def test_root_endpoint_includes_rule_review_service(self):
        from fastapi.testclient import TestClient
        from app import app

        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "services" in data
        assert "rule_review" in data["services"]
        assert data["services"]["rule_review"]["path"] == "/v1/rule-review"
        assert "GET /v1/rule-review" in data["services"]["rule_review"]["endpoints"]

    def test_rule_review_status_endpoint(self):
        from fastapi.testclient import TestClient
        from app import app

        client = TestClient(app)
        response = client.get("/v1/rule-review")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "rule_review"
        assert data["status"] == "infrastructure_ready"

    def test_rule_review_router_registered(self):
        from app import app

        paths = []
        for route in app.routes:
            if hasattr(route, "path"):
                paths.append(route.path)
            # FastAPI include_router 会产生 _IncludedRouter，
            # 需要再展开其原始 router 中的路径。
            original_router = getattr(route, "original_router", None)
            if original_router is not None:
                paths.extend(
                    getattr(r, "path", "") for r in original_router.routes
                )

        assert any("/v1/rule-review" in p for p in paths)


class TestRuleReviewPackageInit:
    """验证 src/rule_review/__init__.py 的副作用符合预期。"""

    def test_data_directories_created(self):
        assert os.path.isdir(settings.RULE_DOCUMENTS_DIR)
        assert os.path.isdir(settings.RULE_INDEX_DIR)

    def test_version_defined(self):
        assert __version__ == "0.1.0"
