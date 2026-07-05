"""
电力规则审查系统 - API 路由

按设计文档 §5 接口设计实现：
- POST /v1/rule-review          规则审查主接口（SSE 流式 / 非流式）
- POST /v1/rule-review/documents 上传规则 PDF → 解析 → chunk → 索引
- GET  /v1/rule-review/documents 列出已入库文档
- DELETE /v1/rule-review/documents/{doc_id} 删除文档 + 清理索引
- GET  /v1/rule-review/health    健康检查
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from src.rule_review.document_store import DocumentStore
from src.rule_review.pipeline import RuleReviewPipeline, get_default_pipeline
from src.rule_review.schemas import DocumentUploadResponse, RuleReviewRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/rule-review", tags=["规则审查"])

# 支持的文件类型
_ALLOWED_EXTENSIONS = {".pdf"}
_MAX_UPLOAD_SIZE_MB = 50

# 默认单例
_pipeline: Optional[RuleReviewPipeline] = None
_document_store: Optional[DocumentStore] = None


def _get_pipeline() -> RuleReviewPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = get_default_pipeline()
    return _pipeline


def _get_document_store() -> DocumentStore:
    global _document_store
    if _document_store is None:
        _document_store = DocumentStore()
    return _document_store


# ---------------------------------------------------------------------------
# 规则审查主接口
# ---------------------------------------------------------------------------


@router.post("", summary="规则审查查询")
async def rule_review(request: RuleReviewRequest):
    """电力规则审查主接口。

    接收自然语言问题，经改写→澄清→检索→LLM推理后返回结构化审查结果。

    - **stream=true**：SSE 流式输出（默认）
    - **stream=false**：一次性返回完整结果 JSON
    """
    if request.stream:
        return StreamingResponse(
            _generate_sse(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
            },
        )

    # 非流式模式
    pipeline = _get_pipeline()
    result = await pipeline.execute(request)

    if "clarification" in result:
        return JSONResponse(content=result)

    return JSONResponse(content=result)


async def _generate_sse(request: RuleReviewRequest):
    """SSE 流式生成器，逐条 yield SSE 格式字符串。"""
    try:
        pipeline = _get_pipeline()
        async for sse_line in pipeline.execute_stream(request):
            yield sse_line
    except Exception as e:
        logger.error(f"[router] SSE 流式异常: {e}", exc_info=True)
        error_payload = json.dumps(
            {
                "type": "error",
                "answer": json.dumps(
                    {
                        "decision": "无法判断",
                        "reason": f"规则审查服务异常: {str(e)}",
                        "evidence": [],
                        "confidence": 0.0,
                    },
                    ensure_ascii=False,
                ),
            },
            ensure_ascii=False,
        )
        yield f"event: message\ndata: {error_payload}\n\n"
        yield f"event: done\ndata: {json.dumps({'done': True})}\n\n"


# ---------------------------------------------------------------------------
# 文档管理接口
# ---------------------------------------------------------------------------


@router.post("/documents", summary="上传规则文档")
async def upload_document(
    file: UploadFile = File(..., description="规则 PDF 文件"),
    force: bool = Form(default=False, description="是否覆盖同名文档"),
) -> JSONResponse:
    """上传规则 PDF 文件，自动完成解析、chunk 切分、embedding 和 FAISS 索引。

    支持文本 PDF 和扫描版 PDF（需安装 PaddleOCR）。

    返回文档元信息，包含 chunk_count 等。
    """
    # 校验文件类型
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    ext = Path(file.filename).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型 '{ext}'，仅支持: {', '.join(_ALLOWED_EXTENSIONS)}",
        )

    # 校验文件大小
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > _MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大 ({size_mb:.1f}MB)，最大允许 {_MAX_UPLOAD_SIZE_MB}MB",
        )

    # 解析并入库
    try:
        store = _get_document_store()
        resp = store.ingest(content, filename=file.filename)

        # 刷新 HybridRetriever 的 BM25 索引
        pipeline = _get_pipeline()
        pipeline.retriever.refresh_bm25()

        logger.info(
            "[router] 文档上传成功: %s, %d 页, %d chunks",
            resp.file_name, resp.page_count, resp.chunk_count,
        )
        return JSONResponse(
            content={"status": "success", "data": resp.model_dump()},
            status_code=201,
        )
    except Exception as e:
        logger.error(f"[router] 文档上传失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"文档解析失败: {str(e)}")


@router.get("/documents", summary="列出已入库文档")
async def list_documents() -> JSONResponse:
    """列出所有已入库的规则文档及其元信息。"""
    store = _get_document_store()
    docs = store.list_documents()
    return JSONResponse(
        content={
            "status": "success",
            "data": [
                {
                    "doc_id": d.doc_id,
                    "file_name": d.file_name,
                    "page_count": d.page_count,
                    "chunk_count": d.chunk_count,
                    "created_at": d.created_at,
                }
                for d in docs
            ],
        }
    )


@router.delete("/documents/{doc_id}", summary="删除规则文档")
async def delete_document(
    doc_id: str,
    force: bool = Query(default=False, description="即使不存在也不报错"),
) -> JSONResponse:
    """删除指定文档及其所有 chunk 和向量索引。"""
    store = _get_document_store()
    success = store.delete(doc_id)

    if not success:
        if force:
            return JSONResponse(
                content={"status": "ok", "deleted": False, "message": f"文档 {doc_id} 不存在或已删除"},
            )
        raise HTTPException(status_code=404, detail=f"文档 {doc_id} 不存在")

    # 刷新 BM25 索引
    pipeline = _get_pipeline()
    pipeline.retriever.refresh_bm25()

    logger.info("[router] 文档已删除: %s", doc_id)
    return JSONResponse(
        content={"status": "success", "deleted": True, "doc_id": doc_id}
    )


# ---------------------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------------------


@router.get("/health", summary="规则审查服务健康检查")
async def health_check() -> JSONResponse:
    """检查各组件状态。"""
    store = _get_document_store()
    doc_count = len(store.list_documents())
    chunk_count = len(store._chunks)

    return JSONResponse(
        content={
            "service": "rule_review",
            "status": "healthy",
            "version": "0.1.0",
            "components": {
                "document_store": {
                    "documents": doc_count,
                    "chunks": chunk_count,
                    "index_size": store._index.ntotal if store._index is not None else 0,
                },
            },
        }
    )
