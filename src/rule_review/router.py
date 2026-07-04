"""
电力规则审查系统 - API 路由

当前为基础设施占位路由，具体端点将在后续功能中逐步实现：
- POST /v1/rule-review          规则审查主接口
- POST /v1/rule-review/documents 上传 PDF
- GET  /v1/rule-review/documents 列出已入库文档
- DELETE /v1/rule-review/documents/{doc_id} 删除文档
"""

from fastapi import APIRouter

router = APIRouter(prefix="/v1/rule-review", tags=["规则审查"])


@router.get("", summary="规则审查服务状态")
async def rule_review_status():
    """基础设施占位接口，用于验证路由已正确注册。"""
    return {
        "service": "rule_review",
        "status": "infrastructure_ready",
        "version": "0.1.0",
    }
