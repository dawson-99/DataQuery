"""
电力规则审查系统 - Pydantic 数据模型

遵循设计文档 §6 数据模型设计，支撑 /v1/rule-review 接口的请求/响应。
"""

from typing import Optional

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    """审查结论引用的单条规则证据"""

    source: str = Field(default="", description="文档名称")
    section: str = Field(default="", description="章节标题")
    page: int = Field(default=0, description="页码", ge=0)
    text: str = Field(default="", description="原文引用")
    chunk_id: str = Field(default="", description="来源 chunk ID")


class RuleReviewRequest(BaseModel):
    """规则审查查询请求"""

    question: str = Field(description="用户输入的自然语言规则审查问题")
    stream: bool = Field(default=True, description="是否使用 SSE 流式输出")
    sessionId: str = Field(default="", description="会话唯一标识")
    userInfo: Optional[dict] = Field(default=None, description="用户信息")
    top_k: int = Field(default=10, ge=1, le=50, description="送入 LLM 的 chunk 数量")


class RuleReviewResult(BaseModel):
    """最终审查结果"""

    decision: str = Field(
        description="审查结论", examples=["符合", "不符合", "部分符合", "无法判断"]
    )
    reason: str = Field(description="推理过程")
    evidence: list[EvidenceItem] = Field(default_factory=list, description="证据列表")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="置信度 0.0-1.0"
    )


class LLMOutput(BaseModel):
    """LLM 输出 JSON 结构（含可选 tool_calls，Phase 2 使用）"""

    decision: str = ""
    reason: str = ""
    evidence: list[EvidenceItem] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    tool_calls: list[dict] = Field(default_factory=list)
    not_found: bool = Field(default=False, description="是否在文档中未找到相关规则")
    tool_unsolved: bool = Field(default=False, description="工具调用是否未在限定轮次内解决")
    tool_unsolved_reason: str = ""
    judge_skipped: bool = Field(default=False, description="是否跳过 Judge 校验")
    judge_skipped_reason: str = ""


class ToolCallLog(BaseModel):
    """工具调用日志（调试 + 训练数据收集）"""

    query_id: str
    round: int = Field(ge=1, description="第几轮 LLM 推理")
    tool_name: str
    args: dict
    result: dict
    timestamp: str
    latency_ms: float = Field(ge=0.0)


class DocumentUploadResponse(BaseModel):
    """文档上传响应"""

    doc_id: str
    file_name: str
    page_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    uploaded_at: str


class ClarificationResponse(BaseModel):
    """问题澄清响应"""

    needs_clarification: bool
    missing: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class NotFoundResponse(BaseModel):
    """检索无结果兜底响应"""

    decision: str = "无法判断"
    reason: str = "规则文档库中未检索到与您问题相关的规则内容，无法进行审查判断。请确认是否已上传相关规则文档。"
    evidence: list[EvidenceItem] = Field(default_factory=list)
    confidence: float = 0.0
    not_found: bool = True
