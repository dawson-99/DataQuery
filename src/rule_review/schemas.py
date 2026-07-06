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


# ---------------------------------------------------------------------------
# 审计追溯模型（设计文档 §4.7）
# ---------------------------------------------------------------------------


class RetrievalAudit(BaseModel):
    """检索阶段审计信息"""

    bm25_k: int = Field(default=0, description="BM25 召回数量")
    vector_k: int = Field(default=0, description="向量召回数量")
    sparse_k: int = Field(default=0, description="bge-m3 sparse 召回数量")
    fusion_method: str = Field(default="RRF", description="融合方法")
    final_k: int = Field(default=0, description="最终送入 LLM 的 chunk 数")
    search_expanded: bool = Field(default=False, description="是否触发了扩大搜索兜底")
    retrieval_latency_ms: float = Field(default=0.0, description="检索耗时（毫秒）")


class LLMGenerationAudit(BaseModel):
    """LLM 推理阶段审计信息"""

    model: str = Field(default="", description="使用的模型")
    tok_input: int = Field(default=0, description="输入 token 数")
    tok_output: int = Field(default=0, description="输出 token 数")
    latency_ms: float = Field(default=0.0, description="推理耗时（毫秒）")
    not_found: bool = Field(default=False, description="是否判定'文档中无相关规则'")


class JudgeAudit(BaseModel):
    """Judge 校验阶段审计信息"""

    model: str = Field(default="", description="使用的模型")
    verified: bool = Field(default=False, description="校验是否通过")
    hallucinated_count: int = Field(default=0, description="检测到的幻觉数")
    skipped: bool = Field(default=False, description="是否跳过了校验")
    skipped_reason: str = Field(default="", description="跳过原因")
    latency_ms: float = Field(default=0.0, description="校验耗时（毫秒）")


class SourceTrace(BaseModel):
    """单条溯源信息：答案中的某段结论 → 原始文档位置"""

    result_field: str = Field(default="", description="对应结果的哪个字段")
    result_excerpt: str = Field(default="", description="结果中的原文片段")
    source_doc: str = Field(default="", description="来源于哪个文档")
    source_section: str = Field(default="", description="来源于哪个章节")
    source_page: int = Field(default=0, description="来源于哪一页")
    source_text: str = Field(default="", description="原始文档中的原文")
    source_chunk_id: str = Field(default="", description="来源于哪个 chunk")
    match_type: str = Field(
        default="llm_extracted",
        description="匹配类型：exact | fuzzy | llm_extracted",
    )


class AuditRecord(BaseModel):
    """单次审查的完整审计记录"""

    query_id: str = Field(description="审查唯一标识")
    timestamp: str = Field(description="审查时间")

    # 用户输入
    original_query: str = Field(default="", description="原始问题")
    rewritten_query: str = Field(default="", description="改写后问题")

    # 检索过程
    retrieval: RetrievalAudit = Field(default_factory=RetrievalAudit)

    # LLM 推理过程
    llm_generation: LLMGenerationAudit = Field(default_factory=LLMGenerationAudit)

    # Tool 调用过程
    tool_executions: list[ToolCallLog] = Field(default_factory=list)

    # Judge 校验过程
    judge_verification: JudgeAudit | None = Field(default=None)

    # 最终输出
    final_result: dict = Field(default_factory=dict)

    # 溯源信息
    source_traceability: list[SourceTrace] = Field(default_factory=list)
