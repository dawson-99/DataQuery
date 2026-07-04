from pydantic import BaseModel
from typing import Optional

class ChatMessageRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    user: Optional[str] = None
    api_base_url: Optional[str] = None
    stream: bool = True
    enable_chart: bool = True  # 是否启用图表生成功能, 默认为 True

class InterProvincialQueryRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    api_base_url: Optional[str] = None
    stream: bool = True

class QueryRequest(BaseModel):
    """通用智能查询请求"""
    # query: str
    # conversation_id: Optional[str] = None
    # user: Optional[str] = None
    # stream: bool = True


    question: str
    aiModel: str
    sessionId: str
    showThinkProcess: bool = False
    platform: Optional[str] = None
    userInfo: Optional[dict] = None
    stream: bool = True

class SSEPayload(BaseModel):
    """统一 SSE Payload 结构"""
    conversation_id: str
    data: dict

    @classmethod
    def message(cls, conversation_id: str, content: str):
        return cls(conversation_id=conversation_id, data={"content": content})

    @classmethod
    def echart(cls, conversation_id: str, content: str):
        return cls(conversation_id=conversation_id, data={"content": content})

    @classmethod
    def error(cls, conversation_id: str, content: str):
        return cls(conversation_id=conversation_id, data={"content": content})

    @classmethod
    def done(cls, conversation_id: str):
        return cls(conversation_id=conversation_id, data={})

class UnifiedQueryResponse(BaseModel):
    """统一非流式响应结构"""
    status: str
    conversation_id: str
    data: dict = {}
    message: Optional[str] = None


class ReturnQuery(BaseModel):
    answer: str = ""
    suggestions: Optional[list] = None
    out_of_scope: bool = False
    isEnd: bool = False
    conversationId: str = ""  # sessionId

    isNotProcess: bool = True

    traceId: str = ""
    appId: str = ""
    globalTraceId: str = ""
    metadata: Optional[dict] = None
    messageId: str = ""
    isSensitiveWord: bool = False
    interruptId: Optional[dict] = None
    isMultiConversation: bool = False

    echart_holder: bool = False

