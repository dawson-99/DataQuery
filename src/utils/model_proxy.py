import json
import httpx
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
    AIMessageChunk,
)
from langchain_core.runnables import RunnableConfig

class ProxyChatModel:
    """模型代理：代替 ChatQwen 向独立部署的请求中转服务发起调用。"""

    def __init__(
        self,
        model: str,
        api_key: str = "",
        base_url: str = "",
        enable_thinking: bool = False,
        timeout: float = 180.0,
        **kwargs,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.enable_thinking = enable_thinking
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._bound_tools = None
        self._bound_tool_kwargs = {}

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
            self._client = httpx.AsyncClient(
                limits=limits,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    def _build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _convert_message(msg: BaseMessage) -> Dict[str, Any]:
        """将 LangChain 消息对象转换为可 JSON 序列化的字典"""
        if isinstance(msg, HumanMessage):
            return {"role": "user", "content": msg.content}
        elif isinstance(msg, AIMessage):
            return {"role": "assistant", "content": msg.content}
        elif isinstance(msg, SystemMessage):
            return {"role": "system", "content": msg.content}
        elif isinstance(msg, ToolMessage):
            return {"role": "tool", "content": msg.content}
        else:
            # 回退：尝试使用 .content 属性
            return {"role": "user", "content": str(msg.content)}

    @staticmethod
    def _convert_messages(messages: List[Any]) -> List[Dict[str, Any]]:
        """处理可能混合了字典和 LangChain 消息的列表"""
        converted = []
        for m in messages:
            if isinstance(m, BaseMessage):
                converted.append(ProxyChatModel._convert_message(m))
            elif isinstance(m, dict):
                converted.append(m)
            else:
                converted.append({"role": "user", "content": str(m)})
        return converted

    def _build_request_body(self, messages: List[Any], **kwargs) -> Dict:
        body = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 4096),
            "stream": kwargs.get("stream", False),
        }
        if self.enable_thinking:
            body["enable_thinking"] = True
        if self._bound_tools:
            body["tools"] = self._bound_tools
            body["tool_choice"] = "auto"
        return body

    # ---------- bind / with_structured_output ----------
    def bind(self, *args, **kwargs) -> "ProxyChatModel":
        clone = ProxyChatModel(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            enable_thinking=self.enable_thinking,
            timeout=self.timeout,
        )
        clone._client = self._client
        if args:
            clone._bound_tools = args[0] if isinstance(args[0], list) else [args[0]]
        clone._bound_tool_kwargs = kwargs
        return clone

    def with_structured_output(self, *args, **kwargs):
        return self

    # ---------- 核心调用 ----------
    async def ainvoke(
        self,
        messages: List[Any],
        config: Optional[RunnableConfig] = None,
        **kwargs,
    ) -> AIMessage:
        client = await self._get_client()
        body = self._build_request_body(messages, **kwargs)
        url = f"{self.base_url}"
        resp = await client.post(url, json=body, headers=self._build_headers())
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return AIMessage(content=content)

    async def astream(
        self,
        messages: List[Any],
        config: Optional[RunnableConfig] = None,
        **kwargs,
    ) -> AsyncGenerator[AIMessageChunk, None]:
        client = await self._get_client()
        body = self._build_request_body(messages, stream=True, **kwargs)
        url = f"{self.base_url}"
        async with client.stream("POST", url, json=body, headers=self._build_headers()) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        obj = json.loads(data_str)
                        delta = obj["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield AIMessageChunk(content=content)
                    except json.JSONDecodeError:
                        continue

    def __repr__(self):
        return f"ProxyChatModel(model={self.model})"