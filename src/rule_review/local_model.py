"""
电力规则审查系统 - 本地模型配置（Phase 3）

提供 vLLM 本地部署模型的客户端封装：
- vLLM OpenAI-compatible API 客户端
- 模型注册表（Qwen3B、DeepSeek-v4 等本地部署实例）
- 健康检查与模型切换
- 与现有 ProxyChatModel / ChatQwen 同接口，无缝替换

部署方式：
  docker run --gpus all -p 8000:8000 \\
    vllm/vllm-openai:latest \\
    --model Qwen/Qwen3-32B \\
    --tensor-parallel-size 2

配置（.env）:
  VLLM_BASE_URL=http://localhost:8000/v1
  VLLM_MODEL_NAME=Qwen/Qwen3-32B
  VLLM_API_KEY=not-needed
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncGenerator

from src.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 模型注册表
# ---------------------------------------------------------------------------


@dataclass
class LocalModelConfig:
    """本地模型部署配置。"""
    name: str
    base_url: str
    model_name: str  # vLLM 中注册的模型名
    api_key: str = "not-needed"
    max_tokens: int = 4096
    temperature: float = 0.1
    top_p: float = 0.9


# 预定义模型配置
LOCAL_MODEL_REGISTRY: dict[str, LocalModelConfig] = {
    "qwen3-32b": LocalModelConfig(
        name="qwen3-32b",
        base_url="http://localhost:8000/v1",
        model_name="Qwen/Qwen3-32B",
        max_tokens=4096,
        temperature=0.1,
    ),
    "deepseek-v4": LocalModelConfig(
        name="deepseek-v4",
        base_url="http://localhost:8001/v1",
        model_name="deepseek-ai/DeepSeek-V4",
        max_tokens=4096,
        temperature=0.0,
    ),
    "bge-m3": LocalModelConfig(
        name="bge-m3",
        base_url="http://localhost:8002/v1",
        model_name="BAAI/bge-m3",
        max_tokens=512,
        temperature=0.0,
    ),
}


# ---------------------------------------------------------------------------
# vLLM 客户端（OpenAI-compatible）
# ---------------------------------------------------------------------------


class VLLMClient:
    """vLLM OpenAI-compatible API 客户端。

    接口与现有的 ProxyChatModel / ChatQwen 保持一致，
    支持 ainvoke（非流式）和 astream（流式）。

    作为现有模型调用路径的第四条路径：
        DashScope 直连 → 网关代理 → 内网模型直连 → vLLM 本地部署
    """

    def __init__(self, config: LocalModelConfig | None = None):
        """
        Args:
            config: 本地模型配置，None 时从环境变量构建。
        """
        if config is None:
            config = self._config_from_env()

        self.config = config
        self._http_client = None
        self._available: bool | None = None  # None = 未检查

    @staticmethod
    def _config_from_env() -> LocalModelConfig:
        return LocalModelConfig(
            name=settings.VLLM_MODEL_NAME or "vllm-local",
            base_url=settings.VLLM_BASE_URL or "http://localhost:8000/v1",
            model_name=settings.VLLM_MODEL_NAME or "",
            api_key=settings.VLLM_API_KEY or "not-needed",
        )

    @property
    def is_available(self) -> bool:
        """检查 vLLM 服务是否可用（缓存结果）。"""
        if self._available is not None:
            return self._available
        return False  # 需要显式调用 check_health() 后才准确

    async def check_health(self) -> bool:
        """检查 vLLM 服务健康状态。"""
        import aiohttp

        url = f"{self.config.base_url.rstrip('/')}/models"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    self._available = resp.status == 200
                    if self._available:
                        data = await resp.json()
                        models = [m.get("id", "") for m in data.get("data", [])]
                        logger.info(
                            "[vLLM] 健康检查通过: %s, 可用模型: %s",
                            url, models,
                        )
                    return self._available
        except Exception as e:
            self._available = False
            logger.warning("[vLLM] 健康检查失败: %s", e)
            return False

    async def ainvoke(self, messages: list[dict], **kwargs) -> Any:
        """非流式调用。"""
        import aiohttp

        url = f"{self.config.base_url.rstrip('/')}/chat/completions"

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "stream": False,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"vLLM 请求失败 (HTTP {resp.status}): {text[:500]}")

                data = await resp.json()
                content = data["choices"][0]["message"]["content"]
                return _create_ai_message(content)

    async def astream(self, messages: list[dict], **kwargs) -> AsyncGenerator[Any, None]:
        """流式调用。"""
        import aiohttp

        url = f"{self.config.base_url.rstrip('/')}/chat/completions"

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "stream": True,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"vLLM 流式请求失败 (HTTP {resp.status}): {text[:500]}")

                async for line in resp.content:
                    line_str = line.decode("utf-8").strip()
                    if not line_str or not line_str.startswith("data: "):
                        continue
                    data_str = line_str[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield _create_ai_message_chunk(content)
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue


def _create_ai_message(content: str) -> Any:
    """创建 AIMessage（兼容 LangChain）。"""
    try:
        from langchain_core.messages import AIMessage
        return AIMessage(content=content)
    except ImportError:
        # Fallback: 简单 dict
        return type("FakeAIMessage", (), {"content": content})()


def _create_ai_message_chunk(content: str) -> Any:
    """创建 AIMessageChunk（兼容 LangChain）。"""
    try:
        from langchain_core.messages import AIMessageChunk
        return AIMessageChunk(content=content)
    except ImportError:
        return type("FakeAIMessageChunk", (), {"content": content})()


# ---------------------------------------------------------------------------
# 默认工厂
# ---------------------------------------------------------------------------


_default_vllm_client: VLLMClient | None = None


def get_default_vllm_client() -> VLLMClient:
    global _default_vllm_client
    if _default_vllm_client is None:
        _default_vllm_client = VLLMClient()
    return _default_vllm_client
