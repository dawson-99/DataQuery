import json
import re
from typing import Any, AsyncGenerator

# from exceptiongroup import catch
from uvicorn.main import logger
from src.utils.filter_think_tags import filter_think_tags_async, filter_think_tags_simple, filter_think_tags_async_v2

import aiohttp

from src.agents import prompts
from src.config import settings


class InnerModelAgent:
    """通过流式 API 调用模型并按片段返回内容。"""

    def __init__(
        self,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self.model = settings.INNER_MODEL
        self.url = settings.INNER_MODEL_URL.rstrip("/")
        self.authorization = settings.INNER_MODEL_AUTH_TOKEN
        self._session = session

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            client_timeout = aiohttp.ClientTimeout(total=settings.INNER_MODEL_TIMEOUT)
            self._session = aiohttp.ClientSession(timeout=client_timeout)
        return self._session


    @property
    def api_url(self) -> str:
        return self.url

    async def astream(
        self,
        messages: list[dict[str, Any]]
    ) -> AsyncGenerator[str, None]:
        """异步流式调用模型"""

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            # "temperature": 0.6,
            "max_tokens": 8000,
        }
        headers = {
            "Content-Type": "application/json;charset=utf-8",
            "Authorization": self.authorization,
        }

        session = await self._get_session()


        async def _raw_stream() -> AsyncGenerator[str, None]:
            try:
                async with session.post(self.api_url, headers=headers, json=payload) as response:
                    response.raise_for_status()

                    async for raw_line in response.content:
                        line = raw_line.decode("utf-8", errors="ignore").strip()
                        if not line or not line.startswith("data:"):
                            continue

                        data = line[5:].strip()
                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(self._fix_invalid_json_string(data))
                        except json.JSONDecodeError:
                            # 输出包含 '\n'（单个换行符）→ 非法 JSON → 必须用 .replace('\n', '\\n')
                            # logger.info(f"json解析失败：repr(data): {repr(data)}")
                            logger.warning(f"invalid json: {data}")

                            continue

                        text = self._extract_content(chunk)
                        if text:
                            yield text
            except aiohttp.ClientResponseError as e:
                logger.error(f"HTTP 请求失败：状态码={e.status}, 信息={e.message}")
                yield f"抱歉，数据查询遇到一点小问题，暂时无法为您返回结果，请您稍后再试～"

            # 捕获网络连接、超时、DNS 错误
            except aiohttp.ClientError as e:
                logger.error(f"网络请求异常：{str(e)}")
                yield f"抱歉，数据查询遇到一点小问题，暂时无法为您返回结果，请您稍后再试～  "

            # 捕获其他未知异常
            except Exception as e:
                logger.error(f"流式请求未知异常：{str(e)}", exc_info=True)
                yield f"抱歉，数据查询遇到一点小问题，暂时无法为您返回结果，请您稍后再试～  "

        async for filtered_text in filter_think_tags_async_v2(_raw_stream()):
            if filtered_text:
                yield filtered_text

    async def ainvoke(self, messages: list[dict[str, Any]]) -> str:
        """异步非流式调用模型，并返回完整文本。"""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            # "temperature": 0.6,
            "max_tokens": 8000,
        }
        headers = {
            "Content-Type": "application/json;charset=utf-8",
            "Authorization": self.authorization,
        }

        session = await self._get_session()
        try:
            async with session.post(self.api_url, headers=headers, json=payload) as response:
                response.raise_for_status()
                result = await response.json(content_type=None)

            return filter_think_tags_simple(self._extract_message_content(result))

        except aiohttp.ClientResponseError as e:
            logger.error(f"HTTP 请求失败：状态码={e.status}, 信息={e.message}")
            return f"抱歉，数据查询遇到一点小问题，暂时无法为您返回结果，请您稍后再试～  "

        # 捕获网络连接、超时、DNS 错误
        except aiohttp.ClientError as e:
            logger.error(f"网络请求异常：{str(e)}")
            return f"抱歉，数据查询遇到一点小问题，暂时无法为您返回结果，请您稍后再试～  "

        # 捕获其他未知异常
        except Exception as e:
            logger.error(f"流式请求未知异常：{str(e)}", exc_info=True)
            return f"抱歉，数据查询遇到一点小问题，暂时无法为您返回结果，请您稍后再试～  "

    async def ainvoke_for_echarts(
        self,
        user_query: str = "",
        data: str = "",
    ) -> str:
        messages: list[dict[str, Any]] = []
        echarts_prompt = prompts.ECharts_Generation_Prompt
        content = f"{echarts_prompt}\n\nUSER_QUERY={user_query}\nQUERY_DATA={data}\n</no thinking>"
        messages.append({"role": "user", "content": content})
        return await self.ainvoke(messages)


    async def ainvoke_for_format(
        self,
        user_query: str,
        data: str,
        prompt_template: str,
    ) -> str:
        messages: list[dict[str, Any]] = []

        content = f"{prompt_template}\n\nUSER_QUERY={user_query}\nINPUT_DATA:{data}</no thinking>"
        messages.append({"role": "user", "content": content})
        return await self.ainvoke(messages)


    async def astream_for_format(
        self,
        user_query: str,
        data: str,
        prompt_template: str,
    ) -> AsyncGenerator[str, None]:
        """异步流式调用模型"""
        messages: list[dict[str, Any]] = []

        content = f"{prompt_template}\n\nUSER_QUERY={user_query}\nINPUT_DATA:{data}</no thinking>"
        messages.append({"role": "user", "content": content})

        async for chunk in self.astream(messages):
            yield chunk

    @staticmethod
    def _extract_content(chunk: dict[str, Any]) -> str:
        """从流式返回块中提取 delta.content。"""
        choices = chunk.get("choices")
        if not choices:
            return ""

        if not isinstance(choices, list):
            return ""

        first_choice = choices[0] if choices else None
        if not isinstance(first_choice, dict):
            return ""

        delta = first_choice.get("delta")
        if not isinstance(delta, dict):
            return ""

        content = delta.get("content")
        return content if isinstance(content, str) else ""

    @staticmethod
    def _extract_message_content(result: dict[str, Any]) -> str:
        """从非流式返回结果中提取 message.content。"""
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            return ""

        message = first_choice.get("message")
        if not isinstance(message, dict):
            return ""

        content = message.get("content")
        return content if isinstance(content, str) else ""

    @staticmethod
    def _fix_invalid_json_string(s: str) -> str:
        return s
        # return (
        # s
        # .replace('\\', '\\\\')   # 先转义反斜杠（重要！）
        # .replace('\n', '\\n')
        # .replace('\r', '\\r')
        # .replace('\t', '\\t')
        # .replace('\b', '\\b')
        # .replace('\f', '\\f')
    # )