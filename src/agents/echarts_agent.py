from collections.abc import Generator, AsyncGenerator
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_qwq import ChatQwen
from langchain_core.runnables import RunnableConfig
from src.agents import prompts
from src.config import settings


class EChartsAgent:
    def __init__(
        self, 
        model: ChatQwen,
    ) -> None:
        self.model = model
        self._system_prompt = prompts.ECharts_Generation_Prompt
        self.base_prompt_len = len(self._system_prompt)


    def _build_config(self, context: dict | None) -> RunnableConfig | None:
        thread_id = (context or {}).get("thread_id")
        if not thread_id:
            return None
        return {"configurable": {"thread_id": str(thread_id)}}

    async def ainvoke(
        self,
        user_query: str,
        data: str,
        context: dict | None = None,
    ) -> str:
        """异步非流式调用方法

        Args:
            content: 用户输入内容
            context: 上下文信息

        Returns:
            Any: 解析后的结果
        """
        config = self._build_config(context)
        content = f"USER_QUERY={user_query}\nQUERY_DATA={data}"

        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=content)
        ]
        result = await self.model.ainvoke(messages, config=config)

        # 兼容：Agent返回dict / 模型返回AIMessage
        if isinstance(result, dict) and 'messages' in result:
            return result['messages'][-1].content
        elif hasattr(result, 'content'):
            return result.content
        elif isinstance(result, list) and len(result) > 0:
            return result[-1].content if hasattr(result[-1], 'content') else str(result[-1])

        return str(result)

    async def astream(
        self,
        user_query: str,
        data: str,
        context: dict | None = None,
    ) -> AsyncGenerator[str, None]:
        """异步流式调用方法

        Args:
            content: 用户输入内容
            context: 上下文信息（含thread_id等）
            stream_mode: 流式模式

        Yields:
            str: 流式输出的内容片段
        """
        config = self._build_config(context)
        content = f"USER_QUERY={user_query}\nQUERY_DATA={data}"

        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=content)
        ]
        async for message, meta in self.model.astream(input={"messages": messages}, stream_mode="messages",
                                                       config=config):
            if hasattr(message, 'content') and message.content:
                yield message.content
