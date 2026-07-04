from typing import Any
from collections.abc import Generator, AsyncGenerator
from langchain_core.messages import HumanMessage
from langchain_qwq import ChatQwen
from langchain.agents.middleware import before_model
from langchain.agents import AgentState, create_agent
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from src.agents.utils import trim_messages_by_length


class FormatAgent:
    """通用结果格式化Agent
    
    将查询结果转换为易读的Markdown格式
    """
    
    def __init__(self, model: ChatQwen, middleware: list | None = None) -> None:
        """
        Args:
            model: LLM模型实例
            middleware: 中间件列表
        """
        self.model = model
        self.tools: list = []
        self.checkpointer = InMemorySaver()
        # 基础提示词长度（用于动态裁剪）
        self.base_prompt_len = 1000  # 预估值

        @before_model
        def dynamic_trim(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
            return trim_messages_by_length(state, self.base_prompt_len)

        self.middleware = [dynamic_trim] if middleware is None else middleware + [dynamic_trim]

    def _build_prompt(self, prompt_template: str, user_query: str, data: str) -> str:
        """构建格式化提示词
        
        Args:
            prompt_template: 提示词模板
            user_query: 用户问题
            data: 查询结果数据（JSON字符串）
        
        Returns:
            完整的提示词
        """
        return prompt_template.replace('{user_query}', user_query).replace('{data}', data)

    def _build_config(self, context: dict | None) -> RunnableConfig | None:
        thread_id = (context or {}).get("thread_id")
        if not thread_id:
            return None
        return {"configurable": {"thread_id": str(thread_id)}}

    def invoke(
        self,
        user_query: str,
        data: str,
        prompt_template: str,
        context: dict | None = None,
    ) -> Any:
        """同步非流式调用方法
        
        Args:
            user_query: 用户问题
            data: 查询结果数据（JSON字符串）
            prompt_template: 提示词模板
            context: 上下文信息
        
        Returns:
            格式化后的Markdown字符串
        """
        system_prompt = self._build_prompt(prompt_template, user_query, data)
        self.base_prompt_len = len(system_prompt)
        agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=system_prompt,
            middleware=self.middleware,
            checkpointer=self.checkpointer,
        )
        config = self._build_config(context)
        messages = [HumanMessage(content=user_query)]

        result = agent.invoke({"messages": messages}, config=config)
        
        # 提取最后一条消息的内容
        if isinstance(result, dict) and 'messages' in result:
            return result['messages'][-1].content
        elif isinstance(result, list) and len(result) > 0:
            return result[-1].content if hasattr(result[-1], 'content') else str(result[-1])
        return result

    def stream(
        self,
        user_query: str,
        data: str,
        prompt_template: str,
        context: dict | None = None,
        stream_mode: str = "messages",
    ) -> Generator[str, None, None]:
        """同步流式调用方法
        
        Args:
            user_query: 用户问题
            data: 查询结果数据（JSON字符串）
            prompt_template: 提示词模板
            context: 上下文信息
            stream_mode: 流式模式
        
        Yields:
            格式化输出的内容片段
        """
        system_prompt = self._build_prompt(prompt_template, user_query, data)
        self.base_prompt_len = len(system_prompt)
        agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=system_prompt,
            middleware=self.middleware,
            checkpointer=self.checkpointer,
        )
        config = self._build_config(context)
        messages = [HumanMessage(content=user_query)]

        for message, meta in agent.stream(input={"messages": messages}, stream_mode=stream_mode, config=config):
            if hasattr(message, 'content') and message.content:
                yield message.content

    async def ainvoke(
        self,
        user_query: str,
        data: str,
        prompt_template: str,
        context: dict | None = None,
    ) -> Any:
        """异步非流式调用方法

        Args:
            user_query: 用户问题
            data: 查询结果数据（JSON字符串）
            prompt_template: 提示词模板
            context: 上下文信息
        
        Returns:
            格式化后的Markdown字符串
        """
        system_prompt = self._build_prompt(prompt_template, user_query, data)
        # 用实际 system_prompt 长度更新，确保 trim 中间件准确计算可用空间
        self.base_prompt_len = len(system_prompt)
        agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=system_prompt,
            middleware=self.middleware,
            checkpointer=self.checkpointer,
        )
        config = self._build_config(context)
        messages = [HumanMessage(content=user_query)]

        result = await agent.ainvoke({"messages": messages}, config=config)

        # ✅ 优先从 state 中取解析结果
        if isinstance(result, dict) and 'parsed_output' in result:
            return result['parsed_output']
        # 降级处理：取最后一条消息内容（未被覆盖的原始字符串）
        if isinstance(result, dict) and 'messages' in result:
            return result['messages'][-1].content
            
        return result

    async def astream(
        self,
        user_query: str,
        data: str,
        prompt_template: str,
        context: dict | None = None,
        stream_mode: str = "messages",
    ) -> AsyncGenerator[str, None]:
        """异步流式调用方法

        Args:
            user_query: 用户问题
            data: 查询结果数据（JSON字符串）
            prompt_template: 提示词模板
            context: 上下文信息
            stream_mode: 流式模式
        
        Yields:
            格式化输出的内容片段
        """
        system_prompt = self._build_prompt(prompt_template, user_query, data)
        self.base_prompt_len = len(system_prompt)
        agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=prompt_template,
            middleware=self.middleware,
            checkpointer=self.checkpointer,
        )
        config = self._build_config(context)
        content = f"USER_QUERY={user_query}\nINPUT_DATA={data}"
        messages = [HumanMessage(content=content)]

        async for message, meta in agent.astream(input={"messages": messages}, stream_mode=stream_mode, config=config):
            if hasattr(message, 'content') and message.content:
                yield message.content

