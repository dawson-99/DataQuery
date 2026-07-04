from src.agents import prompts

from typing import Any
from collections.abc import Generator, AsyncGenerator
from langchain_core.messages import HumanMessage
from langchain_qwq import ChatQwen
from src.utils.output_parser import parse_json_block
from langchain.agents.middleware import before_model, after_model
from langchain.agents import AgentState, create_agent
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from src.agents.utils import trim_messages_by_length


class IntentAgent:
    """通用意图识别Agent
    
    识别用户意图，判断具体的业务场景：
    例如
    - 交易结果查询：交易结果信息数据的查询
    """
    
    def __init__(self, model: ChatQwen, prompt_template: str, middleware: list | None = None) -> None:
        """
        Args:
            model: LLM模型实例
            prompt_template: 意图识别提示词模板
            middleware: 中间件列表
        """
        self.model = model
        self.prompt_template = prompt_template
        self.tools: list = []
        self.checkpointer = InMemorySaver()
        self.base_prompt_len = len(prompt_template)

        @before_model
        def dynamic_trim(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
            return trim_messages_by_length(state, self.base_prompt_len)

        @after_model
        def parse_block(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
            """在模型输出后，统一解析 JSON"""
            if not state.get("messages"):
                return None
            last_message = state["messages"][-1]
            content = getattr(last_message, "content", None)
            if not isinstance(content, str):
                return None

            parsed = parse_json_block(content)
            if parsed is None:
                return None

            # ✅ 不覆盖 content，存入独立字段
            return {"parsed_output": parsed}

        self.middleware = [dynamic_trim, parse_block] if middleware is None else middleware + [dynamic_trim, parse_block]

    def _build_prompt(self, context: dict | None) -> str:
        """构建提示词"""
        return self.prompt_template

    def _build_config(self, context: dict | None) -> RunnableConfig | None:
        thread_id = (context or {}).get("thread_id")
        if not thread_id:
            return None
        return {"configurable": {"thread_id": str(thread_id)}}

    def invoke(
        self,
        content: str,
        context: dict | None = None,
    ) -> Any:
        """同步非流式调用方法
        
        注意：@after_model 中间件已经处理了 JSON 解析，无需在此重复解析
        """
        agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=self._build_prompt(context),
            middleware=self.middleware,
            checkpointer=self.checkpointer,
        )
        config = self._build_config(context)
        messages = [HumanMessage(content=content)]

        result = agent.invoke({"messages": messages}, config=config)
        
        # 提取最后一条消息的解析过的内容
        if isinstance(result, dict) and 'messages' in result:
            return result['messages'][-1].content
        elif isinstance(result, list) and len(result) > 0:
            return result[-1].content if hasattr(result[-1], 'content') else str(result[-1])
        return result

    def stream(
        self,
        content: str,
        context: dict | None = None,
        stream_mode: str = "mes   sages",
    ) -> Generator[str, None, None]:
        """同步流式调用方法"""
        agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=self._build_prompt(context),
            middleware=self.middleware,
            checkpointer=self.checkpointer,
        )
        config = self._build_config(context)
        messages = [HumanMessage(content=content)]

        for message, meta in agent.stream(input={"messages": messages}, stream_mode=stream_mode, config=config):
            if hasattr(message, 'content') and message.content:
                yield message.content

    
    async def ainvoke(
        self,
        content: str,
        context: dict | None = None,
    ) -> Any:
        """异步非流式调用方法

        Args:
            content: 用户输入内容
            context: 上下文信息
        
        Returns:
            Any: 解析后的结果
        """
        agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=self._build_prompt(context),
            middleware=self.middleware,
            checkpointer=self.checkpointer,
        )
        config = self._build_config(context)
        messages = [HumanMessage(content=content)]

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
        content: str,
        context: dict | None = None,
        stream_mode: str = "messages",
    ) -> AsyncGenerator[str, None]:
        """异步流式调用方法

        Args:
            content: 用户输入内容
            context: 上下文信息（含thread_id等）
            stream_mode: 流式模式
        
        Yields:
            str: 流式输出的内容片段
        """
        agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=self._build_prompt(context),
            middleware=self.middleware,
            checkpointer=self.checkpointer,
        )
        config = self._build_config(context)
        messages = [HumanMessage(content=content)]

        async for message, meta in agent.astream(input={"messages": messages}, stream_mode=stream_mode, config=config):
            if hasattr(message, 'content') and message.content:
                yield message.content

