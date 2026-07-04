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
import json


class ParameterAgent:
    """通用参数提取Agent
    
    从用户问题中提取业务查询所需的参数
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
        self.base_prompt_len = 2000  # 预估值

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

    def _build_prompt(self, prompt_template: str, user_query: str, intent_result: dict) -> str:
        """构建参数提取提示词
        
        Args:
            prompt_template: 提示词模板
            user_query: 用户问题
            intent_result: 意图识别结果
        
        Returns:
            完整的提示词
        """
        return prompt_template.replace(
            "__USER_QUERY__", user_query
        ).replace(
            "__INTENT_RESULT__", json.dumps(intent_result, ensure_ascii=False)
        )

    def _build_config(self, context: dict | None) -> RunnableConfig | None:
        thread_id = (context or {}).get("thread_id")
        if not thread_id:
            return None
        return {"configurable": {"thread_id": str(thread_id)}}

    def invoke(
        self,
        user_query: str,
        intent_result: dict,
        prompt_template: str,
        context: dict | None = None,
    ) -> Any:
        """同步非流式调用方法
        
        Args:
            user_query: 用户问题
            intent_result: 意图识别结果
            prompt_template: 提示词模板
            context: 上下文信息
        
        Returns:
            提取的参数字典
        """
        agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=self._build_prompt(prompt_template, user_query, intent_result),
            middleware=self.middleware,
            checkpointer=self.checkpointer,
        )
        config = self._build_config(context)
        messages = [HumanMessage(content=user_query)]

        result = agent.invoke({"messages": messages}, config=config)
        
        # 提取最后一条消息的解析过的内容
        if isinstance(result, dict) and 'messages' in result:
            return result['messages'][-1].content
        elif isinstance(result, list) and len(result) > 0:
            return result[-1].content if hasattr(result[-1], 'content') else str(result[-1])
        return result

    async def ainvoke(
        self,
        user_query: str,
        intent_result: dict,
        prompt_template: str,
        context: dict | None = None,
    ) -> Any:
        """异步非流式调用方法

        Args:
            user_query: 用户问题
            intent_result: 意图识别结果
            prompt_template: 提示词模板
            context: 上下文信息
        
        Returns:
            提取的参数字典
        """
        agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=self._build_prompt(prompt_template, user_query, intent_result),
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

