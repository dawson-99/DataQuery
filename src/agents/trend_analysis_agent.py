import json
from typing import Any
from collections.abc import AsyncGenerator

from langchain_core.messages import HumanMessage
from langchain_qwq import ChatQwen

from src.agents import prompts


class TrendAnalysisAgent:
    """通用数据分析 Agent。

    - 第一阶段：根据数据概要生成可执行 Python 分析代码
    - 第二阶段：基于沙箱统计结果生成简要分析总结
    """

    def __init__(self, model: ChatQwen) -> None:
        self.model = model

    def _build_code_prompt(self, user_query: str, overview: dict[str, Any]) -> str:
        overview_json = json.dumps(overview, ensure_ascii=False, indent=2)
        return prompts.Trend_Analysis_Code_Generation_Prompt.format(
            user_query=user_query,
            data_overview=overview_json,
        )

    def _build_summary_prompt(
        self,
        user_query: str,
        overview: dict[str, Any],
        analysis_result: dict[str, Any],
    ) -> str:
        overview_json = json.dumps(overview, ensure_ascii=False, indent=2)
        analysis_result_json = json.dumps(analysis_result, ensure_ascii=False, indent=2)
        return prompts.Trend_Analysis_Summary_Prompt.format(
            user_query=user_query,
            data_overview=overview_json,
            analysis_result=analysis_result_json,
        )

    async def generate_code(self, user_query: str, overview: dict[str, Any]) -> str:
        prompt = self._build_code_prompt(user_query, overview)
        response = await self.model.ainvoke([HumanMessage(content=prompt)])
        return response.content if hasattr(response, "content") else str(response)

    async def summarize(self, user_query: str, overview: dict[str, Any], analysis_result: dict[str, Any]) -> str:
        prompt = self._build_summary_prompt(user_query, overview, analysis_result)
        response = await self.model.ainvoke([HumanMessage(content=prompt)])
        return response.content if hasattr(response, "content") else str(response)

    async def summarize_stream(
        self,
        user_query: str,
        overview: dict[str, Any],
        analysis_result: dict[str, Any],
    ) -> AsyncGenerator[str, None]:
        prompt = self._build_summary_prompt(user_query, overview, analysis_result)
        async for chunk in self.model.astream([HumanMessage(content=prompt)]):
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content
            elif isinstance(chunk, str) and chunk:
                yield chunk

