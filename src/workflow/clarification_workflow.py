# src/workflow/clarification_workflow.py
import json
from typing import Any, Dict, Optional
from collections.abc import AsyncGenerator
from src.workflow.base_workflow import BaseWorkflow


class ClarificationWorkflow(BaseWorkflow):
    """问题澄清工作流（待澄清场景）

    仅透传意图路由模型生成的澄清信息（clarification.message 和 suggestions）。
    不包含任何 fallback 逻辑，完全依赖模型输出。
    """

    def __init__(self, conversation_id: str, parameter_model, format_model, echarts_model=None, trend_analysis_model=None, interface_name: str = ""):
        super().__init__(
            conversation_id=conversation_id,
            parameter_model=parameter_model,
            format_model=format_model,
            echarts_model=echarts_model,
            trend_analysis_model=trend_analysis_model,
            interface_name=interface_name,
        )

    async def _call_api_impl(self, params: Dict[str, Any]) -> Any:
        return None

    def get_parameter_prompt(self) -> str:
        return ""

    def get_format_prompt(self) -> str:
        return ""

    def validate_params(self, params: Dict[str, Any]) -> bool:
        return True

    # async def execute(self, user_query: str, intent_result: Optional[Dict] = None) -> Dict[str, Any]:
    #     """执行澄清工作流：直接透传模型生成的澄清信息"""
    #     intent_result = intent_result or {}
    #
    #     clarification = intent_result.get("clarification")
    #     if not isinstance(clarification, dict) or "message" not in clarification or "suggestions" not in clarification:
    #         # 理论上模型应该总是提供有效的 clarification，如果没有，返回一个兜底提示（但不做复杂生成）
    #         clarification = {
    #             "message": "您的问题信息不足，请补充业务类型和时间范围（例如：上个月的交易结果）。",
    #             "suggestions": ["上个月的交易结果", "本月的联络线输电功率", "今天的发电总出力"]
    #         }
    #
    #     return {
    #         "intent": "待澄清",
    #         "message": {
    #             "is_fuzzy": True,
    #             "clarification": clarification,
    #             "missing_fields": intent_result.get("missing_fields"),
    #             "time_range": intent_result.get("time_range"),
    #         }
    #     }

    async def execute_stream(self, user_query: str, intent_result: Optional[Dict] = None) -> AsyncGenerator[str, None]:
        result = await self.execute(user_query, intent_result)
        yield json.dumps(result, ensure_ascii=False)