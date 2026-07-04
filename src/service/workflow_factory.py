from typing import Any, Dict, Optional

from langchain_qwq import ChatQwen

from src.config import settings
from src.utils.model_proxy import ProxyChatModel
from src.workflow.workflow_router import WorkflowRouter
from src.service.session_manager import session_registry
from src.utils.logging_setup import logger


class WorkflowFactory:
    """工作流工厂类，职责：创建和返回各种工作流，并持有进程级共享的模型实例"""

    def __init__(self):
        # 进程级共享模型实例（懒加载），避免每个请求独立创建 httpx 客户端
        self._intent_model: Optional[ChatQwen] = None
        self._parameter_model: Optional[ChatQwen] = None
        self._format_model: Optional[ChatQwen] = None
        self._problem_model: Optional[ChatQwen] = None
        self._echarts_model: Optional[ChatQwen] = None
        self._trend_analysis_model: Optional[ChatQwen] = None

    @staticmethod
    def _create_chat_model(model_name: str, api_key: str, base_url: str) -> Any:
        if model_name in getattr(settings, 'GATEWAY_MODELS', []):
            return ProxyChatModel(
                model=model_name,
                base_url=settings.GATEWAY_BASE_URL,
                enable_thinking=False,
                timeout=getattr(settings, 'REQUEST_TIMEOUT_SECONDS', 180),
            )
        return ChatQwen(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            enable_thinking=False,
            timeout=getattr(settings, 'REQUEST_TIMEOUT_SECONDS', 60)
        )

    def _get_intent_model(self) -> ChatQwen:
        if self._intent_model is None:
            self._intent_model = self._create_chat_model(
                settings.INTENT_MODEL, settings.INTENT_API_KEY, settings.INTENT_API_BASE
            )
        return self._intent_model

    def _get_parameter_model(self) -> ChatQwen:
        if self._parameter_model is None:
            self._parameter_model = self._create_chat_model(
                settings.PARAMETER_MODEL, settings.PARAMETER_API_KEY, settings.PARAMETER_API_BASE
            )
        return self._parameter_model

    def _get_format_model(self) -> ChatQwen:
        if self._format_model is None:
            self._format_model = self._create_chat_model(
                settings.FORMAT_MODEL, settings.FORMAT_API_KEY, settings.FORMAT_API_BASE
            )
        return self._format_model

    def _get_problem_model(self) -> ChatQwen:
        if self._problem_model is None:
            self._problem_model = self._create_chat_model(
                settings.PROBLEM_MODEL, settings.PROBLEM_API_KEY, settings.PROBLEM_API_BASE
            )
        return self._problem_model

    def _get_echarts_model(self) -> ChatQwen:
        if self._echarts_model is None:
            self._echarts_model = self._create_chat_model(
                settings.ECHARTS_MODEL, settings.ECHARTS_API_KEY, settings.ECHARTS_API_BASE
            )
        return self._echarts_model

    def _get_trend_analysis_model(self) -> ChatQwen:
        if self._trend_analysis_model is None:
            self._trend_analysis_model = self._create_chat_model(
                settings.TREND_ANALYSIS_MODEL,
                settings.TREND_ANALYSIS_API_KEY,
                settings.TREND_ANALYSIS_API_BASE,
            )
        return self._trend_analysis_model

    async def create_workflow(self, conversation_id: Optional[str] = None) -> WorkflowRouter:
        """创建工作流路由器，注入进程级共享模型"""
        cid = await session_registry.get_or_create_session(conversation_id)
        logger.info(f"[WorkflowFactory] 创建 WorkflowRouter: conversation_id={cid}")
        return WorkflowRouter(cid, shared_models=self)


workflow_factory = WorkflowFactory()
