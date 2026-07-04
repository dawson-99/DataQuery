from distutils.util import strtobool
from typing import Dict, Any

import aiohttp
from langchain_qwq import ChatQwen

from src.agents.format_agent import FormatAgent
from src.agents.inner_model_agent import InnerModelAgent
from src.config import settings
from src.agents.echarts_agent import EChartsAgent
from src.utils.logging_setup import logger


class AgentFactory:
    """Agent工厂类，职责：创建和管理各种agents"""
    
    def __init__(self):
        self.innerModelEnable = bool(strtobool(settings.INNER_MODEL_ENABLE))
        self.client_timeout = aiohttp.ClientTimeout(total=settings.INNER_MODEL_TIMEOUT)
        self._session: aiohttp.ClientSession | None = None
        self.echartsAgent: EChartsAgent | None = None
        pass

    def _get_or_create_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.client_timeout)
        return self._session

    def create_echarts_agent(
        self, 
        model: ChatQwen
    ):
        """创建ECharts生成agent"""
        if self.echartsAgent is None:
            logger.debug(f"[AgentFactory] 创建 AgentForECharts")
            if self.innerModelEnable:
                session = self._get_or_create_session()
                self.echartsAgent = InnerModelAgent(session)
            else:
                self.echartsAgent = EChartsAgent(model=model)

        return self.echartsAgent

    def create_inner_model_agent(self) -> InnerModelAgent:
        session = self._get_or_create_session()
        return InnerModelAgent(session)


    def create_format_agent(self, model: ChatQwen):
        if self.innerModelEnable:
            session = self._get_or_create_session()
            return InnerModelAgent(session)
        return FormatAgent(model=model)


    async def shutdown(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()


# 全局agent工厂单例
agent_factory = AgentFactory()
