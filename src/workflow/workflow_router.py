"""
工作流路由器 - 并发增强版
支持意图识别、子问题拆分、共享数据优化，并提供并发安全缓存和重试机制
"""

import asyncio
import copy
import json
import re
import time
from datetime import datetime, timedelta
from distutils.util import strtobool
from typing import Any, Dict, List, Optional, Tuple, AsyncGenerator

import httpx
from langchain_qwq import ChatQwen

from src.agents import prompts
from src.agents.echarts_agent import EChartsAgent
from src.agents.intent_agent import IntentAgent
from src.api.routers.shared_cache import _query_cache, sse, _cache_lock  # cache_set 已注释
from src.config import settings
from src.service.agent_factory import agent_factory
from src.utils.echarts_utils import sort_time_fields_batch, is_dense_time_series_records, \
    build_dense_time_series_echarts, has_non_zero_series, add_global_extremes_to_echarts
from src.utils.filter_think_tags import filter_think_tags_simple
from src.utils.logging_setup import logger
from src.utils.model_proxy import ProxyChatModel
from src.workflow.base_workflow import BaseWorkflow, MAX_DISPLAY_ITEMS
from src.workflow.clarification_workflow import ClarificationWorkflow
from src.workflow.emergency_dayahead_workflow import EmergencyDayaheadWorkflow
from src.workflow.question_split_workflow import QuestionSplitWorkflow
from src.workflow.rewrite_workflow import RewriteWorkflow


# ---------- 全局 HTTP 客户端连接池 ----------




# ---------- 重试装饰器（已禁用）----------
# 意图识别阶段不再自动重试，LLM 调用失败直接使用友好提示返回
# def retry_async(max_retries: int = 3, backoff_factor: float = 0.5):
#     """异步重试装饰器，指数退避"""
#     def decorator(func):
#         async def wrapper(*args, **kwargs):
#             last_exception = None
#             for attempt in range(max_retries):
#                 try:
#                     return await func(*args, **kwargs)
#                 except Exception as e:
#                     last_exception = e
#                     if attempt < max_retries - 1:
#                         wait_time = backoff_factor * (2 ** attempt)
#                         logger.warning(
#                             f"[重试] {func.__name__} 失败，{wait_time:.2f}秒后重试 ({attempt+1}/{max_retries}): {e}"
#                         )
#                         await asyncio.sleep(wait_time)
#                     else:
#                         logger.error(f"[重试] {func.__name__} 最终失败: {e}")
#             raise last_exception
#         return wrapper
#     return decorator


class WorkflowRouter:
    """工作流路由器（并发优化版）"""
    INNER_MODEL_ENABLE = bool(strtobool(settings.INNER_MODEL_ENABLE))

    # 工作流注册表
    WORKFLOW_REGISTRY: Dict[str, type] = {
        '省间应急调度交易信息（日前）查询': EmergencyDayaheadWorkflow,
        '待澄清': ClarificationWorkflow,
    }

    # API 端点映射
    API_URL_MAPPING: Dict[str, str] = {
        '省间应急调度交易信息（日前）查询': settings.EMERGENCY_DAYAHEAD_API_URL,
    }

    def __init__(self, conversation_id: str, shared_models: Any = None):
        self.conversation_id = conversation_id
        self.created_at = time.time()
        self.last_updated = time.time()

        self._api_data_cache: Dict[str, Any] = {}
        self._workflow_instances: Dict[str, BaseWorkflow] = {}
        self.current_workflow: Optional[BaseWorkflow] = None

        # 进程级共享模型（由 WorkflowFactory 注入）
        self._shared_models = shared_models

        # 模型实例延迟创建（仅在未注入共享模型时使用）
        self._intent_model: Optional[ChatQwen] = None
        self._parameter_model: Optional[ChatQwen] = None
        self._format_model: Optional[ChatQwen] = None
        self._problem_model: Optional[ChatQwen] = None
        self._echarts_model: Optional[ChatQwen] = None
        self._trend_analysis_model: Optional[ChatQwen] = None
        self._router_echarts_agent: Optional[EChartsAgent] = None

        # 意图识别 Agent 延迟初始化
        self._intent_agent: Optional[IntentAgent] = None
        self._domain_intent_agents: Dict[str, IntentAgent] = {}
        self._split_workflow: Optional[QuestionSplitWorkflow] = None

        # 问题改写工作流
        self._rewrite_workflow: Optional[RewriteWorkflow] = None

    # ---------- 模型懒加载 ----------
    async def _create_chat_model(self, model_name: str, api_key: str, base_url: str) -> Any:
        """
        根据 model_name 是否在 GATEWAY_MODELS 中，决定使用中转代理或直连 ChatQwen
        """
        if model_name in getattr(settings, 'GATEWAY_MODELS', []):
            return ProxyChatModel(
                model=model_name,
                base_url=settings.GATEWAY_BASE_URL,
                enable_thinking=False,
                timeout=getattr(settings, 'REQUEST_TIMEOUT_SECONDS', 180),
            )
        # 原有直连逻辑
        return ChatQwen(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            enable_thinking=False,
            timeout=getattr(settings, 'REQUEST_TIMEOUT_SECONDS', 60)
        )

    async def _get_intent_model(self) -> ChatQwen:
        if self._shared_models:
            return self._shared_models._get_intent_model()
        if self._intent_model is None:
            self._intent_model = await self._create_chat_model(
                settings.INTENT_MODEL, settings.INTENT_API_KEY, settings.INTENT_API_BASE
            )
        return self._intent_model

    async def _get_parameter_model(self) -> ChatQwen:
        if self._shared_models:
            return self._shared_models._get_parameter_model()
        if self._parameter_model is None:
            self._parameter_model = await self._create_chat_model(
                settings.PARAMETER_MODEL, settings.PARAMETER_API_KEY, settings.PARAMETER_API_BASE
            )
        return self._parameter_model

    async def _get_format_model(self) -> ChatQwen:
        if self._shared_models:
            return self._shared_models._get_format_model()
        if self._format_model is None:
            self._format_model = await self._create_chat_model(
                settings.FORMAT_MODEL, settings.FORMAT_API_KEY, settings.FORMAT_API_BASE
            )
        return self._format_model

    async def _get_problem_model(self) -> ChatQwen:
        if self._shared_models:
            return self._shared_models._get_problem_model()
        if self._problem_model is None:
            self._problem_model = await self._create_chat_model(
                settings.PROBLEM_MODEL, settings.PROBLEM_API_KEY, settings.PROBLEM_API_BASE
            )
        return self._problem_model

    async def _get_echarts_model(self) -> ChatQwen:
        if self._shared_models:
            return self._shared_models._get_echarts_model()
        if self._echarts_model is None:
            self._echarts_model = await self._create_chat_model(
                settings.ECHARTS_MODEL, settings.ECHARTS_API_KEY, settings.ECHARTS_API_BASE
            )
        return self._echarts_model

    async def _get_trend_analysis_model(self) -> ChatQwen:
        if self._shared_models:
            return self._shared_models._get_trend_analysis_model()
        if self._trend_analysis_model is None:
            self._trend_analysis_model = await self._create_chat_model(
                settings.TREND_ANALYSIS_MODEL,
                settings.TREND_ANALYSIS_API_KEY,
                settings.TREND_ANALYSIS_API_BASE,
            )
        return self._trend_analysis_model

    async def _get_router_echarts_agent(self) -> EChartsAgent:
        if self._router_echarts_agent is None:
            model = await self._get_echarts_model()
            self._router_echarts_agent = agent_factory.create_echarts_agent(model)
        return self._router_echarts_agent

    async def _get_intent_agent(self) -> IntentAgent:
        if self._intent_agent is None:
            model = await self._get_intent_model()
            self._intent_agent = IntentAgent(model=model, prompt_template=prompts.Unified_Intent_Recognition_Prompt)
        return self._intent_agent

    async def _get_domain_intent_agent(self, intent: str) -> Optional[IntentAgent]:
        if intent not in self._domain_intent_agents:
            prompt_map = {
                '省间应急调度交易信息（日前）查询': prompts.EmergencyDayhead_Intent_Recognition_Prompt,
            }
            template = prompt_map.get(intent)
            if template:
                model = await self._get_intent_model()
                self._domain_intent_agents[intent] = IntentAgent(model=model, prompt_template=template)
        return self._domain_intent_agents.get(intent)

    async def _get_split_workflow(self) -> QuestionSplitWorkflow:
        if self._split_workflow is None:
            model = await self._get_problem_model()
            self._split_workflow = QuestionSplitWorkflow(model=model)
        return self._split_workflow

    async def _get_rewrite_workflow(self) -> RewriteWorkflow:
        if self._rewrite_workflow is None:
            # 使用 format_model 或单独模型均可
            model = await self._get_problem_model()
            self._rewrite_workflow = RewriteWorkflow(model=model)
        return self._rewrite_workflow

    # ---------- 意图识别（重试已禁用）----------
    # @retry_async(max_retries=3, backoff_factor=0.5)
    async def _recognize_intent(self, user_query: str) -> Dict[str, Any]:
        """第一阶段：业务域路由意图识别"""
        logger.info(f"[意图识别] 第一阶段：业务域路由")
        agent = await self._get_intent_agent()
        intent_result = await agent.ainvoke(content=user_query, context={"thread_id": self.conversation_id})
        logger.info(f"[意图识别] 第一阶段结果: {intent_result}")
        parsed_result = await self._parse_intent_result(intent_result)
        parsed_result = self._fill_time_range_from_multiple_date_points(user_query, parsed_result)
        return parsed_result

    def _fill_time_range_from_multiple_date_points(self, user_query: str, intent_result: Dict[str, Any]) -> Dict[str, Any]:
        """当用户问题中包含多个时间点但未识别出 time_range 时，按时间点前后顺序回填 start/end。"""
        if not isinstance(intent_result, dict):
            return intent_result

        if intent_result.get("clarification_needed"):
            return intent_result

        intents = intent_result.get("intents")
        if not isinstance(intents, list) or not intents:
            return intent_result

        time_range = intent_result.get("time_range")
        if isinstance(time_range, dict) and time_range.get("start") and time_range.get("end"):
            return intent_result

        date_points = self._extract_date_points_from_query(user_query)
        if len(date_points) < 2:
            return intent_result

        sorted_points = sorted(date_points)
        filled = dict(intent_result)
        filled["time_range"] = {
            "start": sorted_points[0],
            "end": sorted_points[-1],
        }
        # logger.info(
        #     f"[时间兜底] 从多个时间点回填time_range: {sorted_points} -> {filled['time_range']}"
        # )
        return filled


    @staticmethod
    def _to_user_friendly_stream_error(raw_msg: Optional[str]) -> str:
        msg = (raw_msg or "").strip()
        lower_msg = msg.lower()

        if "请求超时" in msg or "timeout" in lower_msg:
            return "抱歉，数据查询遇到一点小问题，暂时无法为您返回结果，请您稍后再试～"
        if "http错误" in lower_msg or "http " in lower_msg:
            return "抱歉，数据查询遇到一点小问题，暂时无法为您返回结果，请您稍后再试～"
        if "网关业务失败" in msg or "success=false" in lower_msg:
            return "抱歉，数据查询遇到一点小问题，暂时无法为您返回结果，请您稍后再试～"
        if "api调用失败" in lower_msg or "业务api地址未配置" in msg:
            return "抱歉，数据查询遇到一点小问题，暂时无法为您返回结果，请您稍后再试～"

        return "抱歉，数据查询遇到一点小问题，暂时无法为您返回结果，请您稍后再试～"

    @staticmethod
    def _extract_date_points_from_query(user_query: str) -> List[str]:
        """从用户原句中提取多个日期点，并对缺失年份的日期做同句年份继承。"""
        if not isinstance(user_query, str) or not user_query.strip():
            return []

        pattern = re.compile(r'(?:(?P<year>20\d{2})年)?(?P<month>1[0-2]|0?[1-9])月(?P<day>3[01]|[12]\d|0?[1-9])(?:日|号)')
        matches = list(pattern.finditer(user_query))
        if not matches:
            return []

        inherited_year = None
        points: List[str] = []
        seen = set()

        for match in matches:
            year_text = match.group('year')
            month = int(match.group('month'))
            day = int(match.group('day'))

            if year_text:
                inherited_year = int(year_text)
                year = inherited_year
            else:
                if inherited_year is None:
                    continue
                year = inherited_year

            try:
                date_str = datetime(year, month, day).strftime("%Y-%m-%d")
            except ValueError:
                continue

            if date_str in seen:
                continue
            seen.add(date_str)
            points.append(date_str)

        return points

    # @retry_async(max_retries=3, backoff_factor=0.5)
    async def _recognize_domain_intent(self, intent: str, user_query: str, intent_result: Dict[str, Any]) -> Dict[str, Any]:
        """第二阶段：专项意图识别"""
        domain_agent = await self._get_domain_intent_agent(intent)
        if not domain_agent:
            return intent_result

        if not isinstance(user_query, str):
            user_query = str(user_query)

        logger.info(f"[意图识别] 第二阶段：{intent}，domain_query={user_query[:100]}")
        unique_thread_id = f"{self.conversation_id}_{intent}_{id(user_query)}"
        detail_result = await domain_agent.ainvoke(content=user_query, context={"thread_id": unique_thread_id})

        # 检查澄清标记
        raw_data = detail_result
        if isinstance(detail_result, str):
            try:
                clean_json = detail_result
                if detail_result.startswith("```"):
                    clean_json = re.sub(r'^```json\n|```$', '', detail_result, flags=re.MULTILINE)
                raw_data = json.loads(clean_json)
            except json.JSONDecodeError:
                raw_data = {}
        elif isinstance(detail_result, dict):
            raw_data = detail_result

        if isinstance(raw_data, dict) and raw_data.get("clarification_needed"):
            logger.info(f"[意图识别] 第二阶段检测到澄清触发")
            if "intents" not in raw_data:
                raw_data["intents"] = []
            return raw_data

        parsed_result = await self._parse_intent_result(detail_result)
        logger.info(f"[意图识别] 第二阶段结果: {parsed_result}")
        return parsed_result


    async def _parse_intent_result(self, result: Any) -> Dict[str, Any]:
        """解析意图识别结果（保持不变）"""
        if isinstance(result, str):
            cleaned = result.strip()
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
            if match:
                cleaned = match.group(1).strip()
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                logger.warning(f"JSON解析失败，原始内容前200字符: {result[:200]}")
                return {"intents": [], "time_range": None}
        else:
            data = result

        if not isinstance(data, dict):
            logger.warning(f"意图识别返回结构异常: {type(data)}，已按空结果处理")
            data = {}

        parsed = dict(data)

        if "question" in parsed and not isinstance(parsed["question"], str):
            parsed["question"] = str(parsed["question"])

        if parsed.get("clarification_needed"):
            if "intents" not in parsed:
                parsed["intents"] = []
            return parsed

        if "intents" in parsed and isinstance(parsed["intents"], list):
            pass
        else:
            single_intent = parsed.get("intent")
            if isinstance(single_intent, list) and single_intent:
                single_intent = single_intent[0]
            if single_intent and single_intent != "待澄清" and await self._is_valid_business_intent(single_intent):
                parsed["intents"] = [single_intent]
            else:
                parsed["intents"] = []

        return parsed

    async def _is_valid_business_intent(self, intent: Optional[str]) -> bool:
        """判断是否为受支持的28类业务意图"""
        if not isinstance(intent, str):
            return False
        # 确保 agents 映射已加载
        if not self._domain_intent_agents:
            # 触发懒加载映射构建（无需实际创建）
            pass
        return intent in self.WORKFLOW_REGISTRY

    # ---------- 工作流获取 ----------
    async def _get_workflow(self, intent: str) -> Optional[BaseWorkflow]:
        """获取或创建工作流实例"""
        if intent not in self.WORKFLOW_REGISTRY:
            return None

        if intent in self._workflow_instances:
            return self._workflow_instances[intent]

        workflow_class = self.WORKFLOW_REGISTRY[intent]
        api_url = self.API_URL_MAPPING.get(intent, "")

        param_model = await self._get_parameter_model()
        format_model = await self._get_format_model()
        echarts_model = await self._get_echarts_model()
        trend_analysis_model = await self._get_trend_analysis_model()

        if intent == '待澄清':
            workflow_instance = workflow_class(
                conversation_id=self.conversation_id,
                parameter_model=param_model,
                format_model=format_model,
                echarts_model=echarts_model,
                trend_analysis_model=trend_analysis_model
            )
        else:
            workflow_instance = workflow_class(
                conversation_id=self.conversation_id,
                parameter_model=param_model,
                format_model=format_model,
                echarts_model=echarts_model,
                trend_analysis_model=trend_analysis_model,
                api_base_url=api_url,
                interface_name=intent
            )

        self._workflow_instances[intent] = workflow_instance
        return workflow_instance

    async def _get_isolated_workflow(self, intent: str) -> Optional[BaseWorkflow]:
        """获取不进入缓存的 workflow 实例，用于并发子问题执行。"""
        if intent not in self.WORKFLOW_REGISTRY:
            return None

        workflow_class = self.WORKFLOW_REGISTRY[intent]
        api_url = self.API_URL_MAPPING.get(intent, "")

        param_model = await self._get_parameter_model()
        format_model = await self._get_format_model()
        echarts_model = await self._get_echarts_model()
        trend_analysis_model = await self._get_trend_analysis_model()

        if workflow_class is ClarificationWorkflow:
            return workflow_class(
                conversation_id=self.conversation_id,
                parameter_model=param_model,
                format_model=format_model,
                echarts_model=echarts_model,
                trend_analysis_model=trend_analysis_model
            )

        return workflow_class(
            conversation_id=self.conversation_id,
            parameter_model=param_model,
            format_model=format_model,
            echarts_model=echarts_model,
            trend_analysis_model=trend_analysis_model,
            api_base_url=api_url,
            interface_name=intent
        )

    # ---------- 子问题拆分相关 ----------
    async def _resolve_sub_items(
            self,
            user_query: str,
            intent_result: Dict[str, Any],
            *,
            skip_split: bool = False,
            log_prefix: str = "[工作流路由]"
    ) -> List[Tuple[str, Optional[str]]]:
        """统一的子问题拆分入口"""
        intents = intent_result.get("intents", [])
        time_range = intent_result.get("time_range", {})
        fallback = [(user_query, intents[0] if intents else None)]

        if skip_split:
            return fallback

        # 策略主线：默认仅多意图拆分；单意图仅在"多主体对比/分别"场景才拆分
        single_intent_should_split = self._should_split_single_intent_query(user_query, intents)
        if len(intents) == 1 and not single_intent_should_split and self._has_forced_multi_subject_split_hint(user_query):
            single_intent_should_split = True
            logger.info(f"{log_prefix} 命中多主体强制拆分兜底策略")
        if len(intents) == 1 and not single_intent_should_split and self._has_multi_date_list_split_hint(user_query, intents):
            single_intent_should_split = True
            logger.info(f"{log_prefix} 命中多日期分别查询拆分策略")
        logger.info(
            f"{log_prefix} 拆分判定: intents={intents}, single_intent_should_split={single_intent_should_split}"
        )
        if len(intents) <= 1 and not single_intent_should_split:
            logger.info(f"{log_prefix} 命中仅多意图拆分策略，当前单意图，跳过问题拆分")
            return fallback

        # 对"单意图多主体"明确拆分，不再被后续直连绕过规则覆盖
        if len(intents) == 1 and single_intent_should_split:
            logger.info(f"{log_prefix} 命中单意图多主体拆分策略，执行问题拆分")
        else:
            bypass_reason = self._get_split_bypass_reason(user_query, intents, time_range)
            if bypass_reason:
                logger.info(f"{log_prefix} 命中{bypass_reason}，跳过问题拆分")
                return fallback

        logger.info(f"{log_prefix} 开始问题拆分...")
        split_workflow = await self._get_split_workflow()
        sub_items = await split_workflow.split(user_query, intent_result)
        logger.info(f"{log_prefix} 拆分结果: {sub_items}")
        if not sub_items:
            return fallback
        return self._normalize_sub_items(sub_items)

    @staticmethod
    def _looks_like_time_token(token: str) -> bool:
        if not isinstance(token, str):
            return False
        t = token.strip().lower()
        if not t:
            return False
        if re.fullmatch(r"v(?:[01]\d|2[0-4])[0-5]\d", t):
            return True
        if re.fullmatch(r"(?:[01]?\d|2[0-4])[:：][0-5]\d", t):
            return True
        if re.fullmatch(r"(?:[01]?\d|2[0-4])点(?:[0-5]\d分?)?", t):
            return True
        return False

    @staticmethod
    def _normalize_subject_candidate(token: str) -> str:
        """用于拆分判定的主体候选归一化：剔除前缀日期、结尾时点等噪音。"""
        if not isinstance(token, str):
            return ""
        t = token.strip()
        if not t:
            return ""

        # 去掉常见引导词
        t = re.sub(r"^(在|请问|帮我|麻烦|查询|统计|计算|比较|对比)+", "", t)
        # 去掉前缀日期片段（如 2025年3月15日冀北）
        t = re.sub(r"^(?:20\d{2}年)?(?:1[0-2]|0?[1-9])月(?:3[01]|[12]\d|0?[1-9])(?:日|号)?", "", t)
        t = re.sub(r"^(?:3[01]|[12]\d|0?[1-9])(?:日|号)", "", t)
        # 清理被截断后残留在开头的"日/号/月/年"等日期单位字
        t = re.sub(r"^[年月日号]+", "", t)
        # 去掉末尾时点片段（如 山西08:00 / 山西8点）
        t = re.sub(r"(?:[01]?\d|2[0-4])[:：][0-5]\d$", "", t)
        t = re.sub(r"(?:[01]?\d|2[0-4])点(?:[0-5]\d分?)?$", "", t)
        # 去掉常见残留连接词/符号
        t = re.sub(r"^[的地得\-_/]+|[的地得\-_/]+$", "", t)
        return t.strip()

    @staticmethod
    def _looks_like_date_fragment(token: str) -> bool:
        """仅当 token 自身形似日期/日期时间片段时判为时间噪音。"""
        if not isinstance(token, str):
            return False
        t = token.strip()
        if not t:
            return False
        patterns = (
            r"^(?:20\d{2}年)?(?:1[0-2]|0?[1-9])月(?:3[01]|[12]\d|0?[1-9])(?:日|号)?$",
            r"^(?:3[01]|[12]\d|0?[1-9])(?:日|号)$",
            r"^(?:20\d{2})[-/](?:1[0-2]|0?[1-9])[-/](?:3[01]|[12]\d|0?[1-9])$",
            r"^(?:[01]?\d|2[0-4])[:：][0-5]\d$",
            r"^(?:[01]?\d|2[0-4])点(?:[0-5]\d分?)?$",
        )
        return any(re.fullmatch(p, t) for p in patterns)

    @classmethod
    def _has_multi_subject_hint(cls, user_query: str) -> bool:
        if not isinstance(user_query, str) or not user_query.strip():
            return False
        q = user_query.replace(" ", "")
        if not re.search(r"(和|与|及|、|,|，)", q):
            return False

        subject_suffix = r"(节点|线路|联络线|电厂|场站|机组|公司|企业|地区|省|市场)"

        # 形如：冀北和蒙东节点 / A和B线路
        shared_suffix_pattern = re.compile(
            rf"(?P<a>[\u4e00-\u9fa5A-Za-z0-9（）()\-]+?)(?:和|与|及|、|,|，)"
            rf"(?P<b>[\u4e00-\u9fa5A-Za-z0-9（）()\-]+?)(?P<s>{subject_suffix})"
        )
        for match in shared_suffix_pattern.finditer(q):
            a = match.group("a")
            b = match.group("b")
            if a and b and a != b and not cls._looks_like_time_token(a) and not cls._looks_like_time_token(b):
                return True

        # 形如：冀北节点和蒙东节点 / 青豫直流和祁韶直流线路
        full_subject_pattern = re.compile(
            rf"(?P<a>[\u4e00-\u9fa5A-Za-z0-9（）()\-]+?{subject_suffix})(?:和|与|及|、|,|，)"
            rf"(?P<b>[\u4e00-\u9fa5A-Za-z0-9（）()\-]+?{subject_suffix})"
        )
        for match in full_subject_pattern.finditer(q):
            a = match.group("a")
            b = match.group("b")
            if a and b and a != b:
                return True

        # 形如：冀北和蒙东全天平均出清电量（无"节点/线路"等后缀）
        # 仅在后续尾部出现业务语义词时触发，避免"8点和9点"这类时点表达误判为多主体。
        plain_pair_pattern = re.compile(
            r"(?P<a>[\u4e00-\u9fa5A-Za-z0-9（）()\-]{2,8})(?:和|与|及|、|,|，)"
            r"(?P<b>[\u4e00-\u9fa5A-Za-z0-9（）()\-]{2,8})(?P<tail>[^，。！？；]{0,24})"
        )
        tail_cues = (
            "节点", "线路", "联络线", "电量", "电价", "出清", "预测", "功率",
            "平均", "全天", "时段", "分别", "各自", "对比", "比较", "排名",
            "信息", "情况", "状态", "数量", "容量", "类型", "名称", "价格",
            "交易", "合同", "公告", "检修", "停运", "申报", "结算",
        )
        noisy_tokens = ("业务", "市场", "查询", "今年", "去年", "本月", "上月", "日期", "时间", "时段", "时刻")
        for match in plain_pair_pattern.finditer(q):
            a = cls._normalize_subject_candidate(match.group("a"))
            b = cls._normalize_subject_candidate(match.group("b"))
            tail = match.group("tail") or ""
            if not a or not b or a == b:
                continue
            if cls._looks_like_time_token(a) or cls._looks_like_time_token(b):
                continue
            if cls._looks_like_date_fragment(a) or cls._looks_like_date_fragment(b):
                continue
            if any(token in a for token in noisy_tokens) or any(token in b for token in noisy_tokens):
                continue
            if any(cue in tail for cue in tail_cues):
                return True

        return False

    @classmethod
    def _should_split_single_intent_query(cls, user_query: str, intents: List[str]) -> bool:
        """单意图默认不拆，仅在多主体（对比/分别/各自）或多步依赖场景拆分。"""
        if not isinstance(user_query, str):
            return False
        if not isinstance(intents, list) or len(intents) != 1:
            return False
        if not (
            cls._has_multi_subject_hint(user_query)
            or cls._has_plain_subject_pair_hint(user_query)
            or cls._has_dependency_chain_hint(user_query)
        ):
            return False

        q = user_query.replace(" ", "")
        q_lower = q.lower()
        aggregate_only_cues = ("总和", "合计", "总计", "汇总", "总体", "整体", "累计", "加总")
        split_cues = (
            "分别", "各自", "各个", "对比", "比较", "相较", "哪个", "哪一个", "谁更",
            "高于", "低于", "差值", "相差", "之差", "乘积", "比值",
            "排序", "排名", "最高", "最低", "最大", "最小", "top", "bottom",
        )

        has_split_cue = any(cue in q_lower for cue in split_cues)
        if any(cue in q for cue in aggregate_only_cues) and not has_split_cue:
            return False
        return True

    @classmethod
    def _has_plain_subject_pair_hint(cls, user_query: str) -> bool:
        """兜底识别"冀北和蒙东全天平均出清电量"这类无后缀多主体表达。"""
        if not isinstance(user_query, str) or not user_query.strip():
            return False

        q = user_query.replace(" ", "")
        pair_pattern = re.compile(
            r"(?P<a>[\u4e00-\u9fa5]{2,6})(?:和|与|及|、|,|，)(?P<b>[\u4e00-\u9fa5]{2,6})"
        )
        tail_cues = (
            "节点", "线路", "联络线", "电量", "电价", "出清", "预测", "功率",
            "平均", "全天", "全日", "分别", "各自", "对比", "比较",
            "最高", "最低", "排名", "排序", "信息", "情况", "状态", "数量",
            "容量", "类型", "名称", "价格", "交易", "合同", "公告", "检修",
            "停运", "申报", "结算",
        )
        stop_words = {
            "业务", "市场", "查询", "数据", "结果", "今天", "昨天", "明天",
            "出清", "电量", "电价", "平均", "全天", "全日", "分别", "对比",
            "日期", "时间", "时段", "时刻",
        }

        for match in pair_pattern.finditer(q):
            a = cls._normalize_subject_candidate(match.group("a"))
            b = cls._normalize_subject_candidate(match.group("b"))
            if not a or not b or a == b:
                continue
            if cls._looks_like_time_token(a) or cls._looks_like_time_token(b):
                continue
            if cls._looks_like_date_fragment(a) or cls._looks_like_date_fragment(b):
                continue
            if any(sw in a for sw in stop_words) or any(sw in b for sw in stop_words):
                continue

            tail = q[match.end(): match.end() + 20]
            if any(cue in tail for cue in tail_cues):
                return True

        return False

    @staticmethod
    def _has_dependency_chain_hint(user_query: str) -> bool:
        if not isinstance(user_query, str) or not user_query.strip():
            return False
        q = user_query.replace(' ', '')
        if not re.search(r'(并|然后|再|接着|同时)', q):
            return False
        if not re.search(r'(该|此|其|上述|前述|对应)', q):
            return False
        if re.search(r'(?:并|然后|再|接着|同时).{0,10}(?:该|此|其|上述|前述|对应)', q):
            return True
        return False

    @staticmethod
    def _has_forced_multi_subject_split_hint(user_query: str) -> bool:
        """More conservative fallback: force split when X和Y + split cues present."""
        if not isinstance(user_query, str) or not user_query.strip():
            return False
        q = user_query.replace(" ", "")
        if not re.search(r"(和|与|及|、|,|，)", q):
            return False
        split_cues = (
            "分别", "各自", "对比", "比较", "哪个", "哪一个", "谁更",
            "高于", "低于", "排名", "排序", "最高", "最低",
        )
        if not any(cue in q for cue in split_cues):
            return False
        pair_pattern = re.compile(
            r"(?P<a>[\u4e00-\u9fa5A-Za-z0-9（）()\-]{2,10})(?:和|与|及|、|,|，)(?P<b>[\u4e00-\u9fa5A-Za-z0-9（）()\-]{2,10})"
        )
        match = pair_pattern.search(q)
        if not match:
            return False
        a = WorkflowRouter._normalize_subject_candidate(match.group("a"))
        b = WorkflowRouter._normalize_subject_candidate(match.group("b"))
        if not a or not b or a == b:
            return False
        if WorkflowRouter._looks_like_time_token(a) or WorkflowRouter._looks_like_time_token(b):
            return False
        # 排除典型时点表达
        if re.search(r"(?:\d{1,2}[:：]\d{2}|\d{1,2}点)", q):
            return False
        return True

    @classmethod
    def _has_multi_date_list_split_hint(cls, user_query: str, intents: List[str]) -> bool:
        """检测"分别列出/查询多个日期下的实体集合"模式，应拆分为每日期一个子问题。

        与 _should_bypass_split_for_spot_multi_date_projection 的区别：
        那条针对"同一实体+同一指标+多日期取值"（如"冀北7月10日和7月20日的电价"），
        本条针对"多日期+分别列出满足条件的实体集合"（如"分别列出7月10日和7月20日参与交易的网省"）。
        """
        if not isinstance(user_query, str) or len(intents) != 1:
            return False

        q = user_query.replace(" ", "")

        # 必须有"分别"语义
        if not any(cue in q for cue in ("分别", "各自")):
            return False

        # 必须有至少 2 个不同日期
        date_points = cls._extract_date_points_from_query(q)
        if len(set(date_points)) < 2:
            return False

        # 必须有列表/枚举语义（问"有哪些"而非问某个具体指标值）
        list_cues = ("有哪些", "哪些", "列出")
        if not any(cue in q for cue in list_cues):
            return False

        return True

    def _get_split_bypass_reason(self, user_query: str, intents: List[str], time_range: Any) -> Optional[str]:
        """判断是否跳过拆分"""
        return None

    @staticmethod
    def _has_time_range(time_range: Any) -> bool:
        if not isinstance(time_range, dict):
            return False
        return bool(time_range.get("start") and time_range.get("end"))

    @staticmethod
    def _normalize_sub_items(sub_items: List[Tuple[Any, Any]]) -> List[Tuple[str, Optional[str]]]:
        normalized = []
        for q, i in sub_items:
            if not isinstance(q, str):
                q = str(q)
            normalized.append((q, i))
        return normalized

    @staticmethod
    def _normalize_date_str(date_str: Any) -> Any:
        if not isinstance(date_str, str):
            return date_str
        raw = date_str.strip()
        if not raw:
            return raw
        candidates = [raw]
        if len(raw) >= 10:
            candidates.append(raw[:10])
        for candidate in candidates:
            try:
                datetime.strptime(candidate, "%Y-%m-%d")
                return candidate
            except ValueError:
                continue
        return raw

    @staticmethod
    def _shift_date_str(date_str: str, days: int = 1) -> str:
        if not isinstance(date_str, str) or not date_str:
            return date_str
        date_str = WorkflowRouter._normalize_date_str(date_str)
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return (dt + timedelta(days=days)).strftime("%Y-%m-%d")
        except ValueError:
            return date_str

    @staticmethod
    def _is_empty_result_data(data: Any) -> bool:
        return data is None or data == [] or data == [{}] or data == {}

    @staticmethod
    def _extract_api_business_error_from_result(workflow: BaseWorkflow, api_result: Dict[str, Any]) -> Optional[str]:
        if not isinstance(api_result, dict):
            return None
        payload = api_result.get("data")
        if not isinstance(payload, dict):
            return None
        checker = getattr(workflow, "_extract_common_api_business_error", None)
        if callable(checker):
            try:
                return checker(payload)
            except Exception:
                return None
        return None

    async def _retry_shared_api_with_lookback_if_empty(
        self,
        workflow: BaseWorkflow,
        params: Dict[str, Any],
        raw_data: Any,
        shared_intent: str,
    ) -> tuple[Any, bool]:
        if not self._is_empty_result_data(raw_data):
            return raw_data, False

        time_range = params.get("time_range")
        if not isinstance(time_range, dict):
            return raw_data, False

        start = self._normalize_date_str(time_range.get("start"))
        if not start:
            return raw_data, False

        try:
            expanded_start = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")
        except ValueError:
            logger.warning(f"[共享数据模式] time_range.start 格式不合法，跳过30天回溯: {start}")
            return raw_data, False

        original_time_range = copy.deepcopy(time_range)
        params["time_range"] = copy.deepcopy(time_range)
        params["time_range"]["start"] = expanded_start
        params["time_range"]["end"] = self._normalize_date_str(params["time_range"].get("end"))
        logger.info(
            f"[共享数据模式] 意图 {shared_intent} 首次查询为空，回溯30天重查: "
            f"{original_time_range.get('start')} -> {expanded_start}"
        )

        retry_result = await workflow.call_api(params)
        if retry_result.get("status") == "error":
            logger.warning(f"[共享数据模式] 回溯30天后二次查询失败，保留原始空结果: {retry_result.get('message')}")
            params["time_range"] = original_time_range
            return raw_data, False

        retry_raw_data = workflow._extract_api_data_payload(
            retry_result,
            workflow_name="WorkflowRouterShared",
            is_retry=True,
        )
        if self._is_empty_result_data(retry_raw_data):
            logger.info(f"[共享数据模式] 回溯30天后二次查询仍为空，按原逻辑返回未查到数据")
            params["time_range"] = original_time_range
            return raw_data, False

        logger.info(f"[共享数据模式] 回溯30天后二次查询命中数据，后续子问题将使用扩展时间范围")
        return retry_raw_data, True

    @staticmethod
    def _extract_single_date_from_detail_result(detail_result: Dict[str, Any]) -> Optional[str]:
        if not isinstance(detail_result, dict):
            return None

        time_range = detail_result.get("time_range")
        if isinstance(time_range, str):
            normalized = WorkflowRouter._normalize_date_str(time_range)
            return normalized if isinstance(normalized, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", normalized) else None

        if isinstance(time_range, list):
            for item in time_range:
                normalized = WorkflowRouter._normalize_date_str(item)
                if isinstance(normalized, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", normalized):
                    return normalized
            return None

        if isinstance(time_range, dict):
            start = WorkflowRouter._normalize_date_str(time_range.get("start"))
            end = WorkflowRouter._normalize_date_str(time_range.get("end"))
            if isinstance(start, str) and isinstance(end, str) and start == end:
                return start
            if isinstance(start, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", start):
                return start

        question = detail_result.get("question")
        if isinstance(question, str):
            match = re.search(r"(20\d{2}-\d{2}-\d{2})", question)
            if match:
                return match.group(1)
        return None

    def _filter_shared_raw_data_for_sub_question(
            self,
            shared_raw_data: Any,
            shared_intent: str,
            detail_result: Dict[str, Any],
            workflow: Optional[BaseWorkflow] = None,
            sub_params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """共享数据模式下，按披露日期过滤子问题数据。"""
        return self._filter_shared_raw_data_by_sub_params(shared_raw_data, workflow, sub_params)

    @staticmethod
    def _merge_shared_and_sub_params(
            shared_params: Dict[str, Any],
            sub_params: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        merged = copy.deepcopy(shared_params) if isinstance(shared_params, dict) else {}
        if not isinstance(sub_params, dict):
            return merged

        for key in ("name_abbreviation", "target_type", "case_id"):
            value = sub_params.get(key)
            if isinstance(value, str) and value.strip():
                merged[key] = value.strip()

        for key in ("time_points", "requested_fields", "post_processing"):
            value = sub_params.get(key)
            if isinstance(value, list) and value:
                merged[key] = value

        return merged

    @staticmethod
    def _is_time_point_field(field: Any) -> bool:
        return isinstance(field, str) and re.fullmatch(r"v(?:[01]\d|2[0-4])[0-5]\d", field, flags=re.IGNORECASE) is not None

    def _normalize_shared_sub_question_plan(
            self,
            operations: Any,
            params: Dict[str, Any],
    ) -> tuple[list, Dict[str, Any]]:
        """共享数据模式下纠正把时点字段误当 filter 的情况，并去掉参数层重复过滤。"""
        if not isinstance(params, dict):
            params = {}

        normalized_params = copy.deepcopy(params)
        existing_time_points = normalized_params.get("time_points", [])
        if not isinstance(existing_time_points, list):
            existing_time_points = []

        seen_time_points = {
            tp for tp in existing_time_points
            if isinstance(tp, str) and self._is_time_point_field(tp)
        }
        normalized_time_points = [tp for tp in existing_time_points if tp in seen_time_points]

        if not isinstance(operations, list):
            if normalized_time_points:
                normalized_params["time_points"] = normalized_time_points
            return [], normalized_params

        normalized_operations = []
        for op in operations:
            if not isinstance(op, dict):
                continue

            if op.get("operation") != "filter":
                normalized_operations.append(op)
                continue

            field = op.get("field")
            operator = str(op.get("operator", "")).strip().lower()
            value = op.get("value")

            if field == "targetType" and normalized_params.get("target_type"):
                continue
            if field == "nameAbbreviation" and normalized_params.get("name_abbreviation"):
                continue
            if field == "caseId" and normalized_params.get("case_id"):
                continue
            if field == "businessTime" and normalized_params.get("time_range"):
                continue

            if self._is_time_point_field(field):
                has_value = value not in (None, "", [], {})
                if not has_value and operator in ("", "eq", "=="):
                    if field not in seen_time_points:
                        seen_time_points.add(field)
                        normalized_time_points.append(field)
                    logger.info(f"[共享模式] 将时点字段误判 filter 纠正为 time_points: {field}")
                    continue

            normalized_operations.append(op)

        if normalized_time_points:
            normalized_params["time_points"] = normalized_time_points

        return normalized_operations, normalized_params

    @staticmethod
    def _filter_shared_raw_data_by_sub_params(
            shared_raw_data: Any,
            workflow: Optional[BaseWorkflow],
            sub_params: Optional[Dict[str, Any]],
    ) -> Any:
        if not isinstance(shared_raw_data, list) or not isinstance(sub_params, dict):
            return shared_raw_data

        filtered = shared_raw_data

        target_type = sub_params.get("target_type")
        if isinstance(target_type, str) and target_type.strip():
            target_type_value = target_type.strip()
            filtered = [
                item for item in filtered
                if isinstance(item, dict) and str(item.get("targetType", "")).strip() == target_type_value
            ]

        name_value = sub_params.get("name_abbreviation")
        if isinstance(name_value, str) and name_value.strip():
            normalize_name = getattr(workflow, "_normalize_name_abbreviation", None)
            normalized_target = normalize_name(name_value) if callable(normalize_name) else name_value.strip()
            filtered = [
                item for item in filtered
                if isinstance(item, dict)
                and isinstance(item.get("nameAbbreviation"), str)
                and (
                    (normalize_name(item.get("nameAbbreviation")) if callable(normalize_name) else item.get("nameAbbreviation").strip())
                    == normalized_target
                )
            ]

        case_id = sub_params.get("case_id")
        if isinstance(case_id, str) and case_id.strip():
            case_value = case_id.strip()
            exact = [
                item for item in filtered
                if isinstance(item, dict) and str(item.get("caseId", "")).strip() == case_value
            ]
            if exact:
                filtered = exact
            else:
                filtered = [
                    item for item in filtered
                    if isinstance(item, dict) and str(item.get("caseId", "")).strip().startswith(case_value)
                ]

        return filtered

    # ---------- 共享数据模式 ----------
    def _can_share_api_call(self, sub_items: List[Tuple[str, Optional[str]]]) -> bool:
        # if len(sub_items) <= 1:
        #     return False
        # intents = {intent for _, intent in sub_items if intent is not None}
        # return len(intents) == 1
        return False

    async def _execute_shared_data_multi_questions_stream(
            self,
            user_query: str,
            sub_items: List[Tuple[str, Optional[str]]],
            intent_result: Dict[str, Any],
            shared_intent: str,
            user_id: str,
            query_id: str
    ) -> AsyncGenerator[Any, None]:
        """流式共享数据处理（优化缓存写入）"""
        logger.info(f"[共享数据模式-流式] 意图 {shared_intent} 下的 {len(sub_items)} 个子问题将共享API，query_id={query_id}")

        workflow = await self._get_workflow(shared_intent)
        if not workflow:
            yield sse({"data": f"不支持的意图: {shared_intent}", "type": "error"})
            return

        time_range = intent_result.get("time_range")
        if not self._has_time_range(time_range):
            yield sse({"data": "缺少有效的时间范围", "type": "error"})
            return

        # 调整 end 日期
        adjusted_time_range = time_range.copy()
        adjusted_time_range["start"] = self._normalize_date_str(adjusted_time_range.get("start"))
        adjusted_time_range["end"] = self._normalize_date_str(adjusted_time_range.get("end"))
        end = adjusted_time_range.get("end")
        try:
            end_dt = datetime.strptime(end, "%Y-%m-%d")
            new_end = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
            adjusted_time_range["end"] = new_end
            logger.info(f"[共享数据模式] time_range.end 调整: {end} → {new_end}")
        except ValueError:
            logger.warning(f"[共享数据模式] time_range.end 格式不合法，跳过调整: {end}")

        params = {"time_range": adjusted_time_range}
        validation = workflow.validate_params(params)
        if not validation['valid']:
            yield sse({"data": validation['message'], "type": "error"})
            return

        api_result = await workflow.call_api(params)
        if api_result['status'] == 'error':
            yield sse({"data": f"API调用失败: {api_result.get('message')}", "type": "error"})
            return

        business_error = self._extract_api_business_error_from_result(workflow, api_result)
        if business_error:
            yield sse({"data": f"API调用失败: {business_error}", "type": "error"})
            return

        raw_data = workflow._extract_api_data_payload(
            api_result,
            workflow_name="WorkflowRouterShared",
            is_retry=False,
        )
        raw_data, expanded_query_applied = await self._retry_shared_api_with_lookback_if_empty(
            workflow=workflow,
            params=params,
            raw_data=raw_data,
            shared_intent=shared_intent,
        )
        shared_raw_data = copy.deepcopy(raw_data)
        effective_root_query = (
            workflow._build_expanded_time_range_query(user_query, params.get("time_range", {}))
            if expanded_query_applied else user_query
        )

        sub_results = []
        for sub_q, sub_intent in sub_items:
            effective_sub_q = (
                workflow._build_expanded_time_range_query(sub_q, params.get("time_range", {}))
                if expanded_query_applied else sub_q
            )
            sub_result = await self._process_sub_question_with_shared_data(
                effective_sub_q, shared_intent, shared_raw_data, params, workflow
            )
            sub_results.append(sub_result)

        valid_results = [r for r in sub_results if "data" in r]
        if not valid_results:
            yield sse({"data": "所有子问题处理均失败", "type": "error"})
            return

        # 汇总元数据
        sub_metadata = []
        for r in valid_results:
            trunc_info = r.get("trunc_info")
            if trunc_info:
                sub_metadata.append({
                    "sub_question": r["sub_question"],
                    "truncated": trunc_info.get("truncated", False),
                    "truncation_types": trunc_info.get("truncation_types", []),
                    "total_count": trunc_info.get("total_count"),
                    "display_count": trunc_info.get("display_count"),
                    "max_time_field": trunc_info.get("max_time_field")
                })
            else:
                sub_metadata.append({"sub_question": r["sub_question"], "truncated": False})

        # 并发安全缓存写入
        async with _cache_lock:
            if user_id not in _query_cache:
                _query_cache[user_id] = {
                # 后续不再缓存全量数据。
                # "full_data": shared_raw_data,
                "sub_metadata": sub_metadata,
                "conversation_id": self.conversation_id,
                "cached_at": datetime.now(),
                "truncated": any(m["truncated"] for m in sub_metadata),
                "truncation_types": list(set(t for m in sub_metadata for t in m.get("truncation_types", []))),
            }

        # 发送截断事件
        event_data = {
            "truncated": any(m["truncated"] for m in sub_metadata),
            "query_id": query_id,
            "conversation_id": self.conversation_id,
            "user": user_id,
            "type": "multi_shared",
            "sub_metadata": sub_metadata
        }
        yield sse({"data": event_data, "type": "data_truncated"})

        # 流式输出各子问题
        for idx, r in enumerate(valid_results, 1):
            sub_q = r["sub_question"]
            data = r["data"]
            workflow._truncated = r.get("trunc_info", {}).get("truncated", False)
            workflow._field_truncated = "field_filter" in r.get("trunc_info", {}).get("truncation_types", [])
            workflow._original_data = shared_raw_data

            yield sse({"data": f"\n\n**{sub_q}**：\n\n", "type": "content"})

            multi_context = {"is_multi": True, "index": idx, "total": len(valid_results)}
            async for chunk in workflow._format_result_stream(sub_q, data, format_context=multi_context):
                yield chunk

            if idx < len(valid_results):
                await asyncio.sleep(0.5)

        unified_chart_data = self._build_unified_chart_dataset(
            [(r["sub_question"], r.get("data")) for r in valid_results]
        )
        unified_echarts = await self._generate_unified_echarts(
            user_query=effective_root_query,
            chart_dataset=unified_chart_data,
            workflow=workflow,
            intent=intent_result.get("intent","")
        )
        if unified_echarts:
            yield sse({"data": unified_echarts, "type": "content"})

    async def _process_sub_question_with_shared_data(
            self, sub_q: str, intent: str, shared_raw_data: Any,
            shared_params: Dict[str, Any], workflow: BaseWorkflow
    ) -> Dict[str, Any]:
        """使用共享数据处理单个子问题（按特定意图定向过滤共享数据）"""
        try:
            sub_intent_result = {"question": sub_q, "intents": [intent]}
            detail_result = await self._recognize_domain_intent(intent, sub_q, sub_intent_result)
            sub_params = await workflow._extract_parameters(sub_q, detail_result)
            merged_params = self._merge_shared_and_sub_params(shared_params, sub_params)
            operations = detail_result.get('operations', [])
            operations, merged_params = self._normalize_shared_sub_question_plan(operations, merged_params)
            operations = workflow.normalize_operations(operations, sub_q, detail_result, merged_params)
            operations = workflow.sanitize_operations(operations, params=merged_params, user_query=sub_q)
            merged_params['operations'] = operations

            filtered_raw_data = self._filter_shared_raw_data_for_sub_question(
                shared_raw_data=shared_raw_data,
                shared_intent=intent,
                detail_result=detail_result,
                workflow=workflow,
                sub_params=merged_params,
            )

            processed_data = workflow.process_data(filtered_raw_data, merged_params, sub_q)

            if operations:
                agg_result = await workflow.aggregation_agent.ainvoke(
                    user_query=sub_q, operations=operations, data=processed_data,
                    context={"thread_id": self.conversation_id}, domain=workflow._get_domain()
                )
                summary = agg_result.get('summary') or []
                aggregated_data = agg_result.get('aggregated_data')
                grouped_data_full = agg_result.get('grouped_data_full')

                _summary_ops = {'sum', 'average', 'weighted_average', 'max', 'min', 'count'}
                _row_ops = {'subtract', 'multiply', 'divide', 'mom_change', 'yoy_change'}
                _sort_group_ops = {'top_n', 'bottom_n', 'group_by'}

                _has_agg = any(op.get('operation') in _summary_ops for op in operations)
                _has_row_ops = any(op.get('operation') in _row_ops for op in operations)
                _has_sort_group = any(op.get('operation') in _sort_group_ops for op in operations)

                _has_agg_ratio = any(
                    op.get('operation') in ('divide', 'subtract', 'multiply')
                    and (
                        (isinstance(op.get('field', ''), str) and op.get('field', '').startswith('sum_'))
                        or (isinstance(op.get('field_b', ''), str) and op.get('field_b', '').startswith('sum_'))
                    )
                    for op in operations
                )

                _has_row_then_agg = (
                    _has_row_ops
                    and any(op.get('operation') in ('max', 'min') for op in operations)
                    and not _has_agg_ratio
                )

                _has_row_then_any_agg = (
                    _has_row_ops
                    and _has_agg
                    and not _has_agg_ratio
                    and not _has_sort_group
                )

                _has_group_then_extreme = (
                    any(op.get('operation') == 'group_by' for op in operations)
                    and any(op.get('operation') in ('max', 'min') for op in operations)
                )

                _has_group_then_count = (
                    any(op.get('operation') == 'group_by' for op in operations)
                    and any(op.get('operation') == 'count' for op in operations)
                    and not any(op.get('operation') in ('top_n', 'bottom_n') for op in operations)
                )

                if _has_agg_ratio and aggregated_data:
                    agg_summary = {}
                    for s in summary:
                        agg_summary[s['field']] = s['result']
                    first = aggregated_data[0] if isinstance(aggregated_data, list) and aggregated_data else {}
                    for key in ('quotient', 'product', 'difference'):
                        if key in first:
                            agg_summary[key] = first[key]
                    processed_data = [agg_summary]
                elif _has_row_then_agg and summary and aggregated_data:
                    extreme_summary = next(
                        (s for s in reversed(summary) if s.get('operation') in ('max', 'min')), None
                    )
                    if extreme_summary:
                        extreme_field = extreme_summary.get('field', '')
                        extreme_val = extreme_summary.get('result')
                        extreme_record = None
                        if extreme_val is not None and isinstance(aggregated_data, list):
                            for record in aggregated_data:
                                if isinstance(record, dict) and record.get(extreme_field) == extreme_val:
                                    extreme_record = record
                                    break
                        processed_data = [extreme_record] if extreme_record else summary
                    else:
                        processed_data = summary
                elif _has_group_then_extreme and summary and aggregated_data:
                    if workflow._should_preserve_grouped_details_with_extreme(sub_q, operations):
                        processed_data = workflow._build_grouped_extreme_payload(
                            grouped_data_full or aggregated_data,
                            summary,
                            aggregated_data
                        )
                    else:
                        extreme_summary = next(
                            (s for s in reversed(summary) if s.get('operation') in ('max', 'min')), None
                        )
                        if extreme_summary:
                            extreme_field = extreme_summary.get('field', '')
                            extreme_val = extreme_summary.get('result')
                            extreme_record = None
                            if extreme_val is not None and isinstance(aggregated_data, list):
                                for record in aggregated_data:
                                    if isinstance(record, dict) and record.get(extreme_field) == extreme_val:
                                        extreme_record = record
                                        break
                            processed_data = [extreme_record] if extreme_record else summary
                        else:
                            processed_data = summary
                elif _has_row_then_any_agg and summary:
                    processed_data = summary
                elif _has_group_then_count and summary:
                    processed_data = summary
                elif (_has_sort_group or _has_row_ops) and aggregated_data:
                    processed_data = aggregated_data
                elif _has_agg and summary:
                    processed_data = summary
                elif agg_result:
                    processed_data = aggregated_data

                if hasattr(workflow, 'post_process_after_aggregation'):
                    processed_data = workflow.post_process_after_aggregation(
                        processed_data=processed_data, user_query=sub_q, intent_result=detail_result,
                        params=merged_params, operations=operations, summary=summary, aggregated_data=aggregated_data,
                        grouped_data_full=grouped_data_full,
                    )

            post_processing = detail_result.get('post_processing', []) or merged_params.get('post_processing', [])
            if post_processing:
                processed_data = workflow.apply_post_processing(processed_data, post_processing, merged_params)

            format_source_data = processed_data

            truncated_rows = False
            full_data_for_trunc = filtered_raw_data
            if isinstance(processed_data, list) and len(
                    processed_data) > BaseWorkflow.MAX_V_FIELDS:
                workflow._truncated = True
                workflow._full_data = full_data_for_trunc
                processed_data = processed_data[:MAX_DISPLAY_ITEMS]
                truncated_rows = True
                logger.info(
                    f"[共享模式] 记录条数截断: {len(full_data_for_trunc)} -> {MAX_DISPLAY_ITEMS}")
            else:
                workflow._truncated = False
                workflow._full_data = None

            # processed_data = workflow._filter_time_fields_by_count(processed_data)

            trunc_info = workflow.get_any_truncation_info()
            if not trunc_info:
                trunc_info = {
                    "truncated": truncated_rows or workflow._field_truncated,
                    # 后续不再对外透传 full_data。
                    # "full_data": full_data_for_trunc,
                    "truncation_types": [],
                    "conversation_id": self.conversation_id
                }
                if truncated_rows:
                    trunc_info["truncation_types"].append("row_count")
                    trunc_info["total_count"] = len(full_data_for_trunc) if isinstance(full_data_for_trunc, list) else 0
                    trunc_info["display_count"] = MAX_DISPLAY_ITEMS
                if workflow._field_truncated:
                    trunc_info["truncation_types"].append("field_filter")
                    trunc_info["max_v_fields_kept"] = BaseWorkflow.MAX_V_FIELDS

            return {
                "sub_question": sub_q,
                "intent": intent,
                "data": processed_data,
                "format_source_data": format_source_data,
                "params": merged_params,
                "trunc_info": trunc_info,
                # 后续不再继续透传 full_data。
                # "full_data": full_data_for_trunc
            }
        except Exception as e:
            logger.error(f"子问题处理失败: {sub_q}, 错误: {e}", exc_info=True)
            return {"sub_question": sub_q, "error": str(e)}

    async def _execute_shared_data_multi_questions_stream_parallel(
            self,
            user_query: str,
            sub_items: List[Tuple[str, Optional[str]]],
            intent_result: Dict[str, Any],
            shared_intent: str,
            user_id: str,
            query_id: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """共享数据流式执行：共享一次 API，并并发处理后统一输出总答案。"""
        logger.info(
            f"[共享数据模式-流式并发] intent={shared_intent}, sub_count={len(sub_items)}, query_id={query_id}"
        )

        shared_workflow = await self._get_workflow(shared_intent)
        if not shared_workflow:
            yield {"data": f"不支持的意图: {shared_intent}", "type": "error"}
            return

        time_range = intent_result.get("time_range")
        if not self._has_time_range(time_range):
            yield {"data": "缺少有效的时间范围", "type": "error"}
            return

        adjusted_time_range = time_range.copy()
        adjusted_time_range["start"] = self._normalize_date_str(adjusted_time_range.get("start"))
        adjusted_time_range["end"] = self._normalize_date_str(adjusted_time_range.get("end"))
        end_value = adjusted_time_range.get("end")
        try:
            end_dt = datetime.strptime(end_value, "%Y-%m-%d")
            adjusted_time_range["end"] = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        except ValueError:
            logger.warning(f"[共享数据模式-流式并发] time_range.end 格式非法，跳过调整: {end_value}")

        params = {"time_range": adjusted_time_range}
        validation = shared_workflow.validate_params(params)
        if not validation['valid']:
            yield {"data": validation['message'], "type": "error"}
            return

        # api_result = await shared_workflow.call_api(params)
        # if api_result['status'] == 'error':
        #     yield {"data": f"API调用失败: {api_result.get('message')}", "type": "error"}
        #     return

        # business_error = self._extract_api_business_error_from_result(shared_workflow, api_result)
        # if business_error:
        #     yield {"data": f"API调用失败: {business_error}", "type": "error"}
        #     return

        api_result = await shared_workflow.call_api(params)
        if api_result['status'] == 'error':
            friendly = self._to_user_friendly_stream_error(api_result.get("message"))
            yield {"data": friendly, "type": "error"}
            return

        business_error = self._extract_api_business_error_from_result(shared_workflow, api_result)
        if business_error:
            friendly = self._to_user_friendly_stream_error(business_error)
            yield {"data": friendly, "type": "error"}
            return

        raw_data = shared_workflow._extract_api_data_payload(
            api_result,
            workflow_name="WorkflowRouterShared",
            is_retry=False,
        )
        raw_data, expanded_query_applied = await self._retry_shared_api_with_lookback_if_empty(
            workflow=shared_workflow,
            params=params,
            raw_data=raw_data,
            shared_intent=shared_intent,
        )
        shared_raw_data = copy.deepcopy(raw_data)
        effective_root_query = (
            shared_workflow._build_expanded_time_range_query(user_query, params.get("time_range", {}))
            if expanded_query_applied else user_query
        )

        concurrency = min(len(sub_items), self._get_subquestion_concurrency(shared=True))
        semaphore = asyncio.Semaphore(concurrency)
        states: List[Dict[str, Any]] = []

        for sub_q, _ in sub_items:
            effective_sub_q = (
                shared_workflow._build_expanded_time_range_query(sub_q, params.get("time_range", {}))
                if expanded_query_applied else sub_q
            )
            states.append({
                "sub_question": sub_q,
                "effective_sub_question": effective_sub_q,
                "workflow": await self._get_isolated_workflow(shared_intent),
            })

        async def process_state(state: Dict[str, Any]) -> None:
            workflow_ref = state.get("workflow")
            if not isinstance(workflow_ref, BaseWorkflow):
                state["error"] = f"不支持的意图: {shared_intent}"
                return

            async with semaphore:
                result = await self._process_sub_question_with_shared_data(
                    state["effective_sub_question"],
                    shared_intent,
                    shared_raw_data,
                    params,
                    workflow_ref,
                )
            state["result"] = result

        yield {"data": "intent recognition II", "type": "messageLabel"}
        await asyncio.gather(*(process_state(state) for state in states))

        valid_states = [
            state for state in states
            if isinstance(state.get("result"), dict) and "data" in state["result"]
        ]
        if not valid_states:
            yield {"data": "所有子问题处理均失败", "type": "error"}
            return

        sub_metadata = [
            self._build_sub_truncation_metadata(
                state["sub_question"],
                state["result"].get("trunc_info"),
            )
            for state in valid_states
        ]
        truncated_any = any(item.get("truncated") for item in sub_metadata)
        if truncated_any:
            # await cache_set(user_id, query_id, {
            #     "sub_metadata": sub_metadata,
            #     "conversation_id": self.conversation_id,
            #     "cached_at": datetime.now(),
            #     "truncated": True,
            #     "truncation_types": list({t for item in sub_metadata for t in item.get("truncation_types", [])}),
            # })
            yield {
                "data": {
                    "truncated": True,
                    "query_id": query_id,
                    "conversation_id": self.conversation_id,
                    "user": user_id,
                    "type": "multi_shared_parallel",
                    "sub_metadata": sub_metadata,
                },
                "type": "data_truncated",
            }

        async def render_state(state: Dict[str, Any]) -> Dict[str, str]:
            workflow_ref = state["workflow"]
            result = state["result"]
            data = result["data"]
            format_source_data = result.get("format_source_data", data)
            sub_params = result.get("params", params)

            workflow_ref._truncated = result.get("trunc_info", {}).get("truncated", False)
            workflow_ref._field_truncated = "field_filter" in result.get("trunc_info", {}).get("truncation_types", [])
            workflow_ref._original_data = shared_raw_data

            async with semaphore:
                if workflow_ref._should_use_format_summary(format_source_data):
                    answer = await workflow_ref._format_result(state["effective_sub_question"], format_source_data)
                else:
                    analysis_result = await workflow_ref._run_data_analysis(
                        user_query=state["effective_sub_question"],
                        raw_data=shared_raw_data,
                        display_data=format_source_data,
                        params=sub_params,
                    )
                    answer = analysis_result.get("summary", "") if analysis_result else ""

            return {
                "sub_question": state["sub_question"],
                "answer": answer,
            }

        sub_answer_items = await asyncio.gather(*(render_state(state) for state in valid_states))
        final_content = await self._generate_merged_answer(effective_root_query, sub_answer_items)
        yield {"data": final_content, "type": "content"}

        unified_chart_data = self._build_unified_chart_dataset([
            (state["sub_question"], state["result"].get("data")) for state in valid_states
        ])
        unified_echarts = await self._generate_unified_echarts(
            user_query=effective_root_query,
            chart_dataset=unified_chart_data,
            workflow=shared_workflow,
            intent=intent_result.get("intent", "")
        )
        if unified_echarts:
            yield {"data": unified_echarts, "type": "content"}

    async def _execute_shared_data_multi_questions_parallel(
            self,
            user_query: str,
            sub_items: List[Tuple[str, Optional[str]]],
            intent_result: Dict[str, Any],
            shared_intent: str,
            user_id: str,
            query_id: str,
    ) -> Dict[str, Any]:
        """共享数据非流式执行：共享一次 API，并对各子问题做受控并发。"""
        logger.info(
            f"[共享数据模式-非流式并发] intent={shared_intent}, sub_count={len(sub_items)}, query_id={query_id}"
        )

        workflow = await self._get_workflow(shared_intent)
        if not workflow:
            return {
                "status": "error",
                "message": {"content": f"不支持的意图: {shared_intent}"},
                "conversation_id": self.conversation_id,
                "query_id": query_id,
                "user": user_id
            }

        time_range = intent_result.get("time_range")
        if not self._has_time_range(time_range):
            return {
                "status": "error",
                "message": {"content": "缺少有效的时间范围"},
                "conversation_id": self.conversation_id,
                "query_id": query_id,
                "user": user_id
            }

        adjusted_time_range = time_range.copy()
        adjusted_time_range["start"] = self._normalize_date_str(adjusted_time_range.get("start"))
        adjusted_time_range["end"] = self._normalize_date_str(adjusted_time_range.get("end"))
        end_value = adjusted_time_range.get("end")
        try:
            end_dt = datetime.strptime(end_value, "%Y-%m-%d")
            adjusted_time_range["end"] = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        except ValueError:
            logger.warning(f"[共享数据模式-非流式并发] time_range.end 格式非法，跳过调整: {end_value}")

        params = {"time_range": adjusted_time_range}
        validation = workflow.validate_params(params)
        if not validation['valid']:
            return {
                "status": "error",
                "message": {"content": validation['message']},
                "conversation_id": self.conversation_id,
                "query_id": query_id,
                "user": user_id
            }

        try:
            api_result = await asyncio.wait_for(
                workflow.call_api(params),
                timeout=getattr(settings, 'REQUEST_TIMEOUT_SECONDS', 60)
            )
        except asyncio.TimeoutError:
            return {
                "status": "error",
                "message": {"content": "API调用超时，请稍后重试"},
                "conversation_id": self.conversation_id,
                "query_id": query_id,
                "user": user_id
            }

        if api_result['status'] == 'error':
            return {
                "status": "error",
                "message": {"content": f"API调用失败: {api_result.get('message')}"},
                "conversation_id": self.conversation_id,
                "query_id": query_id,
                "user": user_id
            }

        business_error = self._extract_api_business_error_from_result(workflow, api_result)
        if business_error:
            return {
                "status": "error",
                "message": {"content": f"API调用失败: {business_error}"},
                "conversation_id": self.conversation_id,
                "query_id": query_id,
                "user": user_id
            }

        raw_data = workflow._extract_api_data_payload(
            api_result,
            workflow_name="WorkflowRouterShared",
            is_retry=False,
        )
        raw_data, expanded_query_applied = await self._retry_shared_api_with_lookback_if_empty(
            workflow=workflow,
            params=params,
            raw_data=raw_data,
            shared_intent=shared_intent,
        )
        shared_raw_data = copy.deepcopy(raw_data)
        effective_root_query = (
            workflow._build_expanded_time_range_query(user_query, params.get("time_range", {}))
            if expanded_query_applied else user_query
        )

        concurrency = min(len(sub_items), self._get_subquestion_concurrency(shared=True))
        semaphore = asyncio.Semaphore(concurrency)
        states: List[Dict[str, Any]] = []

        for sub_q, _ in sub_items:
            effective_sub_q = (
                workflow._build_expanded_time_range_query(sub_q, params.get("time_range", {}))
                if expanded_query_applied else sub_q
            )
            states.append({
                "sub_question": sub_q,
                "effective_sub_question": effective_sub_q,
                "workflow": await self._get_isolated_workflow(shared_intent),
            })

        async def process_state(state: Dict[str, Any]) -> None:
            workflow_ref = state.get("workflow")
            if not isinstance(workflow_ref, BaseWorkflow):
                state["error"] = f"不支持的意图: {shared_intent}"
                return

            async with semaphore:
                result = await self._process_sub_question_with_shared_data(
                    state["effective_sub_question"],
                    shared_intent,
                    shared_raw_data,
                    params,
                    workflow_ref,
                )
            state["result"] = result

        await asyncio.gather(*(process_state(state) for state in states))

        valid_states = [
            state for state in states
            if isinstance(state.get("result"), dict) and "data" in state["result"]
        ]
        if not valid_states:
            return {
                "status": "error",
                "message": {"content": "所有子问题处理均失败"},
                "conversation_id": self.conversation_id,
                "query_id": query_id,
                "user": user_id
            }

        sub_metadata = [
            self._build_sub_truncation_metadata(
                state["sub_question"],
                state["result"].get("trunc_info"),
            )
            for state in valid_states
        ]
        # await cache_set(user_id, query_id, {
        #     "sub_metadata": sub_metadata,
        #     "conversation_id": self.conversation_id,
        #     "cached_at": datetime.now(),
        #     "truncated": any(item["truncated"] for item in sub_metadata),
        #     "truncation_types": list({t for item in sub_metadata for t in item.get("truncation_types", [])}),
        # })

        async def render_state(state: Dict[str, Any]) -> Dict[str, str]:
            workflow_ref = state["workflow"]
            result = state["result"]
            data = result["data"]
            format_source_data = result.get("format_source_data", data)
            sub_params = result.get("params", params)

            workflow_ref._truncated = result.get("trunc_info", {}).get("truncated", False)
            workflow_ref._field_truncated = "field_filter" in result.get("trunc_info", {}).get("truncation_types", [])
            workflow_ref._original_data = shared_raw_data

            async with semaphore:
                if workflow_ref._should_use_format_summary(format_source_data):
                    answer = await workflow_ref._format_result(state["effective_sub_question"], format_source_data)
                else:
                    analysis_result = await workflow_ref._run_data_analysis(
                        user_query=state["effective_sub_question"],
                        raw_data=shared_raw_data,
                        display_data=format_source_data,
                        params=sub_params,
                    )
                    answer = analysis_result.get("summary", "") if analysis_result else ""

            return {
                "sub_question": state["sub_question"],
                "answer": answer,
            }

        sub_answer_items = await asyncio.gather(*(render_state(state) for state in valid_states))
        unified_chart_sources = [
            (state["sub_question"], state["result"].get("data"))
            for state in valid_states
        ]

        final_content = await self._generate_merged_answer(effective_root_query, sub_answer_items)
        unified_chart_data = self._build_unified_chart_dataset(unified_chart_sources)
        unified_echarts = await self._generate_unified_echarts(
            user_query=effective_root_query,
            chart_dataset=unified_chart_data,
            workflow=workflow,
            intent=intent_result.get("intent", "")
        )
        message = {"content": final_content}
        if unified_echarts:
            message["echarts"] = unified_echarts

        return {
            "status": "success",
            "conversation_id": self.conversation_id,
            "workflow_type": "multi_shared_api",
            "message": message,
            "sub_questions_count": len(sub_items),
            "truncated": any(item["truncated"] for item in sub_metadata),
            "query_id": query_id,
            "user": user_id,
            "collected_params": params,
            "sub_metadata": sub_metadata,
            "sub_answers": sub_answer_items,
        }

    @staticmethod
    def _extract_result_content(result: Dict[str, Any]) -> str:
        """从工作流结果中提取用户可见内容。"""
        if not isinstance(result, dict):
            return str(result)

        message = result.get("message", {})
        if isinstance(message, dict):
            analysis_summary = message.get("analysis_summary")
            if isinstance(analysis_summary, str) and analysis_summary.strip():
                return analysis_summary.strip()

            trend_summary = message.get("trend_summary")
            if isinstance(trend_summary, str) and trend_summary.strip():
                return trend_summary.strip()

            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()

            clarification = message.get("clarification")
            if isinstance(clarification, dict):
                clarification_msg = clarification.get("message")
                if isinstance(clarification_msg, str) and clarification_msg.strip():
                    return clarification_msg.strip()

            suggestions = message.get("suggestions")
            if suggestions:
                return f"需要补充信息: {json.dumps(suggestions, ensure_ascii=False)}"

        if isinstance(message, str) and message.strip():
            return message.strip()

        error = result.get("error")
        if isinstance(error, str) and error.strip():
            return f"子问题执行失败: {error.strip()}"

        return "子问题未返回可用内容。"

    @staticmethod
    def _is_structured_chart_data(data: Any) -> bool:
        if isinstance(data, list):
            return len(data) > 0 and data != [{}]
        if isinstance(data, dict):
            return len(data) > 0
        return False

    @staticmethod
    def _extract_result_chart_data(result: Dict[str, Any]) -> Any:
        if not isinstance(result, dict):
            return None

        merged_chart_data = result.get("_merged_chart_data")
        if WorkflowRouter._is_structured_chart_data(merged_chart_data):
            return merged_chart_data

        message = result.get("message")
        if not isinstance(message, dict):
            return None

        for key in ("analysis_detail", "trend_analysis"):
            chart_data = message.get(key)
            if WorkflowRouter._is_structured_chart_data(chart_data):
                return chart_data
        return None

    @staticmethod
    def _build_unified_chart_dataset(items: List[Tuple[str, Any]]) -> List[Dict[str, Any]]:
        dataset: List[Dict[str, Any]] = []
        for sub_question, data in items:
            if not isinstance(sub_question, str):
                continue
            if not WorkflowRouter._is_structured_chart_data(data):
                continue
            dataset.append({
                "sub_question": sub_question,
                "data": data,
            })
        return dataset

    async def _generate_unified_echarts(
        self,
        user_query: str,
        chart_dataset: List[Dict[str, Any]],
        workflow: Optional[BaseWorkflow] = None,
        intent: str = "",
    ) -> str:
        if not chart_dataset:
            return ""

        sorted_data = sort_time_fields_batch(chart_dataset)

        # 判断是否生成他图
        if isinstance(sorted_data, dict):
            return f""
        if isinstance(sorted_data, list) and len(sorted_data) <= 1:
            return f""

        if is_dense_time_series_records(sorted_data):
            dense_option = build_dense_time_series_echarts(sorted_data, intent=intent)
            if not dense_option:
                return f""
            echarts_filled = json.dumps(dense_option, ensure_ascii=False)
            logger.info(f"[base_workflow] userQuery:{user_query}  ——>  判断[代码]生成明细图表")
            return f"\n\n```echarts\n{echarts_filled}```"

        # 模型生成图
        data_str = json.dumps(sorted_data, ensure_ascii=False, default=str)
        echarts_agent = workflow.echarts_agent if workflow is not None else await self._get_router_echarts_agent()

        if self.INNER_MODEL_ENABLE:
            echarts_task_result = await echarts_agent.ainvoke_for_echarts(user_query=user_query, data=data_str)
        else:
            echarts_task_result = await echarts_agent.ainvoke(user_query=user_query, data=data_str)
        echarts_task_result = filter_think_tags_simple(echarts_task_result)

        try:
            s = echarts_task_result.strip()
            pattern = r"^```(?:json)?\s*\n?(.*?)\n?```$"
            match = re.search(pattern, s, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
            else:
                json_str = s

            echarts_config = json.loads(json_str)
        except Exception:
            echarts_config = {}
            logger.info(f"echarts配置项生成格式错误，echarts_task_result:{echarts_task_result}")

        logger.info(
            f"[base_workflow] userQuery:{user_query} ——>  判断[模型]生成明细图表\n模型输出echarts配置：{echarts_config}")

        series = echarts_config.get('series', []) if isinstance(echarts_config, dict) else []
        # 判断图数据是否均为non_zero
        if not has_non_zero_series(series):
            return f""

        # 最高点、最低点添加
        config = add_global_extremes_to_echarts(echarts_config)
        echarts_filled = json.dumps(config, ensure_ascii=False)
        return f"\n\n```echarts\n{echarts_filled}```"


    @staticmethod
    def _normalize_stream_chunk(chunk: Any) -> Optional[Dict[str, Any]]:
        """将内部流式输出统一为 payload dict，兼容历史 SSE 字符串。"""
        if chunk is None:
            return None
        if isinstance(chunk, dict) and "data" in chunk and "type" in chunk:
            return chunk
        if isinstance(chunk, str):
            stripped = chunk.strip()
            if stripped.startswith("data:"):
                payload_text = stripped[5:].strip()
                try:
                    payload = json.loads(payload_text)
                    if isinstance(payload, dict) and "data" in payload and "type" in payload:
                        return payload
                except json.JSONDecodeError:
                    pass
            return {"data": chunk, "type": "content"}
        return {"data": chunk, "type": "content"}

    @staticmethod
    def _extract_sub_result_error(result: Any) -> str:
        if isinstance(result, dict):
            error = result.get("error")
            if isinstance(error, str) and error.strip():
                return error.strip()
            message = result.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
            elif isinstance(message, str) and message.strip():
                return message.strip()
        return "子问题未返回可用内容。"

    @staticmethod
    def _build_sub_truncation_metadata(sub_question: str, trunc_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        trunc_info = trunc_info or {}
        truncated = bool(trunc_info.get("truncated"))
        return {
            "sub_question": sub_question,
            "truncated": truncated,
            "truncation_types": trunc_info.get("truncation_types", []) if truncated else [],
            "total_count": trunc_info.get("total_count"),
            "display_count": trunc_info.get("display_count"),
            "max_time_field": trunc_info.get("max_time_field"),
        }

    @staticmethod
    def _get_subquestion_concurrency(shared: bool = False) -> int:
        key = "MAX_SHARED_SUBQUESTION_CONCURRENCY" if shared else "MAX_SUBQUESTION_CONCURRENCY"
        raw_value = getattr(settings, key, None)
        if raw_value is None:
            raw_value = getattr(settings, "MAX_SUBQUESTION_CONCURRENCY", 4)
        try:
            return max(1, int(raw_value))
        except (TypeError, ValueError):
            return 4

    async def _stream_parallel_subquestion_workers(
            self,
            workers: List[Dict[str, Any]],
            label_owner_index: int = 1,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        并发运行所有子问题，但按子问题段落连续输出。
        label_owner_index 指定的子问题优先输出，确保阶段标签在内容之前。
        """
        stream_end = object()

        async def producer(worker: Dict[str, Any]) -> None:
            queue: asyncio.Queue = worker["queue"]
            try:
                async for chunk in worker["stream"]:
                    payload = self._normalize_stream_chunk(chunk)
                    if payload is not None:
                        await queue.put(payload)
            except Exception as e:
                logger.error(f"[多子问题流式] 子问题流生产失败: {worker['sub_question']}, 错误: {e}", exc_info=True)
                friendly = WorkflowRouter._to_user_friendly_stream_error(str(e))
                await queue.put({"data": friendly, "type": "error"})
            finally:
                await queue.put(stream_end)

        tasks = [asyncio.create_task(producer(worker)) for worker in workers]
        pending_indexes = [worker["index"] for worker in workers]
        worker_map = {worker["index"]: worker for worker in workers}

        is_first_round = True

        try:
            while pending_indexes:
                # 首轮优先等 label_owner_index 产出，确保标签在内容之前
                if is_first_round and label_owner_index in pending_indexes:
                    is_first_round = False
                    first_chunk_tasks = {
                        asyncio.create_task(worker_map[label_owner_index]["queue"].get()): label_owner_index
                    }
                    done, _ = await asyncio.wait(
                        first_chunk_tasks.keys(),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                else:
                    first_chunk_tasks = {
                        asyncio.create_task(worker_map[idx]["queue"].get()): idx
                        for idx in pending_indexes
                    }
                    done, waiting = await asyncio.wait(
                        first_chunk_tasks.keys(),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in waiting:
                        task.cancel()
                    if waiting:
                        await asyncio.gather(*waiting, return_exceptions=True)

                selected_task = next(iter(done))
                selected_idx = first_chunk_tasks[selected_task]
                first_payload = await selected_task
                worker = worker_map[selected_idx]

                if first_payload is stream_end:
                    pending_indexes.remove(selected_idx)
                    continue

                if selected_idx == label_owner_index:
                    # label owner：标签到达时立即输出，标题在首个非标签 chunk 之前插入
                    labels_active = (
                        isinstance(first_payload, dict)
                        and first_payload.get("type") == "messageLabel"
                    )
                    if labels_active:
                        yield first_payload
                    else:
                        yield {"data": f"\n\n**{worker['sub_question']}**\n\n", "type": "content"}
                        yield first_payload

                    while True:
                        payload = await worker["queue"].get()
                        if payload is stream_end:
                            pending_indexes.remove(selected_idx)
                            break
                        is_label = isinstance(payload, dict) and payload.get("type") == "messageLabel"
                        if labels_active and is_label:
                            yield payload
                        elif labels_active and not is_label:
                            labels_active = False
                            yield {"data": f"\n\n**{worker['sub_question']}**\n\n", "type": "content"}
                            yield payload
                        else:
                            yield payload
                else:
                    yield {"data": f"\n\n**{worker['sub_question']}**\n\n", "type": "content"}
                    yield first_payload

                    while True:
                        payload = await worker["queue"].get()
                        if payload is stream_end:
                            pending_indexes.remove(selected_idx)
                            break
                        yield payload
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    def _fallback_merged_answer(original_query: str, sub_answers: List[Dict[str, str]]) -> str:
        usable = [item.get("answer", "").strip() for item in sub_answers if item.get("answer")]
        if not usable:
            return f"围绕你的问题{original_query}，暂未获取到可用结果。"
        if len(usable) == 1:
            return usable[0]
        return f"围绕你的问题{original_query}，综合结果如下：\n\n" + "\n\n".join(usable)

    async def _generate_merged_answer(self, original_query: str, sub_answers: List[Dict[str, str]]) -> str:
        """将多子问题结果整合为一个最终回答。"""
        if len(sub_answers) <= 1:
            return sub_answers[0].get("answer", "") if sub_answers else ""

        prompt = (
            "你是一个多子问题答案整合助手。\n"
            "任务：基于原始问题和若干子问题的回答，输出一个统一、连贯、面向用户的最终回答。\n"
            "要求：\n"
            "1. 必须围绕原始问题作答，不要逐条复述子问题标题。\n"
            "2. 合并重复信息，保留关键结论、数据与约束。\n"
            "3. 如果部分子问题失败，要在最终回答里自然说明缺失点。\n"
            "4. 只输出最终答案正文，不输出分析过程。\n\n"
            f"原始问题：{original_query}\n\n"
            f"子问题结果（JSON）：{json.dumps(sub_answers, ensure_ascii=False, default=str)}"
        )

        try:
            if self.INNER_MODEL_ENABLE:
                inner_model_agent = agent_factory.create_inner_model_agent()
                merged = await inner_model_agent.ainvoke([{"role": "user", "content": prompt}])
            else:
                model = await self._get_format_model()
                response = await model.ainvoke([{"role": "user", "content": prompt}])
                merged = response.content if hasattr(response, "content") else str(response)

            merged = (merged or "").strip()
            if merged.startswith("```"):
                merged = merged.split("```", 1)[1]
                if "```" in merged:
                    merged = merged.split("```", 1)[0]
                merged = merged.strip()
            if merged:
                return merged
        except Exception as e:
            logger.warning(f"[多子问题整合] 模型整合失败，使用兜底策略: {e}")

        return self._fallback_merged_answer(original_query, sub_answers)

    async def _execute_multi_questions_merged(
            self,
            user_query: str,
            sub_items: List[Tuple[str, Optional[str]]],
            intent_result: Dict[str, Any],
            user_id: str,
            query_id: str
    ) -> Dict[str, Any]:
        """多子问题统一执行入口，返回一个整合后的最终结果。"""
        if self._can_share_api_call(sub_items):
            shared_intent = sub_items[0][1]
            logger.info(f"[多子问题整合] 共享数据模式: intent={shared_intent}, sub_count={len(sub_items)}")
            merged = await self._execute_shared_data_multi_questions_parallel(
                user_query, sub_items, intent_result, shared_intent, user_id, query_id
            )
            merged["query_id"] = query_id
            merged["user"] = user_id
            return merged

        logger.info(f"[多子问题整合] 并行执行模式: sub_count={len(sub_items)}")
        tasks = [
            self._execute_single_question_with_intent(
                sub_q,
                sub_intent,
                user_id,
                f"{query_id}_{idx}",
                collect_chart_data=True,
            )
            for idx, (sub_q, sub_intent) in enumerate(sub_items)
        ]
        timeout = getattr(settings, 'REQUEST_TIMEOUT_SECONDS', 60)
        sub_results_raw = await asyncio.gather(
            *[asyncio.wait_for(t, timeout=timeout) for t in tasks],
            return_exceptions=True
        )

        results_with_question = []
        for idx, res in enumerate(sub_results_raw):
            if isinstance(res, Exception):
                res = {
                    "status": "error",
                    "error": str(res),
                    "message": {"content": f"子问题处理超时或失败: {str(res)}"}
                }
            results_with_question.append({
                "sub_question": sub_items[idx][0],
                "result": res
            })

        merged = await self._merge_sub_results(results_with_question, original_query=user_query)
        merged["query_id"] = query_id
        merged["user"] = user_id
        return merged

    # ---------- 流式执行入口 ----------
    async def _execute_multi_questions_stream(
            self,
            user_query: str,
            sub_items: List[Tuple[str, Optional[str]]],
            intent_result: Dict[str, Any],
            user_id: str,
            query_id: str,
    ) -> AsyncGenerator[Any, None]:
        """多子问题流式执行入口：并发处理子问题，并按子问题段落输出。"""
        if self._can_share_api_call(sub_items):
            shared_intent = sub_items[0][1]
            async for chunk in self._execute_shared_data_multi_questions_stream_parallel(
                    user_query=user_query,
                    sub_items=sub_items,
                    intent_result=intent_result,
                    shared_intent=shared_intent,
                    user_id=user_id,
                    query_id=query_id,
            ):
                normalized = self._normalize_stream_chunk(chunk)
                if normalized is not None:
                    yield normalized
            return


        logger.info(f"[多子问题流式] 并发执行模式: sub_count={len(sub_items)}")
        concurrency = min(len(sub_items), self._get_subquestion_concurrency(shared=False))
        semaphore = asyncio.Semaphore(concurrency)
        total = len(sub_items)
        states: List[Dict[str, Any]] = []
        workers: List[Dict[str, Any]] = []
        labels_owner = [1]  # 固定由第一个子问题（index=1）拥有 label 输出权

        for idx, (sub_q, sub_intent) in enumerate(sub_items, 1):
            state: Dict[str, Any] = {
                "index": idx,
                "sub_question": sub_q,
            }
            states.append(state)

            async def bounded_stream(
                    *,
                    sub_question: str = sub_q,
                    sub_intent_value: str = sub_intent or "",
                    idx_value: int = idx,
                    state_ref: Dict[str, Any] = state,
            ) -> AsyncGenerator[Any, None]:
                async with semaphore:
                    stream_end = object()
                    chunk_queue: asyncio.Queue = asyncio.Queue()
                    summary_done = asyncio.Event()

                    async def produce_summary_chunks() -> None:
                        try:
                            async for chunk in self._execute_single_question_stream(
                                    sub_question,
                                    sub_intent_value,
                                    format_context={"is_multi": True, "index": idx_value, "total": total},
                                    isolate_workflow=True,
                                    runtime_state=state_ref,
                            ):
                                if isinstance(chunk, dict) and chunk.get("type") == "messageLabel":
                                    if labels_owner[0] != idx_value:
                                        continue
                                await chunk_queue.put(chunk)
                        except Exception as e:
                            logger.error(f"[多子问题流式] 子问题总结流失败: {sub_question}, 错误: {e}", exc_info=True)
                            friendly = WorkflowRouter._to_user_friendly_stream_error(str(e))
                            await chunk_queue.put({"data": friendly, "type": "error"})
                        finally:
                            summary_done.set()
                            await chunk_queue.put(stream_end)

                    async def build_chart_after_summary() -> Optional[Dict[str, Any]]:
                        await summary_done.wait()
                        workflow_ref = state_ref.get("workflow")
                        if not isinstance(workflow_ref, BaseWorkflow):
                            return None
                        chart_data = getattr(workflow_ref, "_last_chart_data", None)
                        if not self._is_structured_chart_data(chart_data):
                            return None

                        single_chart_dataset = self._build_unified_chart_dataset([
                            (sub_question, copy.deepcopy(chart_data))
                        ])
                        single_echarts = await self._generate_unified_echarts(
                            user_query=sub_question,
                            chart_dataset=single_chart_dataset,
                            workflow=workflow_ref,
                            intent=intent_result.get("intent", "")
                        )
                        if not single_echarts:
                            return None
                        return {"data": single_echarts, "type": "content"}

                    producer_task = asyncio.create_task(produce_summary_chunks())
                    chart_task = asyncio.create_task(build_chart_after_summary())
                    try:
                        while True:
                            payload = await chunk_queue.get()
                            if payload is stream_end:
                                break
                            yield payload

                        # yield {"data": "", "type": "Placeholder_True"}
                        chart_payload = await chart_task
                        # yield {"data": "", "type": "Placeholder_False"}
                        if chart_payload is not None:
                            yield chart_payload
                    finally:
                        if not producer_task.done():
                            producer_task.cancel()
                        if not chart_task.done():
                            chart_task.cancel()
                        await asyncio.gather(producer_task, chart_task, return_exceptions=True)

            workers.append({
                "index": idx,
                "sub_question": sub_q,
                "queue": asyncio.Queue(),
                "stream": bounded_stream(),
            })

        self.current_workflow = None
        async for chunk in self._stream_parallel_subquestion_workers(workers):
            yield chunk

        sub_metadata = []
        for state in states:
            workflow = state.get("workflow")
            if not isinstance(workflow, BaseWorkflow):
                continue

            trunc_info = workflow.get_any_truncation_info()
            sub_metadata.append(self._build_sub_truncation_metadata(state["sub_question"], trunc_info))

        truncated_any = any(item.get("truncated") for item in sub_metadata)
        if truncated_any:
            # await cache_set(user_id, query_id, {
            #     "sub_metadata": sub_metadata,
            #     "conversation_id": self.conversation_id,
            #     "cached_at": datetime.now(),
            #     "truncated": True,
            #     "truncation_types": list({t for item in sub_metadata for t in item.get("truncation_types", [])}),
            # })
            yield {
                "data": {
                    "truncated": True,
                    "query_id": query_id,
                    "conversation_id": self.conversation_id,
                    "user": user_id,
                    "type": "multi_parallel",
                    "sub_metadata": sub_metadata,
                },
                "type": "data_truncated",
            }

    async def execute_stream(self, user_query: str, user_id: str, query_id: str) -> AsyncGenerator[Any, None]:
        """执行工作流路由（流式）"""
        intents = []
        try:
            logger.info(f"[工作流路由-流式] 开始处理请求, conversation_id={self.conversation_id}")

            # 问题改写
            yield {"data": "problem rewriting", "type": "messageLabel"}
            try:
                rewrite_wf = await self._get_rewrite_workflow()
                user_query = await asyncio.wait_for(rewrite_wf.rewrite(user_query), timeout=30)
            except asyncio.TimeoutError:
                logger.warning("问题改写超时，使用原始问题")
            except Exception as e:
                logger.warning(f"问题改写异常: {e}，使用原始问题")

            # 意图识别第一阶段
            yield {"data": "finish", "type": "messageLabel"}
            yield {"data": "intent recognition Ⅰ", "type": "messageLabel"}
            intent_result = await self._recognize_intent(user_query)
            intents = intent_result.get("intents", [])
            out_of_scope = intent_result.get("out_of_scope", False)

            # 不属于任意意图的情况
            if out_of_scope:
                logger.info("[无效问题] 无法匹配任意意图")
                yield {"data": "finish", "type": "messageLabel"}
                yield {
                    "data": {"out_of_scope": out_of_scope},
                    "type": "out_of_scope",
                }
                return

            # 待澄清
            if not intents:
                logger.info("[待澄清-流式] 未识别到业务意图")
                clarification_workflow = await self._get_workflow('待澄清')
                result = await clarification_workflow.execute(user_query, intent_result)
                clarification = result.get("message", {}).get("clarification", {})
                message = clarification.get("message", "请提供更多信息")
                suggestions = clarification.get("suggestions", [])
                yield sse({"data": message, "type": "content"})
                yield sse({"data": {"suggestions": suggestions}, "type": "metadata"})
                return

            # 向上层传递intents 信息
            yield {"intents": intents}

            yield {"data": "finish", "type": "messageLabel"}
            yield {"data": "problem breakdown", "type": "messageLabel"}
            sub_items = await self._resolve_sub_items(user_query, intent_result, log_prefix="[工作流路由-流式]")

            if len(sub_items) == 1:
                sub_q, sub_intent = sub_items[0]

                if not sub_intent or not await self._is_valid_business_intent(sub_intent):
                    logger.warning(f"[流式] 子问题意图无效: {sub_intent}，回退")
                    intent_result = await self._recognize_intent(sub_q)
                    intents = intent_result.get("intents", [])
                    if not intents:
                        yield sse({"data": "无法识别该子问题的业务意图。", "type": "error"})
                        return
                    sub_intent = intents[0]
                async for chunk in self._execute_single_question_stream(sub_q, sub_intent):
                    yield chunk
            else:
                async for chunk in self._execute_multi_questions_stream(
                        user_query=user_query,
                        sub_items=sub_items,
                        intent_result=intent_result,
                        user_id=user_id,
                        query_id=query_id,
                ):
                    yield chunk

        except Exception as e:
            logger.error(f"[工作流路由-流式] 执行错误: {e}", exc_info=True)
            friendly = self._to_user_friendly_stream_error(str(e))
            yield sse({"data": friendly, "type": "error"})

    async def _execute_single_question_stream(
            self,
            user_query: str,
            intent: str,
            format_context: Optional[Dict] = None,
            workflow_instance: Optional[BaseWorkflow] = None,
            isolate_workflow: bool = False,
            runtime_state: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Any, None]:
        """执行单个子问题（流式）"""
        try:
            if not isinstance(user_query, str):
                user_query = str(user_query)

            placeholder = {"question": user_query}
            yield {"data": "finish", "type": "messageLabel"}
            yield {"data": "intent recognition II", "type": "messageLabel"}
            detail_result = await self._recognize_domain_intent(intent, user_query, placeholder)

            if detail_result.get("clarification_needed"):
                clarification = detail_result.get("clarification", {})
                message = clarification.get("message", "")
                suggestions = clarification.get("suggestions", [])
                yield sse({"data": message, "type": "content"})
                yield sse({"data": {"suggestions": suggestions}, "type": "metadata"})
                return

            detail_intent = detail_result.get("intent") if isinstance(detail_result, dict) else None
            if detail_intent and await self._is_valid_business_intent(detail_intent):
                intent = detail_intent

            workflow = workflow_instance
            if workflow is not None and getattr(workflow, "interface_name", None) != intent:
                workflow = None

            if workflow is None:
                workflow = await (
                    self._get_isolated_workflow(intent)
                    if isolate_workflow else self._get_workflow(intent)
                )
            if workflow is None:
                yield sse({"data": f"抱歉，暂不支持 {intent} 相关的查询。", "type": "error"})
                return

            if isinstance(runtime_state, dict):
                runtime_state["workflow"] = workflow
                runtime_state["intent"] = intent

            if not isolate_workflow and workflow_instance is None:
                self.current_workflow = workflow
            async for chunk in workflow.execute_stream(user_query, detail_result, format_context=format_context):
                yield chunk   
        except Exception as e:
            logger.error(f"[子问题流式执行] 错误: {e}", exc_info=True)
            friendly = self._to_user_friendly_stream_error(str(e))
            yield sse({"data": friendly, "type": "error"})

    # ---------- 非流式执行----------
    async def execute(self, user_query: str, user_id: str = "anonymous", query_id: Optional[str] = None) -> Dict[
        str, Any]:
        """
        执行工作流路由（非流式，并发增强版）

        Args:
            user_query: 用户查询
            user_id: 用户标识，用于缓存隔离
            query_id: 可选查询ID，若不传则自动生成

        Returns:
            标准响应字典，包含 status、data、conversation_id 等字段
        """
        import uuid
        if query_id is None:
            query_id = str(uuid.uuid4())

        try:
            logger.info(f"[工作流路由-非流式] 开始处理请求, conversation_id={self.conversation_id}")

            # 问题改写
            try:
                rewrite_wf = await self._get_rewrite_workflow()
                user_query = await asyncio.wait_for(rewrite_wf.rewrite(user_query), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("问题改写超时，使用原始问题")
            except Exception as e:
                logger.warning(f"问题改写异常: {e}，使用原始问题")

            # 第一阶段意图识别（带重试）
            intent_result = await self._recognize_intent(user_query)
            intents = intent_result.get("intents", [])
            out_of_scope = intent_result.get("out_of_scope", False)

            # 不属于任意意图的情况
            if out_of_scope:
                logger.info("[无效问题] 无法匹配任意意图")
                return {
                    "out_of_scope":out_of_scope
                }

            # 待澄清情况
            if not intents:
                logger.info("[待澄清-非流式] 未识别到业务意图")
                clarification_workflow = await self._get_workflow('待澄清')
                result = await clarification_workflow.execute(user_query, intent_result)
                clarification = result.get("message", {}).get("clarification", {})
                message = clarification.get("message", "请提供更多信息")
                suggestions = clarification.get("suggestions", [])
                return {
                    "status": "need_more_info",
                    "intent": "待澄清",
                    "message": {"content": message, "suggestions": suggestions},
                    "conversation_id": self.conversation_id,
                    "query_id": query_id,
                    "user": user_id
                }



            # 子问题拆分
            sub_items = await self._resolve_sub_items(
                user_query,
                intent_result,
                skip_split=False,
                log_prefix="[工作流路由-非流式]"
            )

            # 单子问题处理
            if len(sub_items) == 1:
                sub_q, sub_intent = sub_items[0]

                if not sub_intent or not await self._is_valid_business_intent(sub_intent):
                    intent_result = await self._recognize_intent(sub_q)
                    intents = intent_result.get("intents", [])
                    if not intents:
                        return {
                            "status": "error",
                            "message": {"content": "无法识别该问题的业务意图。"},
                            "conversation_id": self.conversation_id,
                            "query_id": query_id,
                            "user": user_id
                        }
                    sub_intent = intents[0]

                # 执行单个子问题
                result = await self._execute_single_question_with_intent(
                    sub_q, sub_intent, user_id, query_id
                )
                result["query_id"] = query_id
                result["user"] = user_id
                return result

            # 多子问题处理
            else:
                return await self._execute_multi_questions_merged(
                    user_query=user_query,
                    sub_items=sub_items,
                    intent_result=intent_result,
                    user_id=user_id,
                    query_id=query_id,
                )

        except Exception as e:
            logger.error(f"[工作流路由-非流式] 执行错误: {e}", exc_info=True)
            friendly = self._to_user_friendly_stream_error(str(e))
            return {
                "status": "error",
                "error": friendly,
                "message": {"content": friendly},
                "conversation_id": self.conversation_id,
                "query_id": query_id,
                "user": user_id
            }

    async def _execute_single_question_with_intent(
            self,
            user_query: str,
            intent: str,
            user_id: str = "anonymous",
            query_id: str = "",
            collect_chart_data: bool = False,
    ) -> Dict[str, Any]:
        """
        执行单个子问题（非流式），已知意图
        包含缓存写入逻辑
        """
        try:
            # 专项意图识别（带重试）
            placeholder = {"question": user_query, "intents": [intent]}
            detail_result = await self._recognize_domain_intent(intent, user_query, placeholder)

            if detail_result.get("clarification_needed"):
                logger.info(f"[澄清触发] 专项识别返回澄清需求")
                clarification_workflow = await self._get_workflow('待澄清')
                result = await clarification_workflow.execute(user_query, detail_result)
                clarification = result.get("message", {}).get("clarification", {})
                return {
                    "status": "need_more_info",
                    "intent": "待澄清",
                    "message": {"content": clarification.get("message", ""),
                                "suggestions": clarification.get("suggestions", [])},
                    "conversation_id": self.conversation_id,
                    "query_id": query_id,
                    "user": user_id
                }

            # 确定最终意图
            detail_intent = detail_result.get("intent") if isinstance(detail_result, dict) else None
            if detail_intent and await self._is_valid_business_intent(detail_intent):
                intent = detail_intent

            workflow = await self._get_workflow(intent)
            if workflow is None:
                return {
                    "status": "error",
                    "intent": intent,
                    "message": {"content": f"抱歉，暂不支持 {intent} 相关的查询。"},
                    "conversation_id": self.conversation_id,
                    "query_id": query_id,
                    "user": user_id
                }

            self.current_workflow = workflow
            result = await workflow.execute(
                user_query,
                detail_result,
                skip_echarts=collect_chart_data,
            )
            if collect_chart_data and isinstance(result, dict):
                chart_data = getattr(workflow, "_last_chart_data", None)
                if self._is_structured_chart_data(chart_data):
                    result["_merged_chart_data"] = copy.deepcopy(chart_data)
            result['intent'] = intent
            result['query_id'] = query_id
            result['user'] = user_id

            # 处理截断信息，写入缓存
            trunc_info = workflow.get_any_truncation_info()
            if trunc_info:
                # from src.api.routers.shared_cache import cache_set
                # cache_entry = {
                #     "conversation_id": self.conversation_id,
                #     "cached_at": datetime.now(),
                #     "truncation_types": trunc_info.get("truncation_types", [])
                # }
                # if "total_count" in trunc_info:
                #     cache_entry["total_count"] = trunc_info["total_count"]
                #     cache_entry["display_count"] = trunc_info["display_count"]
                # if "max_time_field" in trunc_info:
                #     cache_entry["max_time_field"] = trunc_info["max_time_field"]
                # await cache_set(user_id, query_id, cache_entry)
                result["truncated"] = True
                result["truncation_types"] = trunc_info.get("truncation_types", [])
                # logger.info(f"[缓存] 非流式写入截断数据: user={user_id}, query_id={query_id}")

            return result

        except Exception as e:
            logger.error(f"[单个子问题执行] 错误: {e}", exc_info=True)
            friendly = self._to_user_friendly_stream_error(str(e))
            return {
                "status": "error",
                "error": friendly,
                "message": {"content": friendly},
                "conversation_id": self.conversation_id,
                "query_id": query_id,
                "user": user_id
            }

    # async def _execute_shared_data_multi_questions(
    #         self,
    #         user_query: str,
    #         sub_items: List[Tuple[str, Optional[str]]],
    #         intent_result: Dict[str, Any],
    #         shared_intent: str,
    #         user_id: str,
    #         query_id: str
    # ) -> Dict[str, Any]:
    #     """
    #     非流式共享数据模式：多子问题共享一次API调用
    #     逻辑与流式版本对称，但不输出流式块，而是返回完整响应
    #     """
    #     logger.info(
    #         f"[共享数据模式-非流式] 意图 {shared_intent} 下 {len(sub_items)} 个子问题共享API，query_id={query_id}")
    #
    #     workflow = await self._get_workflow(shared_intent)
    #     if not workflow:
    #         return {
    #             "status": "error",
    #             "message": {"content": f"不支持的意图: {shared_intent}"},
    #             "conversation_id": self.conversation_id,
    #             "query_id": query_id,
    #             "user": user_id
    #         }
    #
    #     time_range = intent_result.get("time_range")
    #     if not self._has_time_range(time_range):
    #         return {
    #             "status": "error",
    #             "message": {"content": "缺少有效的时间范围"},
    #             "conversation_id": self.conversation_id,
    #             "query_id": query_id,
    #             "user": user_id
    #         }
    #
    #     # 调整 end 日期
    #     adjusted_time_range = time_range.copy()
    #     adjusted_time_range["start"] = self._normalize_date_str(adjusted_time_range.get("start"))
    #     adjusted_time_range["end"] = self._normalize_date_str(adjusted_time_range.get("end"))
    #     end = adjusted_time_range.get("end")
    #     try:
    #         end_dt = datetime.strptime(end, "%Y-%m-%d")
    #         new_end = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    #         adjusted_time_range["end"] = new_end
    #         logger.info(f"[共享数据模式] time_range.end 调整: {end} → {new_end}")
    #     except ValueError:
    #         logger.warning(f"[共享数据模式] time_range.end 格式不合法，跳过调整: {end}")
    #
    #     params = {"time_range": adjusted_time_range}
    #     validation = workflow.validate_params(params)
    #     if not validation['valid']:
    #         return {
    #             "status": "error",
    #             "message": {"content": validation['message']},
    #             "conversation_id": self.conversation_id,
    #             "query_id": query_id,
    #             "user": user_id
    #         }
    #
    #     # API调用（带超时）
    #     try:
    #         api_result = await asyncio.wait_for(
    #             workflow.call_api(params),
    #             timeout=getattr(settings, 'REQUEST_TIMEOUT_SECONDS', 60)
    #         )
    #     except asyncio.TimeoutError:
    #         return {
    #             "status": "error",
    #             "message": {"content": "API调用超时，请稍后重试"},
    #             "conversation_id": self.conversation_id,
    #             "query_id": query_id,
    #             "user": user_id
    #         }
    #
    #     if api_result['status'] == 'error':
    #         return {
    #             "status": "error",
    #             "message": {"content": f"API调用失败: {api_result.get('message')}"},
    #             "conversation_id": self.conversation_id,
    #             "query_id": query_id,
    #             "user": user_id
    #         }
    #
    #     business_error = self._extract_api_business_error_from_result(workflow, api_result)
    #     if business_error:
    #         return {
    #             "status": "error",
    #             "message": {"content": f"API调用失败: {business_error}"},
    #             "conversation_id": self.conversation_id,
    #             "query_id": query_id,
    #             "user": user_id
    #         }
    #
    #     raw_data = workflow._extract_api_data_payload(
    #         api_result,
    #         workflow_name="WorkflowRouterShared",
    #         is_retry=False,
    #     )
    #     raw_data, expanded_query_applied = await self._retry_shared_api_with_lookback_if_empty(
    #         workflow=workflow,
    #         params=params,
    #         raw_data=raw_data,
    #         shared_intent=shared_intent,
    #     )
    #     shared_raw_data = copy.deepcopy(raw_data)
    #
    #     effective_root_query = (
    #         workflow._build_expanded_time_range_query(user_query, params.get("time_range", {}))
    #         if expanded_query_applied else user_query
    #     )
    #
    #     # 处理各子问题
    #     sub_results = []
    #     for sub_q, sub_intent in sub_items:
    #         effective_sub_q = (
    #             workflow._build_expanded_time_range_query(sub_q, params.get("time_range", {}))
    #             if expanded_query_applied else sub_q
    #         )
    #         sub_result = await self._process_sub_question_with_shared_data(
    #             effective_sub_q, shared_intent, shared_raw_data, params, workflow
    #         )
    #         sub_results.append(sub_result)
    #
    #     valid_results = [r for r in sub_results if "data" in r]
    #     if not valid_results:
    #         return {
    #             "status": "error",
    #             "message": {"content": "所有子问题处理均失败"},
    #             "conversation_id": self.conversation_id,
    #             "query_id": query_id,
    #             "user": user_id
    #         }
    #
    #     # 汇总截断元数据
    #     sub_metadata = []
    #     for r in valid_results:
    #         trunc_info = r.get("trunc_info")
    #         if trunc_info:
    #             sub_metadata.append({
    #                 "sub_question": r["sub_question"],
    #                 "truncated": trunc_info.get("truncated", False),
    #                 "truncation_types": trunc_info.get("truncation_types", []),
    #                 "total_count": trunc_info.get("total_count"),
    #                 "display_count": trunc_info.get("display_count"),
    #                 "max_time_field": trunc_info.get("max_time_field")
    #             })
    #         else:
    #             sub_metadata.append({"sub_question": r["sub_question"], "truncated": False})
    #
    #     # 写入缓存
    #     await cache_set(user_id, query_id, {
    #         # 后续不再缓存全量数据。
    #         # "full_data": shared_raw_data,
    #         "sub_metadata": sub_metadata,
    #         "conversation_id": self.conversation_id,
    #         "cached_at": datetime.now(),
    #         "truncated": any(m["truncated"] for m in sub_metadata),
    #         "truncation_types": list(set(t for m in sub_metadata for t in m.get("truncation_types", []))),
    #     })
    #
    #     # 汇总每个子问题格式化结果，再整合为一个总回答
    #     sub_answer_items = []
    #     unified_chart_sources: List[Tuple[str, Any]] = []
    #     for r in valid_results:
    #         sub_q = r["sub_question"]
    #         data = r["data"]
    #         format_source_data = r.get("format_source_data", data)
    #         sub_params = r.get("params", params)
    #         workflow._truncated = r.get("trunc_info", {}).get("truncated", False)
    #         workflow._field_truncated = "field_filter" in r.get("trunc_info", {}).get("truncation_types", [])
    #         workflow._original_data = shared_raw_data
    #
    #         if workflow._should_use_format_summary(format_source_data):
    #             answer = await workflow._format_result(sub_q, format_source_data)
    #         else:
    #             analysis_result = await workflow._run_data_analysis(
    #                 user_query=sub_q,
    #                 raw_data=shared_raw_data,
    #                 display_data=format_source_data,
    #                 params=sub_params,
    #             )
    #             answer = analysis_result.get("summary", "") if analysis_result else ""
    #
    #         sub_answer_items.append({
    #             "sub_question": sub_q,
    #             "answer": answer
    #         })
    #         unified_chart_sources.append((sub_q, data))
    #
    #     final_content = await self._generate_merged_answer(effective_root_query, sub_answer_items)
    #     unified_chart_data = self._build_unified_chart_dataset(unified_chart_sources)
    #     unified_echarts = await self._generate_unified_echarts(
    #         user_query=effective_root_query,
    #         chart_dataset=unified_chart_data,
    #         workflow=workflow,
    #     )
    #     message = {"content": final_content}
    #     if unified_echarts:
    #         message["echarts"] = unified_echarts
    #
    #     response = {
    #         "status": "success",
    #         "conversation_id": self.conversation_id,
    #         "workflow_type": "multi_shared_api",
    #         "message": message,
    #         "sub_questions_count": len(sub_items),
    #         "truncated": any(m["truncated"] for m in sub_metadata),
    #         "query_id": query_id,
    #         "user": user_id,
    #         "collected_params": params,
    #         "sub_metadata": sub_metadata,
    #         "sub_answers": sub_answer_items,
    #     }
    #     return response

    async def _merge_sub_results(
            self,
            sub_results: List[Dict],
            original_query: str,
            show_titles: bool = False
    ) -> Dict[str, Any]:
        """合并多个子问题执行结果并产出统一最终回答。"""
        sub_answer_items = []
        sub_metadata = []
        unified_chart_sources: List[Tuple[str, Any]] = []
        for idx, item in enumerate(sub_results, 1):
            sub_q = item["sub_question"]
            result = item["result"]
            content = self._extract_result_content(result)

            if show_titles:
                content = f"【子问题：{sub_q}】\n{content}"

            sub_answer_items.append({
                "index": idx,
                "sub_question": sub_q,
                "answer": content,
                "status": result.get("status", "unknown") if isinstance(result, dict) else "unknown",
            })
            unified_chart_sources.append((sub_q, self._extract_result_chart_data(result)))
            if isinstance(result, dict):
                truncated = bool(result.get("truncated", False))
                sub_metadata.append({
                    "sub_question": sub_q,
                    "truncated": truncated,
                    "truncation_types": result.get("truncation_types", []) if truncated else [],
                    "total_count": result.get("total_count"),
                    "display_count": result.get("display_count"),
                    "max_time_field": result.get("max_time_field"),
                })

        merged_content = await self._generate_merged_answer(original_query, sub_answer_items)
        truncated_any = any(item.get("truncated") for item in sub_metadata)
        unified_chart_data = self._build_unified_chart_dataset(unified_chart_sources)
        unified_echarts = await self._generate_unified_echarts(
            user_query=original_query,
            chart_dataset=unified_chart_data,
        )
        message = {"content": merged_content}
        if unified_echarts:
            message["echarts"] = unified_echarts
        sanitized_sub_results = []
        for item in sub_results:
            if not isinstance(item, dict):
                sanitized_sub_results.append(item)
                continue
            sanitized_item = dict(item)
            result = sanitized_item.get("result")
            if isinstance(result, dict) and "_merged_chart_data" in result:
                result = dict(result)
                result.pop("_merged_chart_data", None)
                sanitized_item["result"] = result
            sanitized_sub_results.append(sanitized_item)

        return {
            "conversation_id": self.conversation_id,
            "status": "success",
            "intent": "多问题组合",
            "original_query": original_query,
            "sub_questions_count": len(sub_results),
            "message": message,
            "sub_results": sanitized_sub_results,
            "sub_answers": sub_answer_items,
            "truncated": truncated_any,
            "sub_metadata": sub_metadata
        }

    def get_any_truncation_info(self) -> Optional[Dict]:
        """获取当前工作流的数据截断信息"""
        if self.current_workflow:
            return self.current_workflow.get_any_truncation_info()
        return None
