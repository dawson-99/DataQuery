import asyncio
import copy
import time
import json
import re
from distutils.util import strtobool

import httpx
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, Optional, List
from collections.abc import AsyncGenerator
from abc import ABC, abstractmethod
#
# from Tools.scripts.dutree import display

from src.agents.parameter_agent import ParameterAgent
from src.agents.format_agent import FormatAgent
from src.agents.aggregation_agent import AggregationAgent
from src.agents.trend_analysis_agent import TrendAnalysisAgent
from src.config import settings
from src.service.agent_factory import agent_factory
from src.utils.echarts_utils import is_dense_time_series_records, build_dense_time_series_echarts, has_non_zero_series, \
    add_global_extremes_to_echarts, sort_time_fields_batch
from src.utils.logging_setup import logger
from src.utils.filter_think_tags import filter_think_tags_async, filter_think_tags_simple, filter_think_tags_async_v2
from src.utils.python_sandbox import PythonSandbox, SandboxExecutionError
from src.utils.data_standard import data_matching
from langchain_qwq import ChatQwen

from src.utils.summary_util import handle_summary_input

MAX_DISPLAY_ITEMS = 24

with open("data/env_variables/time_point_mapping.json", "r", encoding="utf-8") as _f:
    _time_point_mapping = json.load(_f)
_CANONICAL_TIME_POINT_FIELDS: set = {k for k in _time_point_mapping if k != "aliases"}
_CANONICAL_TIME_POINT_COUNT: int = len(_CANONICAL_TIME_POINT_FIELDS)

_shared_http_client: Optional[httpx.AsyncClient] = None
_client_lock = asyncio.Lock()

async def get_shared_http_client(timeout: float = 180.0) -> httpx.AsyncClient:
    global _shared_http_client
    if _shared_http_client is None:
        async with _client_lock:
            if _shared_http_client is None:
                limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
                _shared_http_client = httpx.AsyncClient(
                    limits=limits,
                    timeout=httpx.Timeout(timeout),
                )
                logger.info(f"[BaseWorkflow] 初始化共享 HTTP 客户端，超时={timeout}s")
    return _shared_http_client

class BaseWorkflow(ABC):
    """业务工作流基类
    
    定义了所有业务工作流的通用流程：
    1. 参数提取
    2. 参数验证
    3. API调用
    4. 数据处理
    5. 结果格式化
    """
    TIME_FIELD_PREFIX = "v"
    MAX_V_FIELDS = 8    # 最多保留的 v 字段个数，超过则截断
    INNER_MODEL_ENABLE = bool(strtobool(settings.INNER_MODEL_ENABLE))

    def __init__(
        self,
        conversation_id: str,
        parameter_model: ChatQwen,
        format_model: ChatQwen,
        echarts_model: ChatQwen,
        trend_analysis_model: ChatQwen,
        api_base_url: str = None,
        interface_name: str = ""
    ):
        """
        Args:
            conversation_id: 会话ID
            parameter_model: 参数提取模型
            format_model: 格式化输出模型
            api_base_url: API基础地址
        """
        self.conversation_id = conversation_id
        self.created_at = time.time()
        self.last_updated = time.time()

        # 初始化通用Agent
        self.parameter_agent = ParameterAgent(model=parameter_model)
        # self.format_agent = FormatAgent(model=format_model)
        self.format_agent = agent_factory.create_format_agent(model=format_model)
        self.echarts_agent = agent_factory.create_echarts_agent(model=echarts_model)
        self.aggregation_agent = AggregationAgent()
        self.trend_analysis_agent = TrendAnalysisAgent(model=trend_analysis_model)
        self.python_sandbox = PythonSandbox()

        # API基础地址
        self.api_base_url = api_base_url
        self.interface_name = interface_name

        # 会话状态
        self.session_state = {
            'status': 'idle',
            'current_params': {},
            'messages': []
        }

        self._truncated = False  # 条数截断标志
        self._full_data = None  # 条数截断时的完整数据（实际为原始数据）
        self._field_truncated = False  # 字段截断标志
        self._original_data = None  # 原始数据副本（未经过任何过滤/聚合）
        self._max_time_field_used = None  # 实际使用的过滤阈值
        self._last_chart_data = None  # 最近一次执行用于绘图的数据（供多子问题统一绘图复用）

        self._truncated_front_fields = []
        self._truncated_back_fields = []

        self._api_cache: Dict[str, Any] = {}
        self._cache_ttl = 300  # 秒

    @abstractmethod
    def get_parameter_prompt(self) -> str:
        """获取参数提取提示词模板
        
        子类必须实现此方法，返回该业务场景的参数提取提示词
        """
        pass

    def _get_domain(self) -> str:
        """获取业务域标识
        
        子类可以重写此方法，返回该业务场景的域标识（如 'transmission'、'transaction'）
        默认实现：根据类名推断
        """
        class_name = self.__class__.__name__
        if 'Transmission' in class_name:
            return 'transmission'
        elif 'Transaction' in class_name:
            return 'transaction'
        return ""

    @abstractmethod
    def get_format_prompt(self) -> str:
        """获取结果格式化提示词模板
        
        子类必须实现此方法，返回该业务场景的结果格式化提示词
        """
        pass

    @abstractmethod
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """验证参数是否完整
        
        子类必须实现此方法，定义该业务场景的参数验证规则
        
        Args:
            params: 提取的参数字典
        
        Returns:
            {'valid': bool, 'message': str}
        """
        pass

    async def call_api(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        带缓存和连接池复用的 API 调用入口（模板方法）
        子类不应重写此方法，而应实现 _call_api_impl
        """
        # 1. 检查缓存
        cache_key = self._build_cache_key(params)
        if cached := self._get_cached_result(cache_key):
            logger.info(f"[{self.__class__.__name__}] 使用缓存 API 结果")
            return cached

        # 2. 调用子类实现的真实请求
        result = await self._call_api_impl(params)

        # 3. 存入缓存
        self._set_cached_result(cache_key, result)
        return result

    async def _call_common_api(
            self,
            biz_url: str,
            hash_map: Dict[str, Any],
            timeout: float = 300.0,
            max_retries: int = 3,
            extract_data: Optional[callable] = None,
            request_type: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        通用的统一网关 API 调用封装（带连接复用和重试）

        Args:
            biz_url: 业务 API 地址
            hash_map: 请求参数
            timeout: 超时时间（秒）
            max_retries: 最大重试次数
            extract_data: 可选的数据提取函数，用于从响应中提取 data 部分
            request_type: 统一网关请求体中的 type 字段，可选
        """
        common_api_url = settings.COMMON_API_URL
        if not common_api_url:
            return {"status": "error", "message": "COMMON_API_URL未配置"}
        if not biz_url:
            return {"status": "error", "message": "业务API地址未配置"}

        request_body = {
            "hashMap": hash_map,
            "url": biz_url,
            "interfaceName": self.interface_name,
        }
        if request_type is not None:
            request_body["type"] = request_type

        client = await get_shared_http_client(timeout=timeout)

        for attempt in range(max_retries):
            try:
                response = await client.post(
                    common_api_url,
                    json=request_body,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                raw_data = response.json()

                # 统一网关常见业务失败包识别（HTTP 200 但 success=false / code异常）
                business_error = self._extract_common_api_business_error(raw_data)
                if business_error:
                    logger.error(f"[{self.__class__.__name__}] 网关业务失败: {business_error}")
                    # return {"status": "error", "message": business_error}
                    return {"status": "error", "message": "网关业务失败"}

                # 默认数据提取逻辑
                if extract_data:
                    data = extract_data(raw_data)
                else:
                    # 常见格式：{"data": [...]} 或直接返回数组
                    if isinstance(raw_data, dict) and "data" in raw_data:
                        data = raw_data["data"]
                    else:
                        data = raw_data

                logger.info(f"[{self.__class__.__name__}] API调用成功，返回数据量: {self._count_data(data)}")
                return {"status": "success", "data": data}

            except httpx.TimeoutException as e:
                logger.warning(f"[{self.__class__.__name__}] 超时 (尝试 {attempt + 1}/{max_retries})")
                if attempt == max_retries - 1:
                    # return {"status": "error", "message": f"请求超时: {e}"}
                    return {"status": "error", "message": "请求超时"}
                await asyncio.sleep(2 ** attempt)

            except httpx.HTTPStatusError as e:
                logger.error(f"[{self.__class__.__name__}] HTTP {e.response.status_code}")
                # return {"status": "error", "message": f"HTTP错误 {e.response.status_code}"}
                return {"status": "error", "message": "http错误"}

            except Exception as e:
                logger.warning(f"[{self.__class__.__name__}] 请求异常 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    # return {"status": "error", "message": f"API调用失败: {e}"}
                    return {"status": "error", "message": "api调用失败"}
                await asyncio.sleep(2 ** attempt)

        return {"status": "error", "message": "已达最大重试次数"}

    @staticmethod
    def _to_user_friendly_error(raw_msg: Optional[str]) -> str:
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
    def _extract_common_api_business_error(raw_data: Any) -> Optional[str]:
        """识别统一网关常见的业务失败包（HTTP 200 场景）。"""
        if not isinstance(raw_data, dict):
            return None

        success = raw_data.get("success")
        code = raw_data.get("code")
        message = raw_data.get("message")

        if success is False:
            msg = str(message).strip() if isinstance(message, str) and message.strip() else "上游服务返回 success=false"
            return f"{msg} (code={code})" if code is not None else msg

        if success is True:
            return None

        code_num: Optional[int] = None
        if isinstance(code, int):
            code_num = code
        elif isinstance(code, str) and code.strip().isdigit():
            code_num = int(code.strip())

        has_data = "data" in raw_data and raw_data.get("data") not in (None, "", [], {})
        if code_num is not None and code_num >= 400 and not has_data:
            msg = str(message).strip() if isinstance(message, str) and message.strip() else "上游服务返回错误码"
            return f"{msg} (code={code_num})"

        return None

    @staticmethod
    def _count_data(data: Any) -> int:
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict):
            return len(data)
        return 1

    @abstractmethod
    async def _call_api_impl(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        子类必须实现的具体 API 请求逻辑（可使用共享 HTTP 客户端）
        """
        pass

    def _build_cache_key(self, params: Dict[str, Any]) -> str:
        """生成缓存键（子类可重写以定制）"""
        return f"{self.interface_name}:{json.dumps(params, sort_keys=True)}"

    def _get_cached_result(self, key: str) -> Optional[Dict]:
        """从内存缓存获取结果（可扩展为 Redis）"""
        entry = self._api_cache.get(key)
        if entry and (time.time() - entry["ts"] < self._cache_ttl):
            return entry["data"]
        return None

    def _set_cached_result(self, key: str, data: Dict):
        self._api_cache[key] = {"ts": time.time(), "data": data}



    def _filter_time_fields_by_count(self, data: Any) -> Any:
        """根据时间字段数量进行截断。

        如果数据中时间字段数量超过 MAX_V_FIELDS，则仅保留前4个和后4个（按时间排序），
        并设置 self._field_truncated = True。
        同时保存前4个和后4个字段名，供提示词使用。
        """
        if not isinstance(data, (dict, list)):
            return data

        changed = False
        self._truncated_front_fields = []
        self._truncated_back_fields = []

        def _is_time_field(key: str) -> bool:
            if not isinstance(key, str):
                return False
            # 支持 'v' + 4位数字 或 'val' + 4位数字
            if key.startswith('v') and len(key) == 5 and key[1:].isdigit():
                return True
            if key.startswith('val') and len(key) == 7 and key[3:].isdigit():
                return True
            return False

        def _extract_number(k: str) -> int:
            if k.startswith('val'):
                return int(k[3:])
            else:  # 以 'v' 开头
                return int(k[1:])

        def _process_dict(d: dict) -> dict:
            nonlocal changed
            v_keys = [key for key in d.keys() if _is_time_field(key)]

            if len(v_keys) <= self.MAX_V_FIELDS:
                new_d = {}
                for key, value in d.items():
                    new_d[key] = _process(value)
                return new_d

            changed = True
            v_keys_sorted = sorted(v_keys, key=_extract_number)

            # 取前4个和后4个，使用集合去重（若总数不足8则全保留）
            front = v_keys_sorted[:4]
            back = v_keys_sorted[-4:] if len(v_keys_sorted) >= 4 else []
            keys_to_keep = set(front) | set(back)

            self._truncated_front_fields = front
            self._truncated_back_fields = back

            new_d = {}
            for key, value in d.items():
                if _is_time_field(key) and key not in keys_to_keep:
                    continue
                new_d[key] = _process(value)
            return new_d

        def _process(item: Any) -> Any:
            if isinstance(item, dict):
                return _process_dict(item)
            elif isinstance(item, list):
                return [_process(sub) for sub in item]
            else:
                return item

        filtered = _process(data)
        self._field_truncated = changed
        if changed:
            self._max_time_field_used = f"保留前4后4个时间字段"
        return filtered

    def process_data(self, data: Any, params: Dict[str, Any], user_query: str) -> Any:
        """处理API返回的数据
        
        子类可以重写此方法，实现自定义的数据处理逻辑（如字段筛选、数据转换等）
        默认实现：直接返回原始数据
        
        Args:
            data: API返回的原始数据
            params: 提取的参数
            user_query: 用户原始问题
        
        Returns:
            处理后的数据
        """
        return data

    def _build_data_overview(
        self,
        raw_data: Any,
        display_data: Any,
        user_query: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        raw_records = raw_data if isinstance(raw_data, list) else ([raw_data] if isinstance(raw_data, dict) else [])
        display_records = display_data if isinstance(display_data, list) else ([display_data] if isinstance(display_data, dict) else [])

        sample_record = raw_records[0] if raw_records and isinstance(raw_records[0], dict) else {}
        fields = []
        if isinstance(sample_record, dict):
            fields = [
                {
                    "name": key,
                    "type": type(value).__name__,
                    "sample": value,
                }
                for key, value in sample_record.items()
            ]

        return {
            "user_query": user_query,
            "interface_name": self.interface_name,
            "record_count": len(raw_records),
            "display_record_count": len(display_records),
            "params": params,
            "fields": fields,
            "display_sample": display_records[:3] if isinstance(display_records, list) else display_data,
        }

    def _extract_code_block(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        cleaned = text.strip()
        if "```python" in cleaned:
            return cleaned.split("```python", 1)[1].split("```", 1)[0].strip()
        if "```" in cleaned:
            return cleaned.split("```", 1)[1].split("```", 1)[0].strip()
        return cleaned

    @staticmethod
    def _extract_analysis_records(candidate: Any) -> list[dict[str, Any]]:
        if isinstance(candidate, list):
            return [item for item in candidate if isinstance(item, dict)]

        if isinstance(candidate, dict):
            nested_keys = ("grouped_data", "extreme_scope_data", "data")
            has_nested_payload = any(key in candidate for key in nested_keys)
            for key in nested_keys:
                nested = candidate.get(key)
                if isinstance(nested, list):
                    records = BaseWorkflow._extract_analysis_records(nested)
                    if records:
                        return records
            if has_nested_payload:
                return []
            return [candidate]

        return []

    @staticmethod
    def _is_empty_result_data(data: Any) -> bool:
        return data is None or data == [] or data == [{}] or data == {}

    # @staticmethod
    # def _is_null_aggregation_result(data: Any) -> bool:
    #     """检测聚合操作结果是否全部为 null（接口有数据但聚合处理无结果）。"""
    #     if not isinstance(data, list) or not data:
    #         return False
    #     has_operation = False
    #     for item in data:
    #         if not isinstance(item, dict):
    #             return False
    #         if "operation" not in item:
    #             return False
    #         has_operation = True
    #         if item.get("result") is not None:
    #             return False
    #     return has_operation

    # @staticmethod
    # def _normalize_format_data(data: Any) -> Any:
    #     """将 null 聚合结果转换为格式化提示词可识别的特殊标记，并解包 numerical_value 包装。"""
    #     if BaseWorkflow._is_null_aggregation_result(data):
    #         return {"_data_status": "null_result"}
    #     if isinstance(data, dict) and "numerical_value" in data and len(data) == 1:
    #         inner = data["numerical_value"]
    #         if isinstance(inner, dict):
    #             return {"_data_status": "statistical_summary", **inner}
    #     return data

    @staticmethod
    def _normalize_date_str(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        raw = value.strip()
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
    def _date_to_cn_text(date_str: str) -> str:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return f"{dt.year}年{dt.month}月{dt.day}日"
        except ValueError:
            return date_str

    @staticmethod
    def _build_time_expansion_hint(expansion_info: Optional[Dict[str, str]]) -> str:
        """构建时间范围扩展提示，用于格式化阶段告知 AI 原始范围无数据。"""
        if not expansion_info:
            return ""
        original_start = expansion_info.get("original_start", "")
        original_end = expansion_info.get("original_end", "")
        expanded_start = expansion_info.get("expanded_start", "")
        if not original_start or not original_end or not expanded_start:
            return ""
        try:
            end_dt = datetime.strptime(original_end, "%Y-%m-%d")
            start_dt = datetime.strptime(original_start, "%Y-%m-%d")
            display_end = (end_dt - timedelta(days=1)) if end_dt > start_dt else end_dt
            original_end_cn = BaseWorkflow._date_to_cn_text(display_end.strftime("%Y-%m-%d"))
        except (ValueError, TypeError):
            original_end_cn = BaseWorkflow._date_to_cn_text(original_end)
        original_start_cn = BaseWorkflow._date_to_cn_text(original_start)
        expanded_start_cn = BaseWorkflow._date_to_cn_text(expanded_start)
        return (
            f"\n\n【关键上下文】用户最初查询的是{original_start_cn}至{original_end_cn}的数据，"
            f"但该时间范围内未查到结果。已将起始时间前移30天，实际返回的是{expanded_start_cn}至{original_end_cn}的数据。"
            f"请在回答开头用一句话简要提示用户：您查询的时间段暂无相关数据，已自动为您扩展至 XXXX年XX月XX日 - XXXX年XX月XX日。"
            f"该时段...（后续根据实际扩展的时间范围查询的内容结合用户问题进行补充）"
        )

    @staticmethod
    def _strip_date_tokens_from_query(query: str) -> str:
        cleaned = query
        date_patterns = [
            r"20\d{2}[年/-]\d{1,2}[月/-]\d{1,2}[日号]?",
            r"\d{1,2}月\d{1,2}[日号]?",
        ]
        for pattern in date_patterns:
            cleaned = re.sub(pattern, "", cleaned)
        cleaned = re.sub(r"\s+", "", cleaned)
        cleaned = re.sub(r"^[到至~～\-—]+", "", cleaned)
        return cleaned.strip("，,。:：；;")

    def _build_expanded_time_range_query(self, user_query: str, time_range: Dict[str, Any]) -> str:
        if not isinstance(user_query, str) or not isinstance(time_range, dict):
            return user_query

        start = self._normalize_date_str(time_range.get("start"))
        end = self._normalize_date_str(time_range.get("end"))
        if not start or not end:
            return user_query

        try:
            start_dt = datetime.strptime(start, "%Y-%m-%d")
            end_dt = datetime.strptime(end, "%Y-%m-%d")
        except ValueError:
            return user_query

        # 内部查询 end 为开区间，这里回退一天再组装展示/分析问题。
        display_end_dt = end_dt - timedelta(days=1) if end_dt > start_dt else end_dt
        range_text = f"{self._date_to_cn_text(start_dt.strftime('%Y-%m-%d'))}到{self._date_to_cn_text(display_end_dt.strftime('%Y-%m-%d'))}"

        normalized_query = user_query.strip()
        if normalized_query.startswith("查询"):
            normalized_query = normalized_query[2:]
        normalized_query = self._strip_date_tokens_from_query(normalized_query)
        if normalized_query:
            return f"查询{range_text}{normalized_query}"
        return f"查询{range_text}的数据"

    def _extract_api_data_payload(
        self,
        api_result: Dict[str, Any],
        workflow_name: str,
        is_retry: bool = False,
    ) -> Any:
        raw_data = api_result.get("data", {})
        stage = "二次查询" if is_retry else "首次查询"
        logger.info(
            f"[{workflow_name}] {stage}API返回数据: {type(raw_data).__name__}, "
            f"内容: {raw_data if not isinstance(raw_data, (list, dict)) or len(str(raw_data)) < 200 else '...(数据过长)'}"
        )
        if isinstance(raw_data, dict) and "data" in raw_data and len(raw_data) <= 4:
            actual_data = raw_data.get("data", {})
            logger.info(f"[{workflow_name}] 检测到嵌套 data 结构，提取内层数据: {type(actual_data).__name__}")
            raw_data = actual_data
        return raw_data

    async def _retry_with_expanded_time_range_if_empty(
        self,
        raw_data: Any,
        params: Dict[str, Any],
        user_query: str,
        workflow_name: str,
    ) -> tuple[Any, str]:
        self._time_range_expanded_info = None
        effective_user_query = user_query
        if not self._is_empty_result_data(raw_data):
            return raw_data, effective_user_query

        time_range = params.get("time_range")
        if not isinstance(time_range, dict):
            return raw_data, effective_user_query

        start = self._normalize_date_str(time_range.get("start"))
        if not start:
            return raw_data, effective_user_query

        try:
            expanded_start = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")
        except ValueError:
            logger.warning(f"[{workflow_name}] time_range.start 格式不合法，跳过30天回溯: {start}")
            return raw_data, effective_user_query

        original_time_range = copy.deepcopy(time_range)
        params["time_range"] = copy.deepcopy(time_range)
        params["time_range"]["start"] = expanded_start
        params["time_range"]["end"] = self._normalize_date_str(params["time_range"].get("end"))
        logger.info(
            f"[{workflow_name}] 首次查询为空，自动回溯30天重试: "
            f"{original_time_range.get('start')} -> {expanded_start}"
        )

        retry_result = await self.call_api(params)
        if retry_result.get("status") == "error":
            logger.warning(f"[{workflow_name}] 回溯30天后二次查询失败，保留原始空结果: {retry_result.get('message')}")
            params["time_range"] = original_time_range
            return raw_data, effective_user_query

        retry_raw_data = self._extract_api_data_payload(retry_result, workflow_name, is_retry=True)
        if self._is_empty_result_data(retry_raw_data):
            logger.info(f"[{workflow_name}] 回溯30天后二次查询仍为空，按原逻辑返回未查到数据")
            params["time_range"] = original_time_range
            return raw_data, effective_user_query

        effective_user_query = self._build_expanded_time_range_query(user_query, params.get("time_range", {}))
        self._time_range_expanded_info = {
            "original_start": original_time_range.get("start"),
            "original_end": original_time_range.get("end"),
            "expanded_start": expanded_start,
        }
        logger.info(f"[{workflow_name}] 回溯30天后二次查询命中数据，后续流程改用新问题: {effective_user_query}")
        return retry_raw_data, effective_user_query

    @staticmethod
    def _normalize_output_text(text: Any) -> str:
        if text is None:
            return ""
        normalized = str(text).strip()
        return normalized

    @classmethod
    def _merge_output_sections(cls, *sections: Any) -> str:
        merged: list[str] = []
        for section in sections:
            content = cls._normalize_output_text(section)
            if not content:
                continue

            replaced = False
            for idx, existing in enumerate(merged):
                if content == existing or content in existing:
                    replaced = True
                    break
                if existing in content:
                    merged[idx] = content
                    replaced = True
                    break

            if not replaced:
                merged.append(content)

        return "\n\n".join(merged)

    def _get_chart_subject_label(self) -> str:
        interface_name = (self.interface_name or "").strip()
        subject_map = {
            "日内现货电量查询": "日内现货出清电量",
            "日内现货电价查询": "日内现货出清电价",
            "日前现货电量查询": "日前现货出清电量",
            "日前现货电价查询": "日前现货出清电价",
        }
        if interface_name in subject_map:
            return subject_map[interface_name]
        if interface_name.endswith("查询"):
            return interface_name[:-2]
        return "查询结果"

    def _build_chart_intro(self) -> str:
        subject = self._get_chart_subject_label()
        return f"根据你查询的{subject}部分详情如下："

    def _build_trend_intro(self) -> str:
        subject = self._get_chart_subject_label()
        return f"根据你查询的{subject}趋势总结如下："

    @staticmethod
    def _estimate_summary_data_chars(data: Any) -> int:
        if isinstance(data, (dict, list)):
            return len(json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str))
        return len(str(data))

    def _should_use_format_summary(self, data: Any) -> bool:
        if isinstance(data, dict) and data.get("data_too_large"):
            logger.info(f"[{self.__class__.__name__}] 命中 data_too_large，跳过格式化总结阶段")
            return False

        if self._is_empty_result_data(data):
            return True

        limit = max(0, getattr(settings, "FORMAT_SUMMARY_MAX_DATA_CHARS", 60000))
        estimated_chars = self._estimate_summary_data_chars(data)
        should_use = limit <= 0 or estimated_chars <= limit
        logger.info(
            f"[{self.__class__.__name__}] 格式化总结阈值判断: estimated_chars={estimated_chars}, "
            f"limit={limit}, should_use={should_use}"
        )
        return should_use

    @staticmethod
    def _is_scalar_record_value(value: Any) -> bool:
        return value is None or isinstance(value, (str, int, float, bool, Decimal))

    @staticmethod
    def _normalize_record_value(value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @staticmethod
    def _build_record_signature(record: Dict[str, Any]) -> str:
        normalized = {
            key: BaseWorkflow._normalize_record_value(value)
            for key, value in sorted(record.items())
            if not str(key).startswith("_")
        }
        return json.dumps(normalized, ensure_ascii=False, sort_keys=True, default=str)

    @staticmethod
    def _is_time_measure_field(field_name: Any) -> bool:
        return isinstance(field_name, str) and field_name.startswith("v")

    def _score_hidden_discriminator_field(self, field_name: str, values: list[Any]) -> int:
        lowered = (field_name or "").lower()
        score = 0
        preferred_tokens = (
            "case", "scheme", "plan", "record", "detail", "id", "name", "code", "type",
            "line", "station", "node", "unit", "plant", "province", "region", "voltage",
            "market", "trade", "object", "category"
        )
        if any(token in lowered for token in preferred_tokens):
            score += 100
        if lowered.endswith("id"):
            score += 30
        if lowered.endswith("name") or lowered.endswith("code"):
            score += 20
        if self._is_time_measure_field(field_name):
            score -= 1000
        if lowered in {"success", "code", "message", "data"}:
            score -= 1000

        non_empty_values = [value for value in values if value not in (None, "")]
        if non_empty_values and all(isinstance(value, str) for value in non_empty_values):
            score += 10
        if non_empty_values and max(len(str(value)) for value in non_empty_values) <= 40:
            score += 5
        return score

    def _pick_hidden_discriminator_fields(
        self,
        projected_records: list[dict[str, Any]],
        source_records: list[dict[str, Any]],
        max_fields: int = 3,
    ) -> list[str]:
        if len(projected_records) < 2 or len(projected_records) != len(source_records):
            return []

        projected_keys: set[str] = set()
        for record in projected_records:
            projected_keys.update(record.keys())

        candidate_keys: set[str] = set()
        for record in source_records:
            candidate_keys.update(record.keys())
        candidate_keys.difference_update(projected_keys)

        scored_fields: list[tuple[int, str]] = []
        for field_name in candidate_keys:
            if not isinstance(field_name, str) or field_name.startswith("_"):
                continue

            values = [self._normalize_record_value(record.get(field_name)) for record in source_records]
            if not values or any(not self._is_scalar_record_value(value) for value in values):
                continue

            distinct_values = {
                json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
                for value in values
                if value not in (None, "")
            }
            if len(distinct_values) <= 1:
                continue

            score = self._score_hidden_discriminator_field(field_name, values)
            if score > -1000:
                scored_fields.append((score, field_name))

        scored_fields.sort(key=lambda item: (-item[0], item[1]))
        return [field_name for _, field_name in scored_fields[:max_fields]]

    def _prepare_format_records(self, data: Any, source_data: Any) -> Any:
        return data

        projected_records = self._extract_analysis_records(data)
        source_records = self._extract_analysis_records(source_data)
        if len(projected_records) < 2 or len(projected_records) != len(source_records):
            return data

        projected_signatures = [self._build_record_signature(record) for record in projected_records]
        duplicate_projection = len(set(projected_signatures)) < len(projected_signatures)

        visible_dimension_fields = []
        for field_name in sorted({key for record in projected_records for key in record.keys()}):
            if not isinstance(field_name, str) or field_name.startswith("_") or self._is_time_measure_field(field_name):
                continue
            values = [self._normalize_record_value(record.get(field_name)) for record in projected_records]
            if any(not self._is_scalar_record_value(value) for value in values):
                continue
            distinct_values = {value for value in values if value not in (None, "")}
            if len(distinct_values) > 1:
                visible_dimension_fields.append(field_name)

        ambiguous_multi_record = duplicate_projection or not visible_dimension_fields
        if not ambiguous_multi_record:
            return data

        hidden_fields = self._pick_hidden_discriminator_fields(projected_records, source_records)
        enriched_records: list[dict[str, Any]] = []
        for idx, (projected_record, source_record) in enumerate(zip(projected_records, source_records), start=1):
            enriched_record = dict(projected_record)
            label_parts: list[str] = []
            for field_name in hidden_fields:
                if field_name not in enriched_record and field_name in source_record:
                    enriched_record[field_name] = source_record.get(field_name)
                value = self._normalize_record_value(source_record.get(field_name))
                if value not in (None, ""):
                    label_parts.append(f"{field_name}={value}")

            enriched_record["_record_label"] = "；".join(label_parts) if label_parts else f"第{idx}条记录"
            enriched_records.append(enriched_record)

        logger.info(
            f"[{self.__class__.__name__}] 检测到多记录结果存在歧义，已补充区分字段: "
            f"duplicate_projection={duplicate_projection}, "
            f"visible_dimension_fields={visible_dimension_fields}, hidden_fields={hidden_fields}"
        )
        return enriched_records if isinstance(data, list) else data

    @staticmethod
    def _build_global_format_rules() -> str:
        return """
# 通用输出要求（最高优先级）
- **语言限制（最高优先级）**：只能用中文输出，禁止输出其它国家的语言。
- 本次格式化结果仅用于「结果摘要/直接回答」，禁止输出 Markdown 表格、伪表格、代码块、JSON、字段清单或逐行明细抄录。
- 先结合用户问题直接回答核心结果，再补充必要的范围、口径或单位说明。
- **空数据判定标准（最高优先级）**：仅当 INPUT_DATA 为 `[]`、`{{}}`、`””`或字段完全不存在时，才能判定为「数据为空」。只要 INPUT_DATA 中存在任何非空字段、键名或数值（无论外层是否有陌生包装结构如 `numerical_value`，无论字段名是否为系统生成如 `sum_fields_sum`），都属于查询结果非空，必须基于实际数据直接回答。如果数据内容与用户问题看起来不完全匹配或你无法确定对应关系，也应客观描述数据中已有的内容，**严禁**以按照查询结果数据为空时的提示语作为兜底回复。
- **筛选后无结果规则（最高优先级）**：若 INPUT_DATA 中包含 「筛选结果」 字段且描述为「过滤后无匹配记录」或「聚合操作后无匹配记录」，说明原始API查询成功返回了数据，但经用户问题中的筛选条件（如「电价大于800」）过滤后无记录匹配。此时**严禁**使用通用的「未查询到对应交易数据，可尝试调整日期范围、网省名称、购售方类型等信息重试」空数据提示模板，必须根据用户问题中明确的时间、地点、具体筛选条件等，直接告知用户类似「2026年X月X日没有[地点/节点]的[指标]满足[条件]」的自然语言回复。不要输出「筛选结果」等内部字段。
- 不要输出「如下表所示」「表格如下」「以上数据」等依赖表格的措辞。
- 如果输入数据是多条记录列表，且不同记录对同一查询指标/时间点给出了不同值，禁止将这些记录压缩成单一数值结论；必须先区分记录，再回答。
- 如果数据中包含 `_record_label`，它是系统补充的记录区分标签；回答时应把它视为记录标识使用，但不要原样解释字段名 `_record_label`。
- 如果多条记录无法唯一合并成一句结论，应明确说明存在多条记录/多个方案/多条明细，并按记录标签或区分字段概括，不能只取第一条或前几条样本下结论。
- **分时趋势强制描述规则（最高优先级）**：若 INPUT_DATA 同时包含 `max_point`、`min_point` 和 `avg_nonzero`，则必须先总结最高点和最低点（含对应时间点和数值），再基于 `avg_nonzero` 描述分时趋势变化。**严禁**只输出最高最低点而跳过趋势描述。趋势描述只能用客观词汇（走高、回落、攀升、平稳、波动等），不得推测原因。
  - **关键数值读取规则（强制）**：`max_point` 和 `min_point` 是新结构化字段，格式为 `{{"time": "v1430,v1445", "value": 179.04, "time_sources": [{{"time": "v1430", "sources": [{{"businessTime": "...", "nameAbbreviation": "..."}}]}}, {{"time": "v1445", "sources": [...]}}]}}`。其中：
    - `time` 为极值对应的时间点（v1430=14:30），多个时间点以逗号分隔表示并列；
    - `value` 为最高/最低的**实际数值**，必须直接使用该值作为极值；
    - `time_sources`（可选）为极值溯源信息，每一项包含 `time`（该组来源对应的时刻）和 `sources`（该时刻对应的身份信息列表，如日期、节点名称、购售方等）。必须在回答中根据 `time_sources` 的 time→sources 对应关系分别陈述：不同来源对应不同时刻时须分开描述，禁止将不同来源的时刻混在一起。网省/节点和日期放在"出现在xx年xx月xx日XX的XX时刻"结构中（如"出现在2025年4月4日上海的00:15"）。
    **严禁**用 max_point/min_point 中的时间点去 `avg_nonzero` 中查找对应的平均值来顶替极值。例如 `"max_point": {{"time": "v1430", "value": 179.04}}` → 最高值 = 179.04，**不是** `avg_nonzero` 中 v1430 的值 50.33。"""

    def _select_data_analysis_dataset(self, raw_data: Any, display_data: Any) -> list[dict[str, Any]]:
        if self._is_empty_result_data(display_data):
            return []

        display_records = self._extract_analysis_records(display_data)
        if len(display_records) >= 2:
            return display_records

        raw_records = self._extract_analysis_records(raw_data)
        if len(raw_records) >= 2:
            return raw_records

        return []

    async def _run_data_analysis(
        self,
        user_query: str,
        raw_data: Any,
        display_data: Any,
        params: Dict[str, Any],
    ) -> Dict[str, Any] | None:
        analysis_data = self._select_data_analysis_dataset(raw_data=raw_data, display_data=display_data)
        if len(analysis_data) < 2:
            return None

        overview = self._build_data_overview(
            raw_data=analysis_data,
            display_data=display_data,
            user_query=user_query,
            params=params,
        )
        try:
            generated = await self.trend_analysis_agent.generate_code(user_query, overview)
            python_code = self._extract_code_block(filter_think_tags_simple(generated))
            if not python_code:
                return None

            sandbox_result = self.python_sandbox.execute(code=python_code, dataset=analysis_data)
            analysis_payload = {
                "generated_code": python_code,
                "statistics": sandbox_result.get("analysis_result"),
                "stdout": sandbox_result.get("stdout", ""),
            }
            summary = await self.trend_analysis_agent.summarize(
                user_query=user_query,
                overview=overview,
                analysis_result=analysis_payload,
            )
            return {
                "summary": filter_think_tags_simple(summary),
                "analysis_payload": analysis_payload,
            }
        except SandboxExecutionError as exc:
            logger.error(f"[{self.__class__.__name__}] 数据分析沙箱执行失败: {exc}")
            return None
        except Exception as exc:
            logger.error(f"[{self.__class__.__name__}] 数据分析失败: {exc}", exc_info=True)
            return None

    async def _prepare_data_analysis_stream(
        self,
        user_query: str,
        raw_data: Any,
        display_data: Any,
        params: Dict[str, Any],
    ) -> Dict[str, Any] | None:
        analysis_data = self._select_data_analysis_dataset(raw_data=raw_data, display_data=display_data)
        if len(analysis_data) < 2:
            return None

        overview = self._build_data_overview(
            raw_data=analysis_data,
            display_data=display_data,
            user_query=user_query,
            params=params,
        )
        try:
            generated = await self.trend_analysis_agent.generate_code(user_query, overview)
            python_code = self._extract_code_block(filter_think_tags_simple(generated))
            if not python_code:
                return None

            sandbox_result = self.python_sandbox.execute(code=python_code, dataset=analysis_data)
            analysis_payload = {
                "generated_code": python_code,
                "statistics": sandbox_result.get("analysis_result"),
                "stdout": sandbox_result.get("stdout", ""),
            }
            return {
                "overview": overview,
                "analysis_payload": analysis_payload,
            }
        except SandboxExecutionError as exc:
            logger.error(f"[{self.__class__.__name__}] 数据分析沙箱执行失败: {exc}")
            return None
        except Exception as exc:
            logger.error(f"[{self.__class__.__name__}] 数据分析失败: {exc}", exc_info=True)
            return None

    def _should_preserve_grouped_details_with_extreme(
        self,
        user_query: str,
        operations: list
    ) -> bool:
        """判断 group_by + max/min 场景下是否需要同时保留分组明细和极值结果。"""
        if not isinstance(operations, list):
            return False

        has_group_by = any(isinstance(op, dict) and op.get("operation") == "group_by" for op in operations)
        has_extreme = any(isinstance(op, dict) and op.get("operation") in ("max", "min") for op in operations)
        if not (has_group_by and has_extreme):
            return False

        query = (user_query or "").replace(" ", "")
        detail_cues = (
            "分别是多少", "分别为多少", "各日", "每日", "每天", "逐日", "每一天", "各天",
            "各月", "每月", "逐月", "各年", "每年", "逐年"
        )
        return any(cue in query for cue in detail_cues)

    def _build_grouped_extreme_payload(
        self,
        grouped_data: Any,
        summary: list,
        extreme_source_data: Any | None = None
    ) -> Any:
        """构建“分组明细 + 极值结论”联合输出，避免仅保留极值那一条记录。"""
        if not isinstance(grouped_data, list):
            return grouped_data

        payload: Dict[str, Any] = {"grouped_data": grouped_data}

        extreme_summary = next(
            (s for s in reversed(summary or []) if isinstance(s, dict) and s.get("operation") in ("max", "min")),
            None
        )
        if extreme_summary:
            payload["extreme_summary"] = extreme_summary
            extreme_field = extreme_summary.get("field", "")
            extreme_val = extreme_summary.get("result")
            if extreme_field and extreme_val is not None:
                search_candidates = []
                if isinstance(extreme_source_data, list):
                    search_candidates.append(extreme_source_data)
                if grouped_data is not extreme_source_data:
                    search_candidates.append(grouped_data)

                extreme_record = None
                for candidate_data in search_candidates:
                    extreme_record = next(
                        (
                            record for record in candidate_data
                            if isinstance(record, dict) and record.get(extreme_field) == extreme_val
                        ),
                        None
                    )
                    if extreme_record:
                        break
                if extreme_record:
                    payload["extreme_record"] = extreme_record
                if isinstance(extreme_source_data, list) and extreme_source_data is not grouped_data:
                    payload["extreme_scope_data"] = extreme_source_data

        return payload

    @staticmethod
    def _is_time_point_field(field: Any) -> bool:
        """判断字段是否为标准时点字段（vHHMM）。"""
        return isinstance(field, str) and re.fullmatch(r"v(?:[01]\d|2[0-4])[0-5]\d", field, flags=re.IGNORECASE) is not None

    @staticmethod
    def _is_empty_operation_value(value: Any) -> bool:
        """判定 operation.value 是否为空/无效。"""
        if value is None:
            return True
        if isinstance(value, str):
            text = value.strip().lower()
            return text in ("", "none", "null", "undefined", "nan")
        if isinstance(value, (list, tuple, set, dict)):
            return len(value) == 0
        return False

    def sanitize_operations(
        self,
        operations: Any,
        params: Optional[Dict[str, Any]] = None,
        user_query: str = "",
    ) -> list:
        """通用操作清洗（全接口生效）。

        目标：
        1. 丢弃无效 operation（空 operation、非 dict、非法 filter 条件等）；
        2. 将“时点字段 + 空值filter”纠正为 time_points（例如 v0815 == None）；
        3. 去重完全重复的 operation，减少重复执行与误过滤风险。
        """
        if not isinstance(operations, list):
            return []

        if not isinstance(params, dict):
            params = {}

        raw_time_points = params.get("time_points", [])
        if not isinstance(raw_time_points, list):
            raw_time_points = []
        normalized_time_points = [tp for tp in raw_time_points if self._is_time_point_field(tp)]
        seen_time_points = set(normalized_time_points)

        sanitized: list = []
        seen_signatures: set[str] = set()
        dropped_count = 0
        converted_time_point_filters = 0

        for op in operations:
            if not isinstance(op, dict):
                dropped_count += 1
                continue

            op_type = str(op.get("operation", "")).strip()
            if not op_type:
                dropped_count += 1
                continue

            cleaned_op = dict(op)
            cleaned_op["operation"] = op_type

            if op_type == "filter":
                field = cleaned_op.get("field")
                operator = str(cleaned_op.get("operator", "")).strip().lower()
                value = cleaned_op.get("value")

                requires_value = operator not in ("is_null", "isnull", "not_null", "isnotnull")
                if requires_value and self._is_empty_operation_value(value):
                    if self._is_time_point_field(field):
                        if field not in seen_time_points:
                            seen_time_points.add(field)
                            normalized_time_points.append(field)
                        converted_time_point_filters += 1
                    else:
                        dropped_count += 1
                    continue

                if operator == "between":
                    if not isinstance(value, (list, tuple)) or len(value) != 2:
                        dropped_count += 1
                        continue

                if operator == "in":
                    if not isinstance(value, (list, tuple, set)) or len(value) == 0:
                        dropped_count += 1
                        continue

            # 自动补齐时间点字段：LLM 枚举96个时间点时容易遗漏，兜底补齐
            cleaned_op = self._auto_complete_time_point_fields(cleaned_op)

            signature = json.dumps(cleaned_op, ensure_ascii=False, sort_keys=True, default=str)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            sanitized.append(cleaned_op)

        if normalized_time_points:
            params["time_points"] = normalized_time_points

        if dropped_count or converted_time_point_filters:
            logger.info(
                f"[{self.__class__.__name__}] 操作清洗: input={len(operations)}, output={len(sanitized)}, "
                f"dropped={dropped_count}, time_point_filter_to_time_points={converted_time_point_filters}"
            )

        return sanitized

    @staticmethod
    def _is_time_point_pattern(field_name: str) -> bool:
        """判断字段名是否为 vHHMM 时间点格式。"""
        return isinstance(field_name, str) and bool(re.match(r'^v\d{4}$', field_name))

    @classmethod
    def _auto_complete_time_point_fields(cls, op: dict) -> dict:
        """当 agg_field / field 中的时间点字段覆盖绝大多数时，自动补齐为完整96点。

        仅处理 group_by（agg_field）、average/sum（field）这三类操作。
        覆盖阈值：命中数 >= 总时间点数 - 3（即至少93个）。
        """
        if not isinstance(op, dict):
            return op

        op_type = str(op.get("operation", "")).strip().lower()

        if op_type == "group_by":
            raw = op.get("agg_field")
        elif op_type in ("average", "sum"):
            raw = op.get("field")
        else:
            return op

        if not raw:
            return op

        if isinstance(raw, list):
            field_list = [f.strip() for f in raw if isinstance(f, str) and f.strip()]
        elif isinstance(raw, str) and "," in raw:
            field_list = [f.strip() for f in raw.split(",") if f.strip()]
        elif isinstance(raw, str):
            field_list = [raw.strip()]
        else:
            return op

        if not field_list:
            return op

        matched = [f for f in field_list if f in _CANONICAL_TIME_POINT_FIELDS]
        if not matched:
            return op

        # 只有当当前字段列表中大多数是时间点字段时才触发补齐
        # 条件：匹配数 >= max(93, total-3)
        threshold = max(_CANONICAL_TIME_POINT_COUNT - 3, int(_CANONICAL_TIME_POINT_COUNT * 0.95))
        if len(matched) < threshold:
            return op

        # 已经是完整集合则无需修改
        if set(matched) == _CANONICAL_TIME_POINT_FIELDS:
            return op

        canonical_sorted = sorted(_CANONICAL_TIME_POINT_FIELDS)
        new_op = dict(op)
        field_key = "agg_field" if op_type == "group_by" else "field"

        if isinstance(raw, list):
            new_op[field_key] = canonical_sorted
        else:
            new_op[field_key] = ",".join(canonical_sorted)

        logger.info(
            f"[base] 时间点字段自动补齐: {op_type}.{field_key} "
            f"({len(matched)}/{_CANONICAL_TIME_POINT_COUNT} -> {_CANONICAL_TIME_POINT_COUNT})"
        )
        return new_op

    def normalize_operations(
        self,
        operations: list,
        user_query: str,
        intent_result: Dict[str, Any],
        params: Dict[str, Any]
    ) -> list:
        """对意图识别出的操作做业务侧兜底修正。

        子类可重写此方法，对 operations 做规范化或补全。
        默认实现：原样返回。
        """
        return operations

    def apply_post_processing(self, data: Any, post_processing: list, params: Dict[str, Any]) -> Any:
        """应用后置处理操作到数据上

        Args:
            data: 经过聚合/操作后的最终数据（通常是列表或字典）
            post_processing: 意图识别输出的 post_processing 列表
            params: 提取的参数（可能包含其他上下文）

        Returns:
            处理后数据，可能为原始列表或包含统计信息的字典
        """
        if not post_processing or not isinstance(data, list):
            return data

        result = data
        for step in post_processing:
            action = step.get("action")
            if action == "filter":
                field = step.get("field")
                operator = step.get("operator", "eq")
                value = step.get("value")
                result = self._apply_filter(result, field, operator, value)
            elif action == "count_filtered":
                field = step.get("field")
                operator = step.get("operator", "eq")
                value = step.get("value")
                count = self._count_filtered(result, field, operator, value)
                # 返回包含原始数据和统计信息的结构
                result = {
                    "original_data": result,
                    "count_filtered_result": {
                        "condition": f"{field} {operator} {value}",
                        "count": count
                    }
                }
            elif action == "count_records":  # 新增处理
                # 统计当前数据列表的记录数
                count = len(result) if isinstance(result, list) else 0
                result = {
                    "original_data": result,
                    "count_records_result": {
                        "description": step.get("description", "记录总数"),
                        "count": count
                    }
                }
            elif action == "project":
                fields_list = step.get("fields_list", [])
                result = self._apply_project(result, fields_list)
            elif action == "sort":
                field = step.get("field")
                order = step.get("order", "asc")
                reverse = (order == "desc")
                if isinstance(result, list):
                    result = sorted(result, key=lambda x: x.get(field, 0), reverse=reverse)
            elif action == "limit":
                n = step.get("n")
                if isinstance(result, list) and n is not None:
                    result = result[:n]
            # 可扩展其他 action
        return result

    def _apply_filter(self, data: list, field: str, operator: str, value: Any) -> list:
        """对列表数据应用过滤"""
        if not isinstance(data, list):
            return data
        if operator == "eq":
            return [item for item in data if item.get(field) == value]
        elif operator == "contains":
            return [item for item in data if value in str(item.get(field, ""))]
        elif operator == "gt":
            return [item for item in data if item.get(field, 0) > value]
        elif operator == "lt":
            return [item for item in data if item.get(field, 0) < value]
        elif operator == "gte":
            return [item for item in data if item.get(field, 0) >= value]
        elif operator == "lte":
            return [item for item in data if item.get(field, 0) <= value]
        return data

    def _count_filtered(self, data: list, field: str, operator: str, value: Any) -> int:
        """统计满足条件的记录数"""
        filtered = self._apply_filter(data, field, operator, value)
        return len(filtered)

    def _apply_project(self, data: list, fields: list) -> list:
        """仅保留指定字段"""
        if not isinstance(data, list):
            return data
        return [{k: item.get(k) for k in fields if k in item} for item in data]

    def post_process_after_aggregation(
        self,
        processed_data: Any,
        user_query: str,
        intent_result: Dict[str, Any],
        params: Dict[str, Any],
        operations: list,
        summary: list,
        aggregated_data: Any,
        grouped_data_full: Any | None = None,
    ) -> Any:
        """在聚合执行完成后，对最终传给格式化层的数据做业务侧补充。"""
        return processed_data

    async def execute(
        self,
        user_query: str,
        intent_result: Dict,
        skip_echarts: bool = False,
    ) -> Dict[str, Any]:
        """执行业务工作流（非流式）

        Args:
            user_query: 用户问题
            intent_result: 意图识别结果

        Returns:
            执行结果
        """
        try:
            workflow_name = self.__class__.__name__
            logger.info(f"[{workflow_name}] 开始执行工作流, conversation_id={self.conversation_id}")
            self._last_chart_data = None



            # 1. 参数提取
            logger.info(f"[{workflow_name}] 开始提取参数")
            self.session_state['status'] = 'extracting_params'

            params = await self._extract_parameters(user_query, intent_result)
            self.session_state['current_params'] = params
            logger.info(f"[{workflow_name}] 参数提取结果: {params}")

            # 2. 参数验证
            logger.info(f"[{workflow_name}] 验证参数")
            validation_result = self.validate_params(params)

            if not validation_result['valid']:
                logger.warning(f"[{workflow_name}] 参数验证失败: {validation_result['message']}")
                return {
                    "conversation_id": self.conversation_id,
                    "status": "need_more_info",
                    "workflow_type": workflow_name,
                    "message": {"content": validation_result['message']},
                    "collected_params": params
                }

            # 3. 调用API
            logger.info(f"[{workflow_name}] 调用API")
            self.session_state['status'] = 'calling_api'

            api_result = await self.call_api(params)

            if api_result['status'] == 'error':
                logger.error(f"[{workflow_name}] API调用失败: {api_result.get('message')}")
                return {
                    "conversation_id": self.conversation_id,
                    "status": "error",
                    "workflow_type": workflow_name,
                    "message": {"content": f"查询失败: {api_result.get('message')}"},
                    "collected_params": params
                }

            # 4. 检查数据是否为空
            effective_user_query = user_query
            raw_data = self._extract_api_data_payload(api_result, workflow_name)
            raw_data, effective_user_query = await self._retry_with_expanded_time_range_if_empty(
                raw_data=raw_data,
                params=params,
                user_query=user_query,
                workflow_name=workflow_name,
            )

            logger.info(f"[{workflow_name}] API返回数据: {type(raw_data).__name__}, 内容: {raw_data if not isinstance(raw_data, (list, dict)) or len(str(raw_data)) < 200 else '...(数据过长)'}")

            # 处理嵌套的 data 结构（API 可能返回 {'success': True, 'data': [...]}）
            if isinstance(raw_data, dict) and 'data' in raw_data and len(raw_data) <= 4:
                # 如果是 API 响应对象（通常只有 success/code/message/data 几个字段），提取 data 字段
                actual_data = raw_data.get('data', {})
                logger.info(f"[{workflow_name}] 检测到嵌套 data 结构，提取内层数据: {type(actual_data).__name__}")
                raw_data = actual_data
            self._original_data = copy.deepcopy(raw_data)
            # 判断数据是否为空
            is_empty = self._is_empty_result_data(raw_data)

            if is_empty:
                logger.info(f"[{workflow_name}] API返回空数据，直接跳到格式化步骤")
                format_source_data = []
                chart_data = []
                display_data = []
            else:
                # 4. 数据处理
                logger.info(f"[{workflow_name}] 处理数据")
                # 将操作信息注入 params，供 process_data 中的字段筛选使用
                operations = intent_result.get('operations', []) if isinstance(intent_result, dict) else []
                operations = self.normalize_operations(operations, effective_user_query, intent_result, params)
                operations = self.sanitize_operations(operations, params=params, user_query=effective_user_query)
                params['operations'] = operations
                processed_data = self.process_data(raw_data, params, effective_user_query)

                # 5. 执行操作（过滤、分组、排序、聚合）
                if operations:
                    logger.info(f"[{workflow_name}] 执行操作: {operations}")
                    agg_result = await self.aggregation_agent.ainvoke(
                        user_query=effective_user_query,
                        operations=operations,
                        data=processed_data,
                        context={"thread_id": self.conversation_id},
                        domain=self._get_domain()
                    )
                    summary = agg_result.get('summary') or []
                    aggregated_data = agg_result.get('aggregated_data')
                    grouped_data_full = agg_result.get('grouped_data_full')

                    # 判断操作类型
                    _summary_ops = {'sum', 'average', 'weighted_average', 'max', 'min', 'count'}
                    _row_ops = {'subtract', 'multiply', 'divide', 'mom_change', 'yoy_change'}
                    _sort_group_ops = {'top_n', 'bottom_n', 'group_by'}
                    _ops = {op.get('operation') for op in operations}

                    _has_agg = any(op.get('operation') in _summary_ops for op in operations)
                    _has_row_ops = any(op.get('operation') in _row_ops for op in operations)
                    _has_sort_group = any(op.get('operation') in _sort_group_ops for op in operations)
                    _has_top_bottom = any(op.get('operation') in ('top_n', 'bottom_n') for op in operations)
                    _has_filter = any(op.get('operation') == 'filter' for op in operations)

                    # 检测是否为「先聚合再求比例」场景：
                    # divide/subtract/multiply 的 field 或 field_b 引用了 sum_xxx 格式
                    _has_agg_ratio = any(
                        op.get('operation') in ('divide', 'subtract', 'multiply')
                        and (
                            (isinstance(op.get('field', ''), str) and op.get('field', '').startswith('sum_'))
                            or (isinstance(op.get('field_b', ''), str) and op.get('field_b', '').startswith('sum_'))
                        )
                        for op in operations
                    )

                    # 检测是否为「逐行计算后取极值」场景：
                    # 有逐行操作（multiply/divide/subtract）且后续有 max/min 汇总
                    _has_row_then_agg = (
                        _has_row_ops
                        and any(op.get('operation') in ('max', 'min') for op in operations)
                        and not _has_agg_ratio
                    )

                    # 检测是否为「逐行计算后汇总」场景（宽泛版）：
                    # 有逐行操作且后续有任意汇总操作（sum/average/count/max/min/weighted_average）
                    # 此时应展示 summary 而非全量明细
                    _has_row_then_any_agg = (
                        _has_row_ops
                        and _has_agg
                        and not _has_agg_ratio
                        and not _has_sort_group  # 如果同时有分组，仍展示明细
                    )

                    # 检测是否为「分组后取极值」场景：
                    # group_by 后接 max/min，应只展示极值对应的那条分组记录
                    _has_group_then_extreme = (
                        any(op.get('operation') == 'group_by' for op in operations)
                        and any(op.get('operation') in ('max', 'min') for op in operations)
                    )

                    # 检测是否为「分组/过滤后纯统计」场景：
                    # 包含 group_by 且操作链中有 count（无 top_n/bottom_n），
                    # 用户只需要统计数量，不需要明细数据
                    _has_group_then_count = (
                        any(op.get('operation') == 'group_by' for op in operations)
                        and any(op.get('operation') == 'count' for op in operations)
                        and not any(op.get('operation') in ('top_n', 'bottom_n') for op in operations)
                    )

                    # 决策逻辑（按优先级）：
                    # 1. 「先聚合再求比例」：sum_xxx 作为 divide/subtract/multiply 的操作数
                    # 2. 「逐行计算后取极值」：逐行操作 + max/min，找到极值对应的那条记录
                    # 3. 「逐行计算后汇总」：逐行操作 + sum/average/count 等，展示 summary
                    # 4. 「分组后取极值」：group_by + max/min，找到极值对应的分组记录
                    # 4b.「分组/过滤后纯统计」：group_by + count（无 top_n/bottom_n），只展示统计数
                    # 5. 有排序/分组/逐行操作：展示 aggregated_data（多条明细）
                    # 6. 纯汇总型：展示 summary
                    # 7. 其他（如单独 filter）：展示 aggregated_data
                    if _has_agg_ratio and aggregated_data:
                        # 场景1：先聚合再求比例，构建语义清晰的汇总记录
                        agg_summary = {}
                        for s in summary:
                            agg_summary[s['field']] = s['result']
                        first = aggregated_data[0] if isinstance(aggregated_data, list) and aggregated_data else {}
                        for key in ('quotient', 'product', 'difference'):
                            if key in first:
                                agg_summary[key] = first[key]
                        processed_data = [agg_summary]
                        # logger.info(f"[{workflow_name}] 场景1-先聚合再求比例，折叠为汇总记录: {agg_summary}")
                    elif _has_row_then_agg and summary and aggregated_data:
                        # 场景2：逐行计算后取极值，找到极值对应的那条记录
                        # 找最后一个 max/min 操作的结果
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
                        # logger.info(f"[{workflow_name}] 场景2-逐行后取极值，极值记录: {processed_data}")
                    elif _has_group_then_extreme and summary and aggregated_data:
                        # 场景4：分组后取极值
                        if self._should_preserve_grouped_details_with_extreme(effective_user_query, operations):
                            processed_data = self._build_grouped_extreme_payload(
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
                        # 场景3：逐行计算后汇总（sum/average/count），只展示汇总结果
                        processed_data = summary
                        # logger.info(f"[{workflow_name}] 场景3-逐行计算后汇总，展示 summary: {summary}")
                    elif _has_group_then_count and summary:
                        # 场景4b：分组/过滤后纯统计（group_by + count），用户只需要统计数量
                        # 不展示明细数据，只展示 count 结果
                        processed_data = summary
                        # logger.info(f"[{workflow_name}] 场景4b-分组过滤后纯统计，展示 summary: {summary}")
                    elif (_has_sort_group or _has_row_ops) and aggregated_data and not (_has_agg and summary and not _has_top_bottom):
                        # 场景5：排序/分组/逐行操作，返回多条明细
                        processed_data = aggregated_data
                    elif _has_agg and summary:
                        # 场景6：纯汇总型或 filter+汇总型，展示聚合结果
                        processed_data = summary
                    elif agg_result:
                        # 场景7：其他情况（如单独 filter），展示处理后的完整数据
                        processed_data = aggregated_data
                    # 否则保持 processed_data 为原始数据

                    processed_data = self.post_process_after_aggregation(
                        processed_data=processed_data,
                        user_query=effective_user_query,
                        intent_result=intent_result,
                        params=params,
                        operations=operations,
                        summary=summary,
                        aggregated_data=aggregated_data,
                        grouped_data_full=grouped_data_full,
                    )

                    logger.info(f"[{workflow_name}] 操作执行完成, summary={summary}")

                ## 后置处理
                post_processing = intent_result.get('post_processing', []) if isinstance(intent_result, dict) else []
                print(post_processing, intent_result)
                if post_processing:
                    TIME_POINT_ALIAS_MAP = {"v0000": "v2400"}
                    for step in post_processing:
                        if step.get("action") == "project":
                            fields_list = step.get("fields_list", [])
                            mapped_fields = [TIME_POINT_ALIAS_MAP.get(f, f) for f in fields_list]
                            step["fields_list"] = mapped_fields
                            logger.info(f"[DayaheadPriceWorkflow] project字段映射: {fields_list} -> {mapped_fields}")
                    logger.info(f"[{workflow_name}] 执行后置处理: {post_processing}")
                    processed_data = self.apply_post_processing(processed_data, post_processing, params)

                format_source_data = processed_data
                chart_data = processed_data
                display_data = processed_data
                summary_data = handle_summary_input(format_source_data)

            if isinstance(chart_data, (list, dict)):
                self._last_chart_data = copy.deepcopy(chart_data)

            # 6. 格式化结果
            logger.info(f"[{workflow_name}] 格式化结果")
            if isinstance(display_data, (dict, list)):
                data_preview = json.dumps(display_data, ensure_ascii=False, indent=2, default=str)[:1000]
            else:
                data_preview = str(display_data)[:1000]
            logger.info(f"[{workflow_name}] 输入到格式化的数据: {data_preview}")

            if isinstance(display_data, dict) and display_data.get('data_too_large'):
                # 直接构建响应，确保 questions 作为独立字段
                return {
                    "conversation_id": self.conversation_id,
                    "status": "success",
                    "workflow_type": workflow_name,
                    "message": {
                        "content": display_data['message'],
                        "questions": display_data['questions']
                    },
                    "collected_params": params
                }
            else:
                should_use_format_summary = self._should_use_format_summary(summary_data)
                format_task = self._format_result(effective_user_query, summary_data) if should_use_format_summary else None
                # format_task = self._format_result(effective_user_query, self._normalize_format_data(format_source_data)) if should_use_format_summary else None


                # 用于生成图的数据输入
                data_for_chart = format_source_data

                # 是否绘图逻辑判断（统一基于阶段前数据）
                # Chart visibility is decided from pre-truncation stage data.
                # 是否绘图统一基于截断前的阶段数据。
                should_generate_echarts = self.canEcharts(data_for_chart)

                if should_generate_echarts:
                    echarts_task = self.create_echarts(user_query,intent_result.get("intent",""),data_for_chart)
                    if should_use_format_summary and format_task is not None:
                        formatted_result, echarts_result = await asyncio.gather(format_task, echarts_task)
                    else:
                        formatted_result = ""
                        echarts_result = await echarts_task
                else:
                    formatted_result = await format_task if should_use_format_summary and format_task is not None else ""
                    echarts_result = ""

                formatted_result = filter_think_tags_simple(formatted_result)

                analysis_result = None
                # if not should_use_format_summary:
                #     analysis_result = await self._run_data_analysis(
                #         user_query=effective_user_query,
                #         raw_data=self._original_data,
                #         display_data=format_source_data,
                #         params=params,
                #     )

                chart_intro = self._build_chart_intro() if echarts_result else ""
                # analysis_summary_text = formatted_result if should_use_format_summary and formatted_result else ""
                # trend_summary_text = (
                #     analysis_result.get("summary")
                #     if analysis_result and analysis_result.get("summary")
                #     else ""
                # )
                # trend_intro = self._build_trend_intro() if trend_summary_text and not echarts_result else ""

                response = {
                    "conversation_id": self.conversation_id,
                    "status": "success",
                    "workflow_type": workflow_name,
                    # "message": {
                    #     "content": chart_intro or formatted_result or trend_intro
                    # },
                    "message": {
                        "content": chart_intro or formatted_result
                    },
                    "collected_params": params
                }

                if echarts_result:
                    response["message"]["echarts"] = echarts_result
                # if analysis_summary_text:
                #     response["message"]["analysis_summary"] = analysis_summary_text
                # if trend_summary_text:
                #     response["message"]["trend_summary"] = trend_summary_text
                # if analysis_result and trend_summary_text:
                #     response["message"]["analysis_detail"] = analysis_result.get("analysis_payload")
                #     response["message"]["trend_analysis"] = analysis_result.get("analysis_payload")

                self.session_state['status'] = 'completed'
                self.session_state['messages'].append(user_query)

                trunc_info = self.get_any_truncation_info()
                if trunc_info:
                    # 不再向非流式结果直接输出 full_data。
                    # response.update(trunc_info)
                    response.update({k: v for k, v in trunc_info.items() if k != "full_data"})

                return response

        except Exception as e:
            logger.error(f"[{workflow_name}] 工作流执行错误: {e}")
            import traceback
            traceback.print_exc()
            friendly = self._to_user_friendly_error(str(e))
            return {
                "conversation_id": self.conversation_id,
                "status": "error",
                "workflow_type": workflow_name,
                "error": friendly
            }

    async def execute_stream(self, user_query: str, intent_result: Dict, format_context: Optional[Dict] = None) -> AsyncGenerator[str, None]:
        try:
            workflow_name = self.__class__.__name__
            logger.info(f"[{workflow_name}] 开始执行工作流（流式）, conversation_id={self.conversation_id}")

            # 1. 参数提取
            yield {"data": "finish", "type": "messageLabel"}
            yield {"data": "parameter extraction", "type": "messageLabel"}
            logger.info(f"[{workflow_name}] 开始提取参数")
            self.session_state['status'] = 'extracting_params'
            params = await self._extract_parameters(user_query, intent_result)
            self.session_state['current_params'] = params
            logger.info(f"[{workflow_name}] 参数提取结果: {params}")

            # 2. 参数验证
            logger.info(f"[{workflow_name}] 验证参数")
            validation_result = self.validate_params(params)
            if not validation_result['valid']:
                logger.warning(f"[{workflow_name}] 参数验证失败: {validation_result['message']}")
                yield validation_result['message']
                return      

            # 3. 调用API
            yield {"data": "finish", "type": "messageLabel"}
            yield {"data": "API calling", "type": "messageLabel"}
            logger.info(f"[{workflow_name}] 调用API")
            self.session_state['status'] = 'calling_api'
            api_result = await self.call_api(params)
            if api_result['status'] == 'error':
                raw_msg = api_result.get("message")
                friendly_msg = self._to_user_friendly_error(raw_msg)  # 你新增/复用的统一函数
                # return {"status": "error", "message": friendly_msg}
                yield friendly_msg
                return

            # 4. 检查数据是否为空
            effective_user_query = user_query
            raw_data = self._extract_api_data_payload(api_result, workflow_name)
            raw_data, effective_user_query = await self._retry_with_expanded_time_range_if_empty(
                raw_data=raw_data,
                params=params,
                user_query=user_query,
                workflow_name=workflow_name,
            )
            logger.info(f"[{workflow_name}] API返回数据: {type(raw_data).__name__}, 内容: {raw_data if not isinstance(raw_data, (list, dict)) or len(str(raw_data)) < 200 else '...(数据过长)'}")

            if isinstance(raw_data, dict) and 'data' in raw_data and len(raw_data) <= 4:
                actual_data = raw_data.get('data', {})
                logger.info(f"[{workflow_name}] 检测到嵌套 data 结构，提取内层数据: {type(actual_data).__name__}")
                raw_data = actual_data

            self._original_data = copy.deepcopy(raw_data)

            is_empty = self._is_empty_result_data(raw_data)

            if is_empty:
                logger.info(f"[{workflow_name}] API返回空数据，直接跳到格式化步骤")
                processed_data = []
                format_source_data = []
                chart_data = []
                display_data = []
            else:
                # 5. 数据处理
                yield {"data": "finish", "type": "messageLabel"}
                yield {"data": "data processing", "type": "messageLabel"}
                logger.info(f"[{workflow_name}] 处理数据")
                operations = intent_result.get('operations', []) if isinstance(intent_result, dict) else []
                operations = self.normalize_operations(operations, effective_user_query, intent_result, params)
                operations = self.sanitize_operations(operations, params=params, user_query=effective_user_query)
                params['operations'] = operations
                processed_data = self.process_data(raw_data, params, effective_user_query)


                # 6. 执行操作
                if operations:
                    logger.info(f"[{workflow_name}] 执行操作: {operations}")
                    agg_result = await self.aggregation_agent.ainvoke(
                        user_query=effective_user_query,
                        operations=operations,
                        data=processed_data,
                        context={"thread_id": self.conversation_id},
                        domain=self._get_domain()
                    )
                    summary = agg_result.get('summary') or []
                    aggregated_data = agg_result.get('aggregated_data')
                    grouped_data_full = agg_result.get('grouped_data_full')

                    _summary_ops = {'sum', 'average', 'weighted_average', 'max', 'min', 'count', 'count_if'}
                    _row_ops = {'subtract', 'multiply', 'divide', 'mom_change', 'yoy_change'}
                    _sort_group_ops = {'top_n', 'bottom_n', 'group_by'}

                    _has_agg = any(op.get('operation') in _summary_ops for op in operations)
                    _has_row_ops = any(op.get('operation') in _row_ops for op in operations)
                    _has_sort_group = any(op.get('operation') in _sort_group_ops for op in operations)
                    _has_top_bottom = any(op.get('operation') in ('top_n', 'bottom_n') for op in operations)

                    _has_group_then_count = (
                        any(op.get('operation') == 'group_by' for op in operations)
                        and any(op.get('operation') == 'count' for op in operations)
                        and not any(op.get('operation') in ('top_n', 'bottom_n') for op in operations)
                    )

                    _has_group_then_extreme = (
                        any(op.get('operation') == 'group_by' for op in operations)
                        and any(op.get('operation') in ('max', 'min') for op in operations)
                        and not any(op.get('operation') in ('top_n', 'bottom_n') for op in operations)
                    )

                    if _has_group_then_count and summary:
                        processed_data = summary
                    elif _has_group_then_extreme and summary and aggregated_data:
                        extreme_summary = next(
                            (s for s in reversed(summary) if s.get('operation') in ('max', 'min')), None
                        )
                        if extreme_summary:
                            extreme_field = extreme_summary.get('field', '')
                            extreme_val = extreme_summary.get('result')
                            extreme_record = extreme_summary.get('record')
                            if extreme_record is None and extreme_val is not None and isinstance(aggregated_data, list):
                                for record in aggregated_data:
                                    if isinstance(record, dict) and record.get(extreme_field) == extreme_val:
                                        extreme_record = record
                                        break
                            processed_data = [extreme_record] if extreme_record else summary
                        else:
                            processed_data = summary
                    elif (_has_sort_group or _has_row_ops) and aggregated_data and not (_has_agg and summary and not _has_top_bottom):
                        processed_data = aggregated_data
                    elif _has_agg and summary:
                        processed_data = summary
                    elif aggregated_data:
                        processed_data = aggregated_data
                    elif agg_result:
                        # 兜底：aggregated_data 为空列表/空字典等 falsy 值时也要覆盖
                        processed_data = aggregated_data

                    processed_data = self.post_process_after_aggregation(
                        processed_data=processed_data,
                        user_query=effective_user_query,
                        intent_result=intent_result,
                        params=params,
                        operations=operations,
                        summary=summary,
                        aggregated_data=aggregated_data,
                        grouped_data_full=grouped_data_full,
                    )

                    logger.info(f"[{workflow_name}] 操作执行完成, summary={summary}")

                post_processing = intent_result.get('post_processing', []) if isinstance(intent_result, dict) else []
                if post_processing:
                    TIME_POINT_ALIAS_MAP = {"v0000": "v2400"}
                    for step in post_processing:
                        if step.get("action") == "project":
                            fields_list = step.get("fields_list", [])
                            mapped_fields = [TIME_POINT_ALIAS_MAP.get(f, f) for f in fields_list]
                            step["fields_list"] = mapped_fields
                            logger.info(f"[DayaheadPriceWorkflow] project字段映射: {fields_list} -> {mapped_fields}")
                    logger.info(f"[{workflow_name}] 执行后置处理: {post_processing}")
                    processed_data = self.apply_post_processing(processed_data, post_processing, params)
                # 检测：API 有数据返回但操作（如 filter）后结果为空
                _ops_filtered_to_empty = (
                    (not processed_data or (isinstance(processed_data, list) and len(processed_data) == 0))
                    and operations
                )
                if _ops_filtered_to_empty:
                    _raw_count = len(raw_data) if isinstance(raw_data, list) else 0
                    filter_ops = [op for op in operations if op.get('operation') == 'filter']
                    if filter_ops:
                        _op_cn = {'>': '大于', '<': '小于', '>=': '大于等于', '<=': '小于等于', '==': '等于'}
                        _filter_descs = []
                        for fop in filter_ops:
                            op_cn = _op_cn.get(fop.get('operator', ''), fop.get('operator', ''))
                            _filter_descs.append(f"{op_cn} {fop.get('value', '')}")
                        _filter_text = "、".join(_filter_descs)
                        processed_data = [{"筛选结果": f"API返回{_raw_count}条数据，经筛选条件（{_filter_text}）过滤后无匹配记录"}]
                    else:
                        processed_data = [{"筛选结果": f"API返回{_raw_count}条数据，经聚合操作后无匹配记录"}]

                display_data = processed_data
                format_source_data = processed_data
                display_data = processed_data

            # 7. 流式格式化
            logger.info(f"[{workflow_name}] 流式格式化结果")
            if isinstance(display_data, (dict, list)):
                data_preview = json.dumps(display_data, ensure_ascii=False, indent=2, default=str)[:1000]
                # data_preview = json.dumps(display_data, ensure_ascii=False, indent=2)
            else:
                data_preview = str(display_data)[:1000]
                # data_preview = str(display_data)

            logger.info(f"[{workflow_name}] 流式输入到格式化的数据(阶段前): {data_preview}")


            # 统一维护生成图输入数据
            data_for_chart = format_source_data
            summary_data = handle_summary_input(format_source_data)
            summary_len = len(summary_data) if summary_data is not None else 0
            logger.info(f"[{workflow_name}] 流式输入总结summary_data:\n {summary_data}\n")

            should_generate_echarts = self.canEcharts(data_for_chart)

            should_use_format_summary = self._should_use_format_summary(summary_data)
            chart_intro = self._build_chart_intro() if should_generate_echarts else ""
            trend_intro = self._build_trend_intro()
            result_stream = (
                self._format_result_stream(effective_user_query, summary_data, format_context)
                # self._format_result_stream(effective_user_query, self._normalize_format_data(summary_data), format_context)
                if should_use_format_summary else None
            )

            echarts_queue = asyncio.Queue()
            echarts_error = None

            async def formatted_summary_gen():
                if result_stream is None:
                    return
                async for chunk in result_stream:
                    if hasattr(chunk, 'content'):
                        yield chunk.content
                    elif isinstance(chunk, str):
                        yield chunk
                    else:
                        yield str(chunk)

            async def collect_echarts():
                nonlocal echarts_error
                try:
                    echarts_filled = await self.create_echarts(user_query,intent_result.get("intent",""),data_for_chart)
                    await echarts_queue.put(echarts_filled)
                except Exception as e:
                    echarts_error = e
                    logger.error(f"[{workflow_name}] ECharts生成错误（流式）: {e}")
                finally:
                    await echarts_queue.put(None)

            echarts_task = asyncio.create_task(collect_echarts()) if should_generate_echarts else None
            # data_analysis = None
            # if not should_use_format_summary:
            #     data_analysis = await self._prepare_data_analysis_stream(
            #         user_query=effective_user_query,
            #         raw_data=self._original_data,
            #         display_data=format_source_data,
            #         params=params,
            #     )

            try:
                yield {"data": "finish", "type": "messageLabel"}
                if should_generate_echarts:
                    # yield {"data": chart_intro, "type": "content"}
                    if should_use_format_summary:
                        async for chunk in filter_think_tags_async(formatted_summary_gen()):
                            yield {"data": chunk, "type": "content"}
                    # yield {"data": "ECharts generation", "type": "messageLabel"}
                    yield {"data": "", "type": "Placeholder_True"}
                    while True:
                        chunk = await echarts_queue.get()
                        if chunk is None:
                            break
                        yield {"data": "", "type": "Placeholder_False"}
                        yield {"data": chunk, "type": "content"}
                    if echarts_error:
                        raise echarts_error

                else:
                    if should_use_format_summary:
                        async for chunk in filter_think_tags_async(formatted_summary_gen()):
                            yield {"data": chunk, "type": "content"}
                    # elif data_analysis:
                    #     yield {"data": trend_intro, "type": "content"}

                # if data_analysis:
                #     summary_stream = self.trend_analysis_agent.summarize_stream(
                #         user_query=effective_user_query,
                #         overview=data_analysis["overview"],
                #         analysis_result=data_analysis["analysis_payload"],
                #     )
                #     async for chunk in filter_think_tags_async(summary_stream):
                #         yield {"data": chunk, "type": "content"}
            finally:
                if echarts_task:
                    await echarts_task

            self.session_state['status'] = 'completed'
            self.session_state['messages'].append(user_query)

            logger.info(f"[{workflow_name}] 工作流执行完成（流式）")

        except Exception as e:
            logger.error(f"[{workflow_name}] 工作流执行错误（流式）: {e}")
            import traceback
            traceback.print_exc()
            yield self._to_user_friendly_error(str(e))

    async def _extract_parameters(self, user_query: str, intent_result: Dict) -> Dict[str, Any]:
        """提取参数（内部方法）"""
        result = await self.parameter_agent.ainvoke(
            user_query=user_query,
            intent_result=intent_result,
            prompt_template=self.get_parameter_prompt(),
            context={"thread_id": self.conversation_id}
        )

        # 如果result已经是字典，直接返回
        if isinstance(result, dict):

            #为参数提取的字段的内容做标准化匹配
            params=self._adjust_time_range_end(result)
            if "name_abbreviation" in params:
                params["name_abbreviation"] = data_matching(params["name_abbreviation"], "name_abbreviation")
            if "device_name" in params:
                params["device_name"] = data_matching(params["device_name"], "device_name")
            if "plant_name" in params:
                params["plant_name"] = data_matching(params["plant_name"], "plant_name")
            if "outage_type" in params:
                params["outage_type"] = data_matching(params["outage_type"], "outage_type")
            if "sysName" in params:
                params["sysName"] = data_matching(params["sysName"], "sysName")
            if "sendrecv" in params:
                params["sendrecv"] = data_matching(params["sendrecv"], "sendrecv")
            return params

        # 如果是字符串，尝试解析JSON
        if isinstance(result, str):
            try:
                if "```json" in result:
                    json_str = result.split("```json")[1].split("```")[0].strip()
                elif "```" in result:
                    json_str = result.split("```")[1].split("```")[0].strip()
                else:
                    json_str = result.strip()

                params = json.loads(json_str)
                if 'post_processing' in intent_result:
                    params['post_processing'] = intent_result['post_processing']

                # 为参数提取的字段的内容做标准化匹配
                params = self._adjust_time_range_end(params)
                if "name_abbreviation" in params:
                    params["name_abbreviation"] = data_matching(params["name_abbreviation"], "name_abbreviation")
                if "device_name" in params:
                    params["device_name"] = data_matching(params["device_name"], "device_name")
                if "plant_name" in params:
                    params["plant_name"] = data_matching(params["plant_name"], "plant_name")
                if "outage_type" in params:
                    params["outage_type"] = data_matching(params["outage_type"], "outage_type")
                if "sysName" in params:
                    params["sysName"] = data_matching(params["sysName"], "sysName")
                if "sendrecv" in params:
                    params["sendrecv"] = data_matching(params["sendrecv"], "sendrecv")
                return params

            except (json.JSONDecodeError, IndexError) as e:
                logger.error(f"[参数提取] JSON解析失败: {e}, 原始内容: {result}")
                return {}

        logger.error(f"[参数提取] 未知的返回类型: {type(result)}")
        return {}

    def _adjust_time_range_end(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """将 time_range.end 加一天，确保查询区间包含当天数据"""
        time_range = params.get("time_range")
        if not isinstance(time_range, dict):
            return params

        # 兼容多种时间键名：统一归一到 start/end
        # 常见来源：
        # - start_time/end_time（专项识别结果）
        # - startDate/endDate（部分业务字段命名）
        alias_start = time_range.get("start")
        alias_end = time_range.get("end")
        if alias_start in (None, ""):
            alias_start = time_range.get("start_time")
        if alias_start in (None, ""):
            alias_start = time_range.get("startDate")
        if alias_end in (None, ""):
            alias_end = time_range.get("end_time")
        if alias_end in (None, ""):
            alias_end = time_range.get("endDate")

        if alias_start not in (None, ""):
            params["time_range"]["start"] = alias_start
        if alias_end not in (None, ""):
            params["time_range"]["end"] = alias_end

        raw_start = time_range.get("start")
        raw_end = time_range.get("end")
        start = self._normalize_date_str(raw_start)
        end = self._normalize_date_str(raw_end)
        if start != raw_start or end != raw_end:
            params["time_range"]["start"] = start
            params["time_range"]["end"] = end
            logger.info(
                f"[参数提取] time_range 归一化: start={raw_start} -> {start}, "
                f"end={raw_end} -> {end}"
            )

        if not end:
            return params
        try:
            end_dt = datetime.strptime(end, "%Y-%m-%d")
            new_end = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
            params["time_range"]["end"] = new_end
            logger.info(f"[参数提取] time_range.end 调整: {end} → {new_end}")
        except ValueError:
            logger.warning(f"[参数提取] time_range.end 格式不合法，跳过调整: {end}")
        return params

    # async def _format_result(self, user_query: str, data: Any) -> str:
    #     """格式化结果（非流式，内部方法）"""
    #     # 将数据转换为JSON字符串
    #     if isinstance(data, (dict, list)):
    #         data_str = json.dumps(data, ensure_ascii=False, indent=2)
    #     else:
    #         data_str = str(data)
    #
    #     result = await self.format_agent.ainvoke(
    #         user_query=user_query,
    #         data=data_str,
    #         prompt_template=self.get_format_prompt(),
    #         context={"thread_id": self.conversation_id}
    #     )
    #
    #     return result
    async def _format_result(self, user_query: str, data: Any, format_context: Optional[Dict] = None) -> str:
        """格式化结果（非流式）。
        注意：传入的 data 已经是双重截断后的展示数据，本方法不再做任何截断。
        """
        # 处理后置处理中的计数结果（如果有）
        extra_context = ""
        if isinstance(data, dict) and "count_filtered_result" in data:
            count_info = data["count_filtered_result"]
            extra_context = f"\n【附加统计结果】满足条件 {count_info['condition']} 的记录共有 {count_info['count']} 条。"
            data = data.get("original_data", data)

        if isinstance(data, dict) and "_record_count" in data:
            record_count = data["_record_count"]
            extra_context = f'\n【强制要求】当前查询共返回 {record_count} 条数据。你必须在回复的第一句话中明确告知用户「共查询到{record_count}条数据」，然后再陈述数据内容。这是硬性要求，不可省略。' + extra_context

        if isinstance(data, list) and len(data) > 2:
            list_count = len(data)
            extra_context += f'\n【数据统计】当前查询结果共有 {list_count} 条记录。请严格遵守「条目展示规范」：如果你因输出限制无法逐条列出全部 {list_count} 条记录，则必须在最后一个列出的条目后添加「等」字（如：上海、浙江、四川主网等），禁止只列出部分条目而不加「等」。'

        data_str = json.dumps(data, ensure_ascii=False, indent=2, default=str) if isinstance(data, (dict, list)) else str(data)

        prompt_template = self.get_format_prompt()
        time_note = ('''# 时间格式处理规则
                        - 对于日期时间字段（如 businessTime、createDate、rq），若值为 ISO 8601 格式（如 "2025-06-23T16:00:00.000+00:00"），需转换为北京时间（UTC+8）。
                        - 转换方法：将原始 UTC 时间加 8 小时，然后格式化为 "YYYY-MM-DD"（仅展示日期）。例如：
                        - 输入："2025-06-23T16:00:00.000+00:00" → 输出："2025-06-24"
                        - 输入："2025-06-23T00:00:00.000+00:00" → 输出："2025-06-23"
                        - 若字段值已经是标准日期格式（如 "2025-06-23"），直接使用。''')
        prompt_template = time_note + self._build_global_format_rules() + prompt_template

        if format_context and format_context.get("is_multi"):
            idx = format_context.get("index", 1)
            total = format_context.get("total", 1)
            prompt_template += f"\n\n【注意】这是您提出的多个问题中的第 {idx}/{total} 个。请只针对当前子问题进行回答，不要提及这是第几个问题。"

        truncation_prompt = self._build_truncation_prompt()
        if truncation_prompt:
            prompt_template += truncation_prompt

        if extra_context:
            prompt_template += extra_context

        expansion_hint = self._build_time_expansion_hint(
            getattr(self, "_time_range_expanded_info", None)
        )
        if expansion_hint:
            prompt_template += expansion_hint

        if self.INNER_MODEL_ENABLE:
            result = await self.format_agent.ainvoke_for_format(
                user_query=user_query,
                data=data_str,
                prompt_template=prompt_template,
            )
        else:
            result = await self.format_agent.ainvoke(
                user_query=user_query,
                data=data_str,
                prompt_template=prompt_template,
                context={"thread_id": self.conversation_id}
            )
        return result

    async def _format_result_stream(self, user_query: str, data: Any, format_context: Optional[Dict] = None) -> \
    AsyncGenerator[str, None]:
        """流式格式化，逻辑同 _format_result。"""
        extra_context = ""
        if isinstance(data, dict) and "count_filtered_result" in data:
            count_info = data["count_filtered_result"]
            extra_context = f"\n【附加统计结果】满足条件 {count_info['condition']} 的记录共有 {count_info['count']} 条。"
            data = data.get("original_data", data)

        if isinstance(data, dict) and "_record_count" in data:
            record_count = data["_record_count"]
            extra_context = f'\n【强制要求】当前查询共返回 {record_count} 条数据。你必须在回复的第一句话中明确告知用户「共查询到{record_count}条数据」，然后再陈述数据内容。这是硬性要求，不可省略。' + extra_context

        if isinstance(data, list) and len(data) > 2:
            list_count = len(data)
            extra_context += f'\n【数据统计】当前查询结果共有 {list_count} 条记录。请严格遵守「条目展示规范」：如果你因输出限制无法逐条列出全部 {list_count} 条记录，则必须在最后一个列出的条目后添加「等」字（如：上海、浙江、四川主网等），禁止只列出部分条目而不加「等」。'

        data_str = json.dumps(data, ensure_ascii=False, indent=2, default=str) if isinstance(data, (dict, list)) else str(data)

        prompt_template = self.get_format_prompt()
        time_note = ('''# 时间格式处理规则
                        - 对于日期时间字段（如 businessTime、createDate、rq），若值为 ISO 8601 格式（如 "2025-06-23T16:00:00.000+00:00"），需转换为北京时间（UTC+8）。
                        - 转换方法：将原始 UTC 时间加 8 小时，然后格式化为 "YYYY-MM-DD"（仅展示日期）。例如：
                        - 输入："2025-06-23T16:00:00.000+00:00" → 输出："2025-06-24"
                        - 输入："2025-06-23T00:00:00.000+00:00" → 输出："2025-06-23"
                        - 若字段值已经是标准日期格式（如 "2025-06-23"），直接使用。''')
        prompt_template = time_note + self._build_global_format_rules() + prompt_template

        if format_context and format_context.get("is_multi"):
            idx = format_context.get("index", 1)
            total = format_context.get("total", 1)
            prompt_template += f"\n\n【注意】这是您提出的多个问题中的第 {idx}/{total} 个。请只针对当前子问题进行回答，不要提及这是第几个问题。"

        truncation_prompt = self._build_truncation_prompt()
        if truncation_prompt:
            prompt_template += truncation_prompt

        if extra_context:
            prompt_template += extra_context

        expansion_hint = self._build_time_expansion_hint(
            getattr(self, "_time_range_expanded_info", None)
        )
        if expansion_hint:
            prompt_template += expansion_hint

        # 获取异步生成器并逐块产出
        if self.INNER_MODEL_ENABLE:
            # logger.info(f"[_format_result_stream] 流式调用人工智能平台模型参数：user_query:{user_query}\ndata_str:{data_str}\n")
            stream = self.format_agent.astream_for_format(
                user_query=user_query,
                data=data_str,
                prompt_template=prompt_template,
            )
        else:
            stream = self.format_agent.astream(
                user_query=user_query,
                data=data_str,
                prompt_template=prompt_template,
                context={"thread_id": self.conversation_id}
            )
        async for chunk in stream:
            yield chunk

    def _build_truncation_prompt(self) -> str:
        """根据 self._truncated 和 self._field_truncated 组合构建截断提示"""
        both = self._truncated and self._field_truncated
        only_rows = self._truncated and not self._field_truncated
        only_fields = not self._truncated and self._field_truncated

        if both:
            front = ', '.join(self._truncated_front_fields) if self._truncated_front_fields else '无'
            back = ', '.join(self._truncated_back_fields) if self._truncated_back_fields else '无'
            return f"""
    【【系统指令：数据与字段双截断处理规范】
    **【强制约束】请严格按照以下规范处理截断数据并生成回复，不得自行补充或推测未展示内容。**

    本次查询结果数据量较大，已同时应用两种截断策略：
    1. 记录条数截断：总记录数超过 {MAX_DISPLAY_ITEMS} 条，当前仅展示前 {MAX_DISPLAY_ITEMS} 条。
    2. 时间字段精简：原始数据包含大量连续时间点，已保留代表性时刻（前4个与后4个），中间省略。

    **表格构建规范（需展示表格时）**：
    - 表格顺序：非时间字段 + 前4个字段（{front}）+ 一个单独的 "..." 列+ 后4个字段（{back}）。

    **回复要求**：
    - 回答开头用一句话统一说明“因数据量较大且时间点密集，仅展示部分记录和代表性时刻”。
    - 接着提醒用户点击下方链接获取完整数据表（含全部记录与全部时间点）。
    - 展示表格，需展示全部数据的表格，不得删减。
    - 表格后禁止生成任何内容
    """

        if only_rows:
            return f"""
    【系统指令：记录条数截断处理规范】
    **【强制约束】请严格按照以下规范处理截断数据并生成回复，不得自行补充或推测未展示内容。**

    本次返回的数据总条数超过 {MAX_DISPLAY_ITEMS} 条，当前仅展示前 {MAX_DISPLAY_ITEMS} 条。
    请严格遵循：
    1. 回答开头用一句话指出“因数据总量过大，仅展示部分记录”。
    2. 末尾提醒用户点击下方链接获取完整数据表。
    3. 展示表格，需展示全部数据的表格，不得删减。
    4. 表格后禁止生成任何内容
    """

        if only_fields:
            front = ', '.join(self._truncated_front_fields) if self._truncated_front_fields else '无'
            back = ', '.join(self._truncated_back_fields) if self._truncated_back_fields else '无'
            return f"""
    【系统指令：时间序列字段精简处理规范】
    **【强制约束】请严格按照以下规范处理截断数据并生成回复，不得自行补充或推测未展示内容。**

    原始数据包含大量连续时间点，已保留前4个和后4个代表性时刻，中间列省略。
    **保留字段**：前4个({front})，后4个({back})。

    **表格构建规范**：
    - 非时间字段 + 前4个字段 + "..." 列 + 后4个字段。

    **回复要求**：
    - 回答开头用一句话指出“因时间点较多，仅展示部分时刻”。
    - 提醒用户完整数据可通过点击下方链接获取。
    - 展示表格，需展示全部数据的表格，不得删减。
    - 表格后禁止生成任何内容
    """

        return ""

    def get_any_truncation_info(self) -> dict | None:
        if self._original_data is None:
            return None

        info = {
            "truncated": True,
            "query_id": None,
            "conversation_id": self.conversation_id,
            "truncation_types": [],
            "full_data": self._original_data
        }

        if self._truncated:
            info["truncation_types"].append("row_count")
            info["total_count"] = len(self._full_data) if isinstance(self._full_data, list) else 0
            info["display_count"] = MAX_DISPLAY_ITEMS
        if self._field_truncated:
            info["truncation_types"].append("field_filter")
            info["max_v_fields_kept"] = self.MAX_V_FIELDS
            # 保留 max_time_field_used 用于兼容（但已无实际时间值）
            info["max_time_field"] = f"kept_first_{self.MAX_V_FIELDS}"

        return info

    def get_session_state(self) -> Dict[str, Any]:
        """获取当前会话状态"""
        return self.session_state.copy()

    def reset_session(self):
        """重置会话状态"""
        self.session_state = {
            'status': 'idle',
            'current_params': {},
            'messages': []
        }
        logger.info(f"[会话重置] conversation_id={self.conversation_id}")


    def canEcharts(self, data):
        is_empty = (
                data is None
                or data == []
                or data == [{}]
                or data == {}
        )
        is_single = (isinstance(data, list) and len(data) == 1) or isinstance(
            data, dict)

        return not is_empty and not is_single


    # async def create_echarts(self, user_query, data):
    #         # 对数据中的时间字段进行排序
    #         sorted_data = sort_time_fields_batch(data)
    #         fill_data = build_echarts_data(sorted_data)
    #         handled_data = handle_echarts_input(fill_data)
    #
    #         input_data = handled_data if handled_data else sorted_data
    #         logger.info(
    #             f"echarts配置项数据输入：userQuery:{user_query}\ndata:{data}\nfill_data:{fill_data}\ninput_data:{input_data}")
    #
    #         data_str = json.dumps(input_data, ensure_ascii=False)
    #
    #         if self.INNER_MODEL_ENABLE:
    #             echarts_task_result = await self.echarts_agent.ainvoke_for_echarts(user_query=user_query, data=data_str)
    #         else:
    #             echarts_task_result = await self.echarts_agent.ainvoke(user_query=user_query, data=data_str)
    #         echarts_task_result = filter_think_tags_simple(echarts_task_result)
    #
    #         try:
    #             s = echarts_task_result.strip()  # 去除首尾空白和换行
    #             # 匹配可能的 Markdown 代码块：```json 可选 ... ```
    #             pattern = r"^```(?:json)?\s*\n?(.*?)\n?```$"
    #             match = re.search(pattern, s, re.DOTALL)
    #             if match:
    #                 json_str = match.group(1).strip()
    #             else:
    #                 json_str = s
    #
    #             echarts_config = json.loads(json_str)
    #         except Exception as e:
    #             echarts_config = {}
    #             logger.info(f"echarts配置项生成格式错误，echarts_task_result:{echarts_task_result}")
    #             pass
    #         logger.info(f"[base_workflow] echarts配置项原始：echarts_config：{echarts_config}\nsorted_data:{fill_data}")
    #         # logger.info(f"[base_workflow] echarts配置项原始：echarts_config：{echarts_config}")
    #         blank, config = fill_echarts_config(echarts_config, fill_data)
    #         if not blank:
    #             return f""
    #         else:
    #             echarts_filled = json.dumps(config, ensure_ascii=False)
    #             return f"\n\n```echarts\n{echarts_filled}```"


    async def create_echarts(self, user_query,intent, data):
        sorted_data = sort_time_fields_batch(data)

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
            logger.info(f"[base_workflow] userQuery:{user_query} ——> 判断[代码]生成明细图表")
            return f"\n\n```echarts\n{echarts_filled}```"

        # 模型生成图
        data_str = json.dumps(sorted_data, ensure_ascii=False, default=str)
        if self.INNER_MODEL_ENABLE:
            echarts_task_result = await self.echarts_agent.ainvoke_for_echarts(user_query=user_query, data=data_str)
        else:
            echarts_task_result = await self.echarts_agent.ainvoke(user_query=user_query, data=data_str)

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

        logger.info(f"[base_workflow] userQuery:{user_query} ——> 判断[模型]生成明细图表\n模型输出echarts配置：{echarts_config}")

        series = echarts_config.get('series', []) if isinstance(echarts_config, dict) else []
        # 判断图数据是否均为non_zero
        if not has_non_zero_series(series):
            return f""

        # 最高点、最低点添加
        config = add_global_extremes_to_echarts(echarts_config)
        echarts_filled = json.dumps(config, ensure_ascii=False)
        return f"\n\n```echarts\n{echarts_filled}```"