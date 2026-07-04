import json
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Any, Dict, Set

from src.workflow.base_workflow import BaseWorkflow
from src.agents import prompts
from src.utils.logging_setup import logger
from src.config import settings

# 加载时间点映射（复用）
with open("data/env_variables/time_point_mapping.json", "r", encoding="utf-8") as f:
    _time_point_mapping = json.load(f)
_time_point_aliases: Dict[str, str] = _time_point_mapping.get("aliases", {})
_ALL_TIME_POINT_FIELDS = {k for k in _time_point_mapping if k != "aliases"}

# 加载应急调度字段配置
with open("data/env_variables/emergency_dispatch_fields_config.json", "r", encoding="utf-8") as f:
    _emergency_fields_config = json.load(f)
_ALL_FIELDS: Dict[str, Dict] = _emergency_fields_config["fields"]

# 非时间点字段的关键词映射
_FIELD_KEYWORD_MAP: Dict[str, str] = {}
for _fname, _finfo in _ALL_FIELDS.items():
    if not _finfo.get("is_time_point"):
        for _kw in _finfo.get("keywords", []):
            _FIELD_KEYWORD_MAP[_kw] = _fname


class EmergencyDayaheadWorkflow(BaseWorkflow):
    """应急调度交易信息（日前）查询工作流"""

    def get_parameter_prompt(self) -> str:
        return prompts.EmergencyDispatch_Parameter_Extraction_Prompt

    def get_format_prompt(self) -> str:
        return prompts.EmergencyDispatch_Result_Format_Prompt

    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not params.get("time_range"):
            return {
                "valid": False,
                "message": "请提供查询日期或时间范围（例如：2025年3月15日、今天、本月等）"
            }
        return {"valid": True, "message": "参数验证通过"}

    async def _call_api_impl(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """调用应急调度交易信息 API（统一走 COMMON_API_URL）"""
        common_api_url = settings.COMMON_API_URL
        biz_url = self.api_base_url
        if not common_api_url:
            return {"status": "error", "message": "COMMON_API_URL未配置"}
        if not biz_url:
            return {"status": "error", "message": "应急调度交易信息API地址未配置"}

        try:
            hash_map: Dict[str, Any] = {}

            time_range = params.get("time_range")
            if isinstance(time_range, dict):
                def _shift_date(date_str: str, days: int) -> str:
                    try:
                        return (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=days)).strftime("%Y-%m-%d")
                    except ValueError:
                        return date_str

                start = time_range.get("start")
                end = time_range.get("end")
                if start:
                    start = _shift_date(start, 0)
                if end:
                    end = _shift_date(end, 0)

                if start and end and start == end:
                    hash_map["businessDate"] = start
                    logger.info(f"[EmergencyDayaheadAPI] 单日查询 businessDate={start}")
                else:
                    if start:
                        hash_map["varStartDate"] = start
                    if end:
                        hash_map["varEndDate"] = end
                    logger.info(f"[EmergencyDayaheadAPI] 范围查询 startDate={start} endDate={end}")

            # if params.get("market_id"):
            hash_map["variMarketId"] = "PSGCC"
            if params.get("sys_name"):
                hash_map["varSysName"] = params["sys_name"]
            if params.get("sendrecv"):
                hash_map["varSendRecv"] = params["sendrecv"]
            if params.get("reason"):
                hash_map["varReason"] = params["reason"]


            request_body = {"hashMap": hash_map, "url": biz_url, "interfaceName": self.interface_name}

            logger.info(f"[EmergencyDayaheadAPI] POST {common_api_url}")
            logger.info(f"[EmergencyDayaheadAPI] 请求体: {json.dumps(request_body, ensure_ascii=False)}")

            return await self._call_common_api(
                biz_url=biz_url,
                hash_map=hash_map,
                timeout=120.0,  # 可根据业务调整
            )

        except Exception as e:
            msg = f"API调用异常: {e}"
            logger.error(f"[EmergencyDayaheadAPI] {msg}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": msg}

    def process_data(self, data: Any, params: Dict[str, Any], user_query: str) -> Any:
        actual_data = data
        if isinstance(data, dict) and "data" in data:
            actual_data = data["data"]
        elif isinstance(data, dict) and "list" in data:
            actual_data = data["list"]

        logger.info(f"[EmergencyDayaheadWorkflow] 实际数据条数: {len(actual_data) if isinstance(actual_data, list) else 1}")

        requested_fields = self._resolve_requested_fields(params, user_query)

        if not requested_fields:
            logger.info("[EmergencyDayaheadWorkflow] 未识别到特定字段，返回全部数据")
            return actual_data

        logger.info(f"[EmergencyDayaheadWorkflow] 筛选字段: {requested_fields}")

        if isinstance(actual_data, list):
            return [
                {k: v for k, v in item.items() if k in requested_fields}
                if isinstance(item, dict) else item
                for item in actual_data
            ]
        elif isinstance(actual_data, dict):
            return {k: v for k, v in actual_data.items() if k in requested_fields}
        return actual_data

    def _resolve_requested_fields(self, params: Dict[str, Any], user_query: str) -> Set[str]:
        fields: Set[str] = set()

        time_points = params.get("time_points", [])
        if isinstance(time_points, list):
            fields.update(tp for tp in time_points if tp in _ALL_TIME_POINT_FIELDS)

        requested = params.get("requested_fields", [])
        if isinstance(requested, list):
            fields.update(f for f in requested if f in _ALL_FIELDS)

        for kw, fname in _FIELD_KEYWORD_MAP.items():
            if kw in user_query:
                fields.add(fname)

        for time_str, field_name in _time_point_aliases.items():
            if time_str in user_query:
                fields.add(field_name)

        operations = params.get("operations", [])
        if isinstance(operations, list):
            for op in operations:
                if not isinstance(op, dict):
                    continue
                op_type = op.get("operation", "")
                if op_type == "group_by":
                    group_field = op.get("group_field")
                    if group_field:
                        fields.add(group_field)
                    agg_field = op.get("agg_field")
                    if agg_field:
                        if isinstance(agg_field, list):
                            fields.update(f.strip() for f in agg_field if f.strip())
                        elif isinstance(agg_field, str) and ',' in agg_field:
                            fields.update(f.strip() for f in agg_field.split(',') if f.strip())
                        else:
                            fields.add(agg_field)
                elif op_type in ("max", "min"):
                    f = op.get("field", "")
                    if f and not any(f.startswith(p) for p in ("sum_", "avg_", "average_", "max_", "min_")):
                        fields.add(f)
                else:
                    for key in ("field", "field_b", "time_field", "key_field"):
                        f = op.get(key)
                        if f is None:
                            continue
                        if isinstance(f, list):
                            for item in f:
                                if isinstance(item, str):
                                    fields.add(item)
                        elif isinstance(f, str):
                            fields.add(f)

        if fields:
            for base_field in ("businessTime", "marketId", "sysName", "power", "sendrecv", "planDate", "seller", "buyer", "reason"):
                if base_field in _ALL_FIELDS:
                    fields.add(base_field)

        return fields
