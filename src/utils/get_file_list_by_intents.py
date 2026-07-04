import json
import logging
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

INTENT_DOC_MAPPING: Dict[str, Dict[str, Any]] = {
    "省间应急调度交易信息（日前）查询": {
        "docName": "应急调度交易信息日前计划",
        "pcUrl": "https://pmos.sgcc.com.cn/#/loginTicket?menuPath=/pxf-settlement-outnetpub/columnHomeLeftMenuNew&menuName=综合查询&marketId=PSGCC&guid=8b69a0fc-943c-44ca-a31a-90bf519ed207&ticket=",
        "appUrl": {
            "targetRoute": "/information/home/InformationYjddPage",
            "marketId": "PSGCC",
            "firstTitle": "市场运营",
            "secondTitle": "应急调度交易信息日前计划",
            "targetArguments": {
            "infoCode": "rqyjdu",
            "label": "应急调度交易信息日前计划",
            "title": "应急调度交易信息日前计划",
            "guid": "8b69a0fc-943c-44ca-a31a-90bf519ed207",
            "menuShortIntroduction": "",
            "viewName": "linePointCurve",
            }
        },
    },
}

def get_file_list_by_intents(intents: List[str]) -> List[Dict[str, str]]:
    """根据意图列表获取对应的文档信息列表（fileList），包含PC/App双链接及有效标识flag。"""
    seen = set()
    file_list = []
    for intent in intents:
        doc_info = INTENT_DOC_MAPPING.get(intent)
        if not doc_info:
            logger.warning(f"意图 '{intent}' 未在映射中配置，已跳过")
            continue

        pc_url = doc_info.get("pcUrl")
        app_url = doc_info.get("appUrl")
        dedup_key = (pc_url, json.dumps(app_url, sort_keys=True, ensure_ascii=False))

        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        file_list.append({
            "docName": doc_info.get("docName", ""),
            "pcUrl": pc_url,
            "appUrl": app_url,
            "document_id": "",
            "flag": True
        })
    return file_list

def get_file_list_by_intents_with_ticket(intents: List[str], ticket: str) -> List[Dict[str, str]]:
    """根据意图列表和ticket参数获取对应的文档信息列表（fileList）。"""
    def _append_ticket_to_pc_url(original_pc_url: str, ticket_val: str) -> str:
        if not original_pc_url.endswith("ticket="):
            logger.warning(f"原始pcUrl格式异常，未以'ticket='结尾: {original_pc_url}")
        return original_pc_url + ticket_val

    seen = set()
    file_list = []
    for intent in intents:
        doc_info = INTENT_DOC_MAPPING.get(intent)
        if not doc_info:
            logger.warning(f"意图 '{intent}' 未在映射中配置，已跳过")
            continue

        pc_url = doc_info.get("pcUrl")
        app_url = doc_info.get("appUrl")
        dedup_key = (pc_url, json.dumps(app_url, sort_keys=True, ensure_ascii=False))

        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        new_pc_url = _append_ticket_to_pc_url(pc_url, ticket)

        file_list.append({
            "docName": doc_info.get("docName", ""),
            "pcUrl": new_pc_url,
            "appUrl": app_url,
            "document_id": "",
            "flag": True
        })
    return file_list
