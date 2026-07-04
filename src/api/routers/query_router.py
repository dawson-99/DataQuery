"""
智能数据查询 - 通用查询路由模块（并发增强版）
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.config import settings
from src.service.workflow_factory import workflow_factory
from src.utils.get_file_list_by_intents import get_file_list_by_intents
from src.utils.logging_setup import logger, get_trace_id, set_trace_id
from src.api.schemas.schemas import QueryRequest, ReturnQuery
from uuid import uuid4
from src.api.routers.shared_cache import (
     sse, cache_get, cache_delete,
    cache_delete_user, cache_get_all_stats, cache_cleanup_expired
)

router = APIRouter(prefix="/v1/query", tags=["智能查询"])

# ---------- 缓存管理 ----------
# 缓存结构：user_id -> {
#     query_id -> {
#         "full_data": Any,
#         "total_count": int,
#         "display_count": int,
#         "max_time_field": str,
#         "truncation_types": list,
#         "conversation_id": str,
#         "cached_at": datetime
#     }
# }

_cache_lock = asyncio.Lock()  # 保护共享缓存的异步锁

messagesLabelEnum = {
    "problem rewriting": '- <span style="color:#888888; font-size:12px;">问题改写中...</span>',
    "intent recognition Ⅰ": '- <span style="color:#888888; font-size:12px;">意图识别中 (阶段一)...</span>',
    "intent recognition II": '- <span style="color:#888888; font-size:12px;">意图识别中 (阶段二)...</span>',
    "problem breakdown": '- <span style="color:#888888; font-size:12px;">问题拆解中...</span>',
    "parameter extraction": '- <span style="color:#888888; font-size:12px;">参数提取中...</span>',
    "API calling": '- <span style="color:#888888; font-size:12px;">数据查询中...</span>',
    "data processing": '- <span style="color:#888888; font-size:12px;">数据处理中...</span>',
    "finish": '<span style="color:#888888; font-size:12px;">✔</span>'+ "\n",
    # "ECharts generation": "\n" + '- <span style="color:#888888; font-size:12px;">绘图中...</span>' + "\n",
}

# ---------- 缓存配置 ----------
CACHE_EXPIRY_HOURS = 5 / 60      # 缓存过期时间（小时，5分钟）
CLEANUP_INTERVAL_SECONDS = 60  # 清理间隔（秒）
MAX_CONCURRENT_REQUESTS = 50    # 最大并发请求数
REQUEST_TIMEOUT_SECONDS = getattr(settings, 'REQUEST_TIMEOUT_SECONDS', 60)    # 单个请求超时时间（秒）

_cleanup_task: Optional[asyncio.Task] = None
_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


def build_stream_payload(data: Any, payload_type: str) -> dict:
    """构建统一流式payload"""
    return {"data": data, "type": payload_type}


def emit_answer_start(conversation_id: str) -> str:
    return seeNow(
        "message",
        ReturnQuery(
            conversationId=conversation_id,
            answer="<think></think><answer>"
        )
    )

def emit_echarts_placeholder(conversation_id: str) -> str:
    return seeNow(
        "message",
        ReturnQuery(conversationId=conversation_id, echart_holder=True)
    )

def convert_message_label(raw_chunk: dict) -> str:
    message_tag = messagesLabelEnum.get(raw_chunk.get("data", ""), "")
    return message_tag




def normalize_stream_chunk(chunk: Any) -> dict:
    """将内部流式输出标准化，避免把已格式化的 SSE 字符串再次包进 data 字段。"""
    if isinstance(chunk, dict) and "data" in chunk and "type" in chunk:
        return chunk

    if isinstance(chunk, str):
        stripped = chunk.strip()
        if stripped:
            data_lines = []
            for line in stripped.splitlines():
                if line.startswith("data:"):
                    data_lines.append(line[5:].strip())

            if data_lines:
                try:
                    payload = json.loads("\n".join(data_lines))
                except json.JSONDecodeError:
                    payload = None
                if isinstance(payload, dict) and "data" in payload and "type" in payload:
                    return payload

        return {"data": chunk, "type": "content"}

    return {"data": chunk, "type": "content"}


def seeNow(event: str, content:ReturnQuery):
    if not content.traceId:
        content.traceId = get_trace_id()
    if not content.globalTraceId:
        content.globalTraceId = get_trace_id()
    return (
        f"event: {event}\n"
        f"data: {json.dumps(content.model_dump(), ensure_ascii=False)}\n\n"
    )


def format_output_result_sse(sessionId: str, chunk:Any):
    if not isinstance(chunk, dict) or "data" not in chunk or "type" not in chunk:
        return seeNow("error", ReturnQuery(conversationId=sessionId, isEnd=True))

    type = chunk.get("type")
    data = chunk.get("data")

    if "done" == type:
        return seeNow("done", ReturnQuery(conversationId=sessionId, metadata=data, isEnd=True))

    if isinstance(data, str):
        return seeNow("message", ReturnQuery(conversationId=sessionId, answer=data))

    if isinstance(data, dict):
        if "suggestions" in data:
            suggestions = data.get("suggestions")
            return seeNow("message", ReturnQuery(conversationId=sessionId, suggestions=suggestions))

        if "out_of_scope" in data:
            out_of_scope = data.get("out_of_scope")
            return seeNow("message", ReturnQuery(conversationId=sessionId, out_of_scope=out_of_scope))

        return seeNow("message", ReturnQuery(conversationId=sessionId, metadata=data))


def isIntents(chunk:Any):
    if not isinstance(chunk, dict):
        return False
    data = chunk.get("data")
    if isinstance(data, dict) and "intents" in data:
        return True

    return False


def isEchartsOutput(chunk:Any):
    if not isinstance(chunk, dict):
        return False
    data = chunk.get("data")
    if isinstance(data, str):
        return data.startswith("\n\n```echarts")

    return False


def get_user_id(req: QueryRequest) -> str:
    """获取用户ID，未传则归档到匿名用户档"""
    userInfo = req.userInfo or {}
    return userInfo.get("userId") or "anonymous"


def getX_ticket(userInfo:Optional[dict]) -> str:
    user_info = userInfo or {}
    extend_info = user_info.get("extend") if isinstance(user_info, dict) else None

    if isinstance(extend_info, dict):
        extend_dict = extend_info
    elif isinstance(extend_info, str):
        extend_dict = json.loads(extend_info)
    else:
        return ""
    x_ticket = extend_dict.get("ticket", "")
    return x_ticket

async def cleanup_expired_cache():
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        cleaned = await cache_cleanup_expired(CACHE_EXPIRY_HOURS)
        if cleaned:
            total, _, _ = await cache_get_all_stats()
            logger.info(f"[缓存清理] 清理 {cleaned} 条过期缓存，剩余 {total} 条")


def start_cache_cleanup() -> None:
    """启动后台缓存清理任务（应在应用启动时调用一次）"""
    global _cleanup_task
    if _cleanup_task is None or _cleanup_task.done():
        _cleanup_task = asyncio.create_task(cleanup_expired_cache())
        logger.info(
            f"[缓存清理] 启动后台清理任务，"
            f"清理间隔: {CLEANUP_INTERVAL_SECONDS}秒，过期时间: {CACHE_EXPIRY_HOURS}小时"
        )


async def stop_cache_cleanup() -> None:
    """停止后台缓存清理任务（应在应用关闭时调用）"""
    global _cleanup_task
    if _cleanup_task and not _cleanup_task.done():
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            pass
        logger.info("[缓存清理] 后台清理任务已停止")


# ---------- 路由端点 ----------
@router.post("")
async def query(req: QueryRequest):
    """
    智能查询接口（支持流式和非流式）

    每次请求生成唯一 query_id，用于标识本次查询的截断数据缓存。
    缓存按 user_id 进行隔离，同一用户下不同问题按 query_id 独立存取。

    Returns:
        stream=True:
            data: {"data": "文字片段", "type": "content"}
            data: {"data": {"truncated": true, "query_id": "...", ...}, "type": "data_truncated"}
            data: {"data": {"done": true, ...}, "type": "done"}
        stream=False:
            {"status": "success"|"need_more_info"|"error", "conversation_id": str, ...}
    """
    async with _semaphore:
        try:
            conversation_id = req.sessionId or str(uuid4())
            set_trace_id(conversation_id)
            user_id = get_user_id(req)
            query_id = str(uuid4())
            ticket = getX_ticket(req.userInfo)

            workflow = await workflow_factory.create_workflow(conversation_id=conversation_id)

            if req.stream:
                async def generate():
                    think_tag_checked = False
                    answer_tag_emitted = False
                    captured_intents = None

                    try:
                        async for raw_chunk in workflow.execute_stream(req.question, user_id, query_id):
                            chunk = normalize_stream_chunk(raw_chunk)
                            chunk_type = chunk.get("type", "")

                            if chunk_type == "Placeholder_True":
                                yield emit_echarts_placeholder(conversation_id)
                                continue

                            if chunk_type == "messageLabel":
                                data = convert_message_label(raw_chunk)
                                yield seeNow("message", ReturnQuery(conversationId=conversation_id, answer=data, isNotProcess=False))
                                continue

                            if captured_intents is None and isIntents(chunk):
                                captured_intents = chunk.get("data", {}).get("intents")
                                continue

                            if req.showThinkProcess and not think_tag_checked:
                                think_tag_checked = True
                                if chunk_type != "out_of_scope":
                                    answer_tag_emitted = True
                                    yield emit_answer_start(conversation_id)

                            yield format_output_result_sse(conversation_id, chunk)


                        trunc_info = workflow.get_any_truncation_info()
                        if trunc_info:
                            cache_entry = {
                                # 后续不再缓存全量数据。
                                # "full_data": trunc_info["full_data"],
                                "conversation_id": conversation_id,
                                "cached_at": datetime.now(),
                                "truncation_types": trunc_info["truncation_types"]
                            }
                            event_data = {
                                "truncated": True,
                                "query_id": query_id,
                                "conversation_id": conversation_id,
                                "user": user_id,
                                "truncation_types": trunc_info["truncation_types"]
                            }

                            if "total_count" in trunc_info:
                                cache_entry["total_count"] = trunc_info["total_count"]
                                cache_entry["display_count"] = trunc_info["display_count"]
                                event_data["total_count"] = trunc_info["total_count"]
                                event_data["display_count"] = trunc_info["display_count"]

                            if "max_time_field" in trunc_info:
                                cache_entry["max_time_field"] = trunc_info["max_time_field"]
                                event_data["max_time_field"] = trunc_info["max_time_field"]

                            # await cache_set(user_id, query_id, cache_entry)
                            # total_count, _, _ = await cache_get_all_stats()
                            # logger.info(
                            #     f"[缓存] 写入截断数据: user_id={user_id}, query_id={query_id}, "
                            #     f"类型={trunc_info['truncation_types']}, 当前缓存数={total_count}"
                            # )

                            yield format_output_result_sse(conversation_id,{"data": event_data, "type": "data_truncated"})

                        if answer_tag_emitted:
                            yield seeNow("message", ReturnQuery(conversationId=conversation_id, answer="</answer>"))

                        if captured_intents:
                            yield seeNow("message", ReturnQuery(conversationId=conversation_id,
                                                                answer="\n\n您可以点击下方链接进行相关数据查询。\n"))

                            fileList = get_file_list_by_intents(captured_intents)
                            fileListStr = json.dumps({"fileList":fileList}, ensure_ascii=False)
                            yield seeNow("message", ReturnQuery(conversationId=conversation_id, answer=fileListStr))

                        yield format_output_result_sse(conversation_id,{
                            "data": {
                                "done": True,
                                "conversation_id": conversation_id,
                                "query_id": query_id,
                                "user": user_id
                            },
                            "type": "done"
                        })

                    except Exception as e:
                        logger.error(f"query_stream_error={e}", exc_info=True)
                        yield format_output_result_sse(conversation_id,{
                            "data": {
                                "error": "抱歉，数据查询遇到一点小问题，暂时无法为您返回结果，请您稍后再试～",
                                "conversation_id": conversation_id,
                                "query_id": query_id,
                                "user": user_id
                            },
                            "type": "error"
                        })
                        yield format_output_result_sse(conversation_id,{
                            "data": {
                                "done": True,
                                "conversation_id": conversation_id,
                                "query_id": query_id,
                                "user": user_id
                            },
                            "type": "done"
                        })

                return StreamingResponse(generate(), media_type="text/event-stream")

            else:
                try:
                    result = await asyncio.wait_for(
                        workflow.execute(req.query),
                        timeout=REQUEST_TIMEOUT_SECONDS
                    )
                except asyncio.TimeoutError:
                    logger.error(f"query_timeout: user={user_id}, query={req.query[:100]}")
                    return {
                        "status": "error",
                        "message": "查询超时，请稍后重试",
                        "conversation_id": conversation_id,
                        "query_id": query_id,
                        "user": user_id
                    }

                if result.get("status") == "need_more_info":
                    return {
                        "status": "need_more_info",
                        "data": result,
                        "conversation_id": conversation_id,
                        "query_id": query_id,
                        "user": user_id,
                        "message": result.get("message", {}).get("content", "需要更多信息")
                    }
                elif result.get("status") == "error":
                    return {
                        "status": "error",
                        "message": result.get("error") or result.get("message", {}).get("content", "查询失败"),
                        "conversation_id": conversation_id,
                        "query_id": query_id,
                        "user": user_id
                    }
                else:
                    trunc_info = workflow.get_any_truncation_info()
                    if trunc_info:
                        cache_entry = {
                            # 后续不再缓存全量数据。
                            # "full_data": trunc_info["full_data"],
                            "conversation_id": conversation_id,
                            "cached_at": datetime.now(),
                            "truncation_types": trunc_info["truncation_types"]
                        }

                        if "total_count" in trunc_info:
                            cache_entry["total_count"] = trunc_info["total_count"]
                            cache_entry["display_count"] = trunc_info["display_count"]
                            result["total_count"] = trunc_info["total_count"]
                            result["display_count"] = trunc_info["display_count"]

                        if "max_time_field" in trunc_info:
                            cache_entry["max_time_field"] = trunc_info["max_time_field"]

                        # await cache_set(user_id, query_id, cache_entry)

                        result["truncated"] = True
                        result["query_id"] = query_id
                        result["truncation_types"] = trunc_info["truncation_types"]

                    return {
                        "status": "success",
                        "data": result,
                        "conversation_id": conversation_id,
                        "query_id": query_id,
                        "user": user_id
                    }

        except Exception as e:
            logger.error(f"query_error={e}", exc_info=True)
            friendly = "抱歉，数据查询遇到一点小问题，暂时无法为您返回结果，请您稍后再试～"
            if req.stream:
                async def error_gen():
                    yield format_output_result_sse(conversation_id,build_stream_payload(
                        {
                            "error": friendly,
                            "conversation_id": conversation_id,
                            "query_id": query_id,
                            "user": user_id
                        },
                        "error"
                    ))
                    yield format_output_result_sse(conversation_id,build_stream_payload(
                        {"done": True, "conversation_id": conversation_id, "query_id": query_id, "user": user_id},
                        "done"
                    ))
                return StreamingResponse(error_gen(), media_type="text/event-stream")
            else:
                return {
                    "status": "error",
                    "message": friendly,
                    "conversation_id": conversation_id,
                    "query_id": query_id,
                    "user": user_id
                }


@router.get("/full-data/{query_id}")
async def get_full_data(query_id: str, user: str):
    """
    按 user + query_id 获取被截断的完整数据

    无论发生条数截断还是字段截断，返回的 data 都是原始完整数据
    （包含所有字段和所有记录）。

    Args:
        query_id: 查询唯一ID（由流式/非流式响应中返回）
        user: 用户ID（前端传入）

    Returns:
        成功: {"status": "success", "data": [...], "total_count": int,
               "display_count": int, "conversation_id": str}
        不存在: {"status": "error", "message": "查询ID不存在或已过期"}
    """
    try:
        entry = await cache_get(user, query_id)
        if not entry:
            return {
                "status": "error",
                "message": "查询ID不存在或已过期，请重新查询"
            }
        if "full_data" not in entry:
            return {
                "status": "error",
                "message": "当前版本未缓存全量数据"
            }
        return {
            "status": "success",
            "data": entry["full_data"],
            "total_count": entry.get("total_count"),
            "display_count": entry.get("display_count"),
            "conversation_id": entry["conversation_id"],
            "user": user
        }
    except Exception as e:
        logger.error(f"get_full_data_error={e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@router.delete("/cache/{query_id}")
async def clear_cache(query_id: str, user: str):
    """按 user + query_id 清理单条查询缓存"""
    try:
        deleted = await cache_delete(user, query_id)
        if deleted:
            total_count, _, _ = await cache_get_all_stats()
            logger.info(
                f"[缓存清理] 手动清理查询缓存: user_id={user}, query_id={query_id}, 当前缓存数: {total_count}"
            )
        return {"status": "success", "message": "缓存已清理" if deleted else "缓存不存在"}
    except Exception as e:
        logger.error(f"clear_cache_error={e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@router.delete("/cache/user/{user}")
async def clear_user_cache(user: str):
    """按用户ID清理该用户下所有查询缓存"""
    try:
        removed = await cache_delete_user(user)
        logger.info(f"[缓存清理] 按用户清理缓存: user_id={user}, 清理条数: {removed}")
        return {"status": "success", "message": "用户缓存已清理", "user": user, "removed": removed}
    except Exception as e:
        logger.error(f"clear_user_cache_error={e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@router.get("/cache/stats")
async def get_cache_stats():
    """
    获取当前所有查询缓存的统计信息（按用户分组）

    Returns:
        {
            "status": "success",
            "total_cache_count": int,
            "total_user_count": int,
            "cache_entries": [...]
        }
    """
    try:
        total_count, user_count, entries = await cache_get_all_stats()
        return {
            "status": "success",
            "total_cache_count": total_count,
            "total_user_count": user_count,
            "cache_expiry_hours": CACHE_EXPIRY_HOURS,
            "cleanup_interval_seconds": CLEANUP_INTERVAL_SECONDS,
            "cache_entries": entries
        }
    except Exception as e:
        logger.error(f"get_cache_stats_error={e}", exc_info=True)
        return {"status": "error", "message": str(e)}
