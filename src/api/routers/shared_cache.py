# src/api/routers/shared_cache.py
"""
共享缓存和SSE工具函数（并发安全版）
提供异步锁保护的查询结果缓存，用于流式/非流式截断数据存储
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, Optional, List, Tuple


# ---------- 全局缓存与并发控制 ----------
_query_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}  # user_id -> {query_id -> entry}
_cache_lock = asyncio.Lock()  # 保护 _query_cache 的异步锁


def sse(payload: dict) -> str:
    """生成统一SSE数据格式：data: {"data": ..., "type": ...}"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ---------- 线程/协程安全的缓存操作 ----------
async def cache_get(user: str, query_id: str) -> Optional[Dict[str, Any]]:
    """安全读取缓存条目"""
    async with _cache_lock:
        return _query_cache.get(user, {}).get(query_id)


async def cache_set(user: str, query_id: str, entry: Dict[str, Any]) -> None:
    """安全写入缓存条目"""
    async with _cache_lock:
        if user not in _query_cache:
            _query_cache[user] = {}
        _query_cache[user][query_id] = entry


async def cache_delete(user: str, query_id: str) -> bool:
    """安全删除单个缓存条目，返回是否成功删除"""
    async with _cache_lock:
        user_entries = _query_cache.get(user)
        if user_entries and query_id in user_entries:
            del user_entries[query_id]
            if not user_entries:
                _query_cache.pop(user, None)
            return True
    return False


async def cache_delete_user(user: str) -> int:
    """安全删除用户所有缓存，返回删除条目数"""
    async with _cache_lock:
        removed = len(_query_cache.get(user, {}))
        _query_cache.pop(user, None)
        return removed


async def cache_get_all_stats() -> Tuple[int, int, List[Dict[str, Any]]]:
    """
    获取缓存统计信息（线程安全）
    返回: (总条目数, 用户数, 条目详情列表)
    """
    async with _cache_lock:
        now = datetime.now()
        entries = []
        total_cache_count = 0
        for user_id, user_entries in _query_cache.items():
            for qid, entry in user_entries.items():
                total_cache_count += 1
                age = (now - entry["cached_at"]).total_seconds()
                entries.append({
                    "user": user_id,
                    "query_id": qid,
                    "conversation_id": entry["conversation_id"],
                    "total_count": entry.get("total_count"),
                    "display_count": entry.get("display_count"),
                    "truncation_types": entry.get("truncation_types", []),
                    "max_time_field": entry.get("max_time_field"),
                    "cached_at": entry["cached_at"].isoformat(),
                    "age_seconds": int(age),
                })
        entries.sort(key=lambda x: x["age_seconds"])
        return total_cache_count, len(_query_cache), entries


async def cache_cleanup_expired(expiry_hours: int) -> int:
    """
    清理过期的缓存条目（由后台任务调用）
    返回清理的条目数量
    """
    now = datetime.now()
    to_delete: List[Tuple[str, str]] = []

    # 收集过期条目（短时间持有锁）
    async with _cache_lock:
        for user_id, user_entries in _query_cache.items():
            for qid, entry in user_entries.items():
                if (now - entry["cached_at"]).total_seconds() > expiry_hours * 3600:
                    to_delete.append((user_id, qid))

    cleaned = 0
    for user_id, qid in to_delete:
        if await cache_delete(user_id, qid):
            cleaned += 1

    return cleaned
