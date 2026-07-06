"""
电力规则审查系统 - 双模式缓存层（Phase 3）

Redis 优先 → 内存降级，复用现有 shared_cache.py 的 async Lock 模式。

支持：
- 检索结果缓存（query → chunks）
- LLM 审查结果缓存（query_hash → LLMOutput）
- RRF 精排结果缓存
- TTL 过期 + LRU 淘汰
- 缓存统计与命中率监控
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)

# 默认 TTL（秒）
DEFAULT_TTL = 300  # 5 分钟
DEFAULT_RETRIEVAL_TTL = 600  # 10 分钟（检索结果变化慢）
MAX_MEMORY_ENTRIES = 1000  # 内存模式最大条目数


# ---------------------------------------------------------------------------
# Redis 客户端（可选）
# ---------------------------------------------------------------------------


class RedisClient:
    """Redis 客户端封装，连接失败时静默降级为 None。

    不强制 redis-py 依赖：只有 import 成功且 Redis 可用时才启用。
    """

    def __init__(self):
        self._redis = None
        self._available = False
        self._init_redis()

    def _init_redis(self):
        """尝试连接 Redis，失败则标记不可用。"""
        try:
            import redis as redis_lib

            redis_url = settings.REDIS_URL
            if not redis_url:
                logger.info("[Cache] REDIS_URL 未配置，使用内存缓存")
                return

            self._redis = redis_lib.from_url(
                redis_url,
                socket_connect_timeout=2,
                socket_timeout=2,
                decode_responses=True,
            )
            self._redis.ping()
            self._available = True
            logger.info("[Cache] Redis 连接成功: %s", redis_url)
        except Exception as e:
            logger.info("[Cache] Redis 不可用，降级为内存缓存: %s", e)
            self._redis = None
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available and self._redis is not None

    async def get(self, key: str) -> str | None:
        if not self.is_available:
            return None
        try:
            return await asyncio.to_thread(self._redis.get, key)
        except Exception:
            return None

    async def set(self, key: str, value: str, ttl: int = DEFAULT_TTL) -> bool:
        if not self.is_available:
            return False
        try:
            await asyncio.to_thread(self._redis.setex, key, ttl, value)
            return True
        except Exception:
            return False

    async def delete(self, key: str) -> bool:
        if not self.is_available:
            return False
        try:
            await asyncio.to_thread(self._redis.delete, key)
            return True
        except Exception:
            return False

    async def keys(self, pattern: str) -> list[str]:
        if not self.is_available:
            return []
        try:
            return await asyncio.to_thread(self._redis.keys, pattern)
        except Exception:
            return []

    async def flush_pattern(self, pattern: str) -> int:
        """按模式批量删除。"""
        ks = await self.keys(pattern)
        if not ks:
            return 0
        try:
            return await asyncio.to_thread(self._redis.delete, *ks)
        except Exception:
            return 0

    async def info(self, section: str = "stats") -> dict:
        if not self.is_available:
            return {}
        try:
            return await asyncio.to_thread(self._redis.info, section)
        except Exception:
            return {}


# ---------------------------------------------------------------------------
# 内存 LRU 缓存
# ---------------------------------------------------------------------------


@dataclass
class _MemoryEntry:
    value: Any
    expires_at: float  # Unix timestamp


class MemoryLRUCache:
    """带 TTL + LRU 淘汰的内存缓存。"""

    def __init__(self, max_entries: int = MAX_MEMORY_ENTRIES):
        self._store: OrderedDict[str, _MemoryEntry] = OrderedDict()
        self._max = max_entries
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if time.time() > entry.expires_at:
                del self._store[key]
                return None
            # LRU: 移到末尾
            self._store.move_to_end(key)
            return entry.value

    async def set(self, key: str, value: Any, ttl: int = DEFAULT_TTL) -> None:
        async with self._lock:
            # 淘汰过期条目
            now = time.time()
            expired = [k for k, e in self._store.items() if now > e.expires_at]
            for k in expired:
                del self._store[k]

            # LRU 淘汰：超过上限时删除最旧条目
            while len(self._store) >= self._max:
                self._store.popitem(last=False)

            self._store[key] = _MemoryEntry(
                value=value,
                expires_at=now + ttl,
            )
            self._store.move_to_end(key)

    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    async def flush_pattern(self, prefix: str) -> int:
        """按前缀删除。"""
        async with self._lock:
            to_delete = [k for k in self._store if k.startswith(prefix)]
            for k in to_delete:
                del self._store[k]
            return len(to_delete)

    async def size(self) -> int:
        async with self._lock:
            return len(self._store)

    async def stats(self) -> dict:
        async with self._lock:
            now = time.time()
            expired = sum(1 for e in self._store.values() if now > e.expires_at)
            return {
                "total_entries": len(self._store),
                "expired_entries": expired,
                "max_entries": self._max,
                "mode": "memory",
            }


# ---------------------------------------------------------------------------
# 双模式缓存（核心）
# ---------------------------------------------------------------------------


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    redis_hits: int = 0
    memory_hits: int = 0
    sets: int = 0
    deletes: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / total, 4) if total > 0 else 0.0


class DualModeCache:
    """双模式缓存：Redis 优先 → 内存降级。

    用法（模块级单例）:
        cache = DualModeCache()
        await cache.get("key")
        await cache.set("key", value, ttl=300)
    """

    def __init__(self, namespace: str = "rule_review"):
        self._redis = RedisClient()
        self._memory = MemoryLRUCache()
        self._namespace = namespace
        self._stats = CacheStats()

    # ------------------------------------------------------------------
    # Key 管理
    # ------------------------------------------------------------------

    def _ns_key(self, key: str) -> str:
        return f"{self._namespace}:{key}"

    @staticmethod
    def _make_query_hash(query: str) -> str:
        """对 query 做确定性 hash，用于缓存键。"""
        return hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    async def get(self, key: str) -> Any | None:
        """获取缓存值，优先 Redis → 内存。"""
        ns_key = self._ns_key(key)

        # 尝试 Redis
        raw = await self._redis.get(ns_key)
        if raw is not None:
            self._stats.hits += 1
            self._stats.redis_hits += 1
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return raw

        # 降级到内存
        value = await self._memory.get(ns_key)
        if value is not None:
            self._stats.hits += 1
            self._stats.memory_hits += 1
            return value

        self._stats.misses += 1
        return None

    async def set(self, key: str, value: Any, ttl: int = DEFAULT_TTL) -> None:
        """写入缓存，Redis + 内存双写。"""
        ns_key = self._ns_key(key)
        self._stats.sets += 1

        # 序列化
        try:
            serialized = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            serialized = json.dumps({"raw": str(value)}, ensure_ascii=False)

        # Redis 优先
        redis_ok = await self._redis.set(ns_key, serialized, ttl=ttl)

        # 内存双写（即使 Redis 成功也写，作为热备）
        await self._memory.set(ns_key, value, ttl=ttl)

        if not redis_ok and self._redis.is_available:
            logger.warning("[Cache] Redis 写入失败，仅使用内存缓存")

    async def delete(self, key: str) -> bool:
        """删除缓存。"""
        ns_key = self._ns_key(key)
        self._stats.deletes += 1
        r_ok = await self._redis.delete(ns_key)
        m_ok = await self._memory.delete(ns_key)
        return r_ok or m_ok

    async def get_or_set(
        self,
        key: str,
        factory,
        ttl: int = DEFAULT_TTL,
    ) -> Any:
        """获取缓存，未命中时调用 factory() 生成并缓存。"""
        value = await self.get(key)
        if value is not None:
            return value

        if asyncio.iscoroutinefunction(factory):
            value = await factory()
        else:
            value = factory()

        if value is not None:
            await self.set(key, value, ttl=ttl)
        return value

    async def flush_namespace(self) -> int:
        """清空当前命名空间的所有缓存。"""
        prefix = f"{self._namespace}:"
        r_count = await self._redis.flush_pattern(f"{prefix}*")
        m_count = await self._memory.flush_pattern(prefix)
        return r_count + m_count

    # ------------------------------------------------------------------
    # 便捷方法：按业务场景封装
    # ------------------------------------------------------------------

    async def get_retrieval(self, query: str) -> Any | None:
        """获取检索结果缓存。"""
        key = f"retrieval:{self._make_query_hash(query)}"
        return await self.get(key)

    async def set_retrieval(self, query: str, chunks: Any, ttl: int = DEFAULT_RETRIEVAL_TTL) -> None:
        """缓存检索结果（变化慢，TTL 较长）。"""
        key = f"retrieval:{self._make_query_hash(query)}"
        await self.set(key, chunks, ttl=ttl)

    async def get_llm_result(self, query: str, doc_ids: list[str]) -> Any | None:
        """获取 LLM 审查结果缓存（按 query + doc_ids 联合 hash）。"""
        fingerprint = hashlib.sha256(
            f"{query}|{','.join(sorted(doc_ids))}".encode()
        ).hexdigest()[:16]
        key = f"llm:{fingerprint}"
        return await self.get(key)

    async def set_llm_result(
        self, query: str, doc_ids: list[str], result: Any, ttl: int = DEFAULT_TTL
    ) -> None:
        """缓存 LLM 审查结果。"""
        fingerprint = hashlib.sha256(
            f"{query}|{','.join(sorted(doc_ids))}".encode()
        ).hexdigest()[:16]
        key = f"llm:{fingerprint}"
        await self.set(key, result, ttl=ttl)

    async def invalidate_document(self, doc_id: str) -> int:
        """当文档更新/删除时，失效所有相关缓存。"""
        # 检索缓存和 LLM 缓存都可能依赖此文档，全量刷掉最安全
        return await self.flush_namespace()

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------

    async def stats(self) -> dict:
        """获取缓存统计信息。"""
        redis_info = await self._redis.info()
        mem_stats = await self._memory.stats()

        return {
            "mode": "redis+memory" if self._redis.is_available else "memory",
            "hits": self._stats.hits,
            "misses": self._stats.misses,
            "hit_rate": self._stats.hit_rate,
            "redis_hits": self._stats.redis_hits,
            "memory_hits": self._stats.memory_hits,
            "sets": self._stats.sets,
            "deletes": self._stats.deletes,
            "memory_entries": mem_stats["total_entries"],
            "redis_info": {
                "connected_clients": redis_info.get("connected_clients", 0),
                "used_memory_human": redis_info.get("used_memory_human", "N/A"),
                "uptime_in_seconds": redis_info.get("uptime_in_seconds", 0),
            } if redis_info else None,
        }

    def reset_stats(self) -> None:
        self._stats = CacheStats()


# ---------------------------------------------------------------------------
# 默认工厂
# ---------------------------------------------------------------------------


_default_cache: DualModeCache | None = None


def get_default_cache() -> DualModeCache:
    """获取默认缓存单例。"""
    global _default_cache
    if _default_cache is None:
        _default_cache = DualModeCache()
    return _default_cache
