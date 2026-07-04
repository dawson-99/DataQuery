import asyncio
import uuid
import time
from typing import Any, Dict, Optional, List

from src.config import settings
from src.utils.logging_setup import logger
from langgraph.checkpoint.memory import InMemorySaver

class SessionContext:
    """会话上下文数据类，职责：存储和管理会话相关数据"""
    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.checkpointer = InMemorySaver()  # 每个会话独立的记忆存储
        self.created_at = time.time()
        self.last_access = time.time()
        logger.info(f"[SessionContext] 创建会话上下文: conversation_id={conversation_id}")


class SessionRegistry:
    """会话注册表，职责：全局会话生命周期管理"""

    _CLEANUP_INTERVAL_SECONDS = 300  # 每5分钟触发一次清理检查

    def __init__(self):
        self._sessions: Dict[str, SessionContext] = {}
        self._sessions_lock = asyncio.Lock()
        self._last_access: Dict[str, float] = {}
        self._max_sessions = getattr(settings, 'MAX_SESSIONS', 1000)
        self._session_timeout = getattr(settings, 'SESSION_TIMEOUT', 3600)
        self._last_cleanup_time = time.monotonic()

    def _should_cleanup(self) -> bool:
        """基于时间间隔判断是否需要清理，避免每次请求都检查"""
        return time.monotonic() - self._last_cleanup_time > self._CLEANUP_INTERVAL_SECONDS

    def _cleanup_old_sessions(self):
        """清理过期的会话（需要在锁内调用）"""
        now = time.time()
        expired = [
            cid for cid, last_time in self._last_access.items()
            if now - last_time > self._session_timeout
        ]
        for cid in expired:
            if cid in self._sessions:
                self._sessions.pop(cid, None)
                logger.info(f"[SessionRegistry] 清理过期会话: conversation_id={cid}")
            self._last_access.pop(cid, None)

        # 如果仍然超过最大数量，删除最旧的
        if len(self._sessions) > self._max_sessions:
            sorted_sessions = sorted(self._last_access.items(), key=lambda x: x[1])
            to_remove = len(self._sessions) - self._max_sessions
            for cid, _ in sorted_sessions[:to_remove]:
                if cid in self._sessions:
                    self._sessions.pop(cid, None)
                    logger.info(f"[SessionRegistry] 清理超限会话: conversation_id={cid}")
                self._last_access.pop(cid, None)

    async def get_or_create_session(self, conversation_id: Optional[str]) -> str:
        """获取或创建会话ID"""
        async with self._sessions_lock:
            if self._should_cleanup():
                self._cleanup_old_sessions()
                self._last_cleanup_time = time.monotonic()

            cid = conversation_id or str(uuid.uuid4())
            self._last_access[cid] = time.time()
            return cid

    async def get_session_context(self, conversation_id: str) -> Optional[SessionContext]:
        """获取会话上下文"""
        async with self._sessions_lock:
            if conversation_id in self._sessions:
                self._last_access[conversation_id] = time.time()
                return self._sessions[conversation_id]

        return None

    async def get_or_create_context(self, conversation_id: str) -> SessionContext:
        """获取或创建会话上下文（包含 checkpointer）"""
        async with self._sessions_lock:
            if conversation_id in self._sessions:
                self._last_access[conversation_id] = time.time()
                logger.debug(f"[SessionRegistry] 复用会话上下文: conversation_id={conversation_id}")
                return self._sessions[conversation_id]

            ctx = SessionContext(conversation_id=conversation_id)
            self._sessions[conversation_id] = ctx
            self._last_access[conversation_id] = time.time()
            logger.info(f"[SessionRegistry] 创建新会话上下文: conversation_id={conversation_id}, total_sessions={len(self._sessions)}")
            return ctx

session_registry = SessionRegistry()