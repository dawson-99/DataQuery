"""
电力规则审查系统 - PostgreSQL 数据持久化（Phase 4）

替换 JSON 文件存储，提供：
- 文档元数据表（documents）
- Chunk 表（chunks）
- 审计记录表（audit_records）
- 溯源信息表（source_traces）
- 连接池管理 + 自动建表

PostgreSQL 部署：
  docker run -d --name postgres \\
    -e POSTGRES_USER=dataquery -e POSTGRES_PASSWORD=dataquery \\
    -e POSTGRES_DB=rule_review -p 5432:5432 \\
    postgres:16-alpine

配置（.env）：
  PG_HOST=localhost
  PG_PORT=5432
  PG_USER=dataquery
  PG_PASSWORD=dataquery
  PG_DATABASE=rule_review
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 连接池管理
# ---------------------------------------------------------------------------


class PostgresPool:
    """PostgreSQL 连接池管理器。

    使用 psycopg2 内置连接池（ThreadedConnectionPool）。
    """

    def __init__(self, min_conn: int = 2, max_conn: int = 10):
        self._pool = None
        self._available = False
        self._min_conn = min_conn
        self._max_conn = max_conn
        self._init_pool()

    def _init_pool(self):
        try:
            from psycopg2 import pool as pg_pool

            dsn = (
                f"host={settings.PG_HOST or 'localhost'} "
                f"port={settings.PG_PORT or 5432} "
                f"user={settings.PG_USER or 'dataquery'} "
                f"password={settings.PG_PASSWORD or 'dataquery'} "
                f"dbname={settings.PG_DATABASE or 'rule_review'}"
            )

            self._pool = pg_pool.ThreadedConnectionPool(
                self._min_conn, self._max_conn, dsn
            )
            self._available = True
            logger.info("[PG] 连接池初始化成功 (min=%d, max=%d)", self._min_conn, self._max_conn)
        except Exception as e:
            logger.warning("[PG] 连接失败: %s", e)
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available and self._pool is not None

    @contextmanager
    def connection(self):
        """获取一个连接（上下文管理器，自动归还）。"""
        if not self.is_available:
            raise RuntimeError("PostgreSQL 不可用")
        conn = self._pool.getconn()
        try:
            yield conn
        finally:
            self._pool.putconn(conn)

    def close(self):
        if self._pool:
            self._pool.closeall()
            self._available = False


# ---------------------------------------------------------------------------
# 数据库 Schema
# ---------------------------------------------------------------------------

CREATE_TABLES_SQL = """
-- 文档元数据表
CREATE TABLE IF NOT EXISTS documents (
    doc_id          VARCHAR(64) PRIMARY KEY,
    file_name       VARCHAR(512) NOT NULL,
    page_count      INTEGER DEFAULT 0,
    chunk_count     INTEGER DEFAULT 0,
    file_size_bytes BIGINT DEFAULT 0,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Chunk 表
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    VARCHAR(64) PRIMARY KEY,
    doc_id      VARCHAR(64) NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    text        TEXT NOT NULL,
    source      VARCHAR(512) DEFAULT '',
    section     VARCHAR(512) DEFAULT '',
    page        INTEGER DEFAULT 0,
    chunk_index INTEGER DEFAULT 0,
    tables_json TEXT DEFAULT '[]',
    created_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);

-- 审计记录表
CREATE TABLE IF NOT EXISTS audit_records (
    query_id            VARCHAR(64) PRIMARY KEY,
    timestamp           TIMESTAMP WITH TIME ZONE NOT NULL,
    original_query      TEXT DEFAULT '',
    rewritten_query     TEXT DEFAULT '',
    decision            VARCHAR(32) DEFAULT '',
    confidence          DOUBLE PRECISION DEFAULT 0.0,
    not_found           BOOLEAN DEFAULT FALSE,
    bm25_k              INTEGER DEFAULT 0,
    vector_k            INTEGER DEFAULT 0,
    final_k             INTEGER DEFAULT 0,
    search_expanded     BOOLEAN DEFAULT FALSE,
    retrieval_latency_ms DOUBLE PRECISION DEFAULT 0.0,
    llm_model           VARCHAR(64) DEFAULT '',
    llm_latency_ms      DOUBLE PRECISION DEFAULT 0.0,
    judge_verified      BOOLEAN DEFAULT FALSE,
    judge_skipped       BOOLEAN DEFAULT FALSE,
    judge_hallucinated_count INTEGER DEFAULT 0,
    judge_latency_ms    DOUBLE PRECISION DEFAULT 0.0,
    final_result_json   JSONB DEFAULT '{}',
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_records(timestamp);

-- 溯源信息表
CREATE TABLE IF NOT EXISTS source_traces (
    id              SERIAL PRIMARY KEY,
    query_id        VARCHAR(64) NOT NULL REFERENCES audit_records(query_id) ON DELETE CASCADE,
    result_field    VARCHAR(128) DEFAULT '',
    result_excerpt  TEXT DEFAULT '',
    source_doc      VARCHAR(512) DEFAULT '',
    source_section  VARCHAR(512) DEFAULT '',
    source_page     INTEGER DEFAULT 0,
    source_text     TEXT DEFAULT '',
    source_chunk_id VARCHAR(64) DEFAULT '',
    match_type      VARCHAR(32) DEFAULT 'llm_extracted'
);
CREATE INDEX IF NOT EXISTS idx_traces_query_id ON source_traces(query_id);
"""


# ---------------------------------------------------------------------------
# 数据仓库
# ---------------------------------------------------------------------------


class PostgresStore:
    """PostgreSQL 数据仓库。

    管理 documents、chunks、audit_records、source_traces 四张表。
    所有写操作自动 commit，读操作只读。
    """

    def __init__(self, pool: PostgresPool | None = None):
        self._pool = pool or PostgresPool()
        self._ensure_tables()

    def _ensure_tables(self):
        """首次连接时自动建表。"""
        if not self._pool.is_available:
            return
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(CREATE_TABLES_SQL)
                conn.commit()
            logger.info("[PG] 表结构初始化完成")
        except Exception as e:
            logger.warning("[PG] 表结构初始化失败: %s", e)

    @property
    def is_available(self) -> bool:
        return self._pool.is_available

    # ------------------------------------------------------------------
    # 文档操作
    # ------------------------------------------------------------------

    def save_document(
        self,
        doc_id: str,
        file_name: str,
        page_count: int = 0,
        chunk_count: int = 0,
        file_size_bytes: int = 0,
    ) -> bool:
        if not self.is_available:
            return False
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """INSERT INTO documents (doc_id, file_name, page_count, chunk_count, file_size_bytes)
                           VALUES (%s, %s, %s, %s, %s)
                           ON CONFLICT (doc_id) DO UPDATE SET
                               file_name = EXCLUDED.file_name,
                               page_count = EXCLUDED.page_count,
                               chunk_count = EXCLUDED.chunk_count,
                               updated_at = NOW()""",
                        (doc_id, file_name, page_count, chunk_count, file_size_bytes),
                    )
                conn.commit()
            return True
        except Exception as e:
            logger.error("[PG] 保存文档失败: %s", e)
            return False

    def delete_document(self, doc_id: str) -> bool:
        if not self.is_available:
            return False
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
                conn.commit()
            return True
        except Exception as e:
            logger.error("[PG] 删除文档失败: %s", e)
            return False

    def list_documents(self) -> list[dict]:
        if not self.is_available:
            return []
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT doc_id, file_name, page_count, chunk_count, created_at "
                        "FROM documents ORDER BY created_at DESC"
                    )
                    rows = cur.fetchall()
            return [
                {
                    "doc_id": r[0],
                    "file_name": r[1],
                    "page_count": r[2],
                    "chunk_count": r[3],
                    "created_at": r[4].isoformat() if r[4] else "",
                }
                for r in rows
            ]
        except Exception as e:
            logger.error("[PG] 列出文档失败: %s", e)
            return []

    def get_document(self, doc_id: str) -> dict | None:
        if not self.is_available:
            return None
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT doc_id, file_name, page_count, chunk_count, created_at "
                        "FROM documents WHERE doc_id = %s",
                        (doc_id,),
                    )
                    row = cur.fetchone()
            if row is None:
                return None
            return {
                "doc_id": row[0],
                "file_name": row[1],
                "page_count": row[2],
                "chunk_count": row[3],
                "created_at": row[4].isoformat() if row[4] else "",
            }
        except Exception as e:
            logger.error("[PG] 获取文档失败: %s", e)
            return None

    # ------------------------------------------------------------------
    # Chunk 操作
    # ------------------------------------------------------------------

    def save_chunks(self, chunks: list[dict]) -> int:
        """批量插入 chunk 记录。"""
        if not self.is_available or not chunks:
            return 0
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    for c in chunks:
                        cur.execute(
                            """INSERT INTO chunks (chunk_id, doc_id, text, source, section, page, chunk_index, tables_json)
                               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                               ON CONFLICT (chunk_id) DO UPDATE SET
                                   text = EXCLUDED.text,
                                   source = EXCLUDED.source,
                                   section = EXCLUDED.section,
                                   page = EXCLUDED.page""",
                            (
                                c.get("chunk_id", ""),
                                c.get("doc_id", ""),
                                c.get("text", ""),
                                c.get("source", ""),
                                c.get("section", ""),
                                c.get("page", 0),
                                c.get("chunk_index", 0),
                                json.dumps(c.get("tables", []), ensure_ascii=False),
                            ),
                        )
                conn.commit()
            return len(chunks)
        except Exception as e:
            logger.error("[PG] 保存 chunk 失败: %s", e)
            return 0

    def get_chunk(self, chunk_id: str) -> dict | None:
        if not self.is_available:
            return None
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT chunk_id, doc_id, text, source, section, page, chunk_index, tables_json "
                        "FROM chunks WHERE chunk_id = %s",
                        (chunk_id,),
                    )
                    row = cur.fetchone()
            if row is None:
                return None
            tables = []
            try:
                tables = json.loads(row[7] or "[]")
            except json.JSONDecodeError:
                pass
            return {
                "chunk_id": row[0],
                "doc_id": row[1],
                "text": row[2],
                "source": row[3],
                "section": row[4],
                "page": row[5],
                "chunk_index": row[6],
                "tables": tables,
            }
        except Exception as e:
            logger.error("[PG] 获取 chunk 失败: %s", e)
            return None

    def get_chunks_by_doc_id(self, doc_id: str) -> list[dict]:
        if not self.is_available:
            return []
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT chunk_id, doc_id, text, source, section, page "
                        "FROM chunks WHERE doc_id = %s ORDER BY chunk_index",
                        (doc_id,),
                    )
                    rows = cur.fetchall()
            return [
                {
                    "chunk_id": r[0],
                    "doc_id": r[1],
                    "text": r[2],
                    "source": r[3],
                    "section": r[4],
                    "page": r[5],
                }
                for r in rows
            ]
        except Exception as e:
            logger.error("[PG] 获取chunks失败: %s", e)
            return []

    def delete_chunks_by_doc_id(self, doc_id: str) -> int:
        if not self.is_available:
            return 0
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
                    count = cur.rowcount
                conn.commit()
            return count
        except Exception as e:
            logger.error("[PG] 删除chunks失败: %s", e)
            return 0

    # ------------------------------------------------------------------
    # 审计操作
    # ------------------------------------------------------------------

    def save_audit_record(self, record: dict) -> bool:
        if not self.is_available:
            return False
        try:
            retrieval = record.get("retrieval", {})
            llm_gen = record.get("llm_generation", {})
            judge = record.get("judge_verification") or {}
            final = record.get("final_result", {})

            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """INSERT INTO audit_records (
                               query_id, timestamp, original_query, rewritten_query,
                               decision, confidence, not_found,
                               bm25_k, vector_k, final_k, search_expanded, retrieval_latency_ms,
                               llm_model, llm_latency_ms,
                               judge_verified, judge_skipped, judge_hallucinated_count, judge_latency_ms,
                               final_result_json
                           ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                           ON CONFLICT (query_id) DO UPDATE SET
                               decision = EXCLUDED.decision,
                               confidence = EXCLUDED.confidence,
                               judge_verified = EXCLUDED.judge_verified,
                               judge_skipped = EXCLUDED.judge_skipped""",
                        (
                            record.get("query_id", ""),
                            record.get("timestamp", datetime.now().isoformat()),
                            record.get("original_query", ""),
                            record.get("rewritten_query", ""),
                            final.get("decision", ""),
                            final.get("confidence", 0.0),
                            final.get("not_found", False),
                            retrieval.get("bm25_k", 0),
                            retrieval.get("vector_k", 0),
                            retrieval.get("final_k", 0),
                            retrieval.get("search_expanded", False),
                            retrieval.get("retrieval_latency_ms", 0.0),
                            llm_gen.get("model", ""),
                            llm_gen.get("latency_ms", 0.0),
                            judge.get("verified", False),
                            judge.get("skipped", False),
                            judge.get("hallucinated_count", 0),
                            judge.get("latency_ms", 0.0),
                            json.dumps(final, ensure_ascii=False),
                        ),
                    )

                    # 保存溯源信息
                    traces = record.get("source_traceability", [])
                    if traces:
                        for t in traces:
                            cur.execute(
                                """INSERT INTO source_traces (
                                       query_id, result_field, result_excerpt,
                                       source_doc, source_section, source_page,
                                       source_text, source_chunk_id, match_type
                                   ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                                (
                                    record.get("query_id", ""),
                                    t.get("result_field", ""),
                                    t.get("result_excerpt", ""),
                                    t.get("source_doc", ""),
                                    t.get("source_section", ""),
                                    t.get("source_page", 0),
                                    t.get("source_text", ""),
                                    t.get("source_chunk_id", ""),
                                    t.get("match_type", "llm_extracted"),
                                ),
                            )

                conn.commit()
            return True
        except Exception as e:
            logger.error("[PG] 保存审计记录失败: %s", e)
            return False

    def get_audit_record(self, query_id: str) -> dict | None:
        if not self.is_available:
            return None
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT * FROM audit_records WHERE query_id = %s",
                        (query_id,),
                    )
                    row = cur.fetchone()
            if row is None:
                return None

            cols = [
                "query_id", "timestamp", "original_query", "rewritten_query",
                "decision", "confidence", "not_found",
                "bm25_k", "vector_k", "final_k", "search_expanded", "retrieval_latency_ms",
                "llm_model", "llm_latency_ms",
                "judge_verified", "judge_skipped", "judge_hallucinated_count", "judge_latency_ms",
                "final_result_json", "created_at",
            ]

            return dict(zip(cols, row))
        except Exception as e:
            logger.error("[PG] 获取审计记录失败: %s", e)
            return None

    def get_audit_stats(
        self,
        start_date: str,
        end_date: str,
    ) -> dict:
        if not self.is_available:
            return {"total_reviews": 0}
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """SELECT
                               COUNT(*) as total,
                               COUNT(*) FILTER (WHERE judge_hallucinated_count > 0) as hallucinated,
                               COUNT(*) FILTER (WHERE judge_skipped = TRUE) as judge_skipped,
                               COUNT(*) FILTER (WHERE not_found = TRUE) as not_found,
                               COALESCE(AVG(confidence), 0) as avg_confidence,
                               COALESCE(AVG(retrieval_latency_ms + llm_latency_ms + judge_latency_ms), 0) as avg_latency
                           FROM audit_records
                           WHERE timestamp >= %s AND timestamp <= %s""",
                        (f"{start_date} 00:00:00", f"{end_date} 23:59:59"),
                    )
                    row = cur.fetchone()
            if not row or row[0] == 0:
                return {"total_reviews": 0}

            total = row[0]
            return {
                "total_reviews": total,
                "hallucination_rate": round(row[1] / total, 4) if total else 0.0,
                "judge_skip_rate": round(row[2] / total, 4) if total else 0.0,
                "not_found_rate": round(row[3] / total, 4) if total else 0.0,
                "avg_confidence": round(row[4], 4),
                "avg_latency_ms": round(row[5], 2),
            }
        except Exception as e:
            logger.error("[PG] 获取统计失败: %s", e)
            return {"total_reviews": 0}

    # ------------------------------------------------------------------
    # 全文搜索（PostgreSQL 内置）
    # ------------------------------------------------------------------

    def fulltext_search(self, query: str, limit: int = 10) -> list[dict]:
        """PostgreSQL 全文搜索（替代 BM25）。

        使用 pg_trgm 三元组相似度，支持中文。
        """
        if not self.is_available:
            return []
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    # 创建 pg_trgm 扩展（如果不存在）
                    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

                    cur.execute(
                        """SELECT chunk_id, doc_id, text, source, section, page,
                                  similarity(text, %s) AS sim
                           FROM chunks
                           WHERE text %% %s
                           ORDER BY sim DESC
                           LIMIT %s""",
                        (query, query, limit),
                    )
                    rows = cur.fetchall()
            return [
                {
                    "chunk_id": r[0],
                    "doc_id": r[1],
                    "text": r[2],
                    "source": r[3],
                    "section": r[4],
                    "page": r[5],
                    "score": float(r[6]),
                }
                for r in rows
            ]
        except Exception as e:
            logger.error("[PG] 全文搜索失败: %s", e)
            return []

    def close(self):
        self._pool.close()


# ---------------------------------------------------------------------------
# 默认工厂
# ---------------------------------------------------------------------------


_default_pg_store: PostgresStore | None = None


def get_default_pg_store() -> PostgresStore:
    global _default_pg_store
    if _default_pg_store is None:
        _default_pg_store = PostgresStore()
    return _default_pg_store
