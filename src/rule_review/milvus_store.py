"""
电力规则审查系统 - Milvus 向量数据库集成（Phase 4）

替换 FAISS 本地文件索引，提供：
- 分布式向量检索（无需全量重建索引）
- 实时增删 chunk（不阻塞检索）
- 标量过滤 + 向量检索混合查询
- Collection 分区管理（按文档隔离）
- 与现有 DocumentStore 同接口，无缝替换

Milvus 部署：
  docker run -d --name milvus-standalone \\
    -p 19530:19530 -p 9091:9091 \\
    milvusdb/milvus:latest

配置（.env）：
  MILVUS_HOST=localhost
  MILVUS_PORT=19530
  MILVUS_COLLECTION=rule_documents
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)

# Milvus Collection Schema 字段名
FIELD_ID = "chunk_id"
FIELD_DOC_ID = "doc_id"
FIELD_TEXT = "text"
FIELD_EMBEDDING = "embedding"
FIELD_SOURCE = "source"
FIELD_SECTION = "section"
FIELD_PAGE = "page"
FIELD_CHUNK_INDEX = "chunk_index"
FIELD_TABLES = "tables_json"  # JSON 序列化的表格数据

# 向量维度（bge-m3 = 1024）
VECTOR_DIM = 1024


# ---------------------------------------------------------------------------
# 数据模型（兼容现有 DocumentStore）
# ---------------------------------------------------------------------------


@dataclass
class MilvusChunk:
    """Milvus 中的 chunk 记录。"""
    chunk_id: str = ""
    doc_id: str = ""
    text: str = ""
    source: str = ""
    section: str = ""
    page: int = 0
    chunk_index: int = 0
    embedding: list[float] | None = None
    tables: list[dict] = field(default_factory=list)
    score: float = 0.0


@dataclass
class MilvusSearchResult:
    """向量检索结果。"""
    chunk: MilvusChunk
    score: float


# ---------------------------------------------------------------------------
# Milvus 连接管理
# ---------------------------------------------------------------------------


class MilvusClient:
    """Milvus 客户端封装，连接失败时静默返回不可用。"""

    def __init__(self):
        self._connected = False
        self._client = None
        self._collections: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    def connect(self, host: str = "", port: int = 19530) -> bool:
        """连接 Milvus 服务。"""
        try:
            from pymilvus import connections

            host = host or settings.MILVUS_HOST or "localhost"
            port = int(port or settings.MILVUS_PORT or 19530)

            connections.connect(
                alias="default",
                host=host,
                port=port,
                timeout=10,
            )
            self._connected = True
            logger.info("[Milvus] 连接成功: %s:%d", host, port)
            return True
        except Exception as e:
            logger.warning("[Milvus] 连接失败: %s", e)
            self._connected = False
            return False

    def disconnect(self):
        """断开连接。"""
        try:
            from pymilvus import connections
            connections.disconnect("default")
        except Exception:
            pass
        self._connected = False

    @property
    def is_available(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Collection 管理
    # ------------------------------------------------------------------

    def has_collection(self, name: str) -> bool:
        if not self._connected:
            return False
        try:
            from pymilvus import utility
            return utility.has_collection(name)
        except Exception:
            return False

    def create_collection(
        self,
        name: str,
        dim: int = VECTOR_DIM,
        description: str = "",
    ) -> bool:
        """创建向量集合（Schema: chunk_id + embedding + 标量字段）。"""
        if not self._connected:
            return False
        try:
            from pymilvus import (
                Collection,
                DataType,
                FieldSchema,
                CollectionSchema,
            )

            fields = [
                FieldSchema(name=FIELD_ID, dtype=DataType.VARCHAR, max_length=64, is_primary=True),
                FieldSchema(name=FIELD_DOC_ID, dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name=FIELD_EMBEDDING, dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name=FIELD_TEXT, dtype=DataType.VARCHAR, max_length=8192),
                FieldSchema(name=FIELD_SOURCE, dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name=FIELD_SECTION, dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name=FIELD_PAGE, dtype=DataType.INT64),
                FieldSchema(name=FIELD_CHUNK_INDEX, dtype=DataType.INT64),
                FieldSchema(name=FIELD_TABLES, dtype=DataType.VARCHAR, max_length=16384),
            ]

            schema = CollectionSchema(fields=fields, description=description)
            col = Collection(name=name, schema=schema)

            # 创建向量索引（IP = Inner Product，与 FAISS IndexFlatIP 一致）
            index_params = {
                "metric_type": "IP",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
            }
            col.create_index(
                field_name=FIELD_EMBEDDING,
                index_params=index_params,
            )

            col.load()
            self._collections[name] = col
            logger.info("[Milvus] Collection 创建成功: %s (dim=%d)", name, dim)
            return True
        except Exception as e:
            logger.error("[Milvus] Collection 创建失败: %s", e)
            return False

    def get_collection(self, name: str):
        """获取已有 Collection。"""
        if not self._connected:
            return None
        if name in self._collections:
            return self._collections[name]
        try:
            from pymilvus import Collection
            col = Collection(name=name)
            col.load()
            self._collections[name] = col
            return col
        except Exception as e:
            logger.warning("[Milvus] Collection 获取失败 '%s': %s", name, e)
            return None

    def drop_collection(self, name: str) -> bool:
        if not self._connected:
            return False
        try:
            from pymilvus import utility
            utility.drop_collection(name)
            self._collections.pop(name, None)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # 数据操作
    # ------------------------------------------------------------------

    def insert(self, collection_name: str, chunks: list[MilvusChunk]) -> int:
        """批量插入 chunk。"""
        col = self.get_collection(collection_name)
        if col is None:
            return 0

        data = [
            [c.chunk_id for c in chunks],
            [c.doc_id for c in chunks],
            [c.embedding or [0.0] * VECTOR_DIM for c in chunks],
            [c.text[:8192] for c in chunks],
            [c.source[:512] for c in chunks],
            [c.section[:512] for c in chunks],
            [c.page for c in chunks],
            [c.chunk_index for c in chunks],
            [json.dumps(c.tables, ensure_ascii=False)[:16384] for c in chunks],
        ]

        try:
            col.insert(data)
            col.flush()
            return len(chunks)
        except Exception as e:
            logger.error("[Milvus] 插入失败: %s", e)
            return 0

    def delete_by_doc_id(self, collection_name: str, doc_id: str) -> int:
        """按 doc_id 删除所有 chunk。"""
        col = self.get_collection(collection_name)
        if col is None:
            return 0
        try:
            expr = f'{FIELD_DOC_ID} == "{doc_id}"'
            result = col.delete(expr)
            col.flush()
            count = result.delete_count if hasattr(result, 'delete_count') else 0
            logger.info("[Milvus] 删除 doc_id=%s, %d 条", doc_id, count)
            return count
        except Exception as e:
            logger.error("[Milvus] 删除失败: %s", e)
            return 0

    def search(
        self,
        collection_name: str,
        query_embedding: list[float] | np.ndarray,
        top_k: int = 10,
        filter_expr: str | None = None,
    ) -> list[MilvusSearchResult]:
        """向量相似度检索。"""
        col = self.get_collection(collection_name)
        if col is None:
            return []

        try:
            search_params = {
                "metric_type": "IP",
                "params": {"nprobe": 16},
            }

            output_fields = [
                FIELD_ID, FIELD_DOC_ID, FIELD_TEXT, FIELD_SOURCE,
                FIELD_SECTION, FIELD_PAGE, FIELD_CHUNK_INDEX, FIELD_TABLES,
            ]

            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()

            results = col.search(
                data=[query_embedding],
                anns_field=FIELD_EMBEDDING,
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=output_fields,
            )

            search_results: list[MilvusSearchResult] = []
            for hits in results:
                for hit in hits:
                    entity = hit.entity
                    tables = []
                    try:
                        tables = json.loads(entity.get(FIELD_TABLES, "[]"))
                    except (json.JSONDecodeError, TypeError):
                        pass

                    chunk = MilvusChunk(
                        chunk_id=entity.get(FIELD_ID, ""),
                        doc_id=entity.get(FIELD_DOC_ID, ""),
                        text=entity.get(FIELD_TEXT, ""),
                        source=entity.get(FIELD_SOURCE, ""),
                        section=entity.get(FIELD_SECTION, ""),
                        page=entity.get(FIELD_PAGE, 0),
                        chunk_index=entity.get(FIELD_CHUNK_INDEX, 0),
                        tables=tables,
                        score=hit.score,
                    )
                    search_results.append(MilvusSearchResult(chunk=chunk, score=hit.score))

            return search_results
        except Exception as e:
            logger.error("[Milvus] 检索失败: %s", e)
            return []

    def count(self, collection_name: str) -> int:
        """Collection 中的实体数量。"""
        col = self.get_collection(collection_name)
        if col is None:
            return 0
        try:
            return col.num_entities
        except Exception:
            return 0


# ---------------------------------------------------------------------------
# 默认工厂
# ---------------------------------------------------------------------------


_default_milvus: MilvusClient | None = None


def get_default_milvus() -> MilvusClient:
    global _default_milvus
    if _default_milvus is None:
        _default_milvus = MilvusClient()
    return _default_milvus
