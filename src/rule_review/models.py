"""
电力规则审查系统 - 数据模型

解析分层改造（见 docs/rule-review.md §4.3「解析分层」）后从 document_store.py
拆出独立模块：document_store.py（编排）与 parsers.py（解析器）都需要构造
PageContent/Chunk 等模型，放同一模块可避免顶层循环导入。

字段兼容约定：所有新增字段必须带默认值——旧持久化 JSON 反序列化
（DocumentInfo(**data) / Chunk.from_dict）对缺字段零容忍。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class TextBlock:
    """文本块。"""

    text: str
    bbox: tuple[float, float, float, float]
    confidence: float = 1.0


@dataclass
class TableBlock:
    """表格块。"""

    rows: list[list[str]]
    bbox: tuple[float, float, float, float]
    caption: str | None = None


@dataclass
class PageContent:
    """单页解析结果。"""

    text_blocks: list[TextBlock]
    table_blocks: list[TableBlock]
    avg_confidence: float = 1.0
    # 解析分层新增：解析器把「是否扫描页」判定透传给 chunk 组装
    # （原 DocumentStore._parse_document 用局部变量，解析器抽象后须挂到 PageContent 上）
    is_scanned: bool = False


@dataclass
class Chunk:
    """检索单元。"""

    chunk_id: str
    doc_id: str
    text: str
    tables: list[dict] = field(default_factory=list)
    section: str = ""
    section_hierarchy: list[str] = field(default_factory=list)
    page: int = 0
    ocr_confidence: float = 1.0
    is_scanned: bool = False
    embedding: np.ndarray | None = None
    faiss_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.embedding is not None:
            data["embedding"] = self.embedding.tolist()
        else:
            data["embedding"] = None
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        embedding = data.pop("embedding", None)
        chunk = cls(**data)
        if embedding is not None:
            chunk.embedding = np.array(embedding, dtype=np.float32)
        return chunk


@dataclass
class DocumentInfo:
    """已入库文档元信息。"""

    doc_id: str
    file_name: str
    page_count: int
    chunk_count: int
    created_at: str
    # ---- 解析分层新增字段（均有默认值，旧 JSON 反序列化兼容）----
    importance: str = "low"      # high | low —— 调用方显式标注，不做关键词自动识别
    parse_mode: str = "pymupdf"  # pymupdf | mineru | manual（实际生效模式，含降级后）
    source: str = ""             # 来源说明（手动入库如「政策问答表格人工整理」）


@dataclass
class ChunkSearchResult:
    """向量检索结果。"""

    chunk: Chunk
    score: float
