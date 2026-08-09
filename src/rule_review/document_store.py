"""
电力规则审查系统 - 文档存储与索引模块

按设计文档 Phase 1 步骤 1.3 实现：
- PDF 解析（分层：pymupdf 本地 / MinerU 本地，见 parsers.py 与 docs/rule-review.md §4.3）
- 章节层级检测与表格转 Markdown
- 按标题+表格联合策略切分 chunk
- bge-m3 embedding（可注入）
- FAISS 向量索引持久化

解析分层改造说明：
- 数据模型（TextBlock/TableBlock/PageContent/Chunk/DocumentInfo 等）移至 models.py，
  本模块顶部 re-export，外部 import 全部保持不变
- 解析器（OCRProcessor/PaddleOCRProcessor/PDFParser/PymupdfParser/MinerULocalParser）
  移至 parsers.py；本模块负责解析器选择路由与 chunk 组装
- 手动入库（重要政策问答表格）：ingest_manual()，识别准确率 100%，parse_mode=manual

不调用 LLM，核心逻辑为纯 Python。
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO

import faiss
import numpy as np

from src.config import settings
from src.rule_review.models import (
    Chunk,
    ChunkSearchResult,
    DocumentInfo,
    PageContent,
    TableBlock,
    TextBlock,
)
from src.rule_review.parsers import (
    MinerULocalParser,
    OCRProcessor,
    PDFParser,
    PaddleOCRProcessor,
    PymupdfParser,
    parse_markdown,
)
from src.rule_review.schemas import DocumentUploadResponse

logger = logging.getLogger(__name__)

# chunk 默认参数：约 800 tokens 对应约 600 中文字符
_DEFAULT_CHUNK_SIZE = 600
_DEFAULT_CHUNK_OVERLAP = 150
_DEFAULT_TABLE_MAX_ROWS = 20

# 章节/条款/项目符号正则
_HEADING_PATTERNS: list[tuple[int, re.Pattern]] = [
    (0, re.compile(r"^第[一二三四五六七八九十0-9]+章\s+")),
    (1, re.compile(r"^第[一二三四五六七八九十0-9]+节\s+")),
    (2, re.compile(r"^第[一二三四五六七八九十0-9]+条\s+")),
    (
        3,
        re.compile(
            r"^(?:[一二三四五六七八九十]+、|^（[一二三四五六七八九十0-9]+）|^\([一二三四五六七八九十0-9]+\))\s*"
        ),
    ),
]


# ---------------------------------------------------------------------------
# Embedding 抽象
# ---------------------------------------------------------------------------


class EmbeddingFunction(ABC):
    """Embedding 函数抽象基类。"""

    @abstractmethod
    def __call__(self, texts: list[str]) -> np.ndarray:
        """返回 shape=(len(texts), dim) 的 float32 数组。"""


class BGEM3Embedding(EmbeddingFunction):
    """默认 bge-m3 embedding。"""

    _instances: dict[str, "BGEM3Embedding"] = {}

    def __new__(cls, model_name: str | None = None) -> "BGEM3Embedding":
        name = model_name or settings.EMBEDDING_MODEL
        if name not in cls._instances:
            cls._instances[name] = super().__new__(cls)
        return cls._instances[name]

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.EMBEDDING_MODEL
        if not hasattr(self, "_model"):
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

    def __call__(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)


def _get_default_embedding() -> EmbeddingFunction:
    return BGEM3Embedding()


# ---------------------------------------------------------------------------
# DocumentStore
# ---------------------------------------------------------------------------


class DocumentStore:
    """规则文档存储：解析 PDF、切分 chunk、生成 embedding、维护 FAISS 索引。"""

    def __init__(
        self,
        documents_dir: str | Path | None = None,
        index_dir: str | Path | None = None,
        embedding_fn: EmbeddingFunction | None = None,
        embedding_dim: int | None = None,
        ocr_engine: OCRProcessor | None = None,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
        table_max_rows: int = _DEFAULT_TABLE_MAX_ROWS,
        mineru_parser: PDFParser | None = None,
        mineru_enabled: bool | None = None,
    ) -> None:
        """
        Args:
            documents_dir: 上传 PDF 的存储目录，默认 settings.RULE_DOCUMENTS_DIR。
            index_dir: 索引持久化目录，默认 settings.RULE_INDEX_DIR。
            embedding_fn: 可注入的 embedding 函数；为空时使用 bge-m3。
            embedding_dim: 自定义 embedding 函数时必须传入维度；使用默认时可空。
            ocr_engine: OCR 引擎；为空时尝试 PaddleOCR，失败则不可用。
            chunk_size: 纯文本 chunk 的最大字符数。
            chunk_overlap: 相邻 chunk 重叠字符数。
            table_max_rows: 表格超长时保留的最大行数。
            mineru_parser: 可注入的 MinerU 解析器（测试用 fake；为空时用 MinerULocalParser 单例）。
            mineru_enabled: 是否启用 MinerU 本地解析；为空时读 settings.RULE_REVIEW_MINERU_ENABLED。
        """
        self.documents_dir = Path(documents_dir or settings.RULE_DOCUMENTS_DIR)
        self.index_dir = Path(index_dir or settings.RULE_INDEX_DIR)
        self.chunks_dir = self.index_dir / "chunks"

        for d in (self.documents_dir, self.index_dir, self.chunks_dir):
            d.mkdir(parents=True, exist_ok=True)

        if embedding_fn is None:
            self.embedding_fn = _get_default_embedding()
            self._embedding_dim = embedding_dim or 1024
        else:
            if embedding_dim is None:
                raise ValueError("使用自定义 embedding_fn 时必须指定 embedding_dim")
            self.embedding_fn = embedding_fn
            self._embedding_dim = embedding_dim

        self.ocr_engine = ocr_engine or PaddleOCRProcessor()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.table_max_rows = table_max_rows

        # 解析分层：MinerU 解析器与启用开关（测试可注入 fake）
        self.mineru_parser = mineru_parser
        self.mineru_enabled = (
            mineru_enabled
            if mineru_enabled is not None
            else settings.RULE_REVIEW_MINERU_ENABLED
        )

        # 内存状态
        self._documents: dict[str, DocumentInfo] = {}
        self._chunks: dict[str, Chunk] = {}
        self._faiss_id_to_chunk_id: dict[int, str] = {}
        self._chunk_id_to_faiss_id: dict[str, int] = {}
        self._next_faiss_id = 1
        self._index: faiss.Index | None = None

        self._load()

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def ingest(
        self,
        file: str | Path | bytes | BinaryIO,
        filename: str | None = None,
        doc_id: str | None = None,
        importance: str = "low",
        parse_mode: str = "auto",
    ) -> DocumentUploadResponse:
        """解析并索引一份 PDF 文档（解析分层，见 docs/rule-review.md §4.3）。

        Args:
            file: PDF 文件（路径 / bytes / 文件对象）。
            filename: 文件名；file 为 bytes/文件对象时必填或用于展示。
            doc_id: 自定义文档 ID；为空时自动生成。
            importance: 文档重要程度 high | low（显式标注，不做关键词自动识别）。
            parse_mode: 解析方式 auto | mineru | pymupdf；auto 时 MinerU 可用则用之，
                否则降级 pymupdf（降级后的实际模式写入 DocumentInfo.parse_mode）。
        """
        self._validate_importance(importance)
        data, filename = self._read_file(file, filename)
        doc_id = doc_id or uuid.uuid4().hex

        doc_path = self.documents_dir / f"{doc_id}.pdf"
        doc_path.write_bytes(data)

        chunks, actual_mode, page_count = self._parse_document(
            data, filename, doc_id, parse_mode
        )
        return self._index_chunks(
            chunks,
            doc_id,
            filename,
            page_count,
            importance,
            actual_mode,
        )

    def ingest_manual(
        self,
        markdown: str,
        filename: str | None = None,
        doc_id: str | None = None,
        importance: str = "high",
        source: str = "",
    ) -> DocumentUploadResponse:
        """手动入库：重要政策问答/表格以 Markdown 直接入库。

        不生成 PDF 文件；parse_markdown → _build_chunks 走与 PDF 上传完全相同的
        chunk 组装路径；parse_mode 记为 manual（识别准确率 100%，见 §4.3 评测方法）。

        Args:
            markdown: 规则 Markdown 内容（# 标题、段落、| 表格 |）。
            filename: 文档名称；为空时自动生成。
            doc_id: 自定义文档 ID；为空时自动生成。
            importance: 重要程度，默认 high（手动入库面向重要内容）。
            source: 来源说明（如「政策问答表格人工整理」）。
        """
        self._validate_importance(importance)
        if not markdown or not markdown.strip():
            raise ValueError("markdown 内容不能为空")

        doc_id = doc_id or uuid.uuid4().hex
        pages = parse_markdown(markdown)
        chunks: list[Chunk] = []
        section_stack: list[tuple[int, str]] = []
        for page_number, page_content in enumerate(pages, start=1):
            chunks.extend(
                self._build_chunks(
                    page_content,
                    doc_id,
                    page_number,
                    section_stack,
                    page_content.is_scanned,
                )
            )
        return self._index_chunks(
            chunks,
            doc_id,
            filename=filename or f"{doc_id}.md",
            page_count=len(pages),
            importance=importance,
            parse_mode="manual",
            source=source,
        )

    def delete(self, doc_id: str) -> bool:
        """删除文档及其索引。"""
        if doc_id not in self._documents:
            return False

        # 删除 chunk 元数据与映射
        chunks_to_remove = [c for c in self._chunks.values() if c.doc_id == doc_id]
        for chunk in chunks_to_remove:
            self._chunks.pop(chunk.chunk_id, None)
            fid = self._chunk_id_to_faiss_id.pop(chunk.chunk_id, None)
            if fid is not None:
                self._faiss_id_to_chunk_id.pop(fid, None)

        # 删除持久化文件
        chunk_file = self.chunks_dir / f"{doc_id}.json"
        if chunk_file.exists():
            chunk_file.unlink()
        pdf_file = self.documents_dir / f"{doc_id}.pdf"
        if pdf_file.exists():
            pdf_file.unlink()

        self._documents.pop(doc_id, None)

        # 重建 FAISS 索引
        self._rebuild_index()
        self.save()
        return True

    def list_documents(self) -> list[DocumentInfo]:
        """列出已入库文档。"""
        return list(self._documents.values())

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """按 chunk_id 获取 chunk。"""
        return self._chunks.get(chunk_id)

    def vector_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        doc_filter: str | None = None,
    ) -> list[ChunkSearchResult]:
        """向量检索。"""
        if self._index is None or self._index.ntotal == 0:
            return []

        q = np.ascontiguousarray(query_embedding, dtype=np.float32).reshape(1, -1)
        search_k = top_k if doc_filter is None else top_k * 2
        scores, ids = self._index.search(q, search_k)

        results: list[ChunkSearchResult] = []
        for score, fid in zip(scores[0], ids[0]):
            if fid < 0:
                continue
            chunk_id = self._faiss_id_to_chunk_id.get(int(fid))
            if chunk_id is None:
                continue
            chunk = self._chunks.get(chunk_id)
            if chunk is None:
                continue
            if doc_filter and chunk.doc_id != doc_filter:
                continue
            results.append(ChunkSearchResult(chunk=chunk, score=float(score)))
            if len(results) >= top_k:
                break
        return results

    def save(self) -> None:
        """持久化文档注册表、chunk 元数据和 FAISS 索引。"""
        docs_path = self.index_dir / "documents.json"
        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump(
                {k: v.__dict__ for k, v in self._documents.items()},
                f,
                ensure_ascii=False,
                indent=2,
            )

        for doc_id in self._documents:
            chunks = [c for c in self._chunks.values() if c.doc_id == doc_id]
            chunks.sort(key=lambda c: c.chunk_id)
            with open(self.chunks_dir / f"{doc_id}.json", "w", encoding="utf-8") as f:
                json.dump([c.to_dict() for c in chunks], f, ensure_ascii=False, indent=2)

        # 清理已不存在文档的 chunk 文件
        for p in self.chunks_dir.glob("*.json"):
            if p.stem not in self._documents:
                p.unlink()

        if self._index is not None:
            faiss.write_index(self._index, str(self.index_dir / "vectors.faiss"))

    def load(self) -> None:
        """重新加载持久化数据。"""
        self._load()

    def clear(self) -> None:
        """清空所有文档与索引。"""
        self._documents.clear()
        self._chunks.clear()
        self._faiss_id_to_chunk_id.clear()
        self._chunk_id_to_faiss_id.clear()
        self._next_faiss_id = 1
        self._index = self._create_empty_index()
        for p in self.documents_dir.glob("*.pdf"):
            p.unlink()
        for p in self.chunks_dir.glob("*.json"):
            p.unlink()
        docs_path = self.index_dir / "documents.json"
        if docs_path.exists():
            docs_path.unlink()
        idx_path = self.index_dir / "vectors.faiss"
        if idx_path.exists():
            idx_path.unlink()

    # ------------------------------------------------------------------
    # 内部：文件读取
    # ------------------------------------------------------------------

    @staticmethod
    def _read_file(
        file: str | Path | bytes | BinaryIO,
        filename: str | None = None,
    ) -> tuple[bytes, str]:
        if isinstance(file, (str, Path)):
            path = Path(file)
            data = path.read_bytes()
            name = filename or path.name or "upload.pdf"
        elif isinstance(file, bytes):
            data = file
            name = filename or "upload.pdf"
        else:
            data = file.read()
            name = filename or getattr(file, "name", "upload.pdf") or "upload.pdf"
        return data, name

    # ------------------------------------------------------------------
    # 内部：索引管理
    # ------------------------------------------------------------------

    def _create_empty_index(self) -> faiss.Index:
        base = faiss.IndexFlatIP(self._embedding_dim)
        return faiss.IndexIDMap2(base)

    def _ensure_index(self) -> None:
        if self._index is None:
            self._index = self._create_empty_index()

    def _rebuild_index(self) -> None:
        self._index = self._create_empty_index()
        self._faiss_id_to_chunk_id.clear()
        self._chunk_id_to_faiss_id.clear()
        self._next_faiss_id = 1

        remaining = sorted(self._chunks.values(), key=lambda c: c.chunk_id)
        if not remaining:
            return

        embeddings: list[np.ndarray] = []
        ids: list[int] = []
        for chunk in remaining:
            if chunk.embedding is None:
                continue
            chunk.faiss_id = self._next_faiss_id
            self._next_faiss_id += 1
            embeddings.append(chunk.embedding)
            ids.append(chunk.faiss_id)
            self._faiss_id_to_chunk_id[chunk.faiss_id] = chunk.chunk_id
            self._chunk_id_to_faiss_id[chunk.chunk_id] = chunk.faiss_id

        if embeddings:
            embs = np.ascontiguousarray(np.stack(embeddings, axis=0), dtype=np.float32)
            self._index.add_with_ids(embs, np.array(ids, dtype=np.int64))

    def _load(self) -> None:
        docs_path = self.index_dir / "documents.json"
        if docs_path.exists():
            with open(docs_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._documents = {k: DocumentInfo(**v) for k, v in data.items()}

        # 加载 chunk 元数据并重建映射
        self._chunks.clear()
        self._faiss_id_to_chunk_id.clear()
        self._chunk_id_to_faiss_id.clear()
        self._next_faiss_id = 1

        for doc_id in self._documents:
            chunk_path = self.chunks_dir / f"{doc_id}.json"
            if not chunk_path.exists():
                continue
            with open(chunk_path, "r", encoding="utf-8") as f:
                items = json.load(f)
            for item in items:
                chunk = Chunk.from_dict(item)
                self._chunks[chunk.chunk_id] = chunk
                if chunk.faiss_id is not None:
                    self._faiss_id_to_chunk_id[chunk.faiss_id] = chunk.chunk_id
                    self._chunk_id_to_faiss_id[chunk.chunk_id] = chunk.faiss_id
                    if chunk.faiss_id >= self._next_faiss_id:
                        self._next_faiss_id = chunk.faiss_id + 1

        idx_path = self.index_dir / "vectors.faiss"
        if idx_path.exists():
            self._index = faiss.read_index(str(idx_path))
        else:
            self._index = self._create_empty_index()

    # ------------------------------------------------------------------
    # 内部：PDF 解析
    # ------------------------------------------------------------------

    def _validate_importance(self, importance: str) -> None:
        """校验 importance 取值（high | low）。"""
        if importance not in ("high", "low"):
            raise ValueError(f"importance 仅支持 high/low，收到: {importance}")

    def _select_parser(self, parse_mode: str) -> PDFParser:
        """按 parse_mode 选择解析器；MinerU 不可用时降级 pymupdf。"""
        if parse_mode == "auto":
            if self.mineru_enabled:
                parser = self.mineru_parser or MinerULocalParser()
                if parser.is_available:
                    return parser
                logger.warning("MinerU 不可用（auto 模式），降级为 pymupdf 解析")
            return PymupdfParser(ocr_engine=self.ocr_engine)
        if parse_mode == "mineru":
            parser = self.mineru_parser or MinerULocalParser()
            if parser.is_available:
                return parser
            logger.warning("MinerU 不可用（mineru 模式），降级为 pymupdf 解析")
            return PymupdfParser(ocr_engine=self.ocr_engine)
        if parse_mode == "pymupdf":
            return PymupdfParser(ocr_engine=self.ocr_engine)
        raise ValueError(f"未知解析模式: {parse_mode}，仅支持 auto / mineru / pymupdf")

    def _parse_document(
        self,
        data: bytes,
        filename: str,
        doc_id: str,
        parse_mode: str,
    ) -> tuple[list[Chunk], str, int]:
        """选择解析器解析 PDF → chunks。

        Returns:
            (chunks, 实际解析模式, 页数)。实际模式含降级后的模式，
            写入 DocumentInfo.parse_mode。
        """
        parser = self._select_parser(parse_mode)
        pages = parser.parse(data, filename)

        chunks: list[Chunk] = []
        section_stack: list[tuple[int, str]] = []
        for page_number, page_content in enumerate(pages, start=1):
            chunks.extend(
                self._build_chunks(
                    page_content,
                    doc_id,
                    page_number,
                    section_stack,
                    page_content.is_scanned,
                )
            )
        return chunks, parser.name, len(pages)

    def _index_chunks(
        self,
        chunks: list[Chunk],
        doc_id: str,
        filename: str,
        page_count: int,
        importance: str,
        parse_mode: str,
        source: str = "",
    ) -> DocumentUploadResponse:
        """embedding → FAISS → DocumentInfo → save → 响应（ingest / ingest_manual 共用尾部）。"""
        # 生成 embedding
        if chunks:
            texts = [c.text for c in chunks]
            embeddings = self.embedding_fn(texts)
            self._ensure_index()
            ids: list[int] = []
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = np.ascontiguousarray(emb, dtype=np.float32)
                chunk.faiss_id = self._next_faiss_id
                ids.append(chunk.faiss_id)
                self._next_faiss_id += 1

                self._chunks[chunk.chunk_id] = chunk
                self._faiss_id_to_chunk_id[chunk.faiss_id] = chunk.chunk_id
                self._chunk_id_to_faiss_id[chunk.chunk_id] = chunk.faiss_id

            ids_arr = np.array(ids, dtype=np.int64)
            embs_arr = np.ascontiguousarray(embeddings.astype(np.float32))
            self._index.add_with_ids(embs_arr, ids_arr)

        doc_info = DocumentInfo(
            doc_id=doc_id,
            file_name=filename,
            page_count=page_count,
            chunk_count=len(chunks),
            created_at=datetime.now().isoformat(),
            importance=importance,
            parse_mode=parse_mode,
            source=source,
        )
        self._documents[doc_id] = doc_info
        self.save()

        return DocumentUploadResponse(
            doc_id=doc_id,
            file_name=filename,
            page_count=page_count,
            chunk_count=len(chunks),
            uploaded_at=doc_info.created_at,
            importance=importance,
            parse_mode=parse_mode,
        )

    # ------------------------------------------------------------------
    # 内部：chunk 组装
    # ------------------------------------------------------------------

    def _build_chunks(
        self,
        page_content: PageContent,
        doc_id: str,
        page_number: int,
        section_stack: list[tuple[int, str]],
        is_scanned: bool,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []

        items: list[tuple[str, float, Any]] = []
        for tb in page_content.text_blocks:
            items.append(("text", tb.bbox[1], tb))
        for tbl in page_content.table_blocks:
            items.append(("table", tbl.bbox[1], tbl))
        items.sort(key=lambda x: (x[1], x[2].bbox[0]))

        text_buffer: list[str] = []
        recent_texts: list[str] = []  # 用于表格 caption 回溯

        def _flush_buffer() -> None:
            nonlocal text_buffer
            if not text_buffer:
                return
            text = "\n".join(text_buffer)
            text_buffer = []
            section_title, hierarchy = self._current_section(section_stack)
            for sub_text in self._split_text(text):
                full_text = f"{section_title}\n{sub_text}" if section_title else sub_text
                chunks.append(
                    Chunk(
                        chunk_id=uuid.uuid4().hex,
                        doc_id=doc_id,
                        text=full_text.strip(),
                        section=section_title,
                        section_hierarchy=hierarchy,
                        page=page_number,
                        ocr_confidence=page_content.avg_confidence,
                        is_scanned=is_scanned,
                    )
                )

        for kind, _, obj in items:
            if kind == "text":
                block = obj
                level, title = self._detect_heading(block.text)
                if level >= 0:
                    _flush_buffer()
                    self._update_section_stack(section_stack, level, title)
                    recent_texts = []
                else:
                    text_buffer.append(block.text)
                    recent_texts.append(block.text)
                    if len(recent_texts) > 3:
                        recent_texts.pop(0)
            else:
                # table
                _flush_buffer()
                chunks.append(
                    self._make_table_chunk(
                        obj,
                        doc_id,
                        page_number,
                        section_stack,
                        recent_texts,
                        page_content.avg_confidence,
                        is_scanned,
                    )
                )
                recent_texts = []

        _flush_buffer()

        # 兜底：整页只有标题等元信息时，至少生成一个包含当前段落的 chunk
        if not chunks and section_stack:
            section_title, hierarchy = self._current_section(section_stack)
            chunks.append(
                Chunk(
                    chunk_id=uuid.uuid4().hex,
                    doc_id=doc_id,
                    text=section_title,
                    section=section_title,
                    section_hierarchy=hierarchy,
                    page=page_number,
                    ocr_confidence=page_content.avg_confidence,
                    is_scanned=is_scanned,
                )
            )

        return chunks

    @staticmethod
    def _detect_heading(text: str) -> tuple[int, str]:
        stripped = text.strip()
        for level, pattern in _HEADING_PATTERNS:
            if pattern.match(stripped):
                return level, stripped
        return -1, stripped

    @staticmethod
    def _update_section_stack(
        stack: list[tuple[int, str]],
        level: int,
        title: str,
    ) -> None:
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))

    @staticmethod
    def _current_section(stack: list[tuple[int, str]]) -> tuple[str, list[str]]:
        hierarchy = [title for _, title in stack]
        section = " > ".join(hierarchy)
        return section, hierarchy

    def _split_text(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        parts: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                # 在边界附近寻找合适分隔符
                search_start = max(start + self.chunk_size // 2, start + 1)
                best = end
                for delim in ["\n\n", "。", "；", "\n", " "]:
                    pos = text.rfind(delim, search_start, end)
                    if pos != -1:
                        best = pos + len(delim)
                        break
                end = best
            parts.append(text[start:end])
            start = max(start + 1, end - self.chunk_overlap)
        return parts

    def _make_table_chunk(
        self,
        table: TableBlock,
        doc_id: str,
        page_number: int,
        section_stack: list[tuple[int, str]],
        recent_texts: list[str],
        avg_confidence: float,
        is_scanned: bool,
    ) -> Chunk:
        section_title, hierarchy = self._current_section(section_stack)

        # caption 回溯
        caption = ""
        for t in reversed(recent_texts):
            if "表" in t or t.endswith(("：", ":", "如下")):
                caption = t
                break

        markdown = self._rows_to_markdown(table.rows)
        full_text_parts = [p for p in [section_title, caption, markdown] if p]
        full_text = "\n".join(full_text_parts)

        # 超长截断
        if len(full_text) > self.chunk_size:
            full_text = self._truncate_table_chunk(
                section_title, caption, table.rows
            )

        return Chunk(
            chunk_id=uuid.uuid4().hex,
            doc_id=doc_id,
            text=full_text,
            tables=[
                {
                    "caption": caption,
                    "headers": table.rows[0] if table.rows else [],
                    "rows": table.rows[1:] if len(table.rows) > 1 else [],
                }
            ],
            section=section_title,
            section_hierarchy=hierarchy,
            page=page_number,
            ocr_confidence=avg_confidence,
            is_scanned=is_scanned,
        )

    @staticmethod
    def _rows_to_markdown(rows: list[list[str]]) -> str:
        if not rows:
            return ""
        cleaned = [[(cell or "").strip() for cell in row] for row in rows]
        lines = ["| " + " | ".join(row) + " |" for row in cleaned]
        if len(lines) >= 2:
            header_cols = len(cleaned[0])
            sep = "|" + "|".join([" --- " for _ in range(header_cols)]) + "|"
            lines.insert(1, sep)
        return "\n".join(lines)

    def _truncate_table_chunk(
        self,
        section_title: str,
        caption: str,
        rows: list[list[str]],
    ) -> str:
        if not rows:
            return ""
        header = rows[0]
        kept_rows: list[list[str]] = []
        for row in rows[1:]:
            trial = self._rows_to_markdown([header] + kept_rows + [row])
            parts = [p for p in [section_title, caption, trial] if p]
            if len("\n".join(parts)) > self.chunk_size:
                break
            kept_rows.append(row)
            if len(kept_rows) >= self.table_max_rows:
                break

        if len(kept_rows) < len(rows) - 1:
            ellipsis = ["..."] * len(header)
            kept_rows.append(ellipsis)

        markdown = self._rows_to_markdown([header] + kept_rows)
        full_text_parts = [p for p in [section_title, caption, markdown] if p]
        return "\n".join(full_text_parts)


__all__ = [
    "DocumentStore",
    "Chunk",
    "DocumentInfo",
    "ChunkSearchResult",
    "OCRProcessor",
    "PaddleOCRProcessor",
    "EmbeddingFunction",
    "BGEM3Embedding",
]
