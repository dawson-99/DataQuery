"""
电力规则审查系统 - 文档存储与索引模块

按设计文档 Phase 1 步骤 1.3 实现：
- PDF 解析（文本 PDF + 可选 OCR 扫描 PDF）
- 章节层级检测与表格转 Markdown
- 按标题+表格联合策略切分 chunk
- bge-m3 embedding（可注入）
- FAISS 向量索引持久化

不调用 LLM，核心逻辑为纯 Python。
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO

import faiss
import fitz  # pymupdf
import numpy as np
from PIL import Image

from src.config import settings
from src.rule_review.schemas import DocumentUploadResponse

logger = logging.getLogger(__name__)

# 判定页面为扫描件的最大可识别字符数阈值
_SCANNED_TEXT_THRESHOLD = 50

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
# 数据模型
# ---------------------------------------------------------------------------


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


@dataclass
class ChunkSearchResult:
    """向量检索结果。"""

    chunk: Chunk
    score: float


# ---------------------------------------------------------------------------
# OCR 抽象
# ---------------------------------------------------------------------------


class OCRProcessor(ABC):
    """OCR 处理器抽象基类，扫描 PDF 使用。"""

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """OCR 是否可用。"""

    @abstractmethod
    def process_page(
        self,
        image: Image.Image | np.ndarray,
        page_number: int = 0,
    ) -> PageContent:
        """对单页图片执行 OCR，返回结构化内容。"""


class PaddleOCRProcessor(OCRProcessor):
    """基于 PaddleOCR 的 OCR 实现。import 延迟，失败时 is_available=False。"""

    def __init__(self, use_gpu: bool = False, show_log: bool = False) -> None:
        self._available = False
        self._ocr: Any | None = None
        try:
            from paddleocr import PaddleOCR

            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang="ch",
                use_gpu=use_gpu,
                show_log=show_log,
            )
            self._available = True
        except Exception as exc:  # pragma: no cover - 运行环境未安装 paddleocr 时正常降级
            logger.warning(f"PaddleOCR 初始化失败，扫描 PDF 将跳过: {exc}")

    @property
    def is_available(self) -> bool:
        return self._available

    def process_page(
        self,
        image: Image.Image | np.ndarray,
        page_number: int = 0,
    ) -> PageContent:
        if not self._available or self._ocr is None:
            raise RuntimeError("PaddleOCR 不可用")

        arr = np.array(image) if isinstance(image, Image.Image) else image
        result = self._ocr.ocr(arr, cls=True)

        text_blocks: list[TextBlock] = []
        confidences: list[float] = []
        if result and result[0]:
            for line in result[0]:
                bbox, (text, conf) = line
                text_blocks.append(
                    TextBlock(text=text or "", bbox=tuple(bbox), confidence=float(conf))
                )
                confidences.append(float(conf))

        avg_conf = float(np.mean(confidences)) if confidences else 1.0
        return PageContent(
            text_blocks=text_blocks,
            table_blocks=[],
            avg_confidence=avg_conf,
        )


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
    ) -> DocumentUploadResponse:
        """解析并索引一份 PDF 文档。"""
        data, filename = self._read_file(file, filename)
        doc_id = doc_id or uuid.uuid4().hex

        doc_path = self.documents_dir / f"{doc_id}.pdf"
        doc_path.write_bytes(data)

        try:
            doc = fitz.open(stream=data, filetype="pdf")
        except Exception as exc:
            raise ValueError(f"PDF 解析失败: {exc}") from exc

        page_count = len(doc)
        try:
            chunks = self._parse_document(doc, doc_id)
        finally:
            doc.close()

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
        )
        self._documents[doc_id] = doc_info
        self.save()

        return DocumentUploadResponse(
            doc_id=doc_id,
            file_name=filename,
            page_count=page_count,
            chunk_count=len(chunks),
            uploaded_at=doc_info.created_at,
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

    def _parse_document(self, doc: fitz.Document, doc_id: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        section_stack: list[tuple[int, str]] = []

        for page_number, page in enumerate(doc, start=1):
            is_scanned = self._is_scanned_page(page)

            if is_scanned and self.ocr_engine is not None and self.ocr_engine.is_available:
                page_content = self._ocr_page(page, page_number)
                is_scanned_flag = True
            else:
                if is_scanned:
                    logger.warning(
                        f"第 {page_number} 页疑似扫描页但 OCR 不可用，已跳过"
                    )
                page_content = self._extract_text_page(page, page_number)
                is_scanned_flag = False

            page_chunks = self._build_chunks(
                page_content,
                doc_id,
                page_number,
                section_stack,
                is_scanned_flag,
            )
            chunks.extend(page_chunks)

        return chunks

    @staticmethod
    def _is_scanned_page(page: fitz.Page) -> bool:
        text = page.get_text().strip()
        has_images = bool(page.get_images())
        return len(text) < _SCANNED_TEXT_THRESHOLD and has_images

    def _ocr_page(self, page: fitz.Page, page_number: int) -> PageContent:
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return self.ocr_engine.process_page(img, page_number=page_number)

    def _extract_text_page(self, page: fitz.Page, page_number: int) -> PageContent:
        # 表格检测
        table_finder = page.find_tables()
        table_blocks: list[TableBlock] = []
        table_bboxes: list[tuple[float, float, float, float]] = []
        for table in table_finder.tables:
            rows = table.extract()
            if not rows:
                continue
            bbox = tuple(table.bbox)
            table_blocks.append(TableBlock(rows=rows, bbox=bbox))
            table_bboxes.append(bbox)

        # 文本块，过滤掉与表格区域高度重叠的块
        raw_blocks = page.get_text("blocks")
        text_blocks: list[TextBlock] = []
        for block in raw_blocks:
            x0, y0, x1, y1, text, _, _ = block
            if not text or not text.strip():
                continue
            bbox = (x0, y0, x1, y1)
            if self._bbox_overlaps_any(bbox, table_bboxes):
                continue
            text_blocks.append(TextBlock(text=text.strip(), bbox=bbox))

        return PageContent(text_blocks=text_blocks, table_blocks=table_blocks)

    @staticmethod
    def _bbox_overlaps_any(
        bbox: tuple[float, float, float, float],
        others: list[tuple[float, float, float, float]],
        ratio_threshold: float = 0.5,
    ) -> bool:
        x0, y0, x1, y1 = bbox
        area = max((x1 - x0) * (y1 - y0), 1e-9)
        for ox0, oy0, ox1, oy1 in others:
            ix0, iy0 = max(x0, ox0), max(y0, oy0)
            ix1, iy1 = min(x1, ox1), min(y1, oy1)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            inter = (ix1 - ix0) * (iy1 - iy0)
            if inter / area >= ratio_threshold:
                return True
        return False

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
