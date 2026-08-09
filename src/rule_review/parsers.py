"""
电力规则审查系统 - PDF 解析器抽象（解析分层，见 docs/rule-review.md §4.3）

分层策略（与面试口径一致）:
- 相对不重要的文档 → 本地 MinerU 解析（表格可完整识别，优于 API 方案）
- 重要的政策问答表格 → 手动入库（DocumentStore.ingest_manual，识别准确率 100%）
- MinerU 未安装/不可用时静默降级 pymupdf，与 PaddleOCRProcessor 的降级语义一致

历史方案（已弃用，不写代码）:
- API 解析政策 PDF：成本高（PDF 数量增多后成本不可控）、表格重要内容无法完全准确识别。

所有解析器输出统一的 PageContent 列表，chunk 组装（标题/表格联合切块）
由 DocumentStore 复用同一套 _build_chunks 逻辑，保证分层不改变切块行为。
"""

from __future__ import annotations

import logging
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import fitz  # pymupdf
import numpy as np
from PIL import Image

from src.rule_review.models import PageContent, TableBlock, TextBlock

logger = logging.getLogger(__name__)

# 判定页面为扫描件的最大可识别字符数阈值（原 DocumentStore 常量迁移）
_SCANNED_TEXT_THRESHOLD = 50

# MinerU 整篇 markdown 中页与页之间的分隔符
_MARKDOWN_PAGE_SEPARATOR = "\n\n---\n\n"


# ---------------------------------------------------------------------------
# OCR 抽象（扫描 PDF 用，原 DocumentStore.OCRProcessor 迁移）
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
# PDF 解析器抽象（分层解析入口）
# ---------------------------------------------------------------------------


class PDFParser(ABC):
    """PDF 解析器抽象基类。

    镜像 OCRProcessor 的可用性降级模式：MinerU 未安装时 is_available=False，
    调用方（DocumentStore._select_parser）静默降级 pymupdf。
    """

    name: str = "abstract"  # 实际解析模式，写入 DocumentInfo.parse_mode

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """解析器是否可用（如 MinerU 未安装时 False）。"""

    @abstractmethod
    def parse(self, data: bytes, filename: str) -> list[PageContent]:
        """解析 PDF 字节内容，返回逐页 PageContent（供 chunk 组装复用）。"""


class PymupdfParser(PDFParser):
    """pymupdf 解析实现（原 DocumentStore._parse_document 逻辑迁移，行为零改动）。

    文本 PDF：page.get_text("blocks") + find_tables() 表格检测；
    扫描页且 OCR 可用：300DPI 转图 → OCR。
    """

    name = "pymupdf"

    def __init__(self, ocr_engine: OCRProcessor | None = None) -> None:
        # 每次 ingest 时由 DocumentStore 现构造（透传当时的 self.ocr_engine），
        # 保证测试中 store.ocr_engine = MockOCRProcessor(...) 的替换语义不变
        self.ocr_engine = ocr_engine

    @property
    def is_available(self) -> bool:
        return True

    def parse(self, data: bytes, filename: str) -> list[PageContent]:
        try:
            doc = fitz.open(stream=data, filetype="pdf")
        except Exception as exc:
            raise ValueError(f"PDF 解析失败: {exc}") from exc

        try:
            pages: list[PageContent] = []
            for page_number, page in enumerate(doc, start=1):
                is_scanned = self._is_scanned_page(page)
                if (
                    is_scanned
                    and self.ocr_engine is not None
                    and self.ocr_engine.is_available
                ):
                    page_content = self._ocr_page(page, page_number)
                    page_content.is_scanned = True
                else:
                    if is_scanned:
                        logger.warning(
                            f"第 {page_number} 页疑似扫描页但 OCR 不可用，已跳过"
                        )
                    page_content = self._extract_text_page(page, page_number)
                pages.append(page_content)
            return pages
        finally:
            doc.close()

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


class MinerULocalParser(PDFParser):
    """基于本地 MinerU 2.x 的 PDF 解析器（相对不重要的文档首选）。

    - import 延迟：未安装 mineru 时 is_available=False，调用方静默降级 pymupdf
      （与 PaddleOCRProcessor 同款降级模式）
    - 输出 markdown，经 markdown_to_page_content 还原为 PageContent，复用 chunk 组装
    - 实例缓存：避免每次 ingest 重新加载模型（参考 BGEM3Embedding 单例模式）
    """

    name = "mineru"

    _instances: dict[tuple, "MinerULocalParser"] = {}

    def __new__(
        cls,
        enabled: bool = True,
        method: str = "auto",
        output_dir: str | None = None,
    ) -> "MinerULocalParser":
        key = (enabled, method, output_dir)
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    def __init__(
        self,
        enabled: bool = True,
        method: str = "auto",
        output_dir: str | None = None,
    ) -> None:
        if hasattr(self, "_mineru"):  # 单例已初始化过
            return
        self._available = False
        self._mineru: Any | None = None
        if not enabled:
            return
        try:
            from mineru import MinerU  # 延迟导入：未安装时静默降级

            self._mineru = MinerU(
                method=method,
                output_dir=output_dir
                or f"{tempfile.gettempdir()}/dataquery_mineru",
            )
            self._available = True
        except Exception as exc:  # pragma: no cover - 未安装 mineru 时正常降级
            logger.warning(f"MinerU 初始化失败，将降级为 pymupdf 解析: {exc}")

    @property
    def is_available(self) -> bool:
        return self._available

    def parse(self, data: bytes, filename: str) -> list[PageContent]:
        if not self._available or self._mineru is None:
            raise RuntimeError("MinerU 不可用")

        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / (filename or "input.pdf")
            pdf_path.write_bytes(data)
            try:
                result = self._mineru(str(pdf_path))
            except Exception as exc:
                raise ValueError(f"MinerU 解析失败: {exc}") from exc
            page_markdowns = self._extract_page_markdowns(result)

        return [markdown_to_page_content(md) for md in page_markdowns]

    @staticmethod
    def _extract_page_markdowns(result: Any) -> list[str]:
        """从 MinerU 结果提取逐页 markdown，兼容不同小版本的结果对象差异。

        1) result.pages[i].markdown() 可用 → 逐页取（首选）；
        2) 整体 result.markdown() / result.markdown 属性 → 按页分隔符拆分；
        3) 均失败 → 视为单页整体 markdown。
        """
        try:
            pages = list(result.pages)
            mds = [p.markdown() for p in pages]
            if mds and any(mds):
                return mds
        except Exception as exc:
            logger.warning(f"MinerU 逐页 markdown 获取失败，按页分隔符拆分: {exc}")

        full = (
            result.markdown()
            if callable(getattr(result, "markdown", None))
            else getattr(result, "markdown", "")
        )
        return split_markdown_pages(full or "")


# ---------------------------------------------------------------------------
# Markdown → PageContent（MinerU 输出与手动入库共用）
# ---------------------------------------------------------------------------


def split_markdown_pages(markdown: str) -> list[str]:
    """按 MinerU 页分隔符拆分整篇 markdown；无分隔符时视为单页。"""
    parts = [p.strip() for p in markdown.split(_MARKDOWN_PAGE_SEPARATOR)]
    return [p for p in parts if p]


def markdown_to_page_content(markdown: str) -> PageContent:
    """单页 Markdown → PageContent。

    转换规则:
    - "# " / "## " / "### " 开头的行 → TextBlock（去掉 # 标记；「第一章 总则」交给
      DocumentStore._detect_heading 的章/条正则识别）
    - 连续以 "|" 开头的行 → TableBlock（跳过 |---| 分隔行；按 "|" 切分并 strip）
    - 其余非空行 → TextBlock（空行跳过）
    - bbox 用合成坐标 (0.0, y, 0.0, y+1.0)，y=行序——仅用于 _build_chunks 按 y 排序
      （markdown 无版面坐标，行序即阅读序）

    已知限制（见 docs/rule-review.md §4.3）:
    - 非「第X章/条」结构的 markdown 标题会退化为正文
    - 表格单元格内含 "|" 不做转义处理
    """
    text_blocks: list[TextBlock] = []
    table_blocks: list[TableBlock] = []

    lines = markdown.splitlines()
    y = 0.0
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()

        if line.startswith("|"):
            # 收集连续表格行
            table_lines: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                candidate = lines[i].strip()
                if not _is_table_separator(candidate):
                    table_lines.append(candidate)
                i += 1
            if table_lines:
                rows = [_split_table_row(l) for l in table_lines]
                table_blocks.append(
                    TableBlock(
                        rows=rows,
                        bbox=(0.0, y, 0.0, y + 1.0),
                    )
                )
                y += 1.0
            continue

        if line.startswith("#"):
            # Markdown 标题：去掉 # 标记，交给 _detect_heading 识别章/条结构
            text = raw.lstrip().lstrip("#").strip()
            if text:
                text_blocks.append(
                    TextBlock(text=text, bbox=(0.0, y, 0.0, y + 1.0))
                )
                y += 1.0
            i += 1
            continue

        if line:
            text_blocks.append(TextBlock(text=line, bbox=(0.0, y, 0.0, y + 1.0)))
            y += 1.0
        i += 1

    return PageContent(text_blocks=text_blocks, table_blocks=table_blocks)


def parse_markdown(markdown: str) -> list[PageContent]:
    """整篇 markdown → 逐页 PageContent（split_markdown_pages + markdown_to_page_content）。"""
    return [markdown_to_page_content(p) for p in split_markdown_pages(markdown)]


def _is_table_separator(line: str) -> bool:
    """判定 markdown 表格分隔行（如 |---|---|）。"""
    body = line.strip().strip("|")
    return bool(body) and all(c in "-: |" for c in body) and "-" in body


def _split_table_row(line: str) -> list[str]:
    """按 "|" 切分表格行并逐格 strip（空单元格保留 ""）。"""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]
