"""
电力规则审查系统 - 混合检索引擎

按设计文档 Phase 1 步骤 1.4 实现：
- QueryOptimizer: 术语标准化 + 同义词扩展 + 地名归一化
- BM25 关键词检索（bm25s）
- bge-m3 向量检索（通过 DocumentStore）
- RRF 融合排序
- Cross-Encoder 精排（可注入）
- 空检索兜底策略

不调用 LLM，核心逻辑为纯 Python。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.rule_review.document_store import Chunk, DocumentStore

logger = logging.getLogger(__name__)

# RRF 融合参数
_RRF_K = 60

# 空检索兜底：扩大搜索时的 top_k 倍数
_FALLBACK_K_MULTIPLIER = 3


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


@dataclass
class HybridSearchResult:
    """混合检索单条结果。"""

    chunk: Chunk
    score: float
    bm25_score: float = 0.0
    vector_score: float = 0.0
    rerank_score: float | None = None


@dataclass
class RetrieveResult:
    """检索完整结果。"""

    results: list[HybridSearchResult] = field(default_factory=list)
    not_found: bool = False
    search_expanded: bool = False
    expanded_query: str | None = None
    bm25_hits: int = 0
    vector_hits: int = 0
    fused_hits: int = 0


# ---------------------------------------------------------------------------
# Query 优化器
# ---------------------------------------------------------------------------


class QueryOptimizer:
    """查询优化：地名归一化 + 专业词汇归一化 + 同义词扩展。

    复用现有系统的 data_standard.json 地名库和 rule_terms.json 术语映射。
    """

    def __init__(
        self,
        term_config_path: str = "data/env_variables/rule_terms.json",
        data_standard_path: str = "data/env_variables/data_standard.json",
    ) -> None:
        """
        Args:
            term_config_path: 术语与实体别名映射配置。
            data_standard_path: 标准地名库（name_abbreviation）。
        """
        self.term_config = self._load_json(term_config_path)
        self.data_standard = self._load_json(data_standard_path)

        # 实体别名：alias -> standard
        self.entity_aliases: dict[str, str] = (
            self.term_config.get("entities", {}).get("aliases", {})
        )

        # 术语别名 + 同义词
        self.term_aliases: dict[str, str] = {}  # alias -> standard term
        self.term_synonyms: dict[str, list[str]] = {}  # standard term -> [synonyms]
        for term, info in self.term_config.get("terms", {}).items():
            for alias in info.get("aliases", []):
                self.term_aliases[alias] = term
            self.term_synonyms[term] = info.get("synonyms", [])

        # 按长度降序排列，优先匹配更长、更精确的别名
        self._entity_alias_sorted = sorted(
            self.entity_aliases.items(), key=lambda x: len(x[0]), reverse=True
        )
        self._term_alias_sorted = sorted(
            self.term_aliases.items(), key=lambda x: len(x[0]), reverse=True
        )

    @staticmethod
    def _load_json(path: str) -> Any:
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def normalize(self, query: str) -> str:
        """归一化单条 query：地名 + 术语替换为标准形式。

        使用位置记录 + 重叠过滤避免「日前」这类短别名与
        「日前现货出清电价」等长标准术语重叠匹配导致的二次替换。

        Args:
            query: 原始查询文本。

        Returns:
            归一化后的查询文本。
        """
        result = query

        # 1. 实体名归一化（地名）
        for alias, standard in self._entity_alias_sorted:
            if alias in result:
                result = result.replace(alias, standard)
                break  # 每个别名只替换一次

        # 2. 术语归一化：收集所有匹配位置，过滤重叠（长匹配优先）
        standard_terms = set(self.term_synonyms.keys())
        matches: list[tuple[int, int, str, str]] = []  # (start, end, alias, standard)

        for alias, standard in self._term_alias_sorted:
            if alias not in result:
                continue
            # 跳过已被标准术语覆盖的短别名（防止「日前」匹配在「日前现货出清电价」内部）
            if any(alias in term and alias != term
                   for term in standard_terms if term in result):
                continue
            start = 0
            while True:
                idx = result.find(alias, start)
                if idx == -1:
                    break
                matches.append((idx, idx + len(alias), alias, standard))
                start = idx + 1

        # 过滤重叠匹配：长匹配优先
        matches.sort(key=lambda x: (x[0], x[0] - x[1]))  # 按起始位置升序、长度降序
        filtered: list[tuple[int, int, str, str]] = []
        for m in matches:
            # 与已选匹配是否重叠
            if any(not (m[1] <= f[0] or m[0] >= f[1]) for f in filtered):
                continue
            filtered.append(m)

        # 从右到左应用替换，避免索引偏移
        for start, end, _alias, standard in sorted(filtered, key=lambda x: x[0], reverse=True):
            result = result[:start] + standard + result[end:]

        return result

    def optimize(self, query: str) -> list[str]:
        """优化 query，返回原始 + 变体列表。

        处理顺序:
        1. 先归一化 query
        2. 对归一化后的 query，用同义词替换生成变体

        Args:
            query: 原始查询文本。

        Returns:
            优化后的 query 列表（归一化后的 query + 最多 2 个同义词变体）。
        """
        normalized = self.normalize(query)
        variants = [normalized]

        # 生成同义词变体（最多 2 个）
        synonym_count = 0
        for term, synonyms in self.term_synonyms.items():
            if synonym_count >= 2:
                break
            if term in normalized and synonyms:
                for synonym in synonyms[:2]:
                    if synonym_count >= 2:
                        break
                    variant = normalized.replace(term, synonym)
                    if variant != normalized and variant not in variants:
                        variants.append(variant)
                        synonym_count += 1

        return variants


# ---------------------------------------------------------------------------
# BM25 检索器
# ---------------------------------------------------------------------------


class BM25Retriever:
    """基于 bm25s 的 BM25 关键词检索器。

    在 DocumentStore 的 chunk 集合上构建 BM25 索引，支持增量更新。
    """

    def __init__(self) -> None:
        self._bm25: Any = None
        self._corpus: list[str] = []
        self._chunk_ids: list[str] = []
        self._is_built = False

    @property
    def is_built(self) -> bool:
        return self._is_built

    def build_from_chunks(self, chunks: list[Chunk]) -> None:
        """从 chunk 列表构建 BM25 索引。

        Args:
            chunks: 需要索引的 chunk 列表。
        """
        if not chunks:
            self._bm25 = None
            self._corpus = []
            self._chunk_ids = []
            self._is_built = False
            return

        self._corpus = [c.text for c in chunks]
        self._chunk_ids = [c.chunk_id for c in chunks]

        try:
            import bm25s

            corpus_tokens = bm25s.tokenize(self._corpus)
            self._bm25 = bm25s.BM25()
            self._bm25.index(corpus_tokens)
            self._is_built = True
        except ImportError:
            logger.warning("bm25s 未安装，BM25 检索不可用")
            self._is_built = False

    def search(self, query: str, k: int = 30) -> list[tuple[str, float]]:
        """BM25 检索，返回 (chunk_id, score) 列表。

        Args:
            query: 查询文本。
            k: 返回结果数量（自动限制不超过语料库大小）。

        Returns:
            [(chunk_id, score), ...] 按分数降序排列。
        """
        if not self._is_built or not self._corpus:
            return []

        # 限制 k 不超过语料库大小
        k = min(k, len(self._corpus))

        try:
            import bm25s

            query_tokens = bm25s.tokenize([query])
            results, scores = self._bm25.retrieve(query_tokens, k=k)

            output: list[tuple[str, float]] = []
            for idx, score in zip(results[0], scores[0]):
                if idx < len(self._chunk_ids):
                    output.append((self._chunk_ids[idx], float(score)))
            return output
        except Exception as e:
            logger.warning(f"BM25 检索异常: {e}")
            return []

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """增量添加 chunk 到 BM25 索引（重建方式）。

        Args:
            chunks: 新增的 chunk 列表。
        """
        if not chunks:
            return
        all_chunks_texts = list(self._corpus)
        all_chunk_ids = list(self._chunk_ids)
        for c in chunks:
            if c.chunk_id not in all_chunk_ids:
                all_chunks_texts.append(c.text)
                all_chunk_ids.append(c.chunk_id)

        # 按 chunk_id 重建
        self._corpus = all_chunks_texts
        self._chunk_ids = all_chunk_ids

        try:
            import bm25s

            corpus_tokens = bm25s.tokenize(self._corpus)
            self._bm25 = bm25s.BM25()
            self._bm25.index(corpus_tokens)
            self._is_built = True
        except ImportError:
            self._is_built = False

    def remove_chunks(self, chunk_ids: set[str]) -> None:
        """从 BM25 索引中移除指定 chunk_ids（重建方式）。

        Args:
            chunk_ids: 需要移除的 chunk ID 集合。
        """
        if not chunk_ids:
            return

        remaining = [
            (cid, text)
            for cid, text in zip(self._chunk_ids, self._corpus)
            if cid not in chunk_ids
        ]
        self._chunk_ids = [cid for cid, _ in remaining]
        self._corpus = [text for _, text in remaining]

        if not self._corpus:
            self._bm25 = None
            self._is_built = False
            return

        try:
            import bm25s

            corpus_tokens = bm25s.tokenize(self._corpus)
            self._bm25 = bm25s.BM25()
            self._bm25.index(corpus_tokens)
            self._is_built = True
        except ImportError:
            self._is_built = False


# ---------------------------------------------------------------------------
# Cross-Encoder 抽象（可注入）
# ---------------------------------------------------------------------------


class CrossEncoderReranker:
    """Cross-Encoder 精排器。

    使用 BAAI/bge-reranker-v2-m3 对 (query, chunk) 对逐对打分。
    Phase 1 默认使用 FlagEmbedding 的 FlagReranker，也可注入 mock
    避免下载大模型。
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self.model_name = model_name
        self._reranker: Any = None
        self._available = False
        self._init_reranker()

    def _init_reranker(self) -> None:
        try:
            from FlagEmbedding import FlagReranker

            self._reranker = FlagReranker(self.model_name, use_fp16=True)
            self._available = True
        except ImportError:
            logger.warning(
                "FlagEmbedding 未安装，Cross-Encoder 精排不可用。"
                "安装方式: pip install FlagEmbedding"
            )
            self._available = False
        except Exception as e:
            logger.warning(f"Cross-Encoder 初始化失败: {e}")
            self._available = False

    @property
    def is_available(self) -> bool:
        return getattr(self, "_available", False)

    def rerank(
        self, query: str, documents: list[str]
    ) -> list[float]:
        """对 (query, doc) 列表逐对打分。

        Args:
            query: 查询文本。
            documents: 候选文档文本列表。

        Returns:
            每个文档的分数列表（越高越相关），与 documents 同序。
        """
        if not self._available or not documents:
            return [0.0] * len(documents)

        try:
            pairs = [[query, doc] for doc in documents]
            scores = self._reranker.compute_score(pairs)
            if isinstance(scores, float):
                scores = [scores]
            return [float(s) for s in scores]
        except Exception as e:
            logger.warning(f"Cross-Encoder 精排异常: {e}")
            return [0.0] * len(documents)


# ---------------------------------------------------------------------------
# RRF 融合
# ---------------------------------------------------------------------------


def rrf_fusion(
    result_sets: list[list[tuple[str, float]]],
    k: int = _RRF_K,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion (RRF) 多路召回融合。

    对每一路结果集中的每个结果，按其在该集中的排名计算 RRF 分数，
    最后按总分降序排列。

    Args:
        result_sets: 多路召回结果，每路为 [(id, score), ...] 按分数降序。
        k: RRF 平滑参数，默认 60。

    Returns:
        [(id, fused_score), ...] 按融合分数降序排列。
    """
    scores: dict[str, float] = {}
    for results in result_sets:
        for rank, (doc_id, _) in enumerate(results):
            rrf_score = 1.0 / (k + rank + 1)
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score

    # 按分数降序排列
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_items


# ---------------------------------------------------------------------------
# 混合检索引擎
# ---------------------------------------------------------------------------


class HybridRetriever:
    """混合检索引擎：BM25 + 向量检索 → RRF 融合 → Cross-Encoder 精排。

    与 DocumentStore 协同工作，BM25 索引从 DocumentStore 的 chunk 构建。
    向量检索直接委托给 DocumentStore.vector_search()。
    """

    def __init__(
        self,
        document_store: DocumentStore,
        query_optimizer: QueryOptimizer | None = None,
        reranker: CrossEncoderReranker | None = None,
        bm25_retriever: BM25Retriever | None = None,
        embedding_fn: Callable[[list[str]], np.ndarray] | None = None,
    ) -> None:
        """
        Args:
            document_store: 文档存储实例，提供向量检索和 chunk 访问。
            query_optimizer: Query 优化器。None 时自动创建默认实例。
            reranker: Cross-Encoder 精排器。None 时自动创建默认实例。
            bm25_retriever: BM25 检索器。None 时自动创建新实例。
            embedding_fn: embedding 函数，用于将 query 文本转为向量。
                          应与 DocumentStore 使用相同的 embedding 模型。
                          None 时尝试复用 DocumentStore 的 embedding_fn。
        """
        self.doc_store = document_store
        self.query_optimizer = query_optimizer or QueryOptimizer()
        self.reranker = reranker or CrossEncoderReranker()
        self.bm25 = bm25_retriever or BM25Retriever()
        self.embedding_fn = embedding_fn or document_store.embedding_fn

        # 初始化时从 DocumentStore 构建 BM25 索引
        self._sync_bm25()

    def _sync_bm25(self) -> None:
        """从 DocumentStore 的当前 chunk 集合重建 BM25 索引。"""
        chunks = list(self.doc_store._chunks.values())
        self.bm25.build_from_chunks(chunks)

    def _embed_query(self, query: str) -> np.ndarray | None:
        """将查询文本转为向量。

        Args:
            query: 查询文本。

        Returns:
            查询向量，shape 为 (dim,)；embedding_fn 不可用时返回 None。
        """
        if self.embedding_fn is None:
            return None
        try:
            vec = self.embedding_fn([query])
            if isinstance(vec, np.ndarray) and vec.ndim == 2:
                vec = vec[0]
            return np.asarray(vec, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Query embedding 生成失败: {e}")
            return None

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        doc_filter: str | None = None,
        bm25_k: int = 30,
        vector_k: int = 30,
        fusion_k: int = 20,
    ) -> RetrieveResult:
        """混合检索主入口。

        流程:
        1. QueryOptimizer 优化 query → 变体列表
        2. 对每个变体：BM25 + 向量检索 并行召回
        3. RRF 融合所有变体的结果 → top fusion_k
        4. Cross-Encoder 精排 → top_k
        5. 返回 RetrieveResult

        Args:
            query: 查询文本。
            top_k: 最终返回的 chunk 数量。
            doc_filter: 仅检索指定文档（doc_id）。
            bm25_k: BM25 每路召回数量。
            vector_k: 向量检索每路召回数量。
            fusion_k: RRF 融合后保留数量。

        Returns:
            RetrieveResult，包含最终检索结果和元信息。
        """
        if not self.doc_store._chunks:
            return RetrieveResult(not_found=True)

        # Step 1: Query 优化 → 变体
        query_variants = self.query_optimizer.optimize(query)

        # Step 2: 对每个变体分别召回
        all_bm25_results: list[list[tuple[str, float]]] = []
        all_vector_results: list[list[tuple[str, float]]] = []

        for variant in query_variants:
            # BM25 召回
            bm25_hits = self.bm25.search(variant, k=bm25_k)
            if bm25_hits:
                all_bm25_results.append(bm25_hits)

            # 向量召回
            query_vec = self._embed_query(variant)
            if query_vec is not None:
                vector_hits = self.doc_store.vector_search(
                    query_vec, top_k=vector_k, doc_filter=doc_filter
                )
                if vector_hits:
                    all_vector_results.append(
                        [(r.chunk.chunk_id, r.score) for r in vector_hits]
                    )

        bm25_hits = sum(len(r) for r in all_bm25_results)
        vector_hits = sum(len(r) for r in all_vector_results)

        # Step 3: RRF 融合
        all_sets = all_bm25_results + all_vector_results
        if not all_sets:
            return RetrieveResult(
                not_found=True, bm25_hits=0, vector_hits=0, fused_hits=0
            )

        fused = rrf_fusion(all_sets, k=_RRF_K)
        fused = fused[:fusion_k]

        # 收集每个 chunk 的 BM25 和向量最高分
        bm25_best: dict[str, float] = {}
        for results in all_bm25_results:
            for cid, score in results:
                if cid not in bm25_best or score > bm25_best[cid]:
                    bm25_best[cid] = score

        vector_best: dict[str, float] = {}
        for results in all_vector_results:
            for cid, score in results:
                if cid not in vector_best or score > vector_best[cid]:
                    vector_best[cid] = score

        # Step 4: Cross-Encoder 精排（可选）
        chunk_texts = []
        chunk_ids_in_order = []
        for cid, _ in fused:
            chunk = self.doc_store.get_chunk(cid)
            if chunk is not None:
                chunk_texts.append(chunk.text)
                chunk_ids_in_order.append(cid)

        rerank_scores: list[float | None] = [None] * len(chunk_ids_in_order)
        if self.reranker.is_available and chunk_texts:
            try:
                rerank_scores = [
                    float(s) for s in self.reranker.rerank(query, chunk_texts)
                ]
            except Exception as e:
                logger.warning(f"精排异常，跳过: {e}")

        # 组装结果
        results: list[HybridSearchResult] = []
        for i, cid in enumerate(chunk_ids_in_order):
            chunk = self.doc_store.get_chunk(cid)
            if chunk is None:
                continue

            fused_score = fused[i][1] if i < len(fused) else 0.0

            result = HybridSearchResult(
                chunk=chunk,
                score=fused_score,
                bm25_score=bm25_best.get(cid, 0.0),
                vector_score=vector_best.get(cid, 0.0),
                rerank_score=rerank_scores[i] if i < len(rerank_scores) else None,
            )

            # 精排分数可用时，用它作为主分排序依据
            if result.rerank_score is not None:
                result.score = result.rerank_score

            results.append(result)

        # 按最终分数排序
        results.sort(key=lambda x: x.score, reverse=True)
        results = results[:top_k]

        return RetrieveResult(
            results=results,
            not_found=len(results) == 0,
            bm25_hits=bm25_hits,
            vector_hits=vector_hits,
            fused_hits=len(fused),
        )

    def retrieve_with_fallback(
        self,
        query: str,
        top_k: int = 10,
        doc_filter: str | None = None,
    ) -> RetrieveResult:
        """带空检索兜底的混合检索。

        先执行正常检索；如果无结果，依次尝试扩大搜索策略；
        如果仍无结果，返回 not_found=True。

        扩大搜索策略:
        1. 使用原始 query（不经 QueryOptimizer 优化）
        2. 扩大 top_k（vector_search 中的倍增策略）

        Args:
            query: 查询文本。
            top_k: 最终返回的 chunk 数量。
            doc_filter: 仅检索指定文档（doc_id）。

        Returns:
            RetrieveResult，包含最终结果和兜底标记。
        """
        # 第一次：正常检索
        result = self.retrieve(query, top_k=top_k, doc_filter=doc_filter)
        if not result.not_found:
            return result

        # 兜底 1：使用原始 query 直接检索（不经优化）
        logger.info(f"正常检索无结果，尝试原始 query 兜底: {query[:50]}...")
        query_vec = self._embed_query(query)
        if query_vec is not None:
            vector_hits = self.doc_store.vector_search(
                query_vec, top_k=top_k * _FALLBACK_K_MULTIPLIER, doc_filter=doc_filter
            )
            if vector_hits:
                results: list[HybridSearchResult] = []
                for r in vector_hits[:top_k]:
                    results.append(
                        HybridSearchResult(
                            chunk=r.chunk,
                            score=r.score,
                            vector_score=r.score,
                        )
                    )
                return RetrieveResult(
                    results=results,
                    search_expanded=True,
                    expanded_query=query,
                    vector_hits=len(vector_hits),
                )

        # 兜底 2：BM25 原始 query
        bm25_hits = self.bm25.search(query, k=top_k * _FALLBACK_K_MULTIPLIER)
        if bm25_hits:
            results: list[HybridSearchResult] = []
            for cid, score in bm25_hits[:top_k]:
                chunk = self.doc_store.get_chunk(cid)
                if chunk is not None:
                    results.append(
                        HybridSearchResult(
                            chunk=chunk,
                            score=float(score),
                            bm25_score=float(score),
                        )
                    )
            if results:
                return RetrieveResult(
                    results=results,
                    search_expanded=True,
                    expanded_query=query,
                    bm25_hits=len(bm25_hits),
                )

        # 全部失败：返回空
        return RetrieveResult(not_found=True, search_expanded=True)

    def refresh_bm25(self) -> None:
        """手动刷新 BM25 索引（例如在 DocumentStore 增删文档后调用）。"""
        self._sync_bm25()


# ---------------------------------------------------------------------------
# 便捷工厂
# ---------------------------------------------------------------------------


def get_default_retriever(
    document_store: DocumentStore | None = None,
) -> HybridRetriever:
    """获取使用默认配置的 HybridRetriever。

    Args:
        document_store: DocumentStore 实例。None 时创建新的内存实例。

    Returns:
        配置完成的 HybridRetriever。
    """
    if document_store is None:
        document_store = DocumentStore()
    return HybridRetriever(document_store=document_store)
