"""
电力规则审查系统 - 包初始化

提供全局可访问的版本号，并在模块加载时确保规则审查所需的数据目录存在。
"""

import os

from src.config import settings

__version__ = "0.1.0"

# 暴露核心模块，便于 pipeline 与测试使用
from src.rule_review.document_store import DocumentStore
from src.rule_review.generator import RuleReviewGenerator, get_default_generator
from src.rule_review.judge import RuleReviewJudge, get_default_judge
from src.rule_review.pipeline import RuleReviewPipeline, get_default_pipeline
from src.rule_review.query_rewriter import QueryRewriter, get_default_rewriter
from src.rule_review.retriever import (
    HybridRetriever,
    SparseRetriever,
    get_default_retriever,
)
from src.rule_review.tool_executor import ToolExecutor, execute_with_tool_loop

__all__ = [
    "QueryRewriter",
    "get_default_rewriter",
    "DocumentStore",
    "HybridRetriever",
    "SparseRetriever",
    "get_default_retriever",
    "RuleReviewGenerator",
    "get_default_generator",
    "RuleReviewJudge",
    "get_default_judge",
    "RuleReviewPipeline",
    "get_default_pipeline",
    "ToolExecutor",
    "execute_with_tool_loop",
    "__version__",
]


def _ensure_dirs() -> None:
    """确保规则审查子系统所需的数据目录存在。"""
    dirs = [settings.RULE_DOCUMENTS_DIR, settings.RULE_INDEX_DIR]
    for d in dirs:
        if d:
            os.makedirs(d, exist_ok=True)


_ensure_dirs()
