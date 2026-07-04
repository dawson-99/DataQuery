"""
电力规则审查系统 - 包初始化

提供全局可访问的版本号，并在模块加载时确保规则审查所需的数据目录存在。
"""

import os

from src.config import settings

__version__ = "0.1.0"

# 暴露问题改写器与文档存储，便于 pipeline 与测试使用
from src.rule_review.document_store import DocumentStore
from src.rule_review.query_rewriter import QueryRewriter, get_default_rewriter

__all__ = ["QueryRewriter", "get_default_rewriter", "DocumentStore", "__version__"]


def _ensure_dirs() -> None:
    """确保规则审查子系统所需的数据目录存在。"""
    dirs = [settings.RULE_DOCUMENTS_DIR, settings.RULE_INDEX_DIR]
    for d in dirs:
        if d:
            os.makedirs(d, exist_ok=True)


_ensure_dirs()
