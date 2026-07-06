"""
智能数据查询系统 - 配置模块（并发优化版）
支持环境变量加载，包含服务器、模型、API、缓存及并发调优参数
"""

import os
from pathlib import Path
from typing import List, Optional

# 获取项目根目录（相对于本文件的绝对路径）
BASE_DIR = Path(__file__).parent.parent


def load_env(path: str = ".env") -> None:
    """
    加载环境变量文件，跳过已存在的环境变量（系统环境变量优先）
    """
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_env()


class Settings:
    """
    全局配置单例类
    所有配置项均从环境变量读取，提供合理的默认值
    """

    def __init__(self) -> None:
        # ---------- 服务器配置 ----------
        self.SERVER_HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
        self.SERVER_PORT: int = int(os.getenv("SERVER_PORT", "6060"))
        self.DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")

        # ---------- Uvicorn 生产部署配置 ----------
        self.UVICORN_WORKERS: int = int(os.getenv("UVICORN_WORKERS", "1"))
        self.MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "1000"))
        self.BACKLOG: int = int(os.getenv("BACKLOG", "2048"))
        self.TIMEOUT_KEEP_ALIVE: int = int(os.getenv("TIMEOUT_KEEP_ALIVE", "5"))
        # 每个 worker 处理 N 个请求后自动重启，防止慢内存泄漏累积
        self.UVICORN_LIMIT_MAX_REQUESTS: int = int(os.getenv("UVICORN_LIMIT_MAX_REQUESTS", "5000"))
        # 优雅关闭：收到 SIGTERM 后等待现有请求完成的最长时间（秒）
        self.UVICORN_GRACEFUL_SHUTDOWN_TIMEOUT: int = int(os.getenv("UVICORN_GRACEFUL_SHUTDOWN_TIMEOUT", "30"))

        # ---------- 请求处理超时配置 ----------
        self.REQUEST_TIMEOUT_SECONDS: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))
        self.STREAM_CHUNK_TIMEOUT: int = int(os.getenv("STREAM_CHUNK_TIMEOUT", "30"))
        self.MAX_SUBQUESTION_CONCURRENCY: int = int(os.getenv("MAX_SUBQUESTION_CONCURRENCY", "4"))
        self.MAX_SHARED_SUBQUESTION_CONCURRENCY: int = int(
            os.getenv("MAX_SHARED_SUBQUESTION_CONCURRENCY", "2")
        )

        # ---------- 日志配置 ----------
        self.LOG_DIR: str = os.getenv("LOG_DIR", "logs")
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_BACKUP_COUNT: int = int(os.getenv("LOG_BACKUP_COUNT", "30"))
        self.LOG_QUEUE_SIZE: int = int(os.getenv("LOG_QUEUE_SIZE", "10000"))  # 异步日志队列大小

        # ---------- 缓存配置 ----------
        self.CACHE_EXPIRY_HOURS: int = int(os.getenv("CACHE_EXPIRY_HOURS", "1"))
        self.CACHE_CLEANUP_INTERVAL_SECONDS: int = int(os.getenv("CACHE_CLEANUP_INTERVAL_SECONDS", "300"))

        # ---------- 连接池与重试配置 ----------
        self.HTTP_CLIENT_TIMEOUT: int = int(os.getenv("HTTP_CLIENT_TIMEOUT", "30"))
        self.HTTP_CLIENT_MAX_CONNECTIONS: int = int(os.getenv("HTTP_CLIENT_MAX_CONNECTIONS", "100"))
        self.HTTP_CLIENT_MAX_KEEPALIVE: int = int(os.getenv("HTTP_CLIENT_MAX_KEEPALIVE", "20"))
        self.API_RETRY_COUNT: int = int(os.getenv("API_RETRY_COUNT", "3"))
        self.API_RETRY_BACKOFF_FACTOR: float = float(os.getenv("API_RETRY_BACKOFF_FACTOR", "0.5"))

        # ---------- 默认LLM配置（作为后备）----------
        self.DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")
        self.DASHSCOPE_API_BASE: str = os.getenv("DASHSCOPE_API_BASE", "")

        self.INNET_API_BASE: str = os.getenv("INNET_API_BASE", "")
        self.INNET_API_KEY: str = os.getenv("INNET_API_KEY", "")

        # ---------- 意图识别模型配置 ----------
        self.INTENT_MODEL: str = os.getenv("INTENT_MODEL", "qwen3-max")
        self.INTENT_API_KEY: str = os.getenv("INTENT_API_KEY", self.DASHSCOPE_API_KEY)
        self.INTENT_API_BASE: str = os.getenv("INTENT_API_BASE", self.DASHSCOPE_API_BASE)

        # ---------- 参数提取模型配置 ----------
        self.PARAMETER_MODEL: str = os.getenv("PARAMETER_MODEL", "qwen3-max")
        self.PARAMETER_API_KEY: str = os.getenv("PARAMETER_API_KEY", self.DASHSCOPE_API_KEY)
        self.PARAMETER_API_BASE: str = os.getenv("PARAMETER_API_BASE", self.DASHSCOPE_API_BASE)

        # ---------- 格式化输出模型配置 ----------
        self.FORMAT_MODEL: str = os.getenv("FORMAT_MODEL", "qwen3-32b")
        self.FORMAT_API_KEY: str = os.getenv("FORMAT_API_KEY", self.DASHSCOPE_API_KEY)
        self.FORMAT_API_BASE: str = os.getenv("FORMAT_API_BASE", self.DASHSCOPE_API_BASE)
        # 基于序列化后数据长度的保守阈值，用于判断是否继续走格式化总结阶段。
        # 当前项目里格式化阶段使用的是截断前 processed_data：
        # - 记录数展示上限是 24 条
        # - 时点类查询可能包含最多 96 个 vXXXX 字段
        # 这类全量明细一旦超过约 2.8w 字符，就容易逼近 qwen3-32b 在该阶段可稳定消化的上下文空间，
        # 导致格式化总结基于不完整数据或输出不稳定。因此默认阈值收紧到 28000，
        # 小中型结果仍走格式化总结，大结果直接转趋势分析总结。
        self.FORMAT_SUMMARY_MAX_DATA_CHARS: int = int(os.getenv("FORMAT_SUMMARY_MAX_DATA_CHARS", "28000"))

        # ---------- 问题拆分模型配置 ----------
        self.PROBLEM_MODEL: str = os.getenv("PROBLEM_MODEL", "qwen3-max")
        self.PROBLEM_API_KEY: str = os.getenv("PROBLEM_API_KEY", self.DASHSCOPE_API_KEY)
        self.PROBLEM_API_BASE: str = os.getenv("PROBLEM_API_BASE", self.DASHSCOPE_API_BASE)

        # ---------- ECharts 输出模型配置 ----------
        self.ECHARTS_MODEL: str = os.getenv("ECHARTS_MODEL", "qwen3-32b")
        self.ECHARTS_API_KEY: str = os.getenv("ECHARTS_API_KEY", self.DASHSCOPE_API_KEY)
        self.ECHARTS_API_BASE: str = os.getenv("ECHARTS_API_BASE",self.DASHSCOPE_API_BASE)

        # ---------- 数据分析总结模型配置 ----------
        self.TREND_ANALYSIS_MODEL: str = os.getenv("TREND_ANALYSIS_MODEL", "qwen3-32b")
        self.TREND_ANALYSIS_API_KEY: str = os.getenv("TREND_ANALYSIS_API_KEY", self.DASHSCOPE_API_KEY)
        self.TREND_ANALYSIS_API_BASE: str = os.getenv("TREND_ANALYSIS_API_BASE", self.DASHSCOPE_API_BASE)

        # 兼容旧配置：如果设置了LLM_MODEL，则作为默认值
        self.LLM_MODEL: str = os.getenv("LLM_MODEL", self.INTENT_MODEL)

        # ====== 规则审查系统配置 ======
        self.RULE_REVIEW_MODEL: str = os.getenv("RULE_REVIEW_MODEL", "qwen3-max")
        self.RULE_REVIEW_API_KEY: str = os.getenv("RULE_REVIEW_API_KEY", self.DASHSCOPE_API_KEY)
        self.RULE_REVIEW_API_BASE: str = os.getenv("RULE_REVIEW_API_BASE", self.DASHSCOPE_API_BASE)

        self.JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "deepseek-v4")
        self.JUDGE_API_KEY: str = os.getenv("JUDGE_API_KEY", "")
        self.JUDGE_API_BASE: str = os.getenv("JUDGE_API_BASE", "")

        self.EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

        self.RULE_DOCUMENTS_DIR: str = os.getenv("RULE_DOCUMENTS_DIR", "data/rule_documents")
        self.RULE_INDEX_DIR: str = os.getenv("RULE_INDEX_DIR", "data/rule_index")

        # ====== Redis 缓存配置（Phase 3）======
        self.REDIS_URL: str = os.getenv("REDIS_URL", "")
        self.REDIS_CACHE_TTL: int = int(os.getenv("REDIS_CACHE_TTL", "300"))
        self.REDIS_RETRIEVAL_TTL: int = int(os.getenv("REDIS_RETRIEVAL_TTL", "600"))

        # ====== vLLM 本地模型配置（Phase 3）======
        self.VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "")
        self.VLLM_MODEL_NAME: str = os.getenv("VLLM_MODEL_NAME", "")
        self.VLLM_API_KEY: str = os.getenv("VLLM_API_KEY", "not-needed")

        # ====== Milvus 向量数据库配置（Phase 4）======
        self.MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
        self.MILVUS_PORT: int = int(os.getenv("MILVUS_PORT", "19530"))
        self.MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION", "rule_documents")

        # ====== PostgreSQL 数据库配置（Phase 4）======
        self.PG_HOST: str = os.getenv("PG_HOST", "localhost")
        self.PG_PORT: int = int(os.getenv("PG_PORT", "5432"))
        self.PG_USER: str = os.getenv("PG_USER", "dataquery")
        self.PG_PASSWORD: str = os.getenv("PG_PASSWORD", "dataquery")
        self.PG_DATABASE: str = os.getenv("PG_DATABASE", "rule_review")

        # 模型请求中转服务配置
        self.GATEWAY_BASE_URL = os.getenv("GATEWAY_BASE_URL", "")
        # 走中转服务的模型名称列表（由环境变量逗号分隔）
        self.GATEWAY_MODELS = [
            m.strip() for m in os.getenv("GATEWAY_MODELS", "").split(",") if m.strip()
        ]

        # ---------- 表格配置 ----------
        self.TABLES_BASE_DIR: str = os.getenv("TABLES_BASE_DIR", str(BASE_DIR))

        # ---------- 业务API端点配置 ----------
        self.EMERGENCY_DAYAHEAD_API_URL: str = os.getenv("EMERGENCY_DAYAHEAD_API_URL", "")
        self.COMMON_API_URL: str = os.getenv("COMMON_API_URL", "")

        # ---------- CORS配置 ----------
        cors_origins: str = os.getenv("CORS_ALLOW_ORIGINS", "*")
        self.CORS_ALLOW_ORIGINS: List[str] = (
            [o.strip() for o in cors_origins.split(",") if o.strip()] if cors_origins else ["*"]
        )

        # ---------- 静态文件配置 ----------
        self.STATIC_DIR: str = os.getenv("STATIC_DIR", "frontend")
        self.STATIC_URL_PREFIX: str = os.getenv("STATIC_URL_PREFIX", "/static")

        # ---------- 会话配置 ----------
        self.MAX_SESSIONS: int = int(os.getenv("MAX_SESSIONS", "1000"))
        self.SESSION_TIMEOUT: int = int(os.getenv("SESSION_TIMEOUT", "3600"))
        self.MAX_CONTEXT_CHARS: int = int(os.getenv("MAX_CONTEXT_CHARS", "100000"))

        #人工智能平台模型调用配置
        self.INNER_MODEL_ENABLE: str = os.getenv("INNER_MODEL_ENABLE", "")
        self.INNER_MODEL_TIMEOUT: int = int(os.getenv("INNER_MODEL_TIMEOUT", "300"))
        self.INNER_MODEL_URL: str = os.getenv("INNER_MODEL_URL", "")
        self.INNER_MODEL_AUTH_TOKEN: str = os.getenv("INNER_MODEL_AUTH_TOKEN", "")
        self.INNER_MODEL: str = os.getenv("INNER_MODEL", "")


# 单例实例
settings = Settings()
