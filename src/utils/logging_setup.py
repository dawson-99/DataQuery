import contextvars
import json
import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from src.config import settings

# ---------- 请求链路追踪 ----------
trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")


def set_trace_id(trace_id: str) -> None:
    trace_id_var.set(trace_id)


def get_trace_id() -> str:
    return trace_id_var.get()


class TraceIdFilter(logging.Filter):
    """将当前协程的 trace_id 注入每条日志记录"""
    def filter(self, record):
        record.trace_id = trace_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """结构化 JSON 日志格式：{ts, level, name, trace_id, msg, **extra}

    通过 extra 传入的结构化字段（stage/latency_ms/query_id 等）会并入 JSON，
    便于可观测性平台直接解析。
    """

    _EXTRA_KEYS = ("stage", "latency_ms", "query_id")

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "name": record.name,
            "trace_id": getattr(record, "trace_id", "-"),
            "msg": record.getMessage(),
        }
        for key in self._EXTRA_KEYS:
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload, ensure_ascii=False)


def _create_handlers():
    log_dir = Path(settings.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = TimedRotatingFileHandler(
        filename=str(log_dir / "app.log"), when="midnight", interval=1, backupCount=settings.LOG_BACKUP_COUNT, encoding="utf-8"
    )
    if settings.LOG_FORMAT == "json":
        formatter = JsonFormatter()  # type: ignore[assignment]
    else:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s [%(trace_id)s] %(message)s")
    file_handler.setFormatter(formatter)
    file_handler.addFilter(TraceIdFilter())
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(TraceIdFilter())
    file_handler.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
    stream_handler.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
    return file_handler, stream_handler


def setup_logging():
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    file_handler, stream_handler = _create_handlers()
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(stream_handler)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.setLevel(level)
        lg.handlers = []
        lg.propagate = True


setup_logging()

logger = logging.getLogger("app")
