import contextvars
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


def _create_handlers():
    log_dir = Path(settings.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = TimedRotatingFileHandler(
        filename=str(log_dir / "app.log"), when="midnight", interval=1, backupCount=settings.LOG_BACKUP_COUNT, encoding="utf-8"
    )
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
