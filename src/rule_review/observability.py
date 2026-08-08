"""
规则审查系统 - 可观测性模块

提供阶段延迟分位数统计：
- LatencyStats：按阶段记录耗时样本，计算 P50/P95/均值，支持 JSON 持久化
- 供 pipeline 各阶段计时与 GET /v1/rule-review/observability/latency 端点使用

设计原则：统计失败绝不阻断主流程（写盘失败仅告警）。
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_STATS_PATH = "data/observability/latency_stats.json"


class LatencyStats:
    """按阶段记录耗时样本，计算分位数（线程安全）。

    样本按阶段名分组保存；percentile/avg/count 在无样本时返回 0.0/0，
    避免调用方做空值判断。
    """

    def __init__(self, path: str | None = None):
        self._path = Path(path) if path else None
        self._lock = threading.Lock()
        self._samples: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # 记录
    # ------------------------------------------------------------------

    def record(self, stage: str, latency_ms: float) -> None:
        """记录单个阶段耗时样本（毫秒）。"""
        with self._lock:
            self._samples.setdefault(stage, []).append(latency_ms)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def percentile(self, stage: str, p: float = 95.0) -> float:
        """计算指定阶段 p 分位数（毫秒，线性插值）。无样本返回 0.0。"""
        with self._lock:
            samples = sorted(self._samples.get(stage, []))
        if not samples:
            return 0.0
        if p <= 0:
            return samples[0]
        if p >= 100:
            return samples[-1]
        idx = (len(samples) - 1) * p / 100.0
        lo = int(idx)
        hi = min(lo + 1, len(samples) - 1)
        frac = idx - lo
        return round(samples[lo] * (1 - frac) + samples[hi] * frac, 2)

    def avg(self, stage: str) -> float:
        """指定阶段平均耗时（毫秒）。无样本返回 0.0。"""
        with self._lock:
            samples = self._samples.get(stage, [])
        if not samples:
            return 0.0
        return round(sum(samples) / len(samples), 2)

    def count(self, stage: str) -> int:
        """指定阶段样本数。"""
        with self._lock:
            return len(self._samples.get(stage, []))

    def summary(self) -> dict[str, dict[str, float]]:
        """返回各阶段统计：count / avg_ms / p50_ms / p95_ms。"""
        with self._lock:
            stages = list(self._samples.keys())
        out: dict[str, dict[str, float]] = {}
        for stage in stages:
            out[stage] = {
                "count": self.count(stage),
                "avg_ms": self.avg(stage),
                "p50_ms": self.percentile(stage, 50),
                "p95_ms": self.percentile(stage, 95),
            }
        return out

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, list[float]]:
        """导出全部原始样本（供持久化/测试）。"""
        with self._lock:
            return {k: list(v) for k, v in self._samples.items()}

    def from_dict(self, data: dict[str, list[float]]) -> None:
        """从字典导入样本（覆盖式）。"""
        with self._lock:
            self._samples = {k: [float(x) for x in v] for k, v in data.items()}

    def save(self, path: str | None = None) -> str | None:
        """保存到 JSON 文件。目录不存在自动创建；失败仅告警不影响主流程。"""
        p = Path(path) if path else self._path
        if p is None:
            return None
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(
                json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return str(p)
        except OSError as e:
            logger.warning("[observability] 延迟统计保存失败: %s", e)
            return None

    def load(self, path: str | None = None) -> bool:
        """从 JSON 文件加载样本。文件不存在或损坏返回 False。"""
        p = Path(path) if path else self._path
        if p is None or not p.exists():
            return False
        try:
            data: Any = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self.from_dict(data)
                return True
            return False
        except (OSError, ValueError) as e:
            logger.warning("[observability] 延迟统计加载失败: %s", e)
            return False


# ---------------------------------------------------------------------------
# 默认工厂（单例）
# ---------------------------------------------------------------------------


_default_stats: LatencyStats | None = None


def get_default_stats() -> LatencyStats:
    """获取默认延迟统计单例（惰性创建，测试可注入覆盖）。"""
    global _default_stats
    if _default_stats is None:
        _default_stats = LatencyStats()
    return _default_stats
