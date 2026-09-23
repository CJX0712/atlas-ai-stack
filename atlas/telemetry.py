"""Structured telemetry: JSONL logs plus in-process counters and timings.

Zero third-party dependency. ``loguru`` is used only if it happens to be
installed; otherwise a stdlib fallback keeps the exact same call surface.

Author: 晨星
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

try:  # optional accelerator, never required
    from loguru import logger as _loguru_logger

    _HAS_LOGURU = True
except Exception:  # pragma: no cover - fallback path
    _loguru_logger = None
    _HAS_LOGURU = False

_log = logging.getLogger("atlas")
if not _log.hasHandlers():
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s :: %(message)s")
    )
    _log.addHandler(_handler)
_log.setLevel(os.environ.get("ATLAS_LOG_LEVEL", "INFO").upper())


def configure(level: str = "INFO") -> None:
    """Set verbosity for both the loguru and the stdlib backends."""
    _log.setLevel(level.upper())
    global _HAS_LOGURU
    if _HAS_LOGURU and _loguru_logger is not None:
        try:
            _loguru_logger.remove()
            _loguru_logger.add(
                sys.stderr,
                level=level.upper(),
                format=(
                    "<green>{time:HH:mm:ss.SSS}</green> "
                    "<level>{level: <7}</level> <cyan>{message}</cyan>"
                ),
            )
        except Exception:  # pragma: no cover
            _HAS_LOGURU = False


def log_event(event: str, **fields: Any) -> None:
    """Emit one structured log line (JSON payload in the message body)."""
    payload = {"event": event, **fields}
    line = json.dumps(payload, ensure_ascii=False, default=str)
    if _HAS_LOGURU and _loguru_logger is not None:
        try:
            _loguru_logger.info(line)
            return
        except Exception:  # pragma: no cover
            pass
    _log.info(line)


@dataclass
class Counter:
    """Named monotone counters."""

    values: dict[str, float] = field(default_factory=dict)

    def inc(self, key: str, amount: float = 1.0) -> None:
        self.values[key] = self.values.get(key, 0.0) + amount

    def get(self, key: str) -> float:
        return self.values.get(key, 0.0)

    def snapshot(self) -> dict[str, float]:
        return dict(self.values)


@dataclass
class Latencies:
    """Bounded latency reservoir per operation name."""

    capacity: int = 512
    samples: dict[str, list[float]] = field(default_factory=dict)

    def observe(self, key: str, ms: float) -> None:
        bucket = self.samples.setdefault(key, [])
        bucket.append(ms)
        if len(bucket) > self.capacity:
            del bucket[: len(bucket) - self.capacity]

    def percentile(self, key: str, q: float) -> float:
        bucket = self.samples.get(key) or []
        if not bucket:
            return 0.0
        ordered = sorted(bucket)
        if len(ordered) == 1:
            return ordered[0]
        pos = (len(ordered) - 1) * max(0.0, min(1.0, q))
        lo, hi = int(pos), min(int(pos) + 1, len(ordered) - 1)
        frac = pos - lo
        return ordered[lo] * (1 - frac) + ordered[hi] * frac

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for key, bucket in self.samples.items():
            out[key] = {
                "count": float(len(bucket)),
                "mean": statistics.fmean(bucket),
                "p50": self.percentile(key, 0.50),
                "p95": self.percentile(key, 0.95),
                "max": max(bucket),
            }
        return out


METRICS = Counter()
LATENCY = Latencies()


@contextmanager
def timed(key: str) -> Iterator[None]:
    """Context manager recording one latency sample under ``key``."""
    start = time.perf_counter()
    try:
        yield
    finally:
        LATENCY.observe(key, (time.perf_counter() - start) * 1000.0)
