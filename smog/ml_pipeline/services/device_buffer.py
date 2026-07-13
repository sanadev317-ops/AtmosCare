"""Rolling device buffer for IoT-driven inference."""

from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from ..utils import get_logger

logger = get_logger(__name__)


class DeviceBufferManager:
    """Keep the last N samples per device in memory and optionally seed from MongoDB."""

    def __init__(self, maxlen: int = 60):
        self.maxlen = int(maxlen)
        self._buffers: Dict[str, deque] = {}
        self._lock = Lock()

    @staticmethod
    def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(row)
        if "datetime" in normalized and "timestamp" not in normalized:
            normalized["timestamp"] = normalized.pop("datetime")
        return normalized

    def append(self, device_id: str, row: Dict[str, Any]) -> None:
        row = self._normalize_row(row)
        with self._lock:
            if device_id not in self._buffers:
                self._buffers[device_id] = deque(maxlen=self.maxlen)
            self._buffers[device_id].append(row)

    def seed(self, device_id: str, rows: Iterable[Dict[str, Any]]) -> None:
        with self._lock:
            buf = self._buffers.get(device_id)
            if buf is None:
                buf = deque(maxlen=self.maxlen)
                self._buffers[device_id] = buf
            for row in rows:
                buf.append(self._normalize_row(row))

    def size(self, device_id: str) -> int:
        with self._lock:
            return len(self._buffers.get(device_id, ()))

    def get_recent_rows(self, device_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._lock:
            rows = list(self._buffers.get(device_id, ()))
        if limit is None:
            return rows
        return rows[-int(limit):]

    def get_recent_frame(self, device_id: str, limit: Optional[int] = None) -> pd.DataFrame:
        rows = self.get_recent_rows(device_id, limit=limit)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        timestamp_col = "timestamp" if "timestamp" in df.columns else "datetime"
        if timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
            df = df.rename(columns={timestamp_col: "datetime"}).set_index("datetime").sort_index()
        return df

    def preload_from_history(self, device_id: str, rows: List[Dict[str, Any]]) -> None:
        if rows:
            self.seed(device_id, rows)
