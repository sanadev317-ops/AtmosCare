"""MongoDB persistence layer for IoT data, predictions, and devices."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from pymongo import ASCENDING, MongoClient
    from pymongo.collection import Collection
    from bson import ObjectId
    HAS_PYMONGO = True
except Exception:  # pragma: no cover - optional dependency
    ASCENDING = None  # type: ignore
    MongoClient = None  # type: ignore
    Collection = Any  # type: ignore
    ObjectId = None  # type: ignore
    HAS_PYMONGO = False

from ..utils import get_logger

logger = get_logger(__name__)


class MongoStore:
    """Thin MongoDB wrapper used by the real-time forecasting service."""

    def __init__(self, config: Dict[str, Any]):
        mongo_cfg = config.get("mongodb", {})
        self.enabled = bool(mongo_cfg.get("enabled", True))
        self.uri = mongo_cfg.get("uri") or "mongodb://localhost:27017"
        self.database_name = mongo_cfg.get("database", "AtmosCareDB")
        self.collection_prefix = mongo_cfg.get("collection_prefix", "")

        self._client: Optional[MongoClient] = None
        self._db = None

        if not self.enabled or not HAS_PYMONGO:
            if not HAS_PYMONGO:
                logger.warning("pymongo is not installed. Running in memory-only mode.")
            logger.warning("MongoDB disabled in configuration. Running in memory-only mode.")
            self.enabled = False
            return

        try:
            self._client = MongoClient(self.uri, serverSelectionTimeoutMS=2500)
            self._client.admin.command("ping")
            self._db = self._client[self.database_name]
            self._ensure_indexes()
            logger.info("MongoDB connected: %s/%s", self.uri, self.database_name)
        except Exception as exc:
            self.enabled = False
            self._client = None
            self._db = None
            logger.warning("MongoDB unavailable, falling back to memory-only mode: %s", exc)

    @property
    def available(self) -> bool:
        return self.enabled and self._db is not None

    def _collection(self, name: str) -> Collection:
        if not self.available:
            raise RuntimeError("MongoDB is not available.")
        return self._db[f"{self.collection_prefix}{name}"]

    def _ensure_indexes(self) -> None:
        try:
            self._collection("iot_data").create_index([("device_id", ASCENDING), ("timestamp", ASCENDING)])
            self._collection("predictions").create_index([("device_id", ASCENDING), ("timestamp", ASCENDING)])
            self._collection("devices").create_index("device_id", unique=True)
        except Exception as exc:
            logger.warning("Unable to create MongoDB indexes: %s", exc)

    def health(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "database": self.database_name,
            "uri": self.uri if self.available else None,
        }

    def upsert_device(self, device_id: str, location: Optional[str] = None, status: str = "active") -> None:
        if not self.available:
            return
        payload = {
            "device_id": device_id,
            "location": location,
            "status": status,
            "updated_at": datetime.utcnow(),
        }
        self._collection("devices").update_one({"device_id": device_id}, {"$set": payload}, upsert=True)

    def store_iot_data(self, payload: Dict[str, Any]) -> Optional[str]:
        if not self.available:
            return None
        result = self._collection("iot_data").insert_one(payload)
        return str(result.inserted_id)

    def store_prediction(self, payload: Dict[str, Any]) -> Optional[str]:
        if not self.available:
            return None
        result = self._collection("predictions").insert_one(payload)
        return str(result.inserted_id)

    def update_prediction_actual(self, prediction_id: str, actual: Any, error: Optional[float] = None) -> None:
        if not self.available:
            return
        if ObjectId is None:
            return
        update = {
            "actual": actual,
            "updated_at": datetime.utcnow(),
        }
        if error is not None:
            update["error"] = error
        self._collection("predictions").update_one(
            {"_id": ObjectId(prediction_id)},
            {"$set": update},
        )

    def get_recent_iot_data(self, device_id: str, limit: int = 60) -> List[Dict[str, Any]]:
        if not self.available:
            return []
        cursor = (
            self._collection("iot_data")
            .find({"device_id": device_id})
            .sort("timestamp", -1)
            .limit(int(limit))
        )
        return list(cursor)[::-1]

    def get_predictions(self, device_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        if not self.available:
            return []
        cursor = (
            self._collection("predictions")
            .find({"device_id": device_id})
            .sort("timestamp", -1)
            .limit(int(limit))
        )
        return list(cursor)[::-1]

    def get_latest_prediction(self, device_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if not self.available:
            return None
        query = {"device_id": device_id} if device_id else {}
        return (
            self._collection("predictions")
            .find_one(query, sort=[("timestamp", -1), ("updated_at", -1)])
        )

    def get_device(self, device_id: str) -> Optional[Dict[str, Any]]:
        if not self.available:
            return None
        return self._collection("devices").find_one({"device_id": device_id})
