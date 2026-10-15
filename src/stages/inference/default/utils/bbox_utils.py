import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Mapping

from common.structs import PhotoExif

_CONNECTIONS: dict[tuple[int, str, int], sqlite3.Connection] = {}


def resolve_target_class_id(
    config: Mapping[str, Any],
    target_name: str,
    *,
    default: int,
) -> int:
    targets = config.get("decision_system", {}).get("targets", {})
    normalized_target_name = str(target_name).strip().upper()

    if isinstance(targets, Mapping):
        for raw_class_id, target_config in targets.items():
            if not isinstance(target_config, Mapping):
                continue
            if (
                str(target_config.get("name", "")).strip().upper()
                == normalized_target_name
            ):
                return int(raw_class_id)

    return int(default)


def resolve_class_id_map(
    config: Mapping[str, Any],
    raw_mapping: Mapping[Any, Any] | None,
) -> dict[int, int]:
    resolved_mapping: dict[int, int] = {}
    if not isinstance(raw_mapping, Mapping):
        return resolved_mapping

    for raw_class_id, target in raw_mapping.items():
        if isinstance(target, str) and not target.strip().lstrip("-").isdigit():
            resolved_mapping[int(raw_class_id)] = resolve_target_class_id(
                config,
                target,
                default=int(raw_class_id),
            )
        else:
            resolved_mapping[int(raw_class_id)] = int(target)

    return resolved_mapping


def remap_bbox_classes(
    bboxes: List[dict],
    class_id_map: Mapping[int, int],
) -> List[dict]:
    remapped_bboxes = []
    for bbox_data in bboxes:
        raw_class = int(bbox_data["class"])
        remapped_bbox_data = dict(bbox_data)
        remapped_bbox_data["raw_class"] = raw_class
        remapped_bbox_data["class"] = int(class_id_map.get(raw_class, raw_class))
        remapped_bboxes.append(remapped_bbox_data)
    return remapped_bboxes


def with_class_id(bboxes: List[dict], class_id: int) -> List[dict]:
    normalized_bboxes = []
    for bbox_data in bboxes:
        normalized_bbox_data = dict(bbox_data)
        normalized_bbox_data.setdefault("raw_class", bbox_data.get("class"))
        normalized_bbox_data["class"] = int(class_id)
        normalized_bboxes.append(normalized_bbox_data)
    return normalized_bboxes


def scale_bboxes_to_full_image(
    bboxes: List[dict],
    inference_shape: tuple[int, ...] | None,
    full_image_shape: tuple[int, ...] | None,
) -> List[dict]:
    if inference_shape is None or full_image_shape is None:
        return bboxes

    inference_height = float(inference_shape[0])
    inference_width = float(inference_shape[1])
    full_height = float(full_image_shape[0])
    full_width = float(full_image_shape[1])

    if inference_height <= 0 or inference_width <= 0:
        return bboxes

    scale_x = full_width / inference_width
    scale_y = full_height / inference_height

    full_image_bboxes = []
    for bbox_data in bboxes:
        scaled_bbox_data = dict(bbox_data)
        x1, y1, x2, y2 = bbox_data["bbox"]
        scaled_bbox_data["bbox"] = [
            float(x1) * scale_x,
            float(y1) * scale_y,
            float(x2) * scale_x,
            float(y2) * scale_y,
        ]
        full_image_bboxes.append(scaled_bbox_data)
    return full_image_bboxes


def build_full_image_bbox_record(
    metadata: PhotoExif,
    bboxes: List[dict],
    inference_shape: tuple[int, ...] | None,
    full_image_shape: tuple[int, ...] | None,
    target_resolution: List[int] | None,
    producer: str,
    mission: str | None,
) -> dict[str, Any]:
    inference_image_shape_payload = None
    full_image_shape_payload = None
    scale = None
    if inference_shape is not None:
        inference_height = int(inference_shape[0])
        inference_width = int(inference_shape[1])
        inference_image_shape_payload = {
            "height": inference_height,
            "width": inference_width,
        }

    if full_image_shape is not None:
        full_height = int(full_image_shape[0])
        full_width = int(full_image_shape[1])
        full_image_shape_payload = {
            "height": full_height,
            "width": full_width,
        }
        if inference_shape is not None:
            inference_height_float = float(inference_shape[0])
            inference_width_float = float(inference_shape[1])
            if inference_height_float > 0 and inference_width_float > 0:
                scale = {
                    "x": full_width / inference_width_float,
                    "y": full_height / inference_height_float,
                }

    return {
        "image": metadata.filename,
        "producer": producer,
        "mission": mission,
        "coordinate_space": "full_image",
        "bbox_format": "xyxy",
        "target_resolution": target_resolution,
        "inference_image_shape": inference_image_shape_payload,
        "full_image_shape": full_image_shape_payload,
        "scale": scale,
        "boxes": bboxes,
    }


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_dumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, default=_json_default, separators=(",", ":"))


def _resolve_db_path(
    metadata: PhotoExif,
    bbox_logging_config: Mapping[str, Any] | None,
) -> Path:
    configured_path = None
    if bbox_logging_config is not None:
        configured_path = bbox_logging_config.get("db_path")

    if configured_path:
        return Path(str(configured_path))

    image_path = Path(metadata.filename)
    parent = image_path.parent if str(image_path.parent) else Path(".")
    return parent / "bboxes.sqlite3"


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bbox_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            image_filename TEXT NOT NULL,
            producer TEXT NOT NULL,
            mission TEXT,
            coordinate_space TEXT NOT NULL,
            bbox_format TEXT NOT NULL,
            target_resolution_json TEXT,
            inference_image_shape_json TEXT,
            full_image_shape_json TEXT,
            scale_json TEXT,
            boxes_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_bbox_records_image_producer
        ON bbox_records (image_filename, producer)
        """
    )


def _get_connection(
    db_path: Path,
    busy_timeout_ms: int,
) -> sqlite3.Connection:
    resolved_path = str(db_path.resolve())
    key = (os.getpid(), resolved_path, busy_timeout_ms)
    conn = _CONNECTIONS.get(key)
    if conn is not None:
        return conn

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        resolved_path,
        timeout=busy_timeout_ms / 1000.0,
        isolation_level=None,
        check_same_thread=False,
    )
    conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    _create_schema(conn)
    _CONNECTIONS[key] = conn
    return conn


def close_bbox_store_connections() -> None:
    for conn in _CONNECTIONS.values():
        conn.close()
    _CONNECTIONS.clear()


def _is_transient_sqlite_error(error: sqlite3.Error) -> bool:
    message = str(error).lower()
    return "locked" in message or "busy" in message


def save_full_image_bboxes(
    metadata: PhotoExif,
    bboxes: List[dict],
    inference_shape: tuple[int, ...] | None,
    full_image_shape: tuple[int, ...] | None,
    target_resolution: List[int] | None,
    *,
    producer: str,
    mission: str | None,
    bbox_logging_config: Mapping[str, Any] | None = None,
    logger: Any | None = None,
) -> bool:
    record = build_full_image_bbox_record(
        metadata=metadata,
        bboxes=bboxes,
        inference_shape=inference_shape,
        full_image_shape=full_image_shape,
        target_resolution=target_resolution,
        producer=producer,
        mission=mission,
    )
    if bbox_logging_config is not None and not bbox_logging_config.get("enabled", True):
        return False

    busy_timeout_ms = 1000
    if bbox_logging_config is not None:
        busy_timeout_ms = int(bbox_logging_config.get("busy_timeout_ms", 1000))

    max_retries = 3
    retry_sleep_s = 0.05
    if bbox_logging_config is not None:
        max_retries = int(bbox_logging_config.get("write_retries", max_retries))
        retry_sleep_s = float(
            bbox_logging_config.get("write_retry_sleep_s", retry_sleep_s)
        )

    db_path = _resolve_db_path(metadata, bbox_logging_config)
    attempts = max(1, max_retries + 1)
    last_error: sqlite3.Error | None = None

    for attempt in range(attempts):
        created_at = datetime.now(timezone.utc).isoformat()
        try:
            conn = _get_connection(db_path, busy_timeout_ms)
            conn.execute(
                """
                INSERT INTO bbox_records (
                    created_at,
                    image_filename,
                    producer,
                    mission,
                    coordinate_space,
                    bbox_format,
                    target_resolution_json,
                    inference_image_shape_json,
                    full_image_shape_json,
                    scale_json,
                    boxes_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    str(record["image"]),
                    str(record["producer"]),
                    record.get("mission"),
                    str(record["coordinate_space"]),
                    str(record["bbox_format"]),
                    _json_dumps(record.get("target_resolution")),
                    _json_dumps(record.get("inference_image_shape")),
                    _json_dumps(record.get("full_image_shape")),
                    _json_dumps(record.get("scale")),
                    _json_dumps(record.get("boxes")) or "[]",
                ),
            )
            return True
        except sqlite3.Error as e:
            last_error = e
            if attempt >= attempts - 1 or not _is_transient_sqlite_error(e):
                break
            time.sleep(retry_sleep_s)

    if logger is not None and last_error is not None:
        logger.warning(
            f"Skipping bbox persistence for {metadata.filename}: {last_error}"
        )
    return False
