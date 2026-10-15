import json
import multiprocessing as mp
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from common.mavlink_structs import PixData
from common.structs import PhotoExif
from stages.inference.default.utils import bbox_utils


def _make_metadata(filename: str) -> PhotoExif:
    return PhotoExif(
        filename=filename,
        timestamp=0.0,
        exposure_time=0.0,
        gain=1.0,
        black_level=0,
        bsl_color_space="sRgb",
        gamma=1,
        light_source_preset="",
        pixel_format="RGB8",
        pix_data=PixData(),
        shm_name=None,
        shm_shape=None,
        shm_dtype=None,
    )


def _write_records_worker(db_path: str, worker_id: int, count: int) -> None:
    config = {
        "enabled": True,
        "db_path": db_path,
        "busy_timeout_ms": 1000,
    }
    for index in range(count):
        metadata = _make_metadata(f"worker_{worker_id}_{index}.npy")
        saved = bbox_utils.save_full_image_bboxes(
            metadata,
            [{"class": worker_id, "conf": 0.9, "bbox": [1.0, 2.0, 3.0, 4.0]}],
            (10, 20, 3),
            (20, 40, 3),
            [20, 10],
            producer=f"producer_{worker_id}",
            mission="NUMBERS",
            bbox_logging_config=config,
        )
        if not saved:
            raise RuntimeError("bbox record was not saved")
    bbox_utils.close_bbox_store_connections()


@pytest.fixture(autouse=True)
def _close_connections_after_test():
    yield
    bbox_utils.close_bbox_store_connections()


def test_build_full_image_bbox_record_includes_producer_and_shapes() -> None:
    metadata = _make_metadata("frame.npy")

    record = bbox_utils.build_full_image_bbox_record(
        metadata=metadata,
        bboxes=[{"class": 7, "conf": 0.8, "bbox": [1.0, 2.0, 3.0, 4.0]}],
        inference_shape=(10, 20, 3),
        full_image_shape=(20, 40, 3),
        target_resolution=[20, 10],
        producer="cnn_inference_engine",
        mission="NUMBERS",
    )

    assert record["image"] == "frame.npy"
    assert record["producer"] == "cnn_inference_engine"
    assert "pid" not in record
    assert record["mission"] == "NUMBERS"
    assert record["coordinate_space"] == "full_image"
    assert record["bbox_format"] == "xyxy"
    assert record["target_resolution"] == [20, 10]
    assert record["inference_image_shape"] == {"height": 10, "width": 20}
    assert record["full_image_shape"] == {"height": 20, "width": 40}
    assert record["scale"] == {"x": 2.0, "y": 2.0}
    assert record["boxes"] == [{"class": 7, "conf": 0.8, "bbox": [1.0, 2.0, 3.0, 4.0]}]


def test_save_full_image_bboxes_creates_sqlite_schema_and_row(tmp_path: Path) -> None:
    db_path = tmp_path / "bboxes.sqlite3"
    metadata = _make_metadata(str(tmp_path / "frame.npy"))

    saved = bbox_utils.save_full_image_bboxes(
        metadata,
        [{"class": 2, "conf": 0.95, "bbox": [1.0, 2.0, 3.0, 4.0]}],
        (10, 20, 3),
        (20, 40, 3),
        [20, 10],
        producer="inference_engine",
        mission="BULLSEYE",
        bbox_logging_config={
            "enabled": True,
            "db_path": str(db_path),
            "busy_timeout_ms": 1000,
        },
    )

    assert saved is True

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT image_filename, producer, mission, boxes_json,
                   target_resolution_json, scale_json
            FROM bbox_records
            """
        ).fetchone()

    assert row[0] == str(tmp_path / "frame.npy")
    assert row[1] == "inference_engine"
    assert row[2] == "BULLSEYE"
    assert json.loads(row[3]) == [
        {"class": 2, "conf": 0.95, "bbox": [1.0, 2.0, 3.0, 4.0]}
    ]
    assert json.loads(row[4]) == [20, 10]
    assert json.loads(row[5]) == {"x": 2.0, "y": 2.0}


def test_save_full_image_bboxes_defaults_db_to_image_directory(
    tmp_path: Path,
) -> None:
    metadata = _make_metadata(str(tmp_path / "frame.npy"))

    saved = bbox_utils.save_full_image_bboxes(
        metadata,
        [],
        (10, 20, 3),
        (10, 20, 3),
        None,
        producer="bullseye_cv_engine",
        mission="BULLSEYE",
        bbox_logging_config={"enabled": True, "db_path": None},
    )

    assert saved is True
    assert (tmp_path / "bboxes.sqlite3").exists()


def test_concurrent_processes_write_to_same_sqlite_store(tmp_path: Path) -> None:
    db_path = tmp_path / "bboxes.sqlite3"
    ctx = mp.get_context("spawn")
    process_count = 4
    records_per_process = 10
    processes = [
        ctx.Process(
            target=_write_records_worker,
            args=(str(db_path), process_id, records_per_process),
        )
        for process_id in range(process_count)
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    with sqlite3.connect(db_path) as conn:
        row_count = conn.execute("SELECT COUNT(*) FROM bbox_records").fetchone()[0]

    assert row_count == process_count * records_per_process


def test_locked_store_logs_warning_and_does_not_raise(tmp_path: Path) -> None:
    db_path = tmp_path / "bboxes.sqlite3"
    lock_conn = sqlite3.connect(db_path, isolation_level=None)
    lock_conn.execute("CREATE TABLE lock_holder (id INTEGER PRIMARY KEY)")
    lock_conn.execute("BEGIN EXCLUSIVE")
    logger = MagicMock()

    try:
        saved = bbox_utils.save_full_image_bboxes(
            _make_metadata(str(tmp_path / "frame.npy")),
            [],
            (10, 20, 3),
            (10, 20, 3),
            None,
            producer="cnn_inference_engine",
            mission="NUMBERS",
            bbox_logging_config={
                "enabled": True,
                "db_path": str(db_path),
                "busy_timeout_ms": 1,
            },
            logger=logger,
        )
    finally:
        lock_conn.rollback()
        lock_conn.close()

    assert saved is False
    logger.warning.assert_called_once()
