import csv
import datetime
import logging
import subprocess
import sys
from dataclasses import replace
from multiprocessing import shared_memory
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest

from common.mavlink_structs import MissionMode, MissionState, PixData
from stages.input_logger.default.input_logger import MetadataSaver
from common.structs import PhotoExif

EXPECTED_COLUMNS = [
    "filename",
    "timestamp",
    "exposure_time",
    "gain",
    "black_level",
    "bsl_color_space",
    "gamma",
    "light_source_preset",
    "pixel_format",
    "camera_trigger_seq",
    "teensy_timestamp",
    "lat",
    "lon",
    "alt_msl",
    "alt_rel",
    "vground",
    "roll",
    "pitch",
    "yaw",
    "mission_mode",
    "mission_state",
]


@pytest.fixture
def metadata():
    return PhotoExif(
        filename="upstream/frame.npy",
        timestamp=1_750_000_000.125,
        exposure_time=1250.0,
        gain=2.5,
        black_level=3,
        bsl_color_space="sRGB",
        gamma=1,
        light_source_preset="Daylight, 6500K",
        pixel_format="BayerBG8",
        pix_data=PixData(
            camera_trigger_seq=42,
            teensy_timestamp=123456,
            lat=41.8,
            lon=12.5,
            alt_msl=135.0,
            alt_rel=75.0,
            roll=1.5,
            pitch=-2.5,
            yaw=90.0,
            mission_mode=MissionMode.IN_MISSION,
            mission_state=MissionState.ACTIVE,
        ),
    )


@pytest.fixture
def saver(tmp_path):
    config = SimpleNamespace(
        config_yaml="sft_config.yaml",
        config_dict={
            "camera": {"img_out_dir": str(tmp_path / "outs")},
            "picture_saver": {
                "metadata_db": str(tmp_path / "outs" / "metadata.csv"),
                "saving_mode": "NPY",
            },
        },
    )
    stage = MetadataSaver(config, logging.getLogger("test.metadata_saver"))
    yield stage
    stage.close()


def read_csv(saver):
    with open(saver.out_metadata, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def test_saves_images_and_exact_metadata_in_legacy_run_layout(saver, metadata):
    assert saver.f is None
    assert not Path(saver.out_dir).exists()
    image = np.arange(24, dtype=np.uint16).reshape(4, 6)
    first = saver.save(image, metadata)
    second = saver.save(image + 1, replace(metadata, timestamp=metadata.timestamp + 1))

    run_dir = Path(saver.out_dir)
    assert run_dir.name == saver.db_creation_time
    datetime.datetime.strptime(run_dir.name, "%Y%m%d-%H%M%S")
    assert Path(saver.out_metadata) == run_dir / f"metadata_{run_dir.name}.csv"
    assert Path(first.filename) == run_dir / f"0_{saver.initial_timestamp}.npy"
    assert Path(second.filename) == run_dir / f"1_{saver.initial_timestamp}.npy"
    assert metadata.filename == "upstream/frame.npy"
    assert saver.saved_img_count == 2

    # Read while the saver is still open: each row must already be flushed.
    fields, rows = read_csv(saver)
    assert fields == EXPECTED_COLUMNS
    assert len(rows) == 2
    np.testing.assert_array_equal(np.load(rows[0]["filename"]), image)
    np.testing.assert_array_equal(np.load(rows[1]["filename"]), image + 1)
    assert rows[0] == {
        "filename": first.filename,
        "timestamp": datetime.datetime.fromtimestamp(metadata.timestamp).isoformat(
            timespec="seconds", sep=" "
        ),
        "exposure_time": "1250.0",
        "gain": "2.5",
        "black_level": "3",
        "bsl_color_space": "sRGB",
        "gamma": "1",
        "light_source_preset": "Daylight, 6500K",
        "pixel_format": "BayerBG8",
        "camera_trigger_seq": "42",
        "teensy_timestamp": "123456",
        "lat": "41.8",
        "lon": "12.5",
        "alt_msl": "135.0",
        "alt_rel": "75.0",
        "vground": "",
        "roll": "1.5",
        "pitch": "-2.5",
        "yaw": "90.0",
        "mission_mode": "1",
        "mission_state": "2",
    }


def test_saves_ground_speed_when_supplied(saver, metadata):
    metadata.pix_data.vground = 13.5
    saver.save(np.zeros((2, 2), dtype=np.uint8), metadata)
    assert read_csv(saver)[1][0]["vground"] == "13.5"


def test_keeps_separate_configured_metadata_root_and_basename(saver, tmp_path):
    saver.configs.config_dict["picture_saver"]["metadata_db"] = str(
        tmp_path / "telemetry" / "flight.csv"
    )
    stage = MetadataSaver(saver.configs, saver.logger)
    stage.in_q = Queue()
    stage.in_q.put(None)
    assert stage.mission() is True
    assert Path(stage.out_metadata) == (
        tmp_path
        / "telemetry"
        / stage.db_creation_time
        / f"flight_{stage.db_creation_time}.csv"
    )
    assert Path(stage.out_dir).is_dir()
    assert read_csv(stage) == (EXPECTED_COLUMNS, [])
    assert stage.f is None


@pytest.mark.parametrize("shutdown", ["sentinel", "stop"])
def test_shutdown_drains_all_buffered_pairs(saver, metadata, shutdown):
    saver.in_q = Queue()
    for index in range(3):
        saver.in_q.put((np.full((2, 3), index, dtype=np.uint8), metadata))
    if shutdown == "sentinel":
        saver.in_q.put(None)
    else:
        saver.stop()

    assert saver.mission() is True
    assert saver.in_q.empty()
    assert saver.f is None
    assert saver.saved_img_count == 3
    for index, row in enumerate(read_csv(saver)[1]):
        np.testing.assert_array_equal(np.load(row["filename"]), np.full((2, 3), index))
    saver.close()
    with pytest.raises(RuntimeError, match="closed"):
        saver.save(np.zeros((1, 1)), metadata)


@pytest.mark.parametrize("write_fails", [False, True])
def test_shared_memory_is_acknowledged_after_write(
    saver, metadata, monkeypatch, write_fails
):
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    shm = shared_memory.SharedMemory(create=True, size=image.nbytes)
    saver.ack_q = Queue()
    try:
        np.copyto(np.ndarray(image.shape, dtype=image.dtype, buffer=shm.buf), image)
        metadata = replace(
            metadata,
            shm_name=shm.name,
            shm_shape=image.shape,
            shm_dtype=str(image.dtype),
        )
        original_save = np.save

        def write(*args, **kwargs):
            assert saver.ack_q.empty()
            if write_fails:
                raise OSError("disk full")
            return original_save(*args, **kwargs)

        monkeypatch.setattr(np, "save", write)
        if write_fails:
            with pytest.raises(OSError, match="disk full"):
                saver.get(metadata)
            assert read_csv(saver)[1] == []
            assert list(Path(saver.out_dir).glob("*.npy")) == []
            assert saver.saved_img_count == 0
        else:
            saver.get(metadata)
            np.testing.assert_array_equal(
                np.load(read_csv(saver)[1][0]["filename"]), image
            )
        assert saver.ack_q.get_nowait() == ("ACK", shm.name, "picture_saver", 0)
        assert saver.ack_q.empty()
        # The saver only releases its view; the garbage collector owns unlinking.
        remaining = shared_memory.SharedMemory(name=shm.name)
        remaining.close()
        assert metadata.shm_name == shm.name
    finally:
        shm.close()
        shm.unlink()


def test_write_failure_closes_mission_csv(saver, metadata, monkeypatch):
    saver.in_q = Queue()
    saver.in_q.put((np.zeros((2, 2)), metadata))

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(np, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        saver.mission()
    assert saver.f is None
    assert saver.saved_img_count == 0
    assert read_csv(saver)[1] == []


def test_existing_run_csv_is_never_truncated(saver, metadata):
    run_dir = Path(saver.out_dir)
    run_dir.mkdir(parents=True)
    database = Path(saver.out_metadata)
    database.write_text("previous run\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        saver.save(np.zeros((2, 2)), metadata)
    assert database.read_text(encoding="utf-8") == "previous run\n"
    assert list(run_dir.glob("*.npy")) == []


def test_stage_starts_with_spawn_and_saves_in_child(tmp_path):
    # Isolate the start method from other tests. Pickling the actual stage catches
    # accidental file handles or writers created in its parent-side constructor.
    script = tmp_path / "spawn_saver.py"
    script.write_text(
        """import csv
import logging
import multiprocessing as mp
from pathlib import Path
import numpy as np
from sft_cv.common.config import SFTConfig
from sft_cv.common.mavlink_structs import PixData
from sft_cv.common.metadata_saver import MetadataSaver
from sft_cv.common.structs import PhotoExif

if __name__ == "__main__":
    mp.set_start_method("spawn")
    Path("config.yaml").write_text("camera:\\n  img_out_dir: outs\\npicture_saver:\\n  metadata_db: outs/metadata.csv\\nutils:\\n  mission_yaml: mission.yaml\\n")
    Path("mission.yaml").write_text("task:\\n  - name: DEFAULT\\n")
    queue = mp.Queue()
    stage = MetadataSaver(SFTConfig("config.yaml"), logging.getLogger("spawn"), in_q=queue)
    assert stage.f is None
    stage.start()
    try:
        image = np.arange(6, dtype=np.uint8).reshape(2, 3)
        metadata = PhotoExif("", 0, 1, 2, 0, "sRGB", 1, "Daylight", "Mono8", PixData())
        queue.put((image, metadata))
        queue.put(None)
        stage.join(timeout=15)
        assert stage.exitcode == 0
        with open(stage.out_metadata, newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 1
        np.testing.assert_array_equal(np.load(rows[0]["filename"]), image)
    finally:
        if stage.is_alive():
            stage.terminate()
            stage.join(timeout=5)
        queue.close()
        queue.join_thread()
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
