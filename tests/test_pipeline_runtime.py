import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("generation_rate_hz, release_delay_range", [
    (5, [0.01, 0.02]),
    (60, [3, 4]),
])
def test_spawned_stages_share_registry_without_pipeline_lease_servicing(
    generation_rate_hz, release_delay_range,
):
    script = """
import multiprocessing as mp
import time
import yaml
from common.config import MISSION_CONFIG_PATH, UTILITY_CONFIG_PATH
from common.new_structs import DATA_TYPE_REGISTRY
from core.pipeline import Pipeline

mp.set_start_method("spawn", force=True)
mission = yaml.safe_load(MISSION_CONFIG_PATH.read_text())
for entry in mission:
    name, implementations = next(iter(entry.items()))
    options = implementations[0]["default"]
    if name == "viewer":
        options.extend([{"display": False}, {"max_frames": max_frames}])
    elif name == "dummy_consumer":
        options.append({"release_delay_range": release_delay_range})
    else:
        options.extend([{"generation_rate_hz": generation_rate_hz}, {"first_capture_id": 0}])
utilities = yaml.safe_load(UTILITY_CONFIG_PATH.read_text())
pipeline = Pipeline(mission, utilities)
close = pipeline.close_registry
blocks = []
def verify_and_close():
    registry = pipeline.registry
    if registry is not None:
        blocks.extend(block.name for helper in registry.data_array.values() for block in helper.buffers)
        close()
        packet = DATA_TYPE_REGISTRY["main_camera"]["data_type"]
        assert registry.last_read["viewer", packet] >= max_frames - 1
        assert registry.last_read["dummy_consumer", packet] >= 0
        assert all(entry.readers == 0 for entry in registry.entries.values())
pipeline.close_registry = verify_and_close
stages = pipeline.build()
started = []
try:
    for stage in stages:
        if generation_rate_hz == 60 and stage.name == "sensor_generator":
            time.sleep(0.1)  # Keep a real arrival offset; matching must handle it.
        stage.start()
        started.append(stage)
    viewer = next(stage for stage in stages if stage.name == "viewer")
    viewer.join(15)
    assert viewer.exitcode == 0
    assert all(stage.is_alive() and not stage.stopped.is_set()
               for stage in stages if stage is not viewer)
finally:
    for stage in started:
        stage.stop()
    for stage in started:
        stage.join(3)
        if stage.is_alive():
            stage.terminate()
            stage.join()
    pipeline.close_registry()
assert all(stage.exitcode == 0 for stage in pipeline.processes)
assert len({stage.pid for stage in pipeline.processes}) == 4
from multiprocessing import shared_memory
for name in blocks:
    try:
        block = shared_memory.SharedMemory(name=name)
    except FileNotFoundError:
        continue
    block.close()
    raise AssertionError("Shared memory was not unlinked")
print("Matched packets consumed")
"""
    script = (f"generation_rate_hz = {generation_rate_hz}\n"
              f"release_delay_range = {release_delay_range!r}\n"
              f"max_frames = {240 if generation_rate_hz == 60 else 8}\n" + script)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert result.returncode == 0, result.stdout + result.stderr[-12000:]
    assert "Matched packets consumed" in result.stdout
    assert "resource_tracker" not in result.stderr
    assert "Terminating stage" not in result.stderr
