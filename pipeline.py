"""
Main pipeline for the Jetson card.

This file spawns a series of processes utilizing the Python
Multiprocessing API, it also allows the various processes
to communicate inbetween themselves through the use of
*message queues*.
"""

from matplotlib import backend_bases
from stages.garbage_collector_shm.garbage_collector_shm import GarbageCollectorStage
import time

import torch.multiprocessing as mp  # Used torch to pass torch objects between processes

from common.config import SFTConfig
from core.logger import SFTLogger
from stages.communicator.mavlink_communicator.mavlink_communicator_stage import MAVLINKCommunicator
from stages.decision_system.default.decision_system_stage import DecisionSystem
from stuff_i_moved_away.inference.bullseye_inference_stage import BullseyeInferenceStage
from stuff_i_moved_away.inference.cv_gcp_inference_stage import CVGCPInferenceStage
from stages.inference.default.inference_model_stage import InferenceStage
from stages.resizer.converter import ConverterStage
from stages.resizer.default.resizer import ResizerStage

# Use 'spawn' start method to avoid issues with forking and pickling on some platforms.
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    # start method was already set; ignore
    pass


def main():
    configs = SFTConfig("sft_config.yaml")
    logger = SFTLogger(configs)
    try:
        _run_pipeline(configs, logger)
    except Exception:
        import traceback

        logger.critical(f"Pipeline crashed:\n{traceback.format_exc()}")
        raise

PROCESSES = 10

def _run_pipeline(configs, logger):
    logger.info("Starting pipeline")
    logger.info(
        f"Config: emulated={configs.config_dict.get('camera', {}).get('emulated')}, "
        f"log_level={configs.config_dict.get('logger', {}).get('log_level')}, "
        f"weights={configs.config_dict.get('inference_engine', {}).get('weights')}"
    )

    mav_q = mp.Queue()
    cam_q = mp.Queue()
    decision_q = mp.Queue()
    cmd_qs = [mp.Queue() for _ in range(PROCESSES)]
    conv_bullseye_q = mp.Queue()  # ConverterStage -> BullseyeInferenceStage
    conv_resizer_q = mp.Queue()  # ConverterStage -> ResizerStage
    resized_q = mp.Queue()  # ResizerStage -> InferenceStage
    infer_q = mp.Queue()  # terminal classified detections -> DecisionSystem
    ack_q = mp.Queue()  # all stages -> GarbageCollectorStage

    if configs.config_dict["camera"]["emulated"] is True:

        if configs.config_dict["mavlink"]["mock_type"] == "simulated":
            from stages.cam.unreal_camera_engine.unreal_camera_engine_stage import (
                UnrealCameraEngine as CameraEngine,
            )

            logger.info("Camera engine: UnrealCameraEngine (simulated)")

        else:
            from tools.mock_camera_engine import MockCameraEngine as CameraEngine

            logger.info("Camera engine: MockCameraEngine")
    else:
        from stages.cam.camera_engine.camera_engine_stage import CameraEngine

        logger.info("Camera engine: CameraEngine (real hardware)")

    # instantiate the communicator class
    comm_obj = MAVLinkCommunicator(
        configs,
        logger,
        idx=0,
        out_q=[mav_q],
        in_q=decision_q,
        in_command_q=cmd_qs,
        out_command_q=cmd_qs,
        ack_q=None,
        stage_name="mavlink",
    )

    processes = [
        CameraEngine(
            configs=configs,
            logger=logger,
            in_q=mav_q,
            out_q=[],
            in_command_q=cmd_qs,
            ack_q=ack_q,
            idx=2,
            stage_name="camera",
        ),
    ]

    for p in processes:
        p.start()
    logger.info(f"All {len(processes)} stage processes started")

    # give it time to setup
    time.sleep(2)
    comm_obj.start()
    logger.info("MAVLink communicator started")
    time.sleep(10000000)
    # Mission ended

    comm_obj.stop()
    processes[0].stop()  # closing camera only
    processes[0].join()
    logger.info("CAMERA SHUT DOWN")
    time.sleep(30)
    for p in processes:
        logger.info(f"Stopping {p.name}")
        p.stop()
        logger.info(f"Waiting for {p.name} to end")
        p.join()

    logger.info("Pipeline finished")


if __name__ == "__main__":
    main()
