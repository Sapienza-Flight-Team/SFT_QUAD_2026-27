# Pipeline Safety Review - 2026-06-06

Scope: local static review of the live pipeline, geometry/angle handling, inference handoff, decision/geolocation, mission routing, and targeted tests. I did not certify the system as safe for life-critical use. The critical findings below should be fixed and validated before autonomous reliance.

## Executive Summary

The original review found multiple high-severity flaws that could either produce wrong geolocations or prevent non-bullseye missions from receiving detections at all. The angle-unit finding has since been addressed in the current working tree; see `docs/angle_workflow_2026-06-06.md`.

Most critical:

1. Angle units were inconsistent in the original review. Current working-tree status: roll, pitch, and yaw are degree-native through the CV pipeline, and camera math is called with `is_deg=True`.
2. `PERIMETER` and `NUMBERS` detections are not forwarded to `DecisionSystem` by the live YOLO stage, and the documented CV-GCP/CNN stages are not instantiated in `pipeline.py`.
3. The live config says `BayerRG8`, but the converter hardcodes `COLOR_BayerBG2RGB`.
4. The default CV-GCP config file is missing.
5. The test suite is not currently a reliable safety gate: the full run is non-interactive-hostile, and a targeted run failed 29 tests.

## Live Pipeline Traced

`pipeline.py` currently wires the runtime stages as:

```text
MAVSDKCommunicator -> CameraEngine -> ConverterStage
                                  -> BullseyeInferenceStage -> DecisionSystem
                                  -> ResizerStage -> InferenceStage -> DecisionSystem
DecisionSystem -> MAVSDKCommunicator
```

`CVGCPInferenceStage` is imported but not instantiated. `CNNInferenceStage` is documented as a post-YOLO refinement stage, but it is not part of the live topology.

## Findings

### Remediated In Working Tree: Angle Units Are Degree-Native

Current working-tree status:

- MAVSDK attitude degrees are stored directly in `PixData.roll`, `PixData.pitch`, and `PixData.yaw`.
- Camera metadata stores roll, pitch, and yaw as degrees.
- `DecisionSystem.input_format()` passes those degree values through.
- `Geoloc.__call__()` calls `Camera.Configuration(..., is_deg=True)`.
- The workflow is documented in `docs/angle_workflow_2026-06-06.md`.
- Regression tests live in `tests/test_angle_units.py`.

Residual requirement:

The math library still converts degrees internally because trigonometric camera math operates in radians. That conversion is now confined to `Camera.Configuration`.

### Remediated In Working Tree: Homography No Longer Guesses Angle Units

Current working-tree status:

- `common.homography.metadata_angles()` returns degree values directly.
- The old `abs(angle) > 2*pi` heuristic has been removed.
- The homography path calls `Camera.Configuration(..., is_deg=True)`.
- The bullseye homography stage no longer carries an angle-unit configuration switch.

### Critical: PERIMETER/NUMBERS Detections Do Not Reach DecisionSystem In The Live Pipeline

Evidence:

- `pipeline.py` instantiates `CameraEngine`, `ConverterStage`, `BullseyeInferenceStage`, `ResizerStage`, `InferenceStage`, and `DecisionSystem`: `pipeline.py:109`.
- It does not instantiate `CVGCPInferenceStage` or `CNNInferenceStage`: `pipeline.py:130` and `pipeline.py:150`.
- YOLO only forwards metadata to decision when current task is `BULLSEYE`: `src/sft_cv/inference/inference_model.py:125`.
- The stage still saves bboxes, but sends only through `_send_handoff_metadata()`, which returns for non-bullseye tasks: `src/sft_cv/inference/inference_model.py:277`.
- Docs claim CV-GCP is instantiated directly in `pipeline.py`: `docs/cv_gcp_inference_stage.md:82`.
- Docs say CNN intended chain is `inference_engine -> cnn_inference_engine -> decision_system`: `docs/cnn_inference_stage.md:73`.

Impact:

`PERIMETER` and `NUMBERS` missions can finalize with no detections from the live pipeline, even though mission config enables classes for those tasks. This is a mission-level functional failure.

Recommended fix:

Define the actual mission topology explicitly and test it. Either:

- forward YOLO detections for `PERIMETER`/`NUMBERS` to decision, or
- instantiate `CVGCPInferenceStage` and `CNNInferenceStage` in the documented chain, or
- add a config-driven topology with integration tests proving each mission reaches decision with expected classes.

### High: Real Camera Bayer Format Mismatch

Evidence:

- Flight config sets `camera.pixel_format: BayerRG8`: `sft_config.yaml:30`.
- `ConverterStage` hardcodes `cv2.COLOR_BayerBG2RGB`: `src/sft_cv/preprocess/converter.py:64`.
- Other modules do inspect the pixel format and choose BayerRG/BG/GR/GB conversions, for example `common.homography._bayer_code()`: `src/sft_cv/common/homography.py:115`.
- Existing converter tests use `BayerBG8`, not the flight config path.

Impact:

Colors can be channel-swapped or demosaiced incorrectly. That can directly affect red bullseye detection and any color-sensitive classifier.

Recommended fix:

Use a shared pixel-format-to-OpenCV conversion helper in converter, CV-GCP, CNN, and tools. Add a test that loads `sft_config.yaml` and verifies `BayerRG8` conversion is used.

### High: Default CV-GCP Detector Config Is Missing

Evidence:

- `find_bw_targets.DEFAULT_CONFIG_PATH` points to `src/sft_cv/inference/bw_targets_config.yaml`: `src/sft_cv/inference/find_bw_targets.py:32`.
- No matching YAML file exists in the repository.
- `CVGCPInferenceStage._load_gcp_config()` falls back to that default when no path/inline config is provided: `src/sft_cv/inference/cv_gcp_inference_stage.py:72`.
- Targeted pytest failed `test_load_gcp_config_uses_bundled_default_when_not_configured` with `FileNotFoundError`.

Impact:

If CV-GCP is wired into the pipeline without explicit config, it will fail at startup.

Recommended fix:

Add the missing `bw_targets_config.yaml`, or make `sft_config.yaml` always provide `inference_engine.bw_targets_config` or `inference_engine.cv_gcp_config`. Test both default and configured paths.

### High: px2body Calibration Is A Large Magic Correction

Evidence:

- Config applies `px2body_error_angle_deg: 176.0` on top of a 90 degree base matrix: `sft_config.yaml:80`.
- `resolve_px2body_matrix()` always composes that error angle when present: `src/sft_math/cam/px2body_matrix.py`.

Impact:

The effective pixel-to-body frame is approximately 266 degrees. This may be a valid empirical calibration, but it is large enough that sign/order/frame mistakes can be hidden behind it. If angle unit fixes are made, this calibration must be revalidated.

Recommended fix:

After fixing angle units, run a ground-truth calibration: image center, left/right/up/down pixels at known yaw/altitude, and compare predicted N/E offsets. Store the convention in config comments and tests.

### Remediated In Working Tree: No Separate Direction Field In CV Pipeline

Current working-tree status:

- `PixData` carries roll, pitch, and yaw only for attitude.
- Camera metadata saves roll, pitch, and yaw only.
- Camera engines no longer contain magnetic-to-true yaw conversion snippets.
- CV and decision stages do not read an alternate yaw source.

Residual requirement:

If yaw frame semantics need to change in the future, do it at the MAVSDK ingestion boundary and keep the stored field name and unit contract unchanged.

### Medium: Real Camera Emits `timestamp=0.0`

Evidence:

- Reader caches `timestamp: time.time_ns()`: `src/sft_cv/cam/camera_engine.py:208`.
- Emitted `PhotoExif` uses `timestamp=0.0`: `src/sft_cv/cam/camera_engine.py:624`.

Impact:

Saved metadata rows can show epoch time and post-flight correlation/replay can be corrupted.

Recommended fix:

Use `img_data["timestamp"]` consistently, and document whether timestamps are seconds, milliseconds, nanoseconds, or Unix epoch.

### Medium: Shared-Memory Helpers Are Brittle In Tests

Evidence:

- `consume_shm_array()` and `fast_ack()` access `self.ack_q` directly: `src/sft_cv/common/pipeline_stage.py:247`.
- Tests that build stages with `__new__` fail before processing because `ack_q` is missing.
- `DecisionSystem.get()` only fast-acks unexpected image SHM and returns: `src/sft_cv/decision_system/decision_system.py:606`.

Impact:

This is less likely in initialized flight stages, but it breaks verification and can hide cleanup regressions. A safety pipeline needs tests that are harder to accidentally invalidate.

Recommended fix:

Use `getattr(self, "ack_q", None)` in helpers, and make unit-test constructors initialize core `PipelineStage` attributes. Add tests for both `ack_q=None` and GC-enabled paths.

### Medium: Shared-Memory Cleanup Depends On Perfect ACKs

Evidence:

- The shared-memory garbage collector only releases a registered segment when `expected > 0 and acks >= expected`: `src/sft_cv/garbage_collector_shm/garbage_collector_shm.py:91`.
- If a consumer crashes before ACK, cleanup is deferred until shutdown dump.
- If a future stage registers with zero expected consumers, the condition never frees the segment.

Impact:

This is an operational reliability risk. In a long mission, leaked shared-memory segments can accumulate and eventually break the camera/inference path.

Recommended fix:

Treat `expected == 0` as immediately releasable, add timeouts for stale registrations, and include GC metrics in logs or health telemetry.

### Medium: Bullseye Worker Queue Has No Backpressure

Evidence:

- `BullseyeInferenceStage` keeps an `active_futures` dictionary: `src/sft_cv/inference/bullseye_inference_stage.py:174`.
- It submits new work to the process pool at `src/sft_cv/inference/bullseye_inference_stage.py:273` and stores it at `src/sft_cv/inference/bullseye_inference_stage.py:290`.
- There is no cap on queued/in-flight futures beyond executor worker count.

Impact:

If images arrive faster than bullseye CV can finish, memory and latency can grow without a deterministic bound.

Recommended fix:

Set a maximum in-flight count, drop or coalesce old frames by mission policy, and log when backpressure activates.

### Medium: Camera Metadata Timing And Sequence Lock Can Drift

Evidence:

- The real camera reader caches a timestamp for image data, but emitted `PhotoExif` uses `timestamp=0.0`: `src/sft_cv/cam/camera_engine.py:624`.
- Hardware sequence offset is locked once from the earliest metadata/image pair: `src/sft_cv/cam/camera_engine.py:543` and `src/sft_cv/cam/camera_engine.py:556`.
- Subsequent matching uses `camera_seq = chunk_id + sequence_offset`: `src/sft_cv/cam/camera_engine.py:584`.

Impact:

Debug/replay timestamps can be useless, and an early dropped or out-of-order frame can poison metadata matching until restart.

Recommended fix:

Emit the real image timestamp with an explicit unit, and allow sequence offset re-locking when repeated mismatches are detected.

### Medium: Survey Angle Convention Needs Operator-Level Validation

Evidence:

- QGC survey generation defines `angle_deg` as the line direction clockwise from geographic north: `src/sft_cv/communicator/qgc_survey_pymavlink.py:471`.
- Config uses `mavlink.decision_output_routing.survey.angle_deg: 0.0` and can optimize from 0 to 180 degrees: `sft_config.yaml:148`.
- A legacy survey helper uses a separate `grid_angle_deg` convention: `src/sft_math/survey.py:112`.

Impact:

The QGC convention is reasonable, but if operators or higher-level planning think `angle_deg` is the grid normal instead of transect direction, the survey rotates by 90 degrees.

Recommended fix:

Add a mission-planning test fixture with a rectangle and expected first/last transect bearings. Put the convention in the operator checklist.

### Medium: Flight Config Safety Gates Are Not Mission-Specific

Evidence:

- `airdrop_bound` spans the full globe: `sft_config.yaml:1`.
- `mavlink.decision_output_routing.enabled` is `false`: `sft_config.yaml:120`.
- `decision_system.geoloc.manifold_center` is in the United States while `real_target_location` is in Italy: `sft_config.yaml:72` and `sft_config.yaml:86`.

Impact:

The broad airdrop bound provides no useful geofence. Disabled decision routing may be intentional for testing, but it means decisions are not appended to the mission. The manifold/target mismatch may be harmless in a Maryland test config, but it is dangerous if this file is reused for the Italian field.

Recommended fix:

Split test/simulation/flight configs. Require a mission-specific config review that checks bounds, manifold center, yaw frame, routing enabled state, and expected mission names before takeoff.

### Positive: Invalid Decision Payloads Are Latched To Hold

Evidence:

- MAVSDK decision routing validates route-specific coordinate count and distinctness before creating mission items: `src/sft_cv/communicator/mavsdk_communicator.py:1192`.
- Invalid routed decisions enter a latched PX4 Hold path: `src/sft_cv/communicator/mavsdk_communicator.py:1249`.

Impact:

This is a good safety behavior and should be preserved while the upstream geometry and topology are fixed.

Recommended fix:

Keep this fail-stop path, add tests for malformed bullseye/perimeter payloads, and make the latch reset procedure explicit for operators.

## Test Results

Full `pdm run pytest` was attempted first, but it did not produce results and appeared to get stuck or leave a stale execution session. A targeted non-interactive test run was then executed:

```bash
pdm run pytest tests/test_pipeline.py tests/test_inference_stage_activation.py tests/test_bullseye_pipeline_integration.py tests/test_cv_gcp_inference_stage.py tests/test_cnn_inference_stage.py tests/test_decision_system_bbox_filtering.py tests/test_perimeter.py tests/test_px2body_matrix.py tests/test_mission_pipeline_config.py tests/test_decision_system_shm_safeguard.py tests/test_bbox_store.py tests/test_instance_splitter.py tests/test_decision_clustering.py tests/test_configs.py
```

Result: 86 passed, 29 failed.

Failure groups:

- Most `tests/test_pipeline.py` converter/resizer failures are due to test stages built with `__new__` lacking `ack_q`, causing `consume_shm_array()` to fail before fanout.
- `tests/test_cv_gcp_inference_stage.py` failed because `bw_targets_config.yaml` is missing.
- `tests/test_cnn_inference_stage.py` failed due missing `ack_q` in `__new__` test stages.
- `tests/test_decision_system_shm_safeguard.py` failed due missing `ack_q` in a `__new__` test stage.

Passing areas in the targeted run:

- Inference activation/handoff tests.
- Bullseye integration tests.
- Bbox filtering tests.
- Perimeter clustering tests.
- px2body matrix tests.
- Bbox store tests.
- Instance splitter and decision clustering tests.
- Basic config tests.

## Minimum Fix Order

1. Fix and test angle units end to end before trusting any geolocation.
2. Fix live pipeline topology so every mission that needs detections sends them to `DecisionSystem`.
3. Fix Bayer conversion to honor `pixel_format`.
4. Add the missing CV-GCP config or require explicit config.
5. Revalidate `px2body_error_angle_deg` after the unit/frame fixes.
6. Repair the test harness so full non-interactive CI is meaningful.
7. Run hardware/simulation validation against known targets before any life-critical use.
8. Freeze a mission-specific config and do a human checklist review before flight.
