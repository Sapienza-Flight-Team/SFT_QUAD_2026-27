from common.pipeline_stage import PipelineStage
from common.structs import PhotoExif, WayPoint
from core.logger import SFTLogger
from common.config import SFTConfig
from sft_math.geo.polar import GPS
from multiprocessing.queues import Queue
from queue import Empty
from multiprocessing import synchronize
from typing import Any, List, Tuple
import multiprocessing as mp
import os, shutil
import traceback
import logging
import time
from sft_math.geo import GeoGPS
import numpy as np
from numpy import typing as np_typing
from scipy.sparse.csgraph import connected_components
import stages.decision_system.default.geoloc as geoloc_module
import stages.decision_system.default.class_handler as class_handler_module
import stages.decision_system.default.decision as decision_module
import stages.decision_system.default.decision.perimeter as perimeter_module


class DecisionSystem(PipelineStage):
    """This class is the main class of the decision system stage.
    This class gets images with UAV's informations and returns the outs
    """

    class Failure(Exception):
        pass

    configs: SFTConfig
    in_q: Queue
    out_q: List[Queue]
    idx: int
    _current_task: str

    geoloc: geoloc_module.Geoloc
    class_handler: class_handler_module.ClassHandler
    decisions: dict[int, decision_module.Decision]
    counter: int

    points: np_typing.NDArray[np.float64]
    covariances: np_typing.NDArray[np.float64]
    outs: tuple[dict[int, np_typing.NDArray[np.float64]] | None, str | None]

    root: str
    stopped: synchronize.Event = (
        mp.Event()
    )  # this flag signs if the class is stopped (see method "stop")
    last_waypoints: List[List[WayPoint]]

    @staticmethod
    def _filter_bboxes_by_size(
        bboxes_list: List,
        min_size: float = 0.0,
    ) -> Tuple[np_typing.NDArray[np.float64], np_typing.NDArray[np.int64]]:
        """Filter detections by bounding box size, removing those with sides < min_size."""
        if len(bboxes_list) == 0:
            return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.int64)

        bboxes_array = np.array([b["bbox"] for b in bboxes_list], dtype=np.float64)
        classes_array = np.array([b["class"] for b in bboxes_list], dtype=np.int64)

        # Compute bbox dimensions
        widths = bboxes_array[:, 2] - bboxes_array[:, 0]
        heights = bboxes_array[:, 3] - bboxes_array[:, 1]
        valid_mask = (widths >= min_size) & (heights >= min_size)

        # Filter and compute centroids
        bboxes_array = bboxes_array[valid_mask]
        classes = classes_array[valid_mask]

        if bboxes_array.shape[0] > 0:
            x_centroids = (bboxes_array[:, 0] + bboxes_array[:, 2]) / 2.0
            y_centroids = (bboxes_array[:, 1] + bboxes_array[:, 3]) / 2.0
            coords = np.column_stack((x_centroids, y_centroids))
        else:
            coords = np.empty((0, 2), dtype=np.float64)

        return coords, classes

    @staticmethod
    def input_format(
        external_input: Any,
        min_size: float = 0.0,
        ref_alt: float = 50.0,
    ) -> Tuple[
        str,
        GPS,
        float,
        Tuple[float, float, float],
        np_typing.NDArray[np.float64],
        np_typing.NDArray[np.int64],
    ]:
        """
        This method converts a PhotoExif with attached inference bounding boxes
        into the decision system internal inputs.
        """

        if not isinstance(external_input, PhotoExif):
            raise TypeError(f"Input is not a PhotoExif, but {type(external_input)}")

        photo = external_input
        if not hasattr(photo, "bboxes"):
            raise TypeError(
                "PhotoExif object does not have attached 'bboxes' attribute"
            )

        pix_data = photo.pix_data

        def require_attr(obj: Any, name: str) -> Any:
            if not hasattr(obj, name):
                raise TypeError(
                    f"PhotoExif.pix_data lacks required telemetry field '{name}'"
                )
            return getattr(obj, name)

        altitude = float(require_attr(pix_data, "alt_rel"))
        roll = float(require_attr(pix_data, "roll"))
        pitch = float(require_attr(pix_data, "pitch"))
        yaw_deg = float(require_attr(pix_data, "yaw"))

        scaled_min_size = min_size
        if min_size > 0 and altitude > 0:
            scaled_min_size = min_size * (ref_alt / altitude)

        bboxes_list = getattr(photo, "bboxes", [])
        coords, classes = DecisionSystem._filter_bboxes_by_size(bboxes_list, min_size=scaled_min_size)

        # Format inputs
        return (
            photo.filename,
            GPS(
                require_attr(pix_data, "lat"),
                require_attr(pix_data, "lon"),
            ),
            altitude,
            (
                roll,
                pitch,
                yaw_deg,
            ),
            coords,
            classes,
        )

    def __init__(
        self,
        configs: SFTConfig,
        logger: SFTLogger,
        in_q: Queue | None,
        out_q: List[Queue] | None,
        idx: int,
        in_command_q: List[Queue] | None = None,
        out_command_q: List[Queue] | None = None,
        ack_q: Queue | None = None,
        stage_name: str = "decision_system",
    ) -> None:
        super().__init__(
            configs=configs,
            logger=logger,
            out_q=out_q,
            in_q=in_q,
            in_command_q=in_command_q,
            out_command_q=out_command_q,
            ack_q=ack_q,
            idx=idx,
            stage_name=stage_name,
        )

        assert in_q is not None, "input queue is None"
        assert out_q is not None, "output queue is None"
        assert 1 <= len(out_q) <= 2, "output queue must be a list of 1 or 2 queues"

        self.received_stop = False

        self.root = self.configs.config_dict["decision_system"]["root"]

        # clean the debug and temporary folders
        if os.path.exists(self.root):
            shutil.rmtree(self.root, ignore_errors=False)
        os.makedirs(self.root, exist_ok=True)

        # set the debugger
        if self.configs.config_dict["decision_system"]["debug"]:
            self.debugger = logging.getLogger("decision_system")
            file_handler = logging.FileHandler(os.path.join(self.root, "debug.log"))
            file_formatter = logging.Formatter(
                "%(asctime)-16s | %(levelname)-8s | '%(funcName)s' in %(pathname)s at line %(lineno)d : %(message)s",
                datefmt="%Y%m%d_%H%M%S",
            )
            file_handler.setFormatter(file_formatter)
            self.debugger.setLevel(logging.DEBUG)
            self.debugger.addHandler(file_handler)
        else:
            self.debugger = logging.getLogger("decision_system")
            self.debugger.addHandler(logging.NullHandler())

        # initialize objects of this class
        self._configure_from_config()

        self.counter = 0
        self.outs = (None, None)

        self.logger.info("Decision System initialized")

    def _configure_from_config(self) -> None:
        decision_config = self.configs.config_dict["decision_system"]
        self.geoloc = geoloc_module.Geoloc(
            self.debugger,
            decision_config["geoloc"],
        )
        self.class_handler = class_handler_module.ClassHandler(
            self.debugger,
            decision_config["class_handler"],
        )
        self.decisions = {
            cls_id: decision_module.Decision(
                self.debugger,
                decision_config["decision"],
            )
            for cls_id in self.class_handler.classes
        }
        self.logger.info(
            f"Decision System configured for mission {self._current_task} "
            f"with classes {self.class_handler.classes}"
        )

    @staticmethod
    def _fit_decision(cls_id, decision_obj, n_clusters: int):
        if decision_obj.data.points.size > 0:
            try:
                return cls_id, decision_obj.fit(n_clusters=n_clusters)
            except decision_module.Decision.Failure:
                return cls_id, np.zeros((0, 2), dtype=np.float64)
        return cls_id, np.zeros((0, 2), dtype=np.float64)

    def _produce_perimeter_output(self) -> None:
        """Localize the 3-4 perimeter GCP corners and emit them.

        Used only for the PERIMETER mission. For each clusterized class it runs
        the robust corner localizer (DBSCAN/HDBSCAN + noise rejection + support
        ranking) on the accumulated detections and reuses the standard ``put()``
        output path. It does not touch the CentroidAnalyzer / instance_splitter
        logic used by the other missions.
        """
        decision_config = self.configs.config_dict["decision_system"]
        targets = decision_config.get("targets", {})
        perimeter_cfg = decision_config.get("perimeter", {})

        all_outs: dict[int, np_typing.NDArray[np.float64]] = {}
        for cls_id, obj in self.decisions.items():
            target_cfg = targets.get(cls_id, {})
            if not target_cfg.get("clusterize", True):
                continue
            spec = target_cfg.get("clusters", 1)
            corners = perimeter_module.localize_corners(
                obj.data.points, obj.data.covs, spec, perimeter_cfg
            )
            if corners.shape[0] > 0:
                all_outs[cls_id] = corners
            self.logger.info(
                f"Perimeter: class {cls_id} -> {corners.shape[0]} corner(s) "
                f"(spec={spec})"
            )

        self.outs = (all_outs, self.outs[1])
        self.put()

    def _produce_centroid_output(self) -> None:
        """Standard CentroidAnalyzer localization path.

        Used by every mission except PERIMETER (i.e. DEFAULT, NUMBERS,
        BULLSEYE). For each clusterized class it clusters the accumulated
        detections with the CentroidAnalyzer (``Decision.fit``), optionally
        splitting the class into spatially separated instances first (only when
        the instance splitter is enabled and more than one cluster is expected),
        then reuses the standard ``put()`` output path.
        """
        import concurrent.futures

        all_outs: dict[int, np_typing.NDArray[np.float64]] = {}
        max_workers = self.configs.config_dict["decision_system"].get("max_workers", 4)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            decision_config = self.configs.config_dict["decision_system"]
            targets = decision_config.get("targets", {})
            splitter_cfg = decision_config.get("instance_splitter", {})
            splitter_enabled = bool(splitter_cfg.get("enabled", False))
            min_instance_distance_m = float(
                splitter_cfg.get("min_instance_distance_m", 15.0)
            )

            futures = {}
            for cls_id, obj in self.decisions.items():
                target_cfg = targets.get(cls_id, {})
                if not target_cfg.get("clusterize", True):
                    continue
                n_clusters = int(target_cfg.get("clusters", 1))

                if splitter_enabled and n_clusters > 1:
                    # multiple instances of the same class are
                    # expected: split the accumulated detections by
                    # distance and clusterize each instance on its
                    # own (one centroid each)
                    sub_decisions = self._split_instances_by_distance(
                        obj, min_instance_distance_m
                    )
                    self.logger.debug(
                        f"Class {cls_id}: instance splitter produced "
                        f"{len(sub_decisions)} instance(s) "
                        f"(up to {n_clusters} expected)"
                    )
                    for sub in sub_decisions:
                        futures[executor.submit(self._fit_decision, cls_id, sub, 1)] = (
                            cls_id
                        )
                else:
                    futures[
                        executor.submit(self._fit_decision, cls_id, obj, n_clusters)
                    ] = cls_id

            for future in concurrent.futures.as_completed(futures):
                cls_id, result = future.result()
                if result.shape[0] > 0:
                    if cls_id in all_outs:
                        all_outs[cls_id] = np.vstack((all_outs[cls_id], result))
                    else:
                        all_outs[cls_id] = result

        self.outs = (all_outs, self.outs[1])
        self.put()

    @staticmethod
    def _split_instances_by_distance(
        decision_obj: "decision_module.Decision", min_distance_m: float
    ) -> List["decision_module.Decision"]:
        """
        Split the accumulated detections of a single class into spatially
        separated instances.

        Two detections belong to the same instance if they can be linked by a
        chain of detections, each within `min_distance_m` metres of the next
        (connected components on a proximity graph). This is a pure distance
        threshold, not a clustering algorithm: the clustering of each instance
        is left untouched to the downstream CentroidAnalyzer.

        The detections are already expressed in local ENU metric coordinates
        (output of Geoloc), so euclidean distances are directly in metres.

        Returns a list of Decision objects, one per detected instance. If there
        are fewer than 2 detections, or they all fall within the threshold, the
        original decision_obj is returned unchanged as a single-element list,
        so the mission proceeds gracefully even when not all instances are
        found.
        """
        points = decision_obj.data.points  # (N, 2) in local ENU metres
        if points.shape[0] <= 1:
            return [decision_obj]

        # proximity adjacency: edge between detections closer than the threshold
        diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        adjacency = dists < float(min_distance_m)

        n_instances, labels = connected_components(adjacency, directed=False)
        if n_instances <= 1:
            return [decision_obj]

        return [decision_obj.subset(labels == k) for k in range(n_instances)]

    def _insert_current_points(self) -> None:
        for cls_id in np.unique(self.classes):
            if cls_id in self.decisions:
                idx_mask = self.classes == cls_id
                self.decisions[cls_id].insert(
                    self.points[idx_mask], self.covariances[idx_mask]
                )
            else:
                self.logger.debug(f"Skipping class {cls_id}: no active decision object")

    def _drain_pending_inputs(self) -> None:
        timeout = self.configs.config_dict["decision_system"].get(
            "finalize_drain_timeout", 2.0
        )
        deadline = time.time() + float(timeout)

        while time.time() < deadline:
            try:
                self.get(timeout=0.2)
                self._insert_current_points()
                deadline = time.time() + float(timeout)
            except Empty:
                return
            except self.Failure:
                self.logger.warning(traceback.format_exc())
            except BaseException as e:
                self.logger.error(
                    f"An unexpected error occurred while draining inputs: {str(e)} \n {traceback.format_exc()}"
                )
                raise e

    def _check_and_emit_fake_payload(self) -> bool:
        """Injects YAML fake coordinates if enabled for the active task."""
        if self._current_task == "DEFAULT":
            return False

        ds_config = self.configs.config_dict.get("decision_system", {})
        fake_config = ds_config.get("fake_decision_inputs", {})

        if not fake_config.get("enabled", False):
            return False

        route = self._current_task.lower()
        if route == "bullseye":
            route_key = "bullseye"
        elif route in ["perimeter", "survey", "numbers"]:
            route_key = "perimeter"
        else:
            return False

        raw_route = fake_config.get(route_key, {})
        if not raw_route.get("enabled", True):
            return False

        active_set = str(raw_route.get("active_set", "default"))

        # Deduplication tracking to ensure we only emit once per trigger
        token = f"{self._current_task}_{active_set}"
        emitted_set = getattr(self, "_emitted_fake_payloads", set())
        self._emitted_fake_payloads = emitted_set

        if token in emitted_set:
            return False

        coordinate_sets = raw_route.get("coordinate_sets")
        if isinstance(coordinate_sets, dict):
            coordinates = coordinate_sets.get(active_set)
        else:
            coordinates = raw_route.get("coordinates")

        if not isinstance(coordinates, list):
            self.logger.warning(
                f"Fake coordinates for {route_key} are invalid or missing."
            )
            return False

        target_name = str(raw_route.get("target_name", self._current_task))

        # Reformat arrays [lat, lon] to dicts {"lat": x, "lon": y} to mirror normal put() output
        formatted_coords = []
        for pt in coordinates:
            if isinstance(pt, dict):
                formatted_coords.append(pt)
            elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                formatted_coords.append({"lat": float(pt[0]), "lon": float(pt[1])})

        payload = {target_name: formatted_coords}

        self.logger.warning(
            f"Injecting YAML fake decision payload for {self._current_task}: "
            f"target={target_name!r}, active_set={active_set!r}, coordinates={len(formatted_coords)}"
        )

        emitted_set.add(token)

        # Emit directly to the communicator queue
        if len(self.out_q) > 0:
            self.out_q[0].put(payload)

        # Clear the task as if we successfully ran the localization
        self._current_task = "DEFAULT"
        self.received_stop = False
        self.counter += 1
        return True

    # @abstractmethod
    def mission(self) -> bool:
        """
        This is the main mission method for the stage. This function will be the
        main loop for the stage. You should be consuming the necessary data
        from `in_q` and sending your results using `PipelineStage.put`. You should also
        check the `stopped` flag to see if the mission has stopped. If the
        mission has stopped, you should wait, as another mission may start soon.

        Returns:
            False:
                - any stop detected
            True:
                - the process detects a stop
        """
        self.logger.debug("mf got here")
        while not self.stopped.is_set():
            self.check_inbound_command()
            if self._check_and_emit_fake_payload():
                continue
            # self.logger.debug(f"Current task: {self._current_task}")
            if self._current_task != "DEFAULT" and self.received_stop:
                try:
                    # devo produrre output
                    self.logger.debug("Producing output")
                    self._drain_pending_inputs()

                    if self._current_task == "PERIMETER":
                        # Robust 3-4 corner localization for the perimeter GCPs
                        # (see decision.perimeter); every other mission uses the
                        # standard CentroidAnalyzer path.
                        self._produce_perimeter_output()
                    else:
                        self._produce_centroid_output()

                    self.logger.debug("Output produced and sent to queues")
                    self.received_stop = False
                    # aggiorno il contatore
                    self.counter += 1
                    self._current_task = "DEFAULT"
                except Exception as e:
                    self.logger.error(
                        f"An unexpected error occurred during fitting: {str(e)} \n {traceback.format_exc()}"
                    )
                    raise self.Failure(str(e)) from e
            elif self._current_task != "DEFAULT":
                # accumulo i dati
                try:
                    self.get()
                    self.logger.debug("Data acquired")
                    try:
                        self._insert_current_points()
                        self.put()
                    except decision_module.Decision.Failure as e:
                        raise self.Failure(str(e)) from e
                    except BaseException as e:
                        self.logger.error(
                            f"An unexpected error occurred: {str(e)} \n {traceback.format_exc()}"
                        )
                        raise e
                except Empty:
                    continue
                    self.logger.debug("No data received, skipping iteration")
                except self.Failure as e:
                    self.logger.warning(traceback.format_exc())
                except BaseException as e:
                    self.logger.error(
                        f"An unexpected error occurred: {str(e)} \n {traceback.format_exc()}"
                    )
                    raise e

        # check if the process is stopped
        if self.stopped.is_set():
            return True

        # mission completed successfully, return False
        return False

    # @abstractmethod
    def update(self) -> None:
        super().update()
        self._configure_from_config()

    # @abstractmethod
    def stop(self) -> None:
        """This method save the temporary folder and the debug folder and reset the class"""
        self.logger.info(
            f"Something stopped {self.__class__.__name__} with {self.in_q.qsize()} metadatas left to compute"
        )
        if not self.stopped.is_set():
            self.stopped.set()
        else:
            self.logger.info(
                f"{self.__class__.__name__} already stopped, nothing to do"
            )
        self._current_task = "DEFAULT"

    # @abstractmethod
    def _dump_data(self) -> None:
        pass

    # @abstractmethod
    def get(self, timeout: float = 5) -> None:
        """get data from the previous stage, could be messages between queues or files
        saved on disk, can be different approach from put

        Raises:
            Mapping.Failure: if the input type is not a pd.Series
            Mapping.Failure: if the serie columns are not the same as metadata headers
        """

        input: Any = self.receive_from_queue(timeout=timeout)

        if not isinstance(input, PhotoExif):
            self.logger.debug(f"Ignoring non-PhotoExif input: {input}")
            self.outs = (None, None)
            self.points = np.empty((0, 2), dtype=np.float64)
            self.covariances = np.empty((0, 2, 2), dtype=np.float64)
            self.classes = np.empty((0,), dtype=np.int64)
            return

        if input.shm_name is not None:
            self.logger.warning(
                f"DecisionSystem received unexpected image shared memory "
                f"{input.shm_name}; acknowledging and ignoring it"
            )
            self.fast_ack(input)
            return

        # Format inputs
        # Retrieve the 'bbox_size_filter_px' setting from the 'decision_system' section of sft_config.yaml
        # If the parameter is missing or the section doesn't exist, it defaults to 0.0, 
        # effectively disabling the bounding box size filtering.
        min_size = float(self.configs.config_dict.get("decision_system", {}).get("bbox_size_filter_px", 0.0))
        ref_alt = float(self.configs.config_dict.get("decision_system", {}).get("bbox_size_filter_ref_alt_m", 50.0))
        
        filename, gps, altitude, (roll, pitch, yaw), coords, classes = (
            self.input_format(input, min_size=min_size, ref_alt=ref_alt)
        )

        if coords.shape[0] == 0:
            self.logger.debug(f"No detections for {filename}, skipping geolocation")
            self.outs = (None, filename)
            self.points = np.empty((0, 2), dtype=np.float64)
            self.covariances = np.empty((0, 2, 2), dtype=np.float64)
            self.classes = np.empty((0,), dtype=np.int64)
            return

        # passo i dati a geoloc
        means, covariances, _ = self.geoloc(
            coords,
            gps,
            altitude,
            roll,
            pitch,
            yaw,
        )

        # save results
        self.outs = (None, filename)
        self.points = means
        self.covariances = covariances
        self.classes = self.class_handler(classes)

    # @abstractmethod
    def put(self) -> None:

        if self.outs[0] is not None and len(self.out_q) > 0:
            manifold_center = self.geoloc.gps
            report_data = {}
            targets = self.configs.config_dict["decision_system"].get("targets", {})

            for cls_id, outs_arr in self.outs[0].items():
                if outs_arr.shape[0] == 0:
                    continue

                outs_vec = GeoGPS.Vector(outs_arr[:, 0], outs_arr[:, 1])
                gps_list = manifold_center.unmap(outs_vec)

                class_waypoints = []
                lats = np.rad2deg(gps_list.lat)
                lons = np.rad2deg(gps_list.lon)
                class_waypoints = [
                    {"lat": float(lat), "lon": float(lon)}
                    for lat, lon in zip(lats, lons)
                ]

                target_key = str(targets.get(cls_id, {}).get("name", cls_id))
                report_data[target_key] = class_waypoints

            import csv

            out_dir = os.path.join("data", "outs")
            os.makedirs(out_dir, exist_ok=True)
            csv_path = os.path.join(out_dir, "SFT_localization_report.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["target_class", "lat", "lon"])
                for target_name, waypoints in report_data.items():
                    for wp in waypoints:
                        writer.writerow([target_name, wp["lat"], wp["lon"]])

            self.logger.info(f"Localization report written to {csv_path}")
            for target_name, waypoints in report_data.items():
                for wp in waypoints:
                    self.logger.info(
                        f"  {target_name}: lat={wp['lat']:.7f}, lon={wp['lon']:.7f}"
                    )

            self.out_q[0].put(report_data)

        if self.outs[1] is not None and len(self.out_q) > 1:
            self.out_q[1].put(self.outs[1])
        self.outs = (None, None)  # reset the outputs

    def ff15(self) -> None:
        pass

    def glhf(self) -> None:
        pass

    def _request_finalization(self, mission_name: str, stop_context: str) -> None:
        """Arm output finalization for the matching active mission.

        The pipeline now emits mission-specific stop commands
        (BULLSEYE_STOP / PERIMETER_STOP / NUMBERS_STOP). Only the stop that
        matches the currently active mission finalizes the decision output;
        a stale or mismatched stop is ignored so it cannot finalize the wrong
        mission with the wrong-shaped report for the communicator.
        """
        current = str(getattr(self, "_current_task", "DEFAULT") or "DEFAULT").upper()
        if current == "DEFAULT":
            self.logger.warning(f"Ignoring {stop_context}: no active decision mission")
            return
        if current != mission_name:
            self.logger.warning(
                f"Ignoring {stop_context}: active mission is {current}, not {mission_name}"
            )
            return
        self.received_stop = True
        self.logger.info(f"Accepted {stop_context}; finalizing {current}")

    def bullseye_stop(self) -> None:
        self._request_finalization("BULLSEYE", "BULLSEYE_STOP")

    def perimeter_stop(self) -> None:
        self._request_finalization("PERIMETER", "PERIMETER_STOP")

    def numbers_stop(self) -> None:
        self._request_finalization("NUMBERS", "NUMBERS_STOP")

    def survey_stop(self) -> None:
        """Legacy alias: finalize whatever mission is currently active."""
        current = str(getattr(self, "_current_task", "DEFAULT") or "DEFAULT").upper()
        if current in {"BULLSEYE", "PERIMETER", "NUMBERS"}:
            self._request_finalization(current, "SURVEY_STOP")
        else:
            self.logger.warning(
                f"Ignoring SURVEY_STOP: unsupported active decision mission {current}"
            )

    def error_handler(self) -> None:
        pass
