from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from pathlib import Path
from queue import Empty
from typing import Any
from xml.etree import ElementTree as ET

import cv2
import numpy as np
from pyproj import CRS, Transformer

from common.pipeline_stage import PipelineStage
from sft_math.cam.px2body_matrix import resolve_px2body_matrix
from common.structs import PhotoExif
from sft_math.cam.affine import Affine
from sft_math.cam.cam import Camera
from sft_math.geo import GeoGPS
from sft_math.geo.polar import GPS
from sft_math.geoloc import GeoLoc

IMAGE_WIDTH = 5320
IMAGE_HEIGHT = 3032
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_POLYGON = np.array(
    [[0, 0], [IMAGE_WIDTH, 0], [IMAGE_WIDTH, IMAGE_HEIGHT], [0, IMAGE_HEIGHT]],
    dtype=np.float32,
)

TARGETS_FILE = Path(__file__).with_name("assets") / "targets.json"
DEFAULT_KML_OUTPUT = Path("data/outs/target_simulator_targets.kml")
KML_NAMESPACE = "http://www.opengis.net/kml/2.2"
ET.register_namespace("", KML_NAMESPACE)

NUMBERED_TARGET_COUNT = 10
NUMBERLESS_TARGET_COUNT = 4
BULLSEYE_INDEX = 10
BULLSEYE_SIDE_M = 5.0
MIN_TARGET_SPACING_M = 4.0
NUMBERED_TARGET_INSET_M = 4.0
RECT_SIZE_RANGE_M = (50.0, 300.0)
RECT_ALPHA_RANGE_DEG = (0.0, 90.0)
TARGET_CORNER_YAWS_DEG = (45, 135, 225, 315)


def kml_tag(name: str) -> str:
    return f"{{{KML_NAMESPACE}}}{name}"


def open_polygon(latlon_points, error_message: str) -> tuple[np.ndarray, bool]:
    points = np.asarray(latlon_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError(error_message)

    was_closed = np.allclose(points[0], points[-1])
    if was_closed:
        points = points[:-1]

    if len(points) < 3:
        raise ValueError(error_message)

    return points, was_closed


@dataclass
class LocalProjector:
    to_xy: Transformer
    to_latlon: Transformer

    @classmethod
    def from_latlon(cls, latlon_points) -> "LocalProjector":
        points = np.asarray(latlon_points, dtype=np.float64)
        lat0 = float(points[:, 0].mean())
        lon0 = float(points[:, 1].mean())
        local_crs = CRS.from_proj4(
            f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
        )
        return cls(
            to_xy=Transformer.from_crs(CRS.from_epsg(4326), local_crs, always_xy=True),
            to_latlon=Transformer.from_crs(
                local_crs, CRS.from_epsg(4326), always_xy=True
            ),
        )

    def xy(self, latlon_points) -> np.ndarray:
        points = np.asarray(latlon_points, dtype=np.float64)
        x, y = self.to_xy.transform(points[:, 1], points[:, 0])
        return np.column_stack([x, y])

    def latlon(self, xy_points) -> np.ndarray:
        points = np.asarray(xy_points, dtype=np.float64)
        lon, lat = self.to_latlon.transform(points[:, 0], points[:, 1])
        return np.column_stack([lat, lon])


def cv_contour(points_xy) -> np.ndarray:
    return np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)


def point_inside(point_xy, contour) -> bool:
    return (
        cv2.pointPolygonTest(
            contour,
            (float(point_xy[0]), float(point_xy[1])),
            False,
        )
        >= 0
    )


def random_points_in_polygon(
    polygon_xy: np.ndarray,
    count: int,
    min_distance_m: float,
    rng: np.random.Generator,
    max_tries: int = 100000,
) -> np.ndarray:
    contour = cv_contour(polygon_xy)
    min_xy = polygon_xy.min(axis=0)
    max_xy = polygon_xy.max(axis=0)
    points: list[np.ndarray] = []

    for _ in range(max_tries):
        if len(points) == count:
            return np.asarray(points, dtype=np.float64)

        candidate = rng.uniform(min_xy, max_xy).astype(np.float64)
        if not point_inside(candidate, contour):
            continue

        if points:
            distances = np.linalg.norm(np.asarray(points) - candidate, axis=1)
            if np.any(distances < min_distance_m):
                continue

        points.append(candidate)

    raise RuntimeError(
        f"Non sono riuscito a generare {count} punti distanti almeno "
        f"{min_distance_m} m. Geofence troppo piccolo o distanza troppo grande."
    )


def offset_polygon_on_bisectors(latlon_points, distance_m: float) -> np.ndarray:
    latlon, was_closed = open_polygon(latlon_points, "Servono almeno 3 vertici")
    projector = LocalProjector.from_latlon(latlon)
    polygon_xy = projector.xy(latlon)
    contour = cv_contour(polygon_xy)
    moved_xy = []

    for index, current in enumerate(polygon_xy):
        previous = polygon_xy[(index - 1) % len(polygon_xy)]
        next_point = polygon_xy[(index + 1) % len(polygon_xy)]

        incoming = _unit(previous - current)
        outgoing = _unit(next_point - current)
        bisector = incoming + outgoing

        if np.linalg.norm(bisector) < 1e-9:
            edge = _unit(next_point - previous)
            bisector = np.array([-edge[1], edge[0]], dtype=np.float64)
        else:
            bisector = _unit(bisector)

        if not point_inside(current + bisector * 0.1, contour):
            bisector = -bisector

        moved_xy.append(current + distance_m * bisector)

    moved_latlon = projector.latlon(np.asarray(moved_xy, dtype=np.float64))
    if was_closed:
        moved_latlon = np.vstack([moved_latlon, moved_latlon[0]])
    return moved_latlon


def rotated_rectangle(width_m: float, height_m: float, alpha_deg: float) -> np.ndarray:
    rectangle = np.array(
        [
            [-width_m / 2.0, -height_m / 2.0],
            [width_m / 2.0, -height_m / 2.0],
            [width_m / 2.0, height_m / 2.0],
            [-width_m / 2.0, height_m / 2.0],
        ],
        dtype=np.float64,
    )
    angle_rad = np.deg2rad(alpha_deg)
    rotation = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ],
        dtype=np.float64,
    )
    return rectangle @ rotation.T


def rectangle_image_corners(width: float, height: float) -> np.ndarray:
    return np.array(
        [[0, 0], [width, 0], [width, height], [0, height]],
        dtype=np.float32,
    )


def geo_gps_to_latlon(geo_gps: GeoGPS | GPS) -> np.ndarray:
    gps = getattr(geo_gps, "gps", geo_gps)
    lat = np.rad2deg(np.asarray(gps.lat, dtype=np.float64))
    lon = np.rad2deg(np.asarray(gps.lon, dtype=np.float64))
    return np.column_stack([lat, lon])


def polygon_kml_coordinates(polygon_latlon) -> str:
    points = np.asarray(polygon_latlon, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("polygon_latlon must have shape Nx2 and at least one point")
    if not np.array_equal(points[0], points[-1]):
        points = np.vstack([points, points[0]])
    return " ".join(f"{lon:.8f},{lat:.8f},0" for lat, lon in points)


def _unit(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < eps:
        raise ValueError("Vector too small to normalize")
    return vector / norm


@dataclass
class Target:
    camera: Camera
    logger: Any
    image_path: str
    side: float
    name: str
    center_gps_coords: np.ndarray
    image: np.ndarray | None = field(init=False, default=None)
    vertexes: dict[str, GeoGPS | GPS] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.side = float(self.side)
        self.center_gps_coords = np.asarray(self.center_gps_coords, dtype=np.float64)
        self.image = cv2.imread(str(self.image_path))
        if self.image is None:
            self.logger.warning(
                f"TargetSimulator could not load target image {self.image_path}"
            )

        center = GeoGPS(GPS(self.center_gps_coords[0], self.center_gps_coords[1]))
        half_diagonal_m = np.hypot(self.side / 2.0, self.side / 2.0)
        self.vertexes["gps"] = center + GeoGPS.Movement(
            yaw=TARGET_CORNER_YAWS_DEG,
            distance=[half_diagonal_m] * len(TARGET_CORNER_YAWS_DEG),
        )

    def warp_for(
        self, metadata: dict[str, float]
    ) -> tuple[np.ndarray, np.ndarray | None] | None:
        if self.image is None:
            return None

        camera_configuration = self.camera.Configuration(
            Affine((0, 0, -metadata["altitude"]), ((), (), ())),
            metadata["yaw"],
            metadata["pitch"],
            metadata["roll"],
        )
        geoloc = GeoLoc(
            self.camera,
            camera_configuration,
            GeoGPS(GPS(metadata["latitude"], metadata["longitude"])),
        )
        projected = geoloc.from_gps(self.vertexes["gps"])
        target_px = np.asarray(
            getattr(projected, "vector", projected), dtype=np.float32
        )

        x = target_px[:, 0]
        y = target_px[:, 1]
        visible = bool(
            np.any((0 <= x) & (x < IMAGE_WIDTH) & (0 <= y) & (y < IMAGE_HEIGHT))
        )
        if not visible:
            return None

        source_px = rectangle_image_corners(self.image.shape[1], self.image.shape[0])
        homography, _ = cv2.findHomography(source_px, target_px)
        if homography is None:
            return None

        self.logger.info(f"Visible target {self.name}")
        warped = cv2.warpPerspective(self.image, homography, IMAGE_SIZE)
        warped_corners = cv2.perspectiveTransform(
            source_px.reshape(-1, 1, 2),
            homography,
        ).reshape(-1, 2)
        _, clipped_polygon = cv2.intersectConvexConvex(
            IMAGE_POLYGON,
            warped_corners.astype(np.float32),
        )
        if clipped_polygon is not None:
            clipped_polygon = cv2.approxPolyDP(clipped_polygon, 1, True)

        return warped, clipped_polygon


class TargetSimulator(PipelineStage):
    Targets = Target

    def __init__(
        self,
        configs,
        logger,
        in_q=None,
        out_q=None,
        in_command_q=None,
        out_command_q=None,
        stage_name="target_simulator",
        idx=6,
    ):
        super().__init__(
            configs=configs,
            logger=logger,
            out_q=out_q,
            in_q=in_q,
            idx=idx,
            in_command_q=in_command_q,
            out_command_q=out_command_q,
            stage_name=stage_name,
        )

        stage_config = self.configs.config_dict.get(self._stage_name, {})
        self.blend_alpha = float(stage_config.get("blend_alpha", 1.0))
        self.unlink_input_shm = bool(stage_config.get("unlink_input_shm", True))
        self.export_kml = bool(stage_config.get("export_kml", True))
        self.kml_output_path = Path(
            stage_config.get("kml_output_path", DEFAULT_KML_OUTPUT)
        )
        configured_seed = stage_config.get("random_seed")
        self.random_seed = (
            None if configured_seed in (None, "") else int(configured_seed)
        )
        self.rng = np.random.default_rng(self.random_seed)
        self.processed_count = 0
        self.visible_target_count = 0

        self.camera = self._build_camera()
        with TARGETS_FILE.open("r", encoding="utf-8") as target_file:
            target_data = json.load(target_file)

        geofence = (
            target_data["geofence"]
            if "geofence" in target_data
            else target_data["boundary_gps_coords"]
        )
        self.geofence = np.asarray(geofence, dtype=np.float64)
        self.targets = target_data["targets"]
        self.targets_objects = self._create_targets()

        if self.export_kml:
            try:
                self._export_targets_kml()
            except Exception as exc:
                self.logger.warning(
                    "TargetSimulator could not export the KML preview to "
                    f"{self.kml_output_path}: {exc}"
                )

    def mission(self) -> bool:
        while not self.stopped.is_set():
            self.check_inbound_command()

            try:
                data = self.get(timeout=1)
            except Empty:
                time.sleep(0.01)
                continue
            except Exception as exc:
                self.logger.error(f"TargetSimulator error reading from queue: {exc}")
                time.sleep(0.01)
                continue

            if data is None:
                time.sleep(0.01)
                continue

            if not isinstance(data, PhotoExif):
                self.put(data)
                continue

            try:
                self.put(self._simulate_targets(data))
                self.processed_count += 1
            except Exception as exc:
                self.logger.error(
                    f"TargetSimulator error while processing image: {exc}"
                )
                self.put(data)

        return True

    def stop(self) -> None:
        self.logger.info("Stopping TargetSimulator")
        self.stopped.set()

    def get(self, timeout: float = 1) -> Any:
        return self.receive_from_queue(timeout=timeout)

    def put(self, data: Any) -> None:
        self.send_to_queues(data)

    def _dump_data(self) -> None:
        self.logger.info(
            "TargetSimulator processed "
            f"{self.processed_count} images and projected "
            f"{self.visible_target_count} visible targets"
        )

    def create_random_waypoint_avoiding_targets(
        self,
        geofence_latlon,
        waypoint_side_m,
        existing_targets,
        seed=None,
        max_tries=100000,
    ) -> np.ndarray:
        if waypoint_side_m <= 0:
            raise ValueError("waypoint_side_m deve essere positivo")

        geofence_latlon, _ = open_polygon(
            geofence_latlon,
            "Il geofence deve avere almeno 3 vertici",
        )
        projector = LocalProjector.from_latlon(geofence_latlon)
        waypoint_radius_m = waypoint_side_m / 2.0
        reduced_geofence_xy = projector.xy(
            offset_polygon_on_bisectors(geofence_latlon, waypoint_radius_m)
        )
        contour = cv_contour(reduced_geofence_xy)
        min_xy = reduced_geofence_xy.min(axis=0)
        max_xy = reduced_geofence_xy.max(axis=0)
        occupied = [
            (projector.xy([target.center_gps_coords])[0], float(target.side) / 2.0)
            for target in existing_targets
        ]

        rng = np.random.default_rng(seed)
        for _ in range(max_tries):
            candidate = rng.uniform(min_xy, max_xy).astype(np.float64)
            if not point_inside(candidate, contour):
                continue
            if any(
                np.linalg.norm(candidate - target_xy)
                < waypoint_radius_m + target_radius_m
                for target_xy, target_radius_m in occupied
            ):
                continue
            return projector.latlon(candidate.reshape(1, 2))[0]

        raise RuntimeError(
            "Non sono riuscito a creare un waypoint valido. "
            "Geofence troppo piccolo oppure target esistenti troppo vicini."
        )

    def move_geofence_vertices_on_bisector(
        self, latlon_points, distance_m
    ) -> np.ndarray:
        return offset_polygon_on_bisectors(latlon_points, distance_m)

    def create_random_rotated_rectangle_in_geofence(
        self,
        geofence_latlon,
        rectangle_size_m,
        alpha_deg,
        seed=None,
    ) -> dict:
        width_m, height_m = np.asarray(rectangle_size_m, dtype=np.float64)
        if not (RECT_SIZE_RANGE_M[0] < width_m < RECT_SIZE_RANGE_M[1]):
            raise ValueError("x_m e y_m devono essere entrambi tra 50 e 300 metri")
        if not (RECT_SIZE_RANGE_M[0] < height_m < RECT_SIZE_RANGE_M[1]):
            raise ValueError("x_m e y_m devono essere entrambi tra 50 e 300 metri")
        if not (RECT_ALPHA_RANGE_DEG[0] < alpha_deg < RECT_ALPHA_RANGE_DEG[1]):
            raise ValueError("alpha_deg deve essere tra 0 e 90 gradi")

        geofence_latlon, _ = open_polygon(
            geofence_latlon,
            "Il geofence deve avere almeno 3 vertici",
        )
        projector = LocalProjector.from_latlon(geofence_latlon)
        rectangle_xy = rotated_rectangle(width_m, height_m, alpha_deg)
        frame_min = rectangle_xy.min(axis=0)
        frame_max = rectangle_xy.max(axis=0)
        frame_size_m = frame_max - frame_min
        half_frame_diag_m = 0.5 * np.hypot(frame_size_m[0], frame_size_m[1])

        reduced_geofence_latlon = offset_polygon_on_bisectors(
            geofence_latlon,
            half_frame_diag_m,
        )
        reduced_geofence_xy = projector.xy(reduced_geofence_latlon)
        center_xy = random_points_in_polygon(
            reduced_geofence_xy,
            count=1,
            min_distance_m=0.0,
            rng=np.random.default_rng(seed),
            max_tries=10000,
        )[0]

        frame_xy = rectangle_image_corners(frame_size_m[0], frame_size_m[1]).astype(
            np.float64
        )
        frame_xy += frame_min

        return {
            "center_latlon": projector.latlon(center_xy.reshape(1, 2))[0],
            "target_vertices_latlon": projector.latlon(rectangle_xy + center_xy),
            "frame_vertices_latlon": projector.latlon(frame_xy + center_xy),
            "reduced_geofence_latlon": reduced_geofence_latlon,
            "half_frame_diag_m": half_frame_diag_m,
            "frame_size_m": frame_size_m,
            "target_size_m": np.array([width_m, height_m]),
            "alpha_deg": alpha_deg,
        }

    def random_points_inside_geofence_min_distance(
        self,
        geofence_latlon,
        min_distance_m,
        n_points=10,
        seed=None,
        max_tries=100000,
    ) -> np.ndarray:
        if min_distance_m <= 0:
            raise ValueError("min_distance_m deve essere positivo")

        geofence_latlon, _ = open_polygon(
            geofence_latlon,
            "Il geofence deve avere almeno 3 vertici",
        )
        projector = LocalProjector.from_latlon(geofence_latlon)
        points_xy = random_points_in_polygon(
            projector.xy(geofence_latlon),
            count=n_points,
            min_distance_m=min_distance_m,
            rng=np.random.default_rng(seed),
            max_tries=max_tries,
        )
        return projector.latlon(points_xy)

    def _simulate_targets(self, metadata: PhotoExif) -> PhotoExif:
        image, input_shm = self._load_image(metadata)
        if image is None:
            self.logger.warning(f"TargetSimulator could not load {metadata.filename}")
            return metadata

        try:
            target_metadata = self._metadata_for_projection(metadata)
            output_image = image.copy()
            visible_targets = 0

            for target in self.targets_objects:
                warped = target.warp_for(target_metadata)
                if warped is None:
                    continue

                warped_image, target_polygon = warped
                mask = self._target_mask(target_polygon, output_image.shape)
                if not np.any(mask):
                    continue

                warped_image = self._fit_image_to_reference(warped_image, output_image)
                alpha = np.clip(self.blend_alpha, 0.0, 1.0)
                blended = (
                    output_image.astype(np.float32) * (1.0 - alpha)
                    + warped_image.astype(np.float32) * alpha
                )
                output_image[mask] = self._cast_like(blended, output_image)[mask]
                visible_targets += 1

            if visible_targets == 0 and input_shm is not None:
                return metadata

            self._store_image(metadata, output_image)
            self.visible_target_count += visible_targets

            if input_shm is not None and self.unlink_input_shm:
                input_shm.close()
                input_shm.unlink()
                input_shm = None

            return metadata
        finally:
            if input_shm is not None:
                input_shm.close()

    def _create_targets(self) -> list[Target]:
        main_rectangle = self.create_random_rotated_rectangle_in_geofence(
            geofence_latlon=self.geofence,
            rectangle_size_m=(
                self.rng.uniform(*RECT_SIZE_RANGE_M),
                self.rng.uniform(*RECT_SIZE_RANGE_M),
            ),
            alpha_deg=self.rng.uniform(*RECT_ALPHA_RANGE_DEG),
            seed=self._next_seed(),
        )
        numbered_region = offset_polygon_on_bisectors(
            main_rectangle["target_vertices_latlon"],
            NUMBERED_TARGET_INSET_M,
        )
        numbered_centers = self.random_points_inside_geofence_min_distance(
            geofence_latlon=numbered_region,
            min_distance_m=MIN_TARGET_SPACING_M,
            n_points=NUMBERED_TARGET_COUNT,
            seed=self._next_seed(),
        )

        targets = [
            self._target_from_info(info, center)
            for info, center in zip(
                self.targets[:NUMBERED_TARGET_COUNT], numbered_centers
            )
        ]
        targets.extend(
            self._target_from_info(info, center)
            for info, center in zip(
                self.targets[-NUMBERLESS_TARGET_COUNT:],
                main_rectangle["target_vertices_latlon"],
            )
        )

        bullseye_center = self.create_random_waypoint_avoiding_targets(
            geofence_latlon=self.geofence,
            waypoint_side_m=BULLSEYE_SIDE_M,
            existing_targets=targets,
            seed=self._next_seed(),
        )
        targets.insert(
            BULLSEYE_INDEX,
            self._target_from_info(self.targets[BULLSEYE_INDEX], bullseye_center),
        )
        return targets

    def _next_seed(self) -> int:
        return int(self.rng.integers(0, np.iinfo(np.uint32).max))

    def _export_targets_kml(self) -> None:
        self.kml_output_path.parent.mkdir(parents=True, exist_ok=True)
        root = ET.Element(kml_tag("kml"))
        document = ET.SubElement(root, kml_tag("Document"))
        ET.SubElement(document, kml_tag("name")).text = "Target simulator"
        ET.SubElement(document, kml_tag("description")).text = (
            "Target positions generated by TargetSimulator"
        )

        for style_id, line_color, fill_color in (
            ("geofence", "ff00ffff", "2200ffff"),
            ("target", "ff0000ff", "220000ff"),
        ):
            style = ET.SubElement(document, kml_tag("Style"), id=style_id)
            line_style = ET.SubElement(style, kml_tag("LineStyle"))
            ET.SubElement(line_style, kml_tag("color")).text = line_color
            ET.SubElement(line_style, kml_tag("width")).text = "2"
            poly_style = ET.SubElement(style, kml_tag("PolyStyle"))
            ET.SubElement(poly_style, kml_tag("color")).text = fill_color

        self._add_kml_polygon(
            document,
            "Geofence",
            self.geofence,
            "Original geofence boundary",
            "#geofence",
        )
        folder = ET.SubElement(document, kml_tag("Folder"))
        ET.SubElement(folder, kml_tag("name")).text = "Target footprints"

        targets_by_name = {target.name: target for target in self.targets_objects}
        export_order = [
            *[f"target_numberless_{index}" for index in range(NUMBERLESS_TARGET_COUNT)],
            *[f"target_{index}" for index in range(NUMBERED_TARGET_COUNT)],
            "target_bullseye",
        ]
        for target_name in export_order:
            target = targets_by_name.get(target_name)
            if target is None:
                continue

            self._add_kml_polygon(
                folder,
                target.name,
                geo_gps_to_latlon(target.vertexes["gps"]),
                f"side={target.side:.2f} m\nimage={Path(target.image_path).name}",
                "#target",
            )

        self._indent_xml(root)
        ET.ElementTree(root).write(
            self.kml_output_path,
            encoding="utf-8",
            xml_declaration=True,
        )
        self.logger.info(f"TargetSimulator saved KML preview to {self.kml_output_path}")

    def _build_camera(self) -> Camera:
        camera_config = self.configs.config_dict["camera"]
        resolution = tuple(camera_config["resolution"])
        resolution_array = np.asarray(resolution, dtype=np.float64)
        geoloc_config = self.configs.config_dict.get("decision_system", {}).get(
            "geoloc", {}
        )

        if "pixel_size" in geoloc_config:
            pixel_size = float(geoloc_config["pixel_size"])
        elif "pixel_size" in camera_config:
            pixel_size = float(camera_config["pixel_size"])
        elif "sensor_width_mm" in camera_config and "sensor_height_mm" in camera_config:
            width_um = (
                float(camera_config["sensor_width_mm"]) * 1000.0 / resolution_array[0]
            )
            height_um = (
                float(camera_config["sensor_height_mm"]) * 1000.0 / resolution_array[1]
            )
            pixel_size = float((width_um + height_um) / 2.0)
        else:
            pixel_size = float(
                np.mean(np.asarray(camera_config["sensor_size"], dtype=np.float64))
            )

        focal_length = camera_config.get(
            "focal_length", camera_config.get("focal_length_mm")
        )
        if focal_length is None:
            raise KeyError("camera config must define focal_length or focal_length_mm")

        px2body_matrix = resolve_px2body_matrix(geoloc_config, camera_config)

        return Camera(
            focal_length,
            pixel_size,
            resolution,
            tuple(coordinate // 2 for coordinate in resolution),
            px2body_matrix,
        )

    def _target_from_info(self, target_info: dict, center_latlon) -> Target:
        return Target(
            camera=self.camera,
            logger=self.logger,
            image_path=target_info["path"],
            side=target_info["side"],
            name=target_info["name"],
            center_gps_coords=center_latlon,
        )

    def _load_image(
        self,
        metadata: PhotoExif,
    ) -> tuple[np.ndarray | None, shared_memory.SharedMemory | None]:
        if metadata.shm_name is not None:
            if metadata.shm_shape is None or metadata.shm_dtype is None:
                raise ValueError("PhotoExif shared memory metadata is incomplete")

            input_shm = shared_memory.SharedMemory(name=metadata.shm_name)
            image = np.ndarray(
                metadata.shm_shape,
                dtype=np.dtype(metadata.shm_dtype),
                buffer=input_shm.buf,
            )
            return np.array(image, copy=True), input_shm

        image_path = Path(metadata.filename)
        if not image_path.exists():
            return None, None
        if image_path.suffix == ".npy":
            return np.load(image_path, allow_pickle=False), None
        return cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED), None

    def _store_image(self, metadata: PhotoExif, image: np.ndarray) -> None:
        output_shm = shared_memory.SharedMemory(create=True, size=image.nbytes)
        output_image = np.ndarray(image.shape, dtype=image.dtype, buffer=output_shm.buf)
        output_image[:] = image
        metadata.shm_name = output_shm.name
        metadata.shm_shape = image.shape
        metadata.shm_dtype = str(image.dtype)
        output_shm.close()

    @staticmethod
    def _metadata_for_projection(metadata: PhotoExif) -> dict[str, float]:
        pix_data = metadata.pix_data
        roll, pitch = np.rad2deg([pix_data.roll, pix_data.pitch]).tolist()
        heading = getattr(pix_data, "heading", None)
        if heading is not None:
            yaw_deg = float(heading)
        else:
            yaw_deg = float(np.rad2deg(pix_data.yaw))

        return {
            "latitude": float(pix_data.lat),
            "longitude": float(pix_data.lon),
            "altitude": float(pix_data.alt_rel),
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw_deg,
        }

    @staticmethod
    def _fit_image_to_reference(
        target_image: np.ndarray, reference_image: np.ndarray
    ) -> np.ndarray:
        ref_h, ref_w = reference_image.shape[:2]
        if (target_image.shape[1], target_image.shape[0]) != (ref_w, ref_h):
            target_image = cv2.resize(
                target_image, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR
            )

        if reference_image.ndim == 2:
            if target_image.ndim == 3:
                return cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
            return target_image

        if target_image.ndim == 2:
            target_image = cv2.cvtColor(target_image, cv2.COLOR_GRAY2BGR)

        ref_channels = reference_image.shape[2]
        target_channels = target_image.shape[2]
        if ref_channels == target_channels:
            return target_image
        if ref_channels == 4 and target_channels == 3:
            return cv2.cvtColor(target_image, cv2.COLOR_BGR2BGRA)
        if ref_channels == 3 and target_channels == 4:
            return target_image[:, :, :3]

        raise ValueError(
            f"Cannot match target with {target_channels} channels "
            f"to image with {ref_channels} channels"
        )

    @staticmethod
    def _target_mask(
        target_polygon: np.ndarray | None, image_shape: tuple[int, ...]
    ) -> np.ndarray:
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        if target_polygon is None:
            return mask.astype(bool)

        polygon = np.asarray(target_polygon, dtype=np.float32).reshape(-1, 2)
        if len(polygon) < 3:
            return mask.astype(bool)

        scale = np.array(
            [image_shape[1] / IMAGE_WIDTH, image_shape[0] / IMAGE_HEIGHT],
            dtype=np.float32,
        )
        cv2.fillPoly(mask, [np.round(polygon * scale).astype(np.int32)], (1,))
        return mask.astype(bool)

    @staticmethod
    def _cast_like(image: np.ndarray, reference_image: np.ndarray) -> np.ndarray:
        if np.issubdtype(reference_image.dtype, np.integer):
            info = np.iinfo(reference_image.dtype)
            image = np.clip(image, info.min, info.max)
        return image.astype(reference_image.dtype)

    @staticmethod
    def _add_kml_polygon(
        parent: ET.Element,
        name: str,
        polygon_latlon,
        description: str,
        style_url: str,
    ) -> None:
        placemark = ET.SubElement(parent, kml_tag("Placemark"))
        ET.SubElement(placemark, kml_tag("name")).text = name
        ET.SubElement(placemark, kml_tag("description")).text = description
        ET.SubElement(placemark, kml_tag("styleUrl")).text = style_url

        polygon = ET.SubElement(placemark, kml_tag("Polygon"))
        ET.SubElement(polygon, kml_tag("tessellate")).text = "1"
        ET.SubElement(polygon, kml_tag("altitudeMode")).text = "clampToGround"
        outer = ET.SubElement(polygon, kml_tag("outerBoundaryIs"))
        ring = ET.SubElement(outer, kml_tag("LinearRing"))
        ET.SubElement(ring, kml_tag("coordinates")).text = polygon_kml_coordinates(
            polygon_latlon
        )

    @staticmethod
    def _indent_xml(element: ET.Element, level: int = 0) -> None:
        indent = "\n" + level * "  "
        if len(element):
            if not element.text or not element.text.strip():
                element.text = indent + "  "
            for child in element:
                TargetSimulator._indent_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = indent
        if level and (not element.tail or not element.tail.strip()):
            element.tail = indent
