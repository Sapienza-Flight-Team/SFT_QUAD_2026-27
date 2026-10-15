#!/usr/bin/env python3
"""
qgc_survey_pymavlink.py

A practical Python port of QGroundControl's core survey expansion logic:
  polygon -> parallel clipped transects -> lawnmower ordering -> individual
  MAVLink mission items -> optional pymavlink upload.

The planner exposes the grid angle and can scan candidate angles to minimize
estimated travelled distance. It intentionally separates geometry planning
from MAVLink upload, so inspect the generated plan before sending it.

Dependencies:
    pip install shapely pyproj pymavlink

Notes:
- Default altitude frame is MAV_FRAME_GLOBAL_RELATIVE_ALT.
- The default clipping mode, "qgc_chord", mirrors QGC's survey behavior for
  simple polygons by using the outermost intersections of each grid line.
  Use "inside_segments" for concave polygons when you do not want a transect
  chord to bridge an indentation.
- This module does not implement QGC terrain queries or terrain-following
  interpolation. Add terrain-adjusted altitudes before mission item creation
  if you need that feature.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator, Literal, Optional, Sequence

from pyproj import CRS, Transformer
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    Point,
    Polygon,
    mapping,
    shape,
)

# MAVLink constants used by QGC for survey expansion.
MAV_FRAME_GLOBAL = 0
MAV_FRAME_MISSION = 2
MAV_FRAME_GLOBAL_RELATIVE_ALT = 3
MAV_FRAME_GLOBAL_TERRAIN_ALT = 10

MAV_CMD_NAV_WAYPOINT = 16
MAV_CMD_DO_SET_CAM_TRIGG_DIST = 206
MAV_CMD_IMAGE_START_CAPTURE = 2000
MAV_CMD_CONDITION_GATE = 4501

MAV_MISSION_TYPE_MISSION = 0
MAV_MISSION_ACCEPTED = 0

LatLon = tuple[float, float]  # latitude, longitude
XY = tuple[float, float]  # easting, northing in local metric projection
NodeKind = Literal["turnaround", "survey_entry", "survey_exit", "hover_trigger"]
ClipMode = Literal["qgc_chord", "inside_segments"]


@dataclass(frozen=True)
class CameraGeometry:
    """Ground footprint and derived survey spacing for a nadir camera."""

    footprint_width_m: float
    footprint_height_m: float
    transect_spacing_m: float
    trigger_distance_m: float


@dataclass(frozen=True)
class SurveyConfig:
    """Planner settings corresponding to the common QGC survey controls."""

    altitude_m: float
    transect_spacing_m: float
    trigger_distance_m: float
    angle_deg: float = 0.0
    turnaround_distance_m: float = 0.0
    trigger_camera: bool = True
    camera_trigger_in_turnarounds: bool = False
    hover_and_capture: bool = False
    hover_delay_s: float = 0.0
    use_condition_gate: bool = False
    altitude_frame: int = MAV_FRAME_GLOBAL_RELATIVE_ALT
    clip_mode: ClipMode = "qgc_chord"

    def validate(self) -> None:
        if self.altitude_m <= 0:
            raise ValueError("altitude_m must be greater than zero")
        if self.transect_spacing_m <= 0:
            raise ValueError("transect_spacing_m must be greater than zero")
        if self.trigger_camera and self.trigger_distance_m <= 0:
            raise ValueError(
                "trigger_distance_m must be greater than zero when camera triggering is enabled"
            )
        if self.turnaround_distance_m < 0:
            raise ValueError("turnaround_distance_m cannot be negative")
        if self.hover_delay_s < 0:
            raise ValueError("hover_delay_s cannot be negative")
        if self.clip_mode not in ("qgc_chord", "inside_segments"):
            raise ValueError(f"Unsupported clip_mode: {self.clip_mode!r}")
        if self.altitude_frame not in (
            MAV_FRAME_GLOBAL,
            MAV_FRAME_GLOBAL_RELATIVE_ALT,
            MAV_FRAME_GLOBAL_TERRAIN_ALT,
        ):
            raise ValueError(
                "altitude_frame must be MAV_FRAME_GLOBAL, GLOBAL_RELATIVE_ALT, or GLOBAL_TERRAIN_ALT"
            )


@dataclass(frozen=True)
class Transect:
    """A single line segment to fly inside the survey area."""

    entry_xy: XY
    exit_xy: XY
    grid_offset_m: float

    @property
    def length_m(self) -> float:
        return distance_xy(self.entry_xy, self.exit_xy)

    def reversed(self) -> "Transect":
        return Transect(self.exit_xy, self.entry_xy, self.grid_offset_m)


@dataclass(frozen=True)
class PathNode:
    """A geometric point in the expanded flight path."""

    kind: NodeKind
    xy: XY
    lat: float
    lon: float
    altitude_m: float


@dataclass
class MissionItem:
    """A MISSION_ITEM_INT payload, kept independent of pymavlink."""

    seq: int
    frame: int
    command: int
    current: int = 0
    autocontinue: int = 1
    param1: float = 0.0
    param2: float = 0.0
    param3: float = 0.0
    param4: float = 0.0
    x: int = 0
    y: int = 0
    z: float = 0.0
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SurveyPlan:
    """Result of survey geometry generation and mission item expansion."""

    config: SurveyConfig
    angle_deg: float
    orientation_name: str
    transects: list[Transect]
    path_nodes: list[PathNode]
    mission_items: list[MissionItem]
    survey_path_distance_m: float
    start_leg_distance_m: float
    return_leg_distance_m: float
    total_distance_m: float
    polygon_latlon: list[LatLon]

    @property
    def transect_count(self) -> int:
        return len(self.transects)

    @property
    def waypoint_count(self) -> int:
        return sum(
            1
            for item in self.mission_items
            if item.command in (MAV_CMD_NAV_WAYPOINT, MAV_CMD_CONDITION_GATE)
        )

    @property
    def camera_command_count(self) -> int:
        return sum(
            1
            for item in self.mission_items
            if item.command
            in (MAV_CMD_DO_SET_CAM_TRIGG_DIST, MAV_CMD_IMAGE_START_CAPTURE)
        )

    def summary(self) -> dict[str, Any]:
        return {
            "angle_deg": round(self.angle_deg, 6),
            "orientation": self.orientation_name,
            "transects": self.transect_count,
            "path_nodes": len(self.path_nodes),
            "mission_items": len(self.mission_items),
            "waypoint_like_items": self.waypoint_count,
            "camera_commands": self.camera_command_count,
            "survey_path_distance_m": round(self.survey_path_distance_m, 3),
            "start_leg_distance_m": round(self.start_leg_distance_m, 3),
            "return_leg_distance_m": round(self.return_leg_distance_m, 3),
            "total_distance_m": round(self.total_distance_m, 3),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "config": asdict(self.config),
            "polygon_latlon": [
                {"lat": lat, "lon": lon} for lat, lon in self.polygon_latlon
            ],
            "transects": [
                {
                    "entry_xy": {"x": t.entry_xy[0], "y": t.entry_xy[1]},
                    "exit_xy": {"x": t.exit_xy[0], "y": t.exit_xy[1]},
                    "grid_offset_m": t.grid_offset_m,
                    "length_m": t.length_m,
                }
                for t in self.transects
            ],
            "path_nodes": [asdict(node) for node in self.path_nodes],
            "mission_items": [item.to_dict() for item in self.mission_items],
        }


@dataclass(frozen=True)
class AngleScore:
    """Compact result from angle optimization."""

    angle_deg: float
    orientation_name: str
    total_distance_m: float
    survey_path_distance_m: float
    start_leg_distance_m: float
    return_leg_distance_m: float
    transect_count: int
    mission_item_count: int

    @classmethod
    def from_plan(cls, plan: SurveyPlan) -> "AngleScore":
        return cls(
            angle_deg=plan.angle_deg,
            orientation_name=plan.orientation_name,
            total_distance_m=plan.total_distance_m,
            survey_path_distance_m=plan.survey_path_distance_m,
            start_leg_distance_m=plan.start_leg_distance_m,
            return_leg_distance_m=plan.return_leg_distance_m,
            transect_count=plan.transect_count,
            mission_item_count=len(plan.mission_items),
        )


@dataclass(frozen=True)
class LocalProjector:
    """Azimuthal-equidistant WGS84 projection centered on the survey polygon."""

    lat0: float
    lon0: float
    forward: Transformer = field(repr=False)
    inverse: Transformer = field(repr=False)

    @classmethod
    def from_latlon(cls, points: Sequence[LatLon]) -> "LocalProjector":
        if len(points) < 3:
            raise ValueError("A survey polygon needs at least three vertices")
        lat0 = sum(lat for lat, _ in points) / len(points)
        lon0 = sum(lon for _, lon in points) / len(points)
        local_crs = CRS.from_proj4(
            f"+proj=aeqd +lat_0={lat0:.12f} +lon_0={lon0:.12f} "
            "+datum=WGS84 +units=m +no_defs"
        )
        forward = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
        inverse = Transformer.from_crs(local_crs, "EPSG:4326", always_xy=True)
        return cls(lat0=lat0, lon0=lon0, forward=forward, inverse=inverse)

    def to_xy(self, latlon: LatLon) -> XY:
        lat, lon = latlon
        x, y = self.forward.transform(lon, lat)
        return float(x), float(y)

    def to_latlon(self, xy: XY) -> LatLon:
        x, y = xy
        lon, lat = self.inverse.transform(x, y)
        return float(lat), float(lon)


def camera_geometry_from_specs(
    *,
    altitude_agl_m: float,
    sensor_width_mm: float,
    sensor_height_mm: float,
    focal_length_mm: float,
    sidelap_percent: float,
    frontlap_percent: float,
    flight_direction_along_sensor_height: bool = True,
) -> CameraGeometry:
    """
    Calculate a simple nadir-camera footprint.

    By default, the aircraft flies along the sensor-height footprint and
    adjacent transects are separated across the sensor-width footprint.
    """
    if (
        altitude_agl_m <= 0
        or sensor_width_mm <= 0
        or sensor_height_mm <= 0
        or focal_length_mm <= 0
    ):
        raise ValueError("Altitude and camera dimensions must be positive")
    if not (0 <= sidelap_percent < 100 and 0 <= frontlap_percent < 100):
        raise ValueError("Overlap percentages must satisfy 0 <= overlap < 100")

    width = altitude_agl_m * sensor_width_mm / focal_length_mm
    height = altitude_agl_m * sensor_height_mm / focal_length_mm

    if flight_direction_along_sensor_height:
        cross_track = width
        along_track = height
    else:
        cross_track = height
        along_track = width

    return CameraGeometry(
        footprint_width_m=width,
        footprint_height_m=height,
        transect_spacing_m=cross_track * (1.0 - sidelap_percent / 100.0),
        trigger_distance_m=along_track * (1.0 - frontlap_percent / 100.0),
    )


def normalize_angle_deg(angle_deg: float) -> float:
    """Survey line orientation modulo 180 degrees."""
    result = float(angle_deg) % 180.0
    if math.isclose(result, 180.0):
        return 0.0
    return result


def distance_xy(a: XY, b: XY) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def path_distance_xy(points: Sequence[XY]) -> float:
    return sum(distance_xy(a, b) for a, b in zip(points, points[1:]))


def _dot(a: XY, b: XY) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _add(a: XY, b: XY) -> XY:
    return a[0] + b[0], a[1] + b[1]


def _sub(a: XY, b: XY) -> XY:
    return a[0] - b[0], a[1] - b[1]


def _mul(a: XY, scalar: float) -> XY:
    return a[0] * scalar, a[1] * scalar


def _unit_vector(a: XY, b: XY) -> XY:
    length = distance_xy(a, b)
    if length <= 1e-9:
        raise ValueError("Cannot create a direction vector from coincident points")
    return (b[0] - a[0]) / length, (b[1] - a[1]) / length


def _extract_lines(geometry: Any) -> list[LineString]:
    """Recursively extract non-empty line segments from a Shapely geometry."""
    if geometry.is_empty:
        return []
    if isinstance(geometry, LineString):
        return [geometry] if geometry.length > 1e-7 else []
    if isinstance(geometry, MultiLineString):
        return [line for line in geometry.geoms if line.length > 1e-7]
    if isinstance(geometry, GeometryCollection):
        lines: list[LineString] = []
        for child in geometry.geoms:
            lines.extend(_extract_lines(child))
        return lines
    return []


def _make_polygon_xy(
    polygon_latlon: Sequence[LatLon], projector: LocalProjector
) -> Polygon:
    coords = [projector.to_xy(point) for point in polygon_latlon]
    polygon = Polygon(coords)
    if polygon.is_empty or polygon.area <= 1e-6:
        raise ValueError("Survey polygon has no usable area")
    if not polygon.is_valid:
        raise ValueError("Survey polygon is invalid or self-intersecting")
    return polygon


def _line_segments_for_offset(
    *,
    polygon: Polygon,
    direction: XY,
    normal: XY,
    offset_m: float,
    min_along: float,
    max_along: float,
    clip_mode: ClipMode,
) -> list[Transect]:
    padding_m = max(100.0, max_along - min_along)
    p1 = _add(_mul(normal, offset_m), _mul(direction, min_along - padding_m))
    p2 = _add(_mul(normal, offset_m), _mul(direction, max_along + padding_m))
    line = LineString([p1, p2])
    clipped = polygon.intersection(line)
    segments = _extract_lines(clipped)
    if not segments:
        return []

    if clip_mode == "qgc_chord":
        endpoints: list[XY] = []
        for segment in segments:
            coords = list(segment.coords)
            endpoints.append((float(coords[0][0]), float(coords[0][1])))
            endpoints.append((float(coords[-1][0]), float(coords[-1][1])))
        endpoints.sort(key=lambda point: _dot(point, direction))
        first, last = endpoints[0], endpoints[-1]
        if distance_xy(first, last) <= 1e-7:
            return []
        return [Transect(first, last, offset_m)]

    result: list[Transect] = []
    segments.sort(
        key=lambda segment: _dot((segment.centroid.x, segment.centroid.y), direction)
    )
    for segment in segments:
        coords = list(segment.coords)
        first: XY = (float(coords[0][0]), float(coords[0][1]))
        last: XY = (float(coords[-1][0]), float(coords[-1][1]))
        if _dot(first, direction) > _dot(last, direction):
            first, last = last, first
        if distance_xy(first, last) > 1e-7:
            result.append(Transect(first, last, offset_m))
    return result


def generate_unoriented_transects_xy(
    polygon_latlon: Sequence[LatLon],
    *,
    transect_spacing_m: float,
    angle_deg: float,
    clip_mode: ClipMode = "qgc_chord",
    projector: Optional[LocalProjector] = None,
) -> tuple[LocalProjector, list[Transect]]:
    """
    Generate clipped parallel survey segments.

    angle_deg is the line direction clockwise from geographic north:
      0 degrees -> north/south transects
      90 degrees -> east/west transects
    """
    if transect_spacing_m <= 0:
        raise ValueError("transect_spacing_m must be greater than zero")
    projector = projector or LocalProjector.from_latlon(polygon_latlon)
    polygon = _make_polygon_xy(polygon_latlon, projector)

    theta = math.radians(normalize_angle_deg(angle_deg))
    direction: XY = (math.sin(theta), math.cos(theta))
    normal: XY = (math.cos(theta), -math.sin(theta))

    vertices = [(float(x), float(y)) for x, y in polygon.exterior.coords[:-1]]
    normal_values = [_dot(vertex, normal) for vertex in vertices]
    along_values = [_dot(vertex, direction) for vertex in vertices]
    min_normal, max_normal = min(normal_values), max(normal_values)
    min_along, max_along = min(along_values), max(along_values)

    # Center the grid on the polygon. QGC generates an expanded line family and
    # clips it; centering gives stable, repeatable results while preserving the
    # same survey semantics.
    center_normal = (min_normal + max_normal) / 2.0
    k_min = math.floor((min_normal - center_normal) / transect_spacing_m) - 1
    k_max = math.ceil((max_normal - center_normal) / transect_spacing_m) + 1

    transects: list[Transect] = []
    for k in range(k_min, k_max + 1):
        offset = center_normal + k * transect_spacing_m
        transects.extend(
            _line_segments_for_offset(
                polygon=polygon,
                direction=direction,
                normal=normal,
                offset_m=offset,
                min_along=min_along,
                max_along=max_along,
                clip_mode=clip_mode,
            )
        )

    if not transects:
        # Narrow polygons still need one center transect.
        transects.extend(
            _line_segments_for_offset(
                polygon=polygon,
                direction=direction,
                normal=normal,
                offset_m=center_normal,
                min_along=min_along,
                max_along=max_along,
                clip_mode=clip_mode,
            )
        )

    if not transects:
        raise ValueError("No survey transects intersect the polygon")

    transects.sort(
        key=lambda item: (item.grid_offset_m, _dot(item.entry_xy, direction))
    )
    return projector, transects


def orient_lawnmower_transects(
    transects: Sequence[Transect],
    *,
    reverse_order: bool,
    reverse_first: bool,
) -> list[Transect]:
    """Apply QGC-style alternating direction to create a lawnmower path."""
    ordered = list(reversed(transects)) if reverse_order else list(transects)
    result: list[Transect] = []
    for index, transect in enumerate(ordered):
        should_reverse = reverse_first ^ bool(index & 1)
        result.append(transect.reversed() if should_reverse else transect)
    return result


def _orientation_variants(
    transects: Sequence[Transect],
) -> Iterator[tuple[str, list[Transect]]]:
    for reverse_order, reverse_first, name in (
        (False, False, "order-forward_entry-forward"),
        (False, True, "order-forward_entry-reversed"),
        (True, False, "order-reversed_entry-forward"),
        (True, True, "order-reversed_entry-reversed"),
    ):
        yield name, orient_lawnmower_transects(
            transects,
            reverse_order=reverse_order,
            reverse_first=reverse_first,
        )


def build_path_nodes(
    transects: Sequence[Transect],
    *,
    projector: LocalProjector,
    config: SurveyConfig,
) -> list[PathNode]:
    """Expand survey entries/exits, optional turnarounds, and hover capture points."""
    nodes: list[PathNode] = []

    def append_node(kind: NodeKind, xy: XY) -> None:
        lat, lon = projector.to_latlon(xy)
        nodes.append(
            PathNode(
                kind=kind,
                xy=xy,
                lat=lat,
                lon=lon,
                altitude_m=config.altitude_m,
            )
        )

    for transect in transects:
        direction = _unit_vector(transect.entry_xy, transect.exit_xy)

        if config.turnaround_distance_m > 0:
            append_node(
                "turnaround",
                _sub(transect.entry_xy, _mul(direction, config.turnaround_distance_m)),
            )

        append_node("survey_entry", transect.entry_xy)

        if config.trigger_camera and config.hover_and_capture:
            length_m = transect.length_m
            multiple = 1
            while multiple * config.trigger_distance_m < length_m - 1e-6:
                append_node(
                    "hover_trigger",
                    _add(
                        transect.entry_xy,
                        _mul(direction, multiple * config.trigger_distance_m),
                    ),
                )
                multiple += 1

        append_node("survey_exit", transect.exit_xy)

        if config.turnaround_distance_m > 0:
            append_node(
                "turnaround",
                _add(transect.exit_xy, _mul(direction, config.turnaround_distance_m)),
            )

    return nodes


def _xy_to_scaled_latlon(node: PathNode) -> tuple[int, int]:
    return int(round(node.lat * 1e7)), int(round(node.lon * 1e7))


def _waypoint_item(
    seq: int, node: PathNode, config: SurveyConfig, *, hold_s: float = 0.0
) -> MissionItem:
    x, y = _xy_to_scaled_latlon(node)
    return MissionItem(
        seq=seq,
        frame=config.altitude_frame,
        command=MAV_CMD_NAV_WAYPOINT,
        param1=float(hold_s),
        param2=0.0,  # no acceptance radius override
        param3=0.0,  # pass through waypoint
        param4=math.nan,  # unchanged yaw
        x=x,
        y=y,
        z=float(node.altitude_m),
        label=f"waypoint:{node.kind}",
    )


def _condition_gate_item(seq: int, node: PathNode, config: SurveyConfig) -> MissionItem:
    x, y = _xy_to_scaled_latlon(node)
    return MissionItem(
        seq=seq,
        frame=config.altitude_frame,
        command=MAV_CMD_CONDITION_GATE,
        param1=0.0,  # gate orthogonal to path
        param2=1.0,  # use altitude
        x=x,
        y=y,
        z=float(node.altitude_m),
        label=f"condition_gate:{node.kind}",
    )


def _camera_trigger_distance_item(seq: int, trigger_distance_m: float) -> MissionItem:
    return MissionItem(
        seq=seq,
        frame=MAV_FRAME_MISSION,
        command=MAV_CMD_DO_SET_CAM_TRIGG_DIST,
        param1=float(trigger_distance_m),
        param2=0.0,  # ignore shutter integration time
        param3=1.0,  # immediate image at update point, like QGC
        label=(
            "camera_trigger_distance:start"
            if trigger_distance_m > 0
            else "camera_trigger_distance:stop"
        ),
    )


def _single_capture_item(seq: int) -> MissionItem:
    return MissionItem(
        seq=seq,
        frame=MAV_FRAME_MISSION,
        command=MAV_CMD_IMAGE_START_CAPTURE,
        param1=0.0,  # all cameras
        param2=0.0,  # no interval
        param3=1.0,  # one photo
        param4=0.0,  # no explicit capture sequence number
        label="camera_single_capture",
    )


def build_mission_items(
    path_nodes: Sequence[PathNode], config: SurveyConfig
) -> list[MissionItem]:
    """
    Convert path nodes to individual mission items.

    This mirrors QGC's common transect behavior:
    - survey entry: waypoint/gate + DO_SET_CAM_TRIGG_DIST(start)
    - survey exit: waypoint/gate + DO_SET_CAM_TRIGG_DIST(stop), unless images
      remain enabled during turnaround arcs
    - hover-and-capture: held waypoint + IMAGE_START_CAPTURE
    """
    items: list[MissionItem] = []

    def append(item: MissionItem) -> None:
        item.seq = len(items)
        items.append(item)

    def append_waypoint(node: PathNode, *, hold_s: float = 0.0) -> None:
        append(_waypoint_item(len(items), node, config, hold_s=hold_s))

    def append_update_point(node: PathNode, trigger_distance_m: float) -> None:
        if config.use_condition_gate:
            append(_condition_gate_item(len(items), node, config))
        else:
            append_waypoint(node)
        append(_camera_trigger_distance_item(len(items), trigger_distance_m))

    add_trigger_at_first_and_last = (
        config.trigger_camera
        and not config.hover_and_capture
        and config.camera_trigger_in_turnarounds
    )
    has_turnarounds = config.turnaround_distance_m > 0

    for index, node in enumerate(path_nodes):
        first_node = index == 0
        last_node = index == len(path_nodes) - 1

        if node.kind == "turnaround":
            if add_trigger_at_first_and_last and (first_node or last_node):
                append_update_point(
                    node,
                    config.trigger_distance_m if first_node else 0.0,
                )
            else:
                append_waypoint(node)
            continue

        if node.kind == "hover_trigger":
            append_waypoint(node, hold_s=config.hover_delay_s)
            append(_single_capture_item(len(items)))
            continue

        if node.kind == "survey_entry":
            if not config.trigger_camera:
                append_waypoint(node)
            elif config.hover_and_capture:
                append_waypoint(node, hold_s=config.hover_delay_s)
                append(_single_capture_item(len(items)))
            else:
                # QGC deliberately restarts the trigger at every entry so a
                # resumed mission can re-fly an individual transect.
                append_update_point(node, config.trigger_distance_m)
            continue

        if node.kind == "survey_exit":
            if not config.trigger_camera:
                append_waypoint(node)
            elif config.hover_and_capture:
                append_waypoint(node, hold_s=config.hover_delay_s)
                append(_single_capture_item(len(items)))
            elif add_trigger_at_first_and_last and not has_turnarounds and last_node:
                append_update_point(node, 0.0)
            elif config.camera_trigger_in_turnarounds:
                append_waypoint(node)
            else:
                append_update_point(node, 0.0)
            continue

        raise ValueError(f"Unsupported node kind: {node.kind}")

    if items:
        items[0].current = 1
    return items


def _evaluate_oriented_plan(
    *,
    polygon_latlon: Sequence[LatLon],
    projector: LocalProjector,
    config: SurveyConfig,
    angle_deg: float,
    orientation_name: str,
    transects: list[Transect],
    start_latlon: Optional[LatLon],
    return_to_start: bool,
) -> SurveyPlan:
    nodes = build_path_nodes(transects, projector=projector, config=config)
    if not nodes:
        raise ValueError("No path nodes were generated")

    xy_path = [node.xy for node in nodes]
    survey_distance = path_distance_xy(xy_path)

    start_leg = 0.0
    return_leg = 0.0
    if start_latlon is not None:
        start_xy = projector.to_xy(start_latlon)
        start_leg = distance_xy(start_xy, xy_path[0])
        if return_to_start:
            return_leg = distance_xy(xy_path[-1], start_xy)

    mission_items = build_mission_items(nodes, config)
    return SurveyPlan(
        config=config,
        angle_deg=normalize_angle_deg(angle_deg),
        orientation_name=orientation_name,
        transects=transects,
        path_nodes=nodes,
        mission_items=mission_items,
        survey_path_distance_m=survey_distance,
        start_leg_distance_m=start_leg,
        return_leg_distance_m=return_leg,
        total_distance_m=survey_distance + start_leg + return_leg,
        polygon_latlon=list(polygon_latlon),
    )


def plan_survey(
    polygon_latlon: Sequence[LatLon],
    *,
    config: SurveyConfig,
    start_latlon: Optional[LatLon] = None,
    return_to_start: bool = False,
) -> SurveyPlan:
    """
    Plan a survey at one angle and select the cheapest of four entry variants.

    Distance includes geometric waypoint-to-waypoint travel and optionally the
    initial leg and return leg to start_latlon. Camera commands add no distance.
    """
    config.validate()
    if return_to_start and start_latlon is None:
        raise ValueError("return_to_start requires start_latlon")
    projector, base_transects = generate_unoriented_transects_xy(
        polygon_latlon,
        transect_spacing_m=config.transect_spacing_m,
        angle_deg=config.angle_deg,
        clip_mode=config.clip_mode,
    )

    candidates: list[SurveyPlan] = []
    for orientation_name, oriented in _orientation_variants(base_transects):
        candidates.append(
            _evaluate_oriented_plan(
                polygon_latlon=polygon_latlon,
                projector=projector,
                config=config,
                angle_deg=config.angle_deg,
                orientation_name=orientation_name,
                transects=oriented,
                start_latlon=start_latlon,
                return_to_start=return_to_start,
            )
        )
    return min(
        candidates, key=lambda plan: (plan.total_distance_m, plan.orientation_name)
    )


def _angle_values(*, start_deg: float, stop_deg: float, step_deg: float) -> list[float]:
    if step_deg <= 0:
        raise ValueError("angle step must be greater than zero")
    if stop_deg <= start_deg:
        raise ValueError("angle stop must be greater than angle start")
    values: list[float] = []
    angle = start_deg
    while angle < stop_deg - 1e-12:
        values.append(normalize_angle_deg(angle))
        angle += step_deg
    # Avoid duplicate normalized angles if callers use a range wider than 180.
    unique: list[float] = []
    seen: set[float] = set()
    for value in values:
        key = round(value, 9)
        if key not in seen:
            unique.append(value)
            seen.add(key)
    return unique


def optimize_survey_angle(
    polygon_latlon: Sequence[LatLon],
    *,
    config: SurveyConfig,
    start_latlon: Optional[LatLon] = None,
    return_to_start: bool = False,
    angle_start_deg: float = 0.0,
    angle_stop_deg: float = 180.0,
    angle_step_deg: float = 1.0,
) -> tuple[SurveyPlan, list[AngleScore]]:
    """Scan survey angles and return the minimum-distance plan and all scores."""
    config.validate()
    scores_and_plans: list[tuple[AngleScore, SurveyPlan]] = []
    for angle in _angle_values(
        start_deg=angle_start_deg,
        stop_deg=angle_stop_deg,
        step_deg=angle_step_deg,
    ):
        angle_config = replace(config, angle_deg=angle)
        plan = plan_survey(
            polygon_latlon,
            config=angle_config,
            start_latlon=start_latlon,
            return_to_start=return_to_start,
        )
        scores_and_plans.append((AngleScore.from_plan(plan), plan))

    if not scores_and_plans:
        raise ValueError("No angle candidates were evaluated")

    scores_and_plans.sort(
        key=lambda item: (
            item[0].total_distance_m,
            item[0].transect_count,
            item[0].angle_deg,
        )
    )
    best_score, best_plan = scores_and_plans[0]
    del best_score
    return best_plan, [score for score, _ in scores_and_plans]


def save_plan_json(plan: SurveyPlan, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(plan.to_dict(), indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )


def save_path_geojson(plan: SurveyPlan, path: str | Path) -> None:
    line = LineString([(node.lon, node.lat) for node in plan.path_nodes])
    features = [
        {
            "type": "Feature",
            "properties": {"name": "survey_path", **plan.summary()},
            "geometry": mapping(line),
        },
        {
            "type": "Feature",
            "properties": {"name": "survey_polygon"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        (lon, lat)
                        for lat, lon in plan.polygon_latlon + [plan.polygon_latlon[0]]
                    ]
                ],
            },
        },
    ]
    # Transect endpoints are stored in local XY coordinates, so recreate the
    # deterministic local projection to export them as WGS84 coordinates.
    projector = LocalProjector.from_latlon(plan.polygon_latlon)
    for index, transect in enumerate(plan.transects):
        entry_lat, entry_lon = projector.to_latlon(transect.entry_xy)
        exit_lat, exit_lon = projector.to_latlon(transect.exit_xy)
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "name": f"transect_{index:03d}",
                    "index": index,
                    "length_m": transect.length_m,
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [(entry_lon, entry_lat), (exit_lon, exit_lat)],
                },
            }
        )
    Path(path).write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _send_mission_count(connection: Any, count: int, mission_type: int) -> None:
    try:
        connection.mav.mission_count_send(
            connection.target_system,
            connection.target_component,
            count,
            mission_type,
        )
    except TypeError:
        # Compatibility with older generated dialects without extension fields.
        connection.mav.mission_count_send(
            connection.target_system,
            connection.target_component,
            count,
        )


def _send_mission_item_int(
    connection: Any, item: MissionItem, mission_type: int
) -> None:
    args = (
        connection.target_system,
        connection.target_component,
        item.seq,
        item.frame,
        item.command,
        item.current,
        item.autocontinue,
        item.param1,
        item.param2,
        item.param3,
        item.param4,
        item.x,
        item.y,
        item.z,
    )
    try:
        connection.mav.mission_item_int_send(*args, mission_type)
    except TypeError:
        connection.mav.mission_item_int_send(*args)


def upload_mission_items(
    connection: Any,
    mission_items: Sequence[MissionItem],
    *,
    mission_type: int = MAV_MISSION_TYPE_MISSION,
    timeout_s: float = 1.5,
    max_retries: int = 5,
) -> Any:
    """
    Upload an already-expanded flat mission list using the MAVLink mission protocol.

    The protocol sends MISSION_COUNT, responds to MISSION_REQUEST_INT with the
    requested MISSION_ITEM_INT, and completes only after MAV_MISSION_ACCEPTED.
    """
    if timeout_s <= 0 or max_retries < 0:
        raise ValueError(
            "timeout_s must be positive and max_retries cannot be negative"
        )

    items = list(mission_items)
    for seq, item in enumerate(items):
        item.seq = seq
        item.current = 1 if seq == 0 else 0

    _send_mission_count(connection, len(items), mission_type)
    last_item: Optional[MissionItem] = None
    retries = 0

    while True:
        msg = connection.recv_match(
            type=["MISSION_REQUEST_INT", "MISSION_REQUEST", "MISSION_ACK"],
            blocking=True,
            timeout=timeout_s,
        )

        if msg is None:
            if retries >= max_retries:
                raise TimeoutError("Mission upload timed out after retry limit")
            retries += 1
            if last_item is None:
                _send_mission_count(connection, len(items), mission_type)
            else:
                _send_mission_item_int(connection, last_item, mission_type)
            continue

        if (
            hasattr(msg, "get_srcSystem")
            and msg.get_srcSystem() != connection.target_system
        ):
            continue

        msg_type = msg.get_type()
        if msg_type in ("MISSION_REQUEST_INT", "MISSION_REQUEST"):
            seq = int(msg.seq)
            if seq < 0 or seq >= len(items):
                raise RuntimeError(
                    f"Vehicle requested invalid mission sequence number {seq}"
                )
            last_item = items[seq]
            _send_mission_item_int(connection, last_item, mission_type)
            retries = 0
            continue

        if msg_type == "MISSION_ACK":
            result = int(msg.type)
            if result != MAV_MISSION_ACCEPTED:
                raise RuntimeError(
                    f"Mission upload rejected with MAV_MISSION_RESULT={result}"
                )
            return msg


def connect_and_upload(
    connection_string: str,
    mission_items: Sequence[MissionItem],
    *,
    baud: int = 115200,
    heartbeat_timeout_s: float = 10.0,
    timeout_s: float = 1.5,
    max_retries: int = 5,
) -> Any:
    """Open a pymavlink connection, wait for PX4 heartbeat, and upload the mission."""
    # MAVLink 2 is strongly preferred for modern mission protocol extension fields.
    os.environ.setdefault("MAVLINK20", "1")
    try:
        from pymavlink import mavutil
    except ImportError as exc:
        raise RuntimeError(
            "pymavlink is required for upload: pip install pymavlink"
        ) from exc

    connection = mavutil.mavlink_connection(
        connection_string,
        baud=baud,
        source_system=255,
        source_component=190,
        dialect="common",
    )
    heartbeat = connection.wait_heartbeat(timeout=heartbeat_timeout_s)
    if heartbeat is None:
        raise TimeoutError("No vehicle heartbeat received")
    return upload_mission_items(
        connection,
        mission_items,
        timeout_s=timeout_s,
        max_retries=max_retries,
    )


def load_polygon_geojson(path: str | Path) -> list[LatLon]:
    """Read the exterior ring of the first Polygon geometry from GeoJSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    def polygon_from_geometry(geometry: dict[str, Any]) -> Optional[Polygon]:
        geometry_type = geometry.get("type")
        if geometry_type == "Polygon":
            return shape(geometry)
        if geometry_type == "MultiPolygon":
            multi = shape(geometry)
            if not multi.geoms:
                return None
            return max(multi.geoms, key=lambda item: item.area)
        return None

    polygon: Optional[Polygon] = None
    if data.get("type") in ("Polygon", "MultiPolygon"):
        polygon = polygon_from_geometry(data)
    elif data.get("type") == "Feature":
        polygon = polygon_from_geometry(data["geometry"])
    elif data.get("type") == "FeatureCollection":
        for feature in data.get("features", []):
            polygon = polygon_from_geometry(feature.get("geometry", {}))
            if polygon is not None:
                break

    if polygon is None:
        raise ValueError("GeoJSON does not contain a Polygon geometry")

    coords = list(polygon.exterior.coords)
    if len(coords) >= 2 and coords[0] == coords[-1]:
        coords = coords[:-1]
    return [(float(lat), float(lon)) for lon, lat in coords]


def _parse_latlon(text: str) -> LatLon:
    try:
        lat_text, lon_text = text.split(",", maxsplit=1)
        latlon = float(lat_text.strip()), float(lon_text.strip())
    except Exception as exc:
        raise argparse.ArgumentTypeError("Expected LAT,LON") from exc
    lat, lon = latlon
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise argparse.ArgumentTypeError("Latitude or longitude is out of range")
    return latlon


def _score_table(scores: Sequence[AngleScore], limit: int = 10) -> str:
    lines = [
        "rank  angle_deg  total_m   survey_m  start_m  return_m  transects  mission_items  orientation",
        "----  ---------  --------  --------  -------  --------  ---------  -------------  -----------",
    ]
    for rank, score in enumerate(scores[:limit], start=1):
        lines.append(
            f"{rank:>4}  {score.angle_deg:>9.2f}  {score.total_distance_m:>8.1f}  "
            f"{score.survey_path_distance_m:>8.1f}  {score.start_leg_distance_m:>7.1f}  "
            f"{score.return_leg_distance_m:>8.1f}  {score.transect_count:>9}  "
            f"{score.mission_item_count:>13}  {score.orientation_name}"
        )
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and optionally upload QGC-style survey mission items with pymavlink."
    )
    parser.add_argument(
        "--polygon", required=True, help="GeoJSON file containing a Polygon"
    )
    parser.add_argument(
        "--altitude", type=float, required=True, help="Survey altitude in meters"
    )
    parser.add_argument(
        "--spacing", type=float, required=True, help="Transect spacing in meters"
    )
    parser.add_argument(
        "--trigger-distance",
        type=float,
        required=True,
        help="Camera trigger distance in meters",
    )
    parser.add_argument(
        "--angle", default="auto", help='Grid direction clockwise from north, or "auto"'
    )
    parser.add_argument(
        "--angle-step", type=float, default=1.0, help="Angle optimizer step in degrees"
    )
    parser.add_argument(
        "--start",
        type=_parse_latlon,
        help="Optional LAT,LON for initial and return leg estimates",
    )
    parser.add_argument(
        "--return-to-start",
        action="store_true",
        help="Include return leg in optimization",
    )
    parser.add_argument(
        "--turnaround-distance",
        type=float,
        default=0.0,
        help="Extend each transect by this many meters",
    )
    parser.add_argument(
        "--images-in-turnarounds",
        action="store_true",
        help="Keep distance triggering enabled through turns",
    )
    parser.add_argument(
        "--condition-gate",
        action="store_true",
        help="Use MAV_CMD_CONDITION_GATE before trigger updates",
    )
    parser.add_argument(
        "--hover-and-capture",
        action="store_true",
        help="Use waypoint holds and single-image commands",
    )
    parser.add_argument(
        "--hover-delay",
        type=float,
        default=0.0,
        help="Waypoint hold duration for hover-and-capture",
    )
    parser.add_argument(
        "--altitude-frame",
        choices=("relative", "absolute", "terrain"),
        default="relative",
        help="Altitude frame for navigation items",
    )
    parser.add_argument(
        "--clip-mode",
        choices=("qgc_chord", "inside_segments"),
        default="qgc_chord",
        help="Polygon clipping behavior for concave polygons",
    )
    parser.add_argument(
        "--output-json",
        default="survey_mission.json",
        help="Write expanded plan JSON here",
    )
    parser.add_argument(
        "--output-geojson",
        default="survey_path.geojson",
        help="Write preview GeoJSON here",
    )
    parser.add_argument(
        "--top", type=int, default=10, help="Print this many optimized angle candidates"
    )
    parser.add_argument("--upload", action="store_true", help="Upload after generation")
    parser.add_argument(
        "--connection",
        default="udpout:127.0.0.1:14540",
        help="pymavlink connection string",
    )
    parser.add_argument(
        "--baud", type=int, default=115200, help="Serial baud rate, if applicable"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    polygon = load_polygon_geojson(args.polygon)
    altitude_frames = {
        "relative": MAV_FRAME_GLOBAL_RELATIVE_ALT,
        "absolute": MAV_FRAME_GLOBAL,
        "terrain": MAV_FRAME_GLOBAL_TERRAIN_ALT,
    }
    config = SurveyConfig(
        altitude_m=args.altitude,
        transect_spacing_m=args.spacing,
        trigger_distance_m=args.trigger_distance,
        turnaround_distance_m=args.turnaround_distance,
        camera_trigger_in_turnarounds=args.images_in_turnarounds,
        hover_and_capture=args.hover_and_capture,
        hover_delay_s=args.hover_delay,
        use_condition_gate=args.condition_gate,
        altitude_frame=altitude_frames[args.altitude_frame],
        clip_mode=args.clip_mode,
    )

    if str(args.angle).lower() == "auto":
        plan, scores = optimize_survey_angle(
            polygon,
            config=config,
            start_latlon=args.start,
            return_to_start=args.return_to_start,
            angle_step_deg=args.angle_step,
        )
        print(_score_table(scores, limit=max(1, args.top)))
    else:
        angle = float(args.angle)
        plan = plan_survey(
            polygon,
            config=replace(config, angle_deg=angle),
            start_latlon=args.start,
            return_to_start=args.return_to_start,
        )

    save_plan_json(plan, args.output_json)
    save_path_geojson(plan, args.output_geojson)

    print("\nSelected plan:")
    print(json.dumps(plan.summary(), indent=2))
    print(f"\nExpanded mission: {args.output_json}")
    print(f"Preview path:     {args.output_geojson}")

    if args.upload:
        print(
            f"\nUploading {len(plan.mission_items)} mission items through {args.connection!r} ..."
        )
        ack = connect_and_upload(args.connection, plan.mission_items, baud=args.baud)
        opaque_id = getattr(ack, "opaque_id", None)
        if opaque_id is None:
            print("Mission accepted.")
        else:
            print(f"Mission accepted. Vehicle opaque mission id: {opaque_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
