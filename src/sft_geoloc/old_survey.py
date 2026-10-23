"""Camera footprints and clipped survey lanes in latitude/longitude degrees."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from shapely.affinity import rotate
from shapely.geometry import LineString, Polygon

from . import check
from .geodesy import LocalMap


@dataclass
class CameraParams:
    """Physical sensor dimensions, image resolution, overlap fractions and height."""

    sensor_width_mm: float
    sensor_height_mm: float
    image_width_px: int
    image_height_px: int
    focal_length_mm: float
    frontal_overlap: float
    side_overlap: float
    altitude_m: float
    landscape: bool = True

    def __post_init__(self) -> None:
        check.survey_camera(self)


@dataclass
class Transect:
    """One flight segment; endpoints contain a single (latitude, longitude) pair."""

    start: list[tuple[float, float]]
    end: list[tuple[float, float]]
    altitude_m: float


def camera_calc(params: CameraParams) -> dict[str, float]:
    """Return lane and trigger spacing in metres after accounting for overlap.

    Both physical sensor dimensions are used, without assuming square pixels.
    Landscape places the sensor width across the flight path.
    """
    check.survey_camera(params)
    width = params.altitude_m * params.sensor_width_mm / params.focal_length_mm
    height = params.altitude_m * params.sensor_height_mm / params.focal_length_mm
    side, frontal = (width, height) if params.landscape else (height, width)
    return {
        "adjusted_footprint_side_m": side * (1 - params.side_overlap),
        "adjusted_footprint_frontal_m": frontal * (1 - params.frontal_overlap),
    }


def build_transects(
    polygon: ArrayLike,
    camera: CameraParams,
    grid_angle_deg: float = 0.0,
) -> tuple[list[Transect], dict[str, float | int]]:
    """Clip serpentine lanes to a simple survey polygon, including concave areas.

    Angle is counterclockwise from east. Lanes are centered in equal-width
    strips with spacing no greater than the adjusted camera footprint.
    Disconnected intersections produce separate segments; connecting flight
    legs between segments are the mission planner's responsibility.
    """
    calc = camera_calc(camera)
    vertices = np.asarray(polygon, dtype=np.float64)
    check.survey_vertices(vertices)
    check.finite_scalar(grid_angle_deg, name="grid_angle_deg")
    local_map = LocalMap(vertices[0])
    boundary = Polygon(local_map.gps_to_ne(vertices)[:, ::-1])
    check.survey_polygon(boundary)
    center = boundary.centroid.coords[0]
    aligned = rotate(boundary, -grid_angle_deg, origin=center)
    xmin, ymin, xmax, ymax = aligned.bounds
    count = max(1, int(np.ceil((ymax - ymin) / calc["adjusted_footprint_side_m"])))
    spacing = (ymax - ymin) / count
    transects = []
    for stripe in range(count):
        y = ymin + (stripe + 0.5) * spacing
        intersection = aligned.intersection(LineString([(xmin - 1, y), (xmax + 1, y)]))
        parts = (
            [intersection]
            if intersection.geom_type == "LineString"
            else intersection.geoms
        )
        segments = sorted(
            (
                part
                for part in parts
                if part.geom_type == "LineString" and part.length > 0
            ),
            key=lambda part: part.bounds[0],
        )
        if stripe % 2:
            segments.reverse()
        for segment in segments:
            endpoints = sorted([segment.coords[0], segment.coords[-1]])
            if stripe % 2:
                endpoints.reverse()
            restored = rotate(LineString(endpoints), grid_angle_deg, origin=center)
            gps = local_map.ne_to_gps(np.asarray(restored.coords)[:, ::-1])
            transects.append(
                Transect([tuple(gps[0])], [tuple(gps[1])], camera.altitude_m)
            )
    return transects, {
        **calc,
        "n_transects": len(transects),
        "trigger_distance_m": calc["adjusted_footprint_frontal_m"],
    }


def single_waypoint(lat: float, lon: float, altitude_m: float) -> list[dict]:
    """Build one MAVLink relative-altitude waypoint; this function performs no I/O."""
    check.waypoint_position(np.asarray([lat, lon], dtype=np.float64), altitude_m)
    return [
        dict(
            frame=3,
            command=16,
            param1=0,
            param2=2,
            param3=0,
            param4=float("nan"),
            lat=lat,
            lon=lon,
            alt=altitude_m,
        )
    ]
