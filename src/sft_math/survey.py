
import math
import time
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class CameraParams:
    sensor_width_mm: float
    sensor_height_mm: float
    image_width_px: float
    image_height_px: float
    focal_length_mm: float
    frontal_overlap: float  # 0-1
    side_overlap: float  # 0-1
    altitude_m: float
    landscape: bool = True


@dataclass
class Transect:
    """Coppia inizio/fine di una striscia di volo."""

    start: list[tuple[float, float]]
    end: list[tuple[float, float]]
    altitude_m: float

EARTH_R = 6_371_000.0

def camera_calc(params: CameraParams) -> dict:
    image_density = (params.altitude_m * params.sensor_width_mm * 100.0) / (
        params.image_width_px * params.focal_length_mm
    )
    if params.landscape:
        fp_side = (params.image_width_px * image_density) / 100.0
        fp_frontal = (params.image_height_px * image_density) / 100.0
    else:
        fp_side = (params.image_height_px * image_density) / 100.0
        fp_frontal = (params.image_width_px * image_density) / 100.0

    return {
        "adjusted_footprint_side_m": fp_side * (1.0 - params.side_overlap),
        "adjusted_footprint_frontal_m": fp_frontal * (1.0 - params.frontal_overlap),
    }

def _to_xy(origin, point):
    lat0 = math.radians(origin[0])
    x = math.radians(point[1] - origin[1]) * EARTH_R * math.cos(lat0)
    y = math.radians(point[0] - origin[0]) * EARTH_R
    return x, y

    
def _to_latlon(origin, x, y):
    lat0 = math.radians(origin[0])
    return [(
        origin[0] + math.degrees(y / EARTH_R),
        origin[1] + math.degrees(x / (EARTH_R * math.cos(lat0))),
    )]


def _rotate(x, y, cx, cy, a):
    s, c = math.sin(a), math.cos(a)
    x -= cx
    y -= cy
    return c * x - s * y + cx, s * x + c * y + cy


def _intersections(x0, y0, x1, y1, poly):
    ts, dx, dy = [], x1 - x0, y1 - y0
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        edx, edy = bx - ax, by - ay
        d = dx * edy - dy * edx
        if abs(d) < 1e-12:
            continue
        t = ((ax - x0) * edy - (ay - y0) * edx) / d
        u = ((ax - x0) * dy - (ay - y0) * dx) / d
        if -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9:
            ts.append(max(0.0, min(1.0, t)))
    return sorted(set(round(t, 10) for t in ts))


def _in_poly(px, py, poly):
    inside, j = False, len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _clip(x0, y0, x1, y1, poly):
    ts = _intersections(x0, y0, x1, y1, poly)
    segs = []
    for i in range(0, len(ts) - 1, 2):
        ta, tb = ts[i], ts[i + 1]
        mx = x0 + (ta + tb) / 2 * (x1 - x0)
        my = y0 + (ta + tb) / 2 * (y1 - y0)
        if _in_poly(mx, my, poly):
            segs.append((x0 + ta * (x1 - x0), x0 + tb * (x1 - x0)))
    return segs


def build_transects(
    polygon: list[tuple[float, float]],
    camera: CameraParams,
    grid_angle_deg: float = 0.0,
) -> Tuple[List[Transect], dict]:
    """
    Genera i transect (inizio/fine striscia) dal poligono e dai parametri camera.
    Restituisce solo gli estremi — il trigger camera avviene in volo autonomamente.
    """
    calc = camera_calc(camera)
    step_side = calc["adjusted_footprint_side_m"]
    step_frontal = calc["adjusted_footprint_frontal_m"]

    origin = polygon[0]
    pts = [_to_xy(origin, p) for p in polygon]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)

    a = math.radians(-grid_angle_deg)
    rot = [_rotate(x, y, cx, cy, a) for x, y in pts]

    xs = [p[0] for p in rot]
    ys = [p[1] for p in rot]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    transects = []
    y = y_min
    stripe = 0
    while y <= y_max + step_side:
        for xa, xb in _clip(x_min - 10, y, x_max + 10, y, rot):
            # Inverte il verso nelle strisce pari (serpentina)
            if stripe % 2 == 0:
                p_start = (xa, y)
                p_end = (xb, y)
            else:
                p_start = (xb, y)
                p_end = (xa, y)

            inv = math.radians(grid_angle_deg)
            sx, sy = _rotate(p_start[0], p_start[1], cx, cy, inv)
            ex, ey = _rotate(p_end[0], p_end[1], cx, cy, inv)

            transects.append(
                Transect(
                    start=_to_latlon(origin, sx, sy),
                    end=_to_latlon(origin, ex, ey),
                    altitude_m=camera.altitude_m,
                )
            )
            stripe += 1
        y += step_side

    return transects, {
        **calc,
        "n_transects": len(transects),
        "trigger_distance_m": step_frontal,
    }

def single_waypoint(
    lat: float,
    lon: float,
    altitude_m: float,
):
    """
    Crea e carica su Pixhawk una missione composta da un solo waypoint.

    lat/lon: gradi decimali
    altitude_m: quota relativa in metri
    """

    item = dict(
        frame=3,  # MAV_FRAME_GLOBAL_RELATIVE_ALT
        command=16,  # MAV_CMD_NAV_WAYPOINT
        param1=0,  # hold time
        param2=2,  # acceptance radius
        param3=0,
        param4=float("nan"),  # yaw non specificato
        lat=lat,
        lon=lon,
        alt=altitude_m,
    )

    items = [item]

    return items