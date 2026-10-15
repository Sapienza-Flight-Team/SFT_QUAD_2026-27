#!/usr/bin/env python3
"""Debug pixel-to-body matrix conventions with a synthetic homography image.

This script creates a blank grid image, projects a few known image points to the
ground, computes the north-up homography, and saves a warped debug image using
the px2body matrix from YAML. It intentionally uses the same functions as
src/sft_cv/inference/bullseye_homo_cv.py so the result matches the CV stage.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.config import SFTConfig
from stuff_i_moved_away.inference import bullseye_homo_cv as homo
from sft_math.cam import Camera
from sft_math.cam.affine import Affine
from sft_math.cam.px2body_matrix import resolve_px2body_matrix


def parse_size(value: str) -> tuple[int, int]:
    width, height = value.lower().replace(",", "x").split("x", maxsplit=1)
    return max(1, int(float(width))), max(1, int(float(height)))


def config_image_size(config: dict[str, Any]) -> tuple[int, int]:
    geoloc_cfg = config.get("decision_system", {}).get("geoloc", {})
    camera_cfg = config.get("camera", {})
    configured = (
        geoloc_cfg.get("image_shape")
        or camera_cfg.get("resolution")
        or camera_cfg.get("photo_shape")
    )
    if configured is not None and len(configured) >= 2:
        return int(configured[0]), int(configured[1])
    return 5320, 3032


def config_lat_lon(config: dict[str, Any]) -> tuple[float, float]:
    geoloc_cfg = config.get("decision_system", {}).get("geoloc", {})
    center = geoloc_cfg.get("manifold_center")
    if center is not None and len(center) >= 2:
        return float(center[0]), float(center[1])
    return 0.0, 0.0


def synthetic_metadata_row(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    lat, lon = config_lat_lon(config)
    return {
        "filename": "<synthetic_blank>",
        "lat": args.lat if args.lat is not None else lat,
        "lon": args.lon if args.lon is not None else lon,
        "alt_rel": args.altitude_m,
        "roll": math.radians(args.roll_deg),
        "pitch": math.radians(args.pitch_deg),
        "yaw": math.radians(args.yaw_deg),
        "heading": args.yaw_deg,
        "pixel_format": "RGB8",
    }


def candidate_matrices(config: dict[str, Any]) -> dict[str, list[list[float]]]:
    geoloc_cfg = config.get("decision_system", {}).get("geoloc", {})
    camera_cfg = config.get("camera", {})
    config_matrix = resolve_px2body_matrix(geoloc_cfg, camera_cfg)
    return {"config_matrix": config_matrix}


def ground_ned_for_points(
    points_xy: np.ndarray,
    row: dict[str, Any],
    config: dict[str, Any],
    image_width: int,
    image_height: int,
) -> np.ndarray:
    altitude = float(row["alt_rel"])
    roll, pitch, yaw = homo.metadata_angles(row, config)
    camera = homo.build_camera(config, image_width, image_height)
    camera_config = Camera.Configuration(
        Affine((0.0, 0.0, -altitude), ((), (), ())),
        roll,
        pitch,
        yaw,
        is_deg=False,
    )

    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    rays = camera.px2ned(Affine(points, np.empty((2, 0))), camera_config)
    origins = np.asarray(rays.vector, dtype=np.float64)
    directions = np.asarray(rays.orthCoos2globalCoos_matrix[..., :, 0], dtype=np.float64)

    if np.any(np.abs(directions[:, 2]) < 1e-9):
        raise ValueError("at least one ray is parallel to the ground")

    distance_along_ray = -origins[:, 2] / directions[:, 2]
    return origins + directions * distance_along_ray[:, None]


def named_points(image_width: int, image_height: int, delta_px: float) -> dict[str, tuple[float, float]]:
    center = ((image_width - 1) / 2.0, (image_height - 1) / 2.0)
    return {
        "center": center,
        "right": (min(image_width - 1.0, center[0] + delta_px), center[1]),
        "left": (max(0.0, center[0] - delta_px), center[1]),
        "down": (center[0], min(image_height - 1.0, center[1] + delta_px)),
        "up": (center[0], max(0.0, center[1] - delta_px)),
        "top_left": (0.0, 0.0),
        "top_right": (image_width - 1.0, 0.0),
        "bottom_right": (image_width - 1.0, image_height - 1.0),
        "bottom_left": (0.0, image_height - 1.0),
    }


def make_debug_image(image_width: int, image_height: int, points: dict[str, tuple[float, float]]) -> np.ndarray:
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    image[:, :] = (22, 22, 22)

    grid_x = max(1, image_width // 8)
    grid_y = max(1, image_height // 8)
    for x in range(0, image_width, grid_x):
        cv2.line(image, (x, 0), (x, image_height - 1), (46, 46, 46), 2)
    for y in range(0, image_height, grid_y):
        cv2.line(image, (0, y), (image_width - 1, y), (46, 46, 46), 2)

    draw_marker(image, points["center"], (255, 255, 255), "center")
    draw_marker(image, points["right"], (0, 0, 255), "right +x")
    draw_marker(image, points["down"], (0, 255, 0), "down +y")
    draw_marker(image, points["up"], (255, 0, 255), "up -y")
    draw_marker(image, points["left"], (255, 255, 0), "left -x")
    return image


def draw_marker(
    image: np.ndarray,
    point_xy: tuple[float, float] | np.ndarray,
    color_bgr: tuple[int, int, int],
    label: str,
) -> None:
    x = int(round(float(point_xy[0])))
    y = int(round(float(point_xy[1])))
    if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
        return
    radius = max(7, min(image.shape[:2]) // 120)
    cv2.circle(image, (x, y), radius, color_bgr, thickness=-1)
    cv2.circle(image, (x, y), radius + 3, (255, 255, 255), thickness=2)
    cv2.putText(
        image,
        label,
        (min(x + radius + 8, image.shape[1] - 1), max(20, y - radius - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color_bgr,
        2,
        cv2.LINE_AA,
    )


def cardinal(delta_north_east: np.ndarray) -> str:
    north, east = float(delta_north_east[0]), float(delta_north_east[1])
    if abs(north) >= abs(east):
        primary = "north" if north >= 0 else "south"
        secondary = "east" if east >= 0 else "west"
    else:
        primary = "east" if east >= 0 else "west"
        secondary = "north" if north >= 0 else "south"
    return f"{primary}, {secondary}"


def matrix_label_filename(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label)


def run_candidate(
    label: str,
    matrix: list[list[float]],
    source_image: np.ndarray,
    points: dict[str, tuple[float, float]],
    row: dict[str, Any],
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    image_height, image_width = source_image.shape[:2]
    points_array = np.asarray(list(points.values()), dtype=np.float32)
    point_names = list(points.keys())

    ground_ned = ground_ned_for_points(
        points_array, row, config, image_width, image_height
    )
    warped_image, computation = homo.warp_image_to_north_up(
        source_image,
        row,
        config,
        output_size=args.output_size,
        max_output_side=args.max_output_side,
        padding_px=args.padding_px,
    )
    warped_points = homo.transform_points(points_array, computation.matrix)

    marked_source = source_image.copy()
    marked_warped = warped_image.copy()
    for name, point in points.items():
        color = {
            "center": (255, 255, 255),
            "right": (0, 0, 255),
            "down": (0, 255, 0),
            "up": (255, 0, 255),
            "left": (255, 255, 0),
        }.get(name, (255, 128, 0))
        draw_marker(marked_source, point, color, name)
    for name, point in zip(point_names, warped_points):
        color = {
            "center": (255, 255, 255),
            "right": (0, 0, 255),
            "down": (0, 255, 0),
            "up": (255, 0, 255),
            "left": (255, 255, 0),
        }.get(name, (255, 128, 0))
        draw_marker(marked_warped, point, color, name)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = matrix_label_filename(label)
    source_path = out_dir / f"{stem}_source.png"
    warped_path = out_dir / f"{stem}_warped.png"
    cv2.imwrite(str(source_path), marked_source)
    cv2.imwrite(str(warped_path), marked_warped)

    center_index = point_names.index("center")
    center_ground = ground_ned[center_index]
    deltas: dict[str, dict[str, Any]] = {}
    for name in ("right", "left", "down", "up"):
        idx = point_names.index(name)
        delta_ne = ground_ned[idx, :2] - center_ground[:2]
        deltas[name] = {
            "north_m": float(delta_ne[0]),
            "east_m": float(delta_ne[1]),
            "direction": cardinal(delta_ne),
            "warped_dx_px": float(warped_points[idx, 0] - warped_points[center_index, 0]),
            "warped_dy_px": float(warped_points[idx, 1] - warped_points[center_index, 1]),
        }

    return {
        "label": label,
        "matrix": matrix,
        "source_path": str(source_path),
        "warped_path": str(warped_path),
        "image_width": image_width,
        "image_height": image_height,
        "homography_output_size_w_h": list(computation.output_size),
        "meters_per_pixel": computation.meters_per_pixel,
        "flat_world_corners_east_minus_north_m": computation.flat_world_corners_m.tolist(),
        "destination_corners_px": computation.destination_corners_px.tolist(),
        "points": {
            name: {
                "pixel_xy": [float(points_array[i, 0]), float(points_array[i, 1])],
                "ground_ned_m": [
                    float(ground_ned[i, 0]),
                    float(ground_ned[i, 1]),
                    float(ground_ned[i, 2]),
                ],
                "warped_xy": [float(warped_points[i, 0]), float(warped_points[i, 1])],
            }
            for i, name in enumerate(point_names)
        },
        "deltas_from_center": deltas,
    }


def print_candidate(result: dict[str, Any]) -> None:
    print()
    print("=" * 88)
    print(result["label"])
    print(f"matrix: {result['matrix']}")
    print(
        "image W,H: "
        f"{result['image_width']},{result['image_height']}  "
        f"homography W,H: {tuple(result['homography_output_size_w_h'])}  "
        f"m/px: {result['meters_per_pixel']:.6f}"
    )
    print(f"source image: {result['source_path']}")
    print(f"warped image: {result['warped_path']}")
    print("pixel-axis movement from center:")
    for name in ("right", "left", "down", "up"):
        delta = result["deltas_from_center"][name]
        print(
            f"  {name:>5}: "
            f"dN={delta['north_m']:>9.3f} m  "
            f"dE={delta['east_m']:>9.3f} m  "
            f"=> {delta['direction']:<14}  "
            f"warped dxy=({delta['warped_dx_px']:>8.2f}, {delta['warped_dy_px']:>8.2f}) px"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic px2body/homography sanity checker."
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "sft_config.yaml"))
    parser.add_argument("--size", default=None, help="Synthetic image size as WIDTHxHEIGHT.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "data" / "outs" / "px2body_debug"))
    parser.add_argument("--output-size", default=None, help="Forced homography output as WIDTHxHEIGHT.")
    parser.add_argument("--max-output-side", type=int, default=1200)
    parser.add_argument("--padding-px", type=int, default=0)
    parser.add_argument("--delta-px", type=float, default=300.0)
    parser.add_argument("--altitude-m", type=float, default=80.0)
    parser.add_argument("--roll-deg", type=float, default=0.0)
    parser.add_argument("--pitch-deg", type=float, default=0.0)
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument(
        "--matrix",
        default="config_matrix",
        choices=("config_matrix",),
        help="Matrix label to run. Only the YAML config matrix is available.",
    )
    args = parser.parse_args()
    args.output_size = parse_size(args.output_size) if args.output_size else None
    return args


def main() -> None:
    args = parse_args()
    config = SFTConfig(str(args.config), setup_logger=False).config_dict
    image_width, image_height = parse_size(args.size) if args.size else config_image_size(config)
    points = named_points(image_width, image_height, args.delta_px)
    row = synthetic_metadata_row(args, config)
    source_image = make_debug_image(image_width, image_height, points)
    matrices = candidate_matrices(config)
    matrices = {args.matrix: matrices[args.matrix]}

    print("Synthetic px2body/homography sanity check")
    print(f"config: {args.config}")
    print(f"blank image shape H,W,C: {source_image.shape}")
    print(f"camera image W,H: {image_width},{image_height}")
    print(
        "metadata: "
        f"alt={row['alt_rel']}m roll={args.roll_deg}deg "
        f"pitch={args.pitch_deg}deg yaw={args.yaw_deg}deg"
    )
    print(
        "Interpretation: ground deltas are NED. In a north-up warped image, "
        "+east should move right and +north should move up."
    )

    results = []
    for label, matrix in matrices.items():
        result = run_candidate(label, matrix, source_image, points, row, config, args)
        results.append(result)
        print_candidate(result)

    report_path = Path(args.output_dir) / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "config": str(args.config),
                "image_width": image_width,
                "image_height": image_height,
                "metadata": row,
                "results": results,
            },
            stream,
            indent=2,
        )
    print()
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
