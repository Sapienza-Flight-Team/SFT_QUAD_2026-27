import argparse
import json
import math
import os
import cv2
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for path in (REPO_ROOT, SRC_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import multiprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm

from common.mavlink_structs import MissionMode, MissionState

PIXHAWK_METADATA_COLUMNS = [
    "filename",
    "timestamp",
    "exposure_time",
    "gain",
    "black_level",
    "bsl_color_space",
    "gamma",
    "light_source_preset",
    "pixel_format",
    "teensy_timestamp",
    "lat",
    "lon",
    "alt_msl",
    "vground",
    "roll",
    "pitch",
    "yaw",
    "heading",
    "mission_mode",
    "mission_state",
]


def _validate_vector(name: str, value: object, image_id: int) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} for image_id={image_id} must be a 3-value list")
    return [float(item) for item in value]


def _annotation_pose(
    image_id: int, annotations: list[dict]
) -> tuple[list[float], list[float]]:
    if not annotations:
        raise ValueError(f"No annotations found for image_id={image_id}")

    first_annotation = annotations[0]
    cam_location = _validate_vector(
        "cam_location", first_annotation.get("cam_location"), image_id
    )
    cam_rotation = _validate_vector(
        "cam_rotation", first_annotation.get("cam_rotation"), image_id
    )

    for annotation in annotations[1:]:
        if (
            _validate_vector("cam_location", annotation.get("cam_location"), image_id)
            != cam_location
        ):
            raise ValueError(f"Inconsistent cam_location for image_id={image_id}")
        if (
            _validate_vector("cam_rotation", annotation.get("cam_rotation"), image_id)
            != cam_rotation
        ):
            raise ValueError(f"Inconsistent cam_rotation for image_id={image_id}")

    return cam_location, cam_rotation


def converted_png_npy_path(source_png: str, converted_npy_dir: str) -> str:
    base_name = os.path.splitext(os.path.basename(source_png))[0]
    return os.path.join(converted_npy_dir, f"{base_name}.npy")


def _rgb_to_bayer_rg8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] >= 3:
        bgr = image[..., :3]
    elif image.ndim == 2:
        bgr = np.stack([image] * 3, axis=-1)
    else:
        raise ValueError("Unsupported image shape for Bayer conversion")

    bayer = np.empty((bgr.shape[0], bgr.shape[1]), dtype=np.uint8)
    r = bgr[..., 2]
    g = bgr[..., 1]
    b = bgr[..., 0]

    bayer[0::2, 0::2] = r[0::2, 0::2]
    bayer[0::2, 1::2] = g[0::2, 1::2]
    bayer[1::2, 0::2] = g[1::2, 0::2]
    bayer[1::2, 1::2] = b[1::2, 1::2]
    return bayer


def _convert_png_to_bayerrg8_npy(args: tuple[str, str, int, bool]) -> tuple[str, str]:
    source_png, converted_npy_dir, upscale_factor, overwrite = args
    converted_npy = converted_png_npy_path(source_png, converted_npy_dir)

    if not overwrite and os.path.exists(converted_npy):
        return source_png, converted_npy

    image = cv2.imread(source_png, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {source_png}")

    if upscale_factor != 1:
        image = cv2.resize(
            image,
            (0, 0),
            fx=upscale_factor,
            fy=upscale_factor,
            interpolation=cv2.INTER_LINEAR,
        )

    bayer_image = _rgb_to_bayer_rg8(image)
    os.makedirs(converted_npy_dir, exist_ok=True)
    np.save(converted_npy, bayer_image)
    return source_png, converted_npy


def convert_pngs_to_bayerrg8_npy(
    source_png_paths: list[str],
    converted_npy_dir: str,
    overwrite: bool = False,
    upscale_factor: int = 2,
    num_workers: int | None = None,
) -> dict[str, str]:
    os.makedirs(converted_npy_dir, exist_ok=True)
    if num_workers is None:
        num_workers = max(1, os.cpu_count() or 1)

    tasks = [
        (source_png, converted_npy_dir, upscale_factor, overwrite)
        for source_png in source_png_paths
    ]

    converted_paths: dict[str, str] = {}
    if num_workers == 1:
        for task in tasks:
            source_png, converted_npy = _convert_png_to_bayerrg8_npy(task)
            converted_paths[source_png] = converted_npy
    else:
        with multiprocessing.Pool(processes=num_workers) as pool:
            for source_png, converted_npy in pool.imap_unordered(
                _convert_png_to_bayerrg8_npy, tasks
            ):
                converted_paths[source_png] = converted_npy

    return converted_paths


def generate_blender_pixhawk_metadata_csv(
    dataset_root: str,
    annotations_json: str,
    converted_npy_dir: str,
    output_csv: str,
    fps: float,
    overwrite_converted: bool = False,
    overwrite_csv: bool = False,
) -> str:
    if fps <= 0:
        raise ValueError("fps must be greater than 0")

    if os.path.exists(output_csv) and not overwrite_csv:
        existing_metadata = pd.read_csv(output_csv)
        if (
            not existing_metadata.empty
            and "filename" in existing_metadata
            and existing_metadata["filename"].map(os.path.exists).all()
        ):
            return output_csv
        raise FileExistsError(
            f"{output_csv} already exists but is not reusable; "
            "pass --overwrite-csv to replace it."
        )

    images_dir = os.path.join(dataset_root, "images")
    with open(annotations_json, "r", encoding="utf-8") as f:
        annotations_data = json.load(f)

    images = sorted(annotations_data["images"], key=lambda image: image["id"])
    annotations_by_image_id: dict[int, list[dict]] = {}
    for annotation in tqdm(
        annotations_data["annotations"],
        desc="Indexing annotations",
        unit="annotation",
    ):
        annotations_by_image_id.setdefault(annotation["image_id"], []).append(
            annotation
        )

    image_rows = []
    source_png_paths = []
    for image in images:
        image_annotations = annotations_by_image_id.get(image["id"], [])
        if not image_annotations:
            continue
        cam_location, cam_rotation = _annotation_pose(image["id"], image_annotations)
        source_png = os.path.join(images_dir, image["file_name"])
        image_rows.append((image, source_png, cam_location, cam_rotation))
        source_png_paths.append(source_png)

    converted_paths = convert_pngs_to_bayerrg8_npy(
        source_png_paths, converted_npy_dir, overwrite=overwrite_converted
    )

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    rows = []
    timestamp_step_ms = int(1000 / fps)
    for idx, (image, source_png, cam_location, cam_rotation) in tqdm(
        enumerate(image_rows),
        total=len(image_rows),
        desc="Building metadata rows",
        unit="image",
    ):
        converted_npy = converted_paths.get(
            source_png, converted_png_npy_path(source_png, converted_npy_dir)
        )
        roll, pitch, yaw = cam_rotation
        alt_msl = cam_location[2]
        rows.append(
            {
                "filename": converted_npy,
                "timestamp": idx * timestamp_step_ms,
                "exposure_time": 0.0,
                "gain": 0.0,
                "black_level": 0,
                "bsl_color_space": "sRgb",
                "gamma": 1,
                "light_source_preset": "Daylight5000K",
                "pixel_format": "BayerRG8",
                "teensy_timestamp": idx * timestamp_step_ms,
                "lat": 0.0,
                "lon": 0.0,
                "alt_msl": alt_msl,
                "vground": 0.0,
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw,
                "heading": math.degrees(yaw) % 360,
                "mission_mode": int(MissionMode.IN_MISSION),
                "mission_state": int(MissionState.ACTIVE),
            }
        )

    pd.DataFrame(rows, columns=PIXHAWK_METADATA_COLUMNS).to_csv(output_csv, index=False)
    return output_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Blender COCO dataset to BayerRG8 NPY files and "
            "Pixhawk-style metadata CSV."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default="/home/sft/drone_dataset_yolo_final/val",
        help="Dataset split folder containing images/ and annotations_val.json.",
    )
    parser.add_argument(
        "--annotations-json",
        default=None,
        help=(
            "Path to the COCO-style annotations JSON. Defaults to "
            "<dataset-root>/annotations_val.json."
        ),
    )
    parser.add_argument(
        "--converted-npy-dir",
        default=None,
        help=(
            "Output folder for BayerRG8 NPY files. Defaults to "
            "<dataset-root>/images_bayerrg8_npy."
        ),
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help=(
            "Output metadata CSV. Defaults to "
            "<dataset-root>/metadata_pixhawk_bayerrg8.csv."
        ),
    )
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--overwrite-converted", action="store_true")
    parser.add_argument("--overwrite-csv", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotations_json = args.annotations_json or os.path.join(
        args.dataset_root, "annotations_val.json"
    )
    converted_npy_dir = args.converted_npy_dir or os.path.join(
        args.dataset_root, "images_bayerrg8_npy"
    )
    output_csv = args.output_csv or os.path.join(
        args.dataset_root, "metadata_pixhawk_bayerrg8.csv"
    )

    csv_path = generate_blender_pixhawk_metadata_csv(
        dataset_root=args.dataset_root,
        annotations_json=annotations_json,
        converted_npy_dir=converted_npy_dir,
        output_csv=output_csv,
        fps=args.fps,
        overwrite_converted=args.overwrite_converted,
        overwrite_csv=args.overwrite_csv,
    )
    print(f"Metadata written to {csv_path}")


if __name__ == "__main__":
    main()
