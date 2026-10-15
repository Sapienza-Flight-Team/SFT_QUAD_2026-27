#!/usr/bin/env python3
import argparse
import torch
from pathlib import Path
from ultralytics import YOLO
import shutil


def main():
    parser = argparse.ArgumentParser(description="Compile YOLO .pt to TensorRT .engine")
    parser.add_argument(
        "--weights", type=str, required=True, help="Path to YOLO .pt model"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Image size (default: 640)"
    )
    parser.add_argument(
        "--batch", type=int, default=14, help="Batch size (default: 14)"
    )

    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at {weights_path}")

    print(f"Loading YOLO model from {weights_path}")
    yolo = YOLO(str(weights_path))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trt_cache_dir = weights_path.parent
    engine_path = trt_cache_dir / (weights_path.stem + f"_trt_b{args.batch}.engine")

    print(f"Exporting to TensorRT...")
    print(f"Target: {engine_path}")

    try:
        yolo.export(
            format="engine",
            imgsz=args.imgsz,
            half=True,
            batch=args.batch,
            device=int(str(device).split(":")[-1]) if device.type != "cpu" else "cpu",
            workspace=4,
        )
    except Exception as e:
        print(f"Export failed natively: {e}")
        return

    # Ultralytics default save path
    default_engine = weights_path.with_suffix(".engine")
    if default_engine.exists():
        if default_engine != engine_path:
            shutil.copy(str(default_engine), str(engine_path))
            print(f"[SUCCESS] YOLO TRT engine renamed and saved -> {engine_path}")
        else:
            print(f"[SUCCESS] YOLO TRT engine saved -> {engine_path}")
    else:
        if engine_path.exists():
            print(f"[SUCCESS] YOLO TRT engine saved -> {engine_path}")
        else:
            print("[ERROR] Engine file was not successfully created by Ultralytics.")


if __name__ == "__main__":
    main()
