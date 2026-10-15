"""
bw_targets_bench.py

Benchmark and evaluation harness for the classical black/white target detector.
Supports both YOLO-style dataset layouts (`split/images`, `split/labels`) and
flat image/label directories through CLI overrides.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stuff_i_moved_away.inference.find_bw_targets import (
    benchmark_config,
    evaluate_on_dataset,
    load_config,
    resolve_dataset_paths,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark del detector CV classico dei piccoli target bianchi/neri."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="bw_targets_config.yaml",
        help="Path al file YAML di configurazione.",
    )
    parser.add_argument(
        "--data-root", type=str, default="", help="Override della dataset root YOLO."
    )
    parser.add_argument(
        "--split", type=str, default="", help="Override dello split YOLO."
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="",
        help="Override diretto della cartella immagini.",
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default="",
        help="Override diretto della cartella label.",
    )
    parser.add_argument(
        "--out", type=str, default="", help="Override del path del report JSON."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Salva overlay e debug JSON intermedi."
    )
    parser.add_argument(
        "--iou-thresh", type=float, default=-1.0, help="Override della soglia IoU."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    bench_cfg = benchmark_config(config)

    if args.image_dir or args.labels_dir:
        if not (args.image_dir and args.labels_dir):
            raise SystemExit(
                "[ERR] Se usi --image-dir o --labels-dir devi specificarli entrambi."
            )
        images_dir = Path(args.image_dir).expanduser()
        labels_dir = Path(args.labels_dir).expanduser()
        split = "flat"
        data_root = images_dir.parent
    else:
        images_dir, labels_dir, split = resolve_dataset_paths(
            config, args.data_root, args.split
        )
        data_root = Path(args.data_root or config["dataset"]["data_root"]).expanduser()

    report_path = Path(args.out or bench_cfg["report_path"])
    debug_dir = bench_cfg["debug_dir"] if args.debug else None
    iou_thresh = (
        args.iou_thresh if args.iou_thresh > 0 else float(bench_cfg["iou_threshold"])
    )

    if not images_dir.exists():
        raise SystemExit(f"[ERR] Nessuna cartella immagini trovata in {images_dir}")
    if not labels_dir.exists():
        raise SystemExit(f"[ERR] Nessuna cartella label trovata in {labels_dir}")

    metrics = evaluate_on_dataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        config=config,
        output_dir=debug_dir,
        iou_thresh=iou_thresh,
        debug=args.debug,
    )

    report = {
        "config_path": args.config,
        "data_root": str(data_root),
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "split": split,
        "iou_threshold": iou_thresh,
        "debug": args.debug,
        "metrics": metrics,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nReport salvato → {report_path}")


if __name__ == "__main__":
    main()
