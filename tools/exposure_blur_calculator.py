from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class ExposureBlurResult:
    altitude_m: float
    speed_m_s: float
    focal_length_mm: float
    pixel_size_um: float
    max_blur_px: float
    gsd_m_per_px: float
    max_exposure_s: float
    max_exposure_us: float
    shutter_denominator: float


def compute_exposure_for_blur(
    *,
    altitude_m: float,
    speed_m_s: float,
    focal_length_mm: float = 16.0,
    pixel_size_um: float = 2.74,
    max_blur_px: float = 1.0,
) -> ExposureBlurResult:
    if altitude_m <= 0:
        raise ValueError("altitude_m must be positive")
    if speed_m_s <= 0:
        raise ValueError("speed_m_s must be positive")
    if focal_length_mm <= 0:
        raise ValueError("focal_length_mm must be positive")
    if pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be positive")
    if max_blur_px <= 0:
        raise ValueError("max_blur_px must be positive")

    pixel_size_mm = pixel_size_um / 1000.0
    gsd_m_per_px = altitude_m * pixel_size_mm / focal_length_mm
    max_exposure_s = max_blur_px * gsd_m_per_px / speed_m_s

    return ExposureBlurResult(
        altitude_m=altitude_m,
        speed_m_s=speed_m_s,
        focal_length_mm=focal_length_mm,
        pixel_size_um=pixel_size_um,
        max_blur_px=max_blur_px,
        gsd_m_per_px=gsd_m_per_px,
        max_exposure_s=max_exposure_s,
        max_exposure_us=max_exposure_s * 1_000_000.0,
        shutter_denominator=1.0 / max_exposure_s,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute exposure time from drone speed, altitude, camera parameters, "
            "and desired max motion blur. Defaults to max 1 px blur."
        )
    )
    parser.add_argument("--altitude-m", type=float, required=True)
    parser.add_argument("--speed-m-s", type=float, required=True)
    parser.add_argument("--focal-mm", type=float, default=16.0)
    parser.add_argument("--pixel-size-um", type=float, default=2.74)
    parser.add_argument("--max-blur-px", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = compute_exposure_for_blur(
        altitude_m=args.altitude_m,
        speed_m_s=args.speed_m_s,
        focal_length_mm=args.focal_mm,
        pixel_size_um=args.pixel_size_um,
        max_blur_px=args.max_blur_px,
    )

    print(f"Altitude: {result.altitude_m:.2f} m")
    print(f"Speed: {result.speed_m_s:.2f} m/s")
    print(f"Focal length: {result.focal_length_mm:.2f} mm")
    print(f"Pixel size: {result.pixel_size_um:.3f} µm")
    print(
        "GSD: "
        f"{result.gsd_m_per_px:.6f} m/px "
        f"({result.gsd_m_per_px * 1000.0:.2f} mm/px)"
    )
    print(f"\nExposure time for <= {result.max_blur_px:.2f} px blur:")
    print(f"  {result.max_exposure_s:.7f} s")
    print(f"  {result.max_exposure_us:.1f} µs")
    print(f"  1/{result.shutter_denominator:.0f} s")


if __name__ == "__main__":
    main()
