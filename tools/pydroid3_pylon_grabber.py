#!/usr/bin/env python3
"""
Very simple Basler capture script for Pydroid3.

Run:
  python pydroid3_pylon_grabber.py
"""

import time
from pathlib import Path

# Change only these if needed.
OUTPUT_DIR = Path("/storage/emulated/0/DCIM/basler")
TIMEOUT_MS = 5000

from pypylon import pylon  # type: ignore


def main() -> None:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = OUTPUT_DIR / f"image_{timestamp}.png"

    try:
        cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        cam.Open()
        cam.StartGrabbingMax(1)
        grab = cam.RetrieveResult(TIMEOUT_MS, pylon.TimeoutHandling_ThrowException)
    except Exception as exc:
        print(f"Could not connect/grab from camera: {exc}")
        return

    try:
        if not grab.GrabSucceeded():
            print(f"Grab failed: {grab.GetErrorCode()} {grab.GetErrorDescription()}")
            return

        image = pylon.PylonImage()
        image.AttachGrabResultBuffer(grab)
        try:
            image.Save(pylon.ImageFileFormat_Png, str(out_path))
        finally:
            image.Release()
    finally:
        grab.Release()
        if cam.IsGrabbing():
            cam.StopGrabbing()
        cam.Close()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
