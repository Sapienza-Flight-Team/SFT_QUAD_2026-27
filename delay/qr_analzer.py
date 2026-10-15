import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from common.metadata_QOL import select_csv

# --------------------------
# CONFIG (hardcoded)
# --------------------------
CSV_PATH = "/path/to/metadata.csv"  # <-- change to your CSV
IMAGE_COLUMN = "filename"  # column in CSV with npy file paths
TIMESTAMP_COLUMN = "teensy_timestamp"  # column in CSV with Pixhawk timestamp

# --------------------------
# FUNCTIONS
# --------------------------


def extract_qr_from_npy(npy_image_path):
    """Load .npy image and extract QR code text using cv2.QRCodeDetector."""
    try:
        img = np.load(npy_image_path)

        # Ensure 8-bit
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Convert grayscale → BGR if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(
                cv2.resize(img, (5320 // 4, 3032 // 4), interpolation=cv2.INTER_LINEAR),
                cv2.COLOR_GRAY2BGR,
            )

        detector = cv2.QRCodeDetector()
        data, bbox, _ = detector.detectAndDecode(img)

        if data == "":
            return 0
        return int(data)

    except Exception as e:
        print(f"[ERROR] QR decode failed for {npy_image_path}: {e}")
        return None


def analyze_metadata(csv_path):
    """Reads CSV, decodes QR from .npy images, writes analyzed CSV."""
    if not os.path.isfile(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    qr_timestamps = []

    for row in df.itertuples(index=False):
        img_path = getattr(row, IMAGE_COLUMN)
        getattr(row, TIMESTAMP_COLUMN)
        print(f"Processing: {img_path}")

        if not os.path.exists(img_path):
            print(f"[WARNING] Missing image: {img_path}")
            qr_timestamps.append(None)
            continue

        qr_time = extract_qr_from_npy(img_path)
        qr_timestamps.append(qr_time)
    df_tee = df.pop(TIMESTAMP_COLUMN)
    df[TIMESTAMP_COLUMN] = df_tee
    df["qr_timestamp"] = qr_timestamps
    df["difference"] = np.array(df_tee) - np.array(qr_timestamps)
    # Output CSV in same folder with "_analyzed.csv"
    folder = os.path.dirname(csv_path)
    base = os.path.splitext(os.path.basename(csv_path))[0]
    out_path = os.path.join(folder, f"{base}_analyzed.csv")
    df.to_csv(out_path, index=False)
    print(f"\nDone! Saved analyzed CSV to: {out_path}")
    plt.figure(figsize=(12, 6))
    plt.scatter([x for x in range(len(df))], df["difference"])
    plt.xlabel("photo")
    plt.ylabel("time_delay")

    plt.savefig("completed_analysis.png")


# --------------------------
# MAIN
# --------------------------
if __name__ == "__main__":
    default = ""
    analyze_metadata(select_csv("data", default))
