import cv2
import numpy as np
from datetime import datetime

GRID_SIZE = 8
BITS = 64


def order_points(pts):
    """Order points: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def read_grid(frame):
    """Detect green square, warp, and read 8x8 grid inside."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Green mask (the square around the grid)
    lower_green = np.array([50, 100, 50])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Take the largest contour (green square)
    largest = max(contours, key=cv2.contourArea)

    # Approximate contour to 4 points
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
    if len(approx) != 4:
        return None

    pts = approx.reshape(4, 2)
    rect = order_points(pts)

    # Warp perspective to perfect square
    side = int(
        max(
            np.linalg.norm(rect[0] - rect[1]),
            np.linalg.norm(rect[1] - rect[2]),
            np.linalg.norm(rect[2] - rect[3]),
            np.linalg.norm(rect[3] - rect[0]),
        )
    )
    dst_pts = np.array(
        [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype="float32"
    )
    M = cv2.getPerspectiveTransform(rect, dst_pts)
    warp = cv2.warpPerspective(frame, M, (side, side))

    # Compute green margin: detect first black/white pixel inside warp
    # Optional: assume margin = 10% of side
    margin = int(side * 0.1)
    grid_start = margin
    grid_end = side - margin
    grid_size = grid_end - grid_start
    cell_size = grid_size / GRID_SIZE

    # Read 8x8 grid
    bits_str = ""
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            x_center = int(grid_start + col * cell_size + cell_size / 2)
            y_center = int(grid_start + row * cell_size + cell_size / 2)
            pixel = warp[y_center, x_center]
            gray_val = int(0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0])
            bits_str += "1" if gray_val > 128 else "0"

    # Convert first 38 bits to timestamp
    timestamp_bits = bits_str[:38]
    timestamp_hundredths = int(timestamp_bits, 2)
    dt = datetime.utcfromtimestamp(timestamp_hundredths / 100)

    return timestamp_hundredths, dt, rect, warp


# ---------------- Real-Time Camera Capture ----------------
cap = cv2.VideoCapture(0)
print("Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    result = read_grid(frame)
    if result is not None:
        ts_val, dt, rect, warp = result
        # Draw detected green square
        for i in range(4):
            pt1 = tuple(rect[i].astype(int))
            pt2 = tuple(rect[(i + 1) % 4].astype(int))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        # Display decoded datetime
        cv2.putText(
            frame,
            dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-4],
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        print("Timestamp:", ts_val, "| Datetime:", dt)
    else:
        print("Grid not found")

    cv2.imshow("Dynamic Grid Reader", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
