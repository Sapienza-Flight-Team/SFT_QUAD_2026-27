import cv2
import numpy as np
import time
from datetime import datetime

GRID_SIZE = 8  # 8x8 grid
BITS = 64
FPS = 10
FRAME_TIME = 1.0 / FPS
CELL_SPACING_RATIO = 0.25  # spacing between cells
GREEN_MARGIN_RATIO = 0.1  # green margin around grid (10% of grid size)

# Screen size (adjust to your screen)
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080


def generate_grid_with_margin():
    # Timestamp in hundredths of a second
    timestamp_hundredths = int(time.time() * 100) & ((1 << 38) - 1)
    bits_str = format(timestamp_hundredths, "038b") + "0" * (BITS - 38)

    # Full red background
    img = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
    img[:] = (0, 0, 255)  # red

    # Compute max grid size (squared)
    max_grid_px = int(min(SCREEN_WIDTH, SCREEN_HEIGHT) * 0.6)  # smaller than screen
    cell_size = max_grid_px // GRID_SIZE
    spacing = int(cell_size * CELL_SPACING_RATIO)
    cell_size -= spacing
    grid_total_size = (cell_size + spacing) * GRID_SIZE - spacing

    # Add green margin
    green_margin = int(grid_total_size * GREEN_MARGIN_RATIO)
    green_square_size = grid_total_size + 2 * green_margin

    # Top-left of green square
    start_x = (SCREEN_WIDTH - green_square_size) // 2
    start_y = (SCREEN_HEIGHT - green_square_size) // 2

    # Draw green square
    cv2.rectangle(
        img,
        (start_x, start_y),
        (start_x + green_square_size, start_y + green_square_size),
        (0, 255, 0),
        thickness=-1,
    )

    # Draw grid inside green square
    grid_start_x = start_x + green_margin
    grid_start_y = start_y + green_margin
    for i in range(BITS):
        row = i // GRID_SIZE
        col = i % GRID_SIZE
        x0 = grid_start_x + col * (cell_size + spacing)
        y0 = grid_start_y + row * (cell_size + spacing)
        x1 = x0 + cell_size
        y1 = y0 + cell_size
        color = (255, 255, 255) if bits_str[i] == "1" else (0, 0, 0)
        img[y0:y1, x0:x1] = color

    return img, timestamp_hundredths


# ------------------- Main loop -------------------
print("Press ESC to exit")
while True:
    start_time = time.time()
    frame, ts_val = generate_grid_with_margin()
    cv2.putText(
        frame,
        f"{datetime.utcfromtimestamp(ts_val/100)}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Grid Generator with Green Margin", frame)

    if cv2.waitKey(1) == 27:
        break

    elapsed = time.time() - start_time
    import time as tm

    tm.sleep(max(0, FRAME_TIME - elapsed))

cv2.destroyAllWindows()
