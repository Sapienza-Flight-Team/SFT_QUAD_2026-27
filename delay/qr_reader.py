import cv2
from datetime import datetime

cap = cv2.VideoCapture(0)
detector = cv2.QRCodeDetector()

print("Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Detect and decode QR
    data, bbox, _ = detector.detectAndDecode(frame)

    if bbox is not None:
        # Draw box around QR
        pts = bbox.astype(int).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (255, 0, 0), 3)

        if data:
            cv2.putText(
                frame,
                f"Decoded: {datetime.utcfromtimestamp(int(data)/100)}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

    cv2.imshow("QR Reader", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
