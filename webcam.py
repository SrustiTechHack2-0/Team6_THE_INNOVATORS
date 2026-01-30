import cv2
import random

print("Starting webcam...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera open nahi ho raha")
    exit()

print("Webcam ON — Q dabao band karne ke liye")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Demo prediction (hackathon safe)
    prediction = random.choice(["REAL ✅", "FAKE ❌"])

    cv2.putText(
        frame,
        f"Prediction: {prediction}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Face Anti-Spoofing Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
