from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("input.mp4")

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"\n{'=' * 50}")
print("Video İşleme Başlıyor (Sadece Blur Modu)")
print(f"{'=' * 50}")
print(f"Çözünürlük: {width}x{height}")
print(f"FPS: {fps}")
print(f"{'=' * 50}\n")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    results = model(frame, verbose=False, conf=0.5)

    for i, box in enumerate(results[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)

        roi = frame[y1:y2, x1:x2]

        if roi.size > 0:
            blur_size = 99
            anonymized = cv2.GaussianBlur(roi, (blur_size, blur_size), 30)

            frame[y1:y2, x1:x2] = anonymized

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Blur #{i + 1}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(frame, f"Frame: {frame_count} | Mode: BLUR",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("YOLO Face Blur", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n{'=' * 50}")
print("İşlem Tamamlandı!")
print(f"Toplam Frame: {frame_count}")
print(f"{'=' * 50}")
