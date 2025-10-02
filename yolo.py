from ultralytics import YOLO
import cv2

# Load your trained YOLO model
model = YOLO("yolov8n.pt")  # Replace with your model path if custom

# Set confidence threshold (lower = more detections)
CONF_THRESHOLD = 0.25

# Open webcam (0) or replace with video path
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model.predict(source=frame, conf=CONF_THRESHOLD, verbose=False)

    # Draw detections
    for result in results:
        for box in result.boxes.xyxy:  # xyxy format
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Optional: display class names and confidence
        if result.boxes.cls is not None:
            for cls, conf in zip(result.boxes.cls, result.boxes.conf):
                label = f"{model.names[int(cls)]} {conf:.2f}"
                x1, y1, _, _ = map(int, result.boxes.xyxy[0])
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # Show frame
    cv2.imshow("YOLO Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
