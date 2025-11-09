from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')  # Make sure yolov8n.pt is installed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for box in results.boxes:
        cls = int(box.cls)
        label = model.names[cls]

        if label == 'person':
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Define feet region
            foot_y1 = int(y2 - 0.25 * (y2 - y1))
            cv2.rectangle(frame, (x1, foot_y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, 'Feet Detected', (x1, foot_y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Feet Detection (Press q to quit)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
