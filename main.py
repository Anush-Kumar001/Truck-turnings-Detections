from ultralytics import YOLO
import cv2

# Load YOLOv8 pre-trained model
model = YOLO('yolov8n.pt')

# Open your webcam or video (0 = default webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    # Draw results on the frame
    annotated_frame = results[0].plot()

    # Display the output
    cv2.imshow("Truck Turning Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
