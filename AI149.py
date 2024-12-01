from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # You can choose different models like 'yolov8m.pt', 'yolov8x.pt', etc.

# Real-time object detection on video
cap = cv2.VideoCapture(0)  # Replace 0 with the video file path if needed

while True:
    ret, frame = cap.read()

    # Perform object detection
    results = model(frame)

    # Visualize the detections
    annotated_frame = results.render()

    cv2.imshow('YOLOv8', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
