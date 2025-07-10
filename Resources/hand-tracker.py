from ultralytics import YOLO
import cv2

# Load a pre-trained model
model = YOLO(r'Model\best.pt')

# Open webcam (0 = default)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run pose prediction on the frame
    results = model.predict(source=frame, conf=0.3, task='pose', verbose=False)

    # Annotate frame with keypoints and bounding boxes
    annotated_frame = results[0].plot()

    # Show the result
    cv2.imshow("Live Pose Detection", annotated_frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()