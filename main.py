# Load YOLO v8 model
from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO('yolov8n.pt')

# Load video
video_path = './man.mp4'
cap = cv2.VideoCapture(video_path)

# Read and process frames
while True:
    ret, frame = cap.read()
    if not ret:
        break 
    # Detect and track objects in the current frame
    results = model.track(frame, persist=True)

    # Plot results on the frame
    frame_ = results[0].plot()

    # Visualize the frame
    cv2.imshow('frame', frame_)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close display window
cap.release()
cv2.destroyAllWindows()
