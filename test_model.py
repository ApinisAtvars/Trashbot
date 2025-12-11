import cv2
import time
from ultralytics import YOLO

MODEL_PATH = 'yolov8n4.pt' 
CAMERA_SOURCE = 0

IMG_SIZE = 640 
CONF_THRESHOLD = 0.5

def main():
    print(f"Loading model: {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH, task='detect')
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_V4L2)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    prev_time = 0
    
    print("Starting inference... Press 'q' to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # half=True uses FP16 (faster on Jetson)
        results = model(frame, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)

        annotated_frame = results[0].plot()
        
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(
            annotated_frame, 
            f"FPS: {int(fps)}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )

        cv2.imshow("YOLOv8n Jetson Inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()