import time
import numpy as np
import onnxruntime as ort
import cv2

# --- CONFIGURATION ---
MODEL_PATH = "yolov8n-opset11-half.onnx"
SCORE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4      
INPUT_SIZE = 640       
SKIP_FRAMES = 1

# YOLO classes are 0-indexed. No "background" class.
CLASSES = ['paper'] 

cap = cv2.VideoCapture(0)
# Force lower resolution for webcam to save USB bandwidth (optional)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# --- LOAD MODEL ---
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
# For Jetson, use TensorRT provider if available:
# providers = [('TensorrtExecutionProvider', {'trt_fp16_enable': True}), 'CUDAExecutionProvider']

print(f"Loading {MODEL_PATH}...")
try:
    sess = ort.InferenceSession(MODEL_PATH, providers=providers)
except Exception as e:
    print(f"GPU provider not found, falling back to CPU. Error: {e}")
    sess = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])

output_name = sess.get_outputs()[0].name
input_name = sess.get_inputs()[0].name

print(f"Active Provider: {sess.get_providers()[0]}")
print("Starting inference loop... Press 'q' to exit.")

# --- STATE VARIABLES ---
frame_count = 0
# We cache the final parsed boxes (x1, y1, x2, y2), class_ids, and scores
cached_boxes = [] 
cached_ids = []
cached_scores = []
last_proc_time = 0.0

# FPS calculation
started=time.time()
last_logged=time.time()
frame_count=0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # --- INFERENCE BLOCK ---
    # Preprocessing (Same as your script)
    # YOLO expects RGB, 0-1 float, 640x640
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False)
    
    # Run ONNX
    inicio = time.time()
    outputs = sess.run([output_name], {input_name: blob})
    inf_time=time.time()-inicio
    # --- POST-PROCESSING (YOLO Specific) ---
    # Output shape is usually (1, 4 + num_classes, 8400)
    # We transpose to (8400, 4 + num_classes) to make it easier to iterate
    output = outputs[0][0].transpose()
    
    # Separate boxes and scores
    # Rows: [x_center, y_center, width, height, class_0_score, class_1_score...]
    box_rows = output[:, :4]
    score_rows = output[:, 4:]
    
    # Get highest class score for each row
    class_ids = np.argmax(score_rows, axis=1)
    max_scores = np.max(score_rows, axis=1)
    
    # 1. Filter by Score Threshold
    mask = max_scores >= SCORE_THRESHOLD
    filtered_boxes = box_rows[mask]
    filtered_scores = max_scores[mask]
    filtered_ids = class_ids[mask]
    
    # 2. Convert Center-WH to TopLeft-WH (For NMS)
    boxes_nms = []
    if len(filtered_boxes) > 0:
        boxes_nms = np.array(filtered_boxes)
        boxes_nms[:, 0] = filtered_boxes[:, 0] - (filtered_boxes[:, 2] / 2) # x
        boxes_nms[:, 1] = filtered_boxes[:, 1] - (filtered_boxes[:, 3] / 2) # y
        
    # 3. Apply NMS (Non-Maximum Suppression)
    indices = []
    if len(boxes_nms) > 0: # <--- CHECK IF BOXES EXIST
        # boxes_nms is a numpy array here, so .tolist() works
        indices = cv2.dnn.NMSBoxes(boxes_nms.tolist(), filtered_scores.tolist(), SCORE_THRESHOLD, NMS_THRESHOLD)
    else:
        indices = [] # No detections

    # 4. Save results for display
    cached_boxes = []
    cached_scores = []
    cached_ids = []
    
    scale_x = orig_w / INPUT_SIZE
    scale_y = orig_h / INPUT_SIZE

    for i in indices:
        # cv2.dnn.NMSBoxes returns a list of [i] or [[i]] depending on version
        idx = i if isinstance(i, (int, np.int32, np.int64)) else i[0]
        
        box_nms = boxes_nms[idx]
        
        # Scale back to original image size
        x = int(box_nms[0] * scale_x)
        y = int(box_nms[1] * scale_y)
        w = int(box_nms[2] * scale_x)
        h = int(box_nms[3] * scale_y)
        
        cached_boxes.append([x, y, x+w, y+h]) 
        cached_scores.append(filtered_scores[idx])
        cached_ids.append(filtered_ids[idx])


    # --- DRAWING BLOCK ---
    if len(cached_boxes) > 0:
        for i in range(len(cached_boxes)):
            box = cached_boxes[i]
            score = cached_scores[i]
            class_id = cached_ids[i]
            
            # Draw Box
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            
            # Draw Label
            label_name = CLASSES[class_id] if class_id < len(CLASSES) else f"ID {class_id}"
            label_text = f"{label_name}: {score:.2f}"
            
            # Label Background
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + w, box[1]), (0, 255, 0), -1)
            cv2.putText(frame, label_text, (box[0], box[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Info
    fps = 1 / inf_time
    cv2.putText(frame, f"FPS: {fps:.1f} | Inf: {last_proc_time*1000:.0f}ms", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('YOLOv8n Detection ONNX half precision', frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()