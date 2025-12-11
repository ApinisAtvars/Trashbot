from ultralytics import YOLO

model = YOLO("yolov8n4.pt")

model.export(format="engine", half=True, imgsz=640, device='0')