from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # nano model (fast)

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=8
)
    