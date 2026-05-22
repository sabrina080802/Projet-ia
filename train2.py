from ultralytics import YOLO

model = YOLO("runs/detect/runs/train/hands_yolo11n-2/weights/best.pt")

model.train(
    data="hands.yolo26/data.yaml",
    epochs=20,
    device="mps"
)