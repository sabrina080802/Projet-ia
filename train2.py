from ultralytics import YOLO
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device utilisé : {device}")

# On repart de ton meilleur modèle actuel
model = YOLO("runs/detect/runs/train/hands_yolo11n-2/weights/best.pt")

model.train(
    data="hands.yolo26/data.yaml",
    epochs=40,
    imgsz=640,
    batch=4,
    device=device,
    workers=0,

    # Amélioration dataset / généralisation
    degrees=10,
    translate=0.15,
    scale=0.6,
    shear=2,
    perspective=0.0005,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.6,
    hsv_v=0.4,

    # Nom clair
    name="hands_boost_perf",
)