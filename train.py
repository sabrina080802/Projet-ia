from ultralytics import YOLO
import torch
from pathlib import Path

# Chemin vers le dataset Roboflow
DATASET_DIR = Path("hands.yolo26")
DATA_YAML = DATASET_DIR / "data.yaml"

# GPU Mac M1 si disponible
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device utilisé : {device}")

# Modèle léger recommandé pour Mac M1
model = YOLO("yolo11n.pt")

model.train(
    data=str(DATA_YAML),
    epochs=50,
    imgsz=640,
    batch=4,
    device=device,
    workers=0,
    project="runs/train",
    name="hands_yolo11n",
)