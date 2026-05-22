from ultralytics import YOLO
import cv2
import time

# Charger le modèle entraîné
model = YOLO("runs/detect/runs/train/hands_yolo11n-2/weights/best.pt")

# Ouvrir la caméra Mac
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

time.sleep(1)

if not cap.isOpened():
    print("Caméra non accessible")
    exit()

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Impossible de lire la caméra")
        continue

    # Détection YOLO
    results = model(
        frame,
        device="mps",
        conf=0.5,
        imgsz=416
    )

    # Dessiner les boîtes
    annotated_frame = results[0].plot()

    # Afficher la fenêtre
    cv2.imshow("YOLO Hands", annotated_frame)

    # Quitter avec Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()