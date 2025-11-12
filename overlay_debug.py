from ultralytics import YOLO
import cv2
import numpy as np

VIDEO_PATH = "side_to_side.mp4"
MODEL_PATH = "best2.pt"

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Could not read video!")
    exit()

print("Running YOLO on first frame...")

results = model(frame)

print("\n=== RAW YOLO OUTPUT ===")
print(results)

print("\n=== BOXES ===")
if results[0].boxes is None:
    print("No boxes returned at all.")
else:
    print("Number of boxes:", len(results[0].boxes))
    for i, box in enumerate(results[0].boxes):
        print(f"Box {i}:")
        print("  cls:", int(box.cls))
        print("  xyxy:", box.xyxy.cpu().numpy())
        print("  conf:", float(box.conf))

print("\nDone.")
