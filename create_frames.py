import cv2
from pathlib import Path
from ultralytics import YOLO
import numpy as np

# --- Config ---
video_path = "short_test_vid.mov"
output_dir = Path("output")
model_path = "best.pt"
confidence_threshold = 0.0
NUM_FRAMES_TO_SAVE = 50

# --- Setup ---
output_dir.mkdir(parents=True, exist_ok=True)
model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Processing video {video_path} ({frame_width}x{frame_height}, {total_frames} frames)")

# Collect metadata only (no images yet)
frame_meta = []  # (frame_id, max_conf, box_info)

frame_id = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    # TEMP ONLY FOR 2 MIN VIDEO
    # if frame_id < 600:
    #     frame_id +=1
    #     continue
    results = model(frame, verbose=False)[0]
    boxes = results.boxes

    if boxes is not None and len(boxes) > 0:
        high_conf_boxes = []
        conf_scores = []
        for box in boxes:
            conf = box.conf.item()
            if conf >= confidence_threshold:
                x1, y1, x2, y2 = box.xyxy[0]
                xc = ((x1 + x2) / 2) / frame_width
                yc = ((y1 + y2) / 2) / frame_height
                w = (x2 - x1) / frame_width
                h = (y2 - y1) / frame_height
                high_conf_boxes.append((0, xc.item(), yc.item(), w.item(), h.item()))
                conf_scores.append(conf)

        if high_conf_boxes:
            max_conf = max(conf_scores)
            frame_meta.append((frame_id, max_conf, high_conf_boxes))

    frame_id += 1
    if frame_id % 500 == 0:
        print(f"{frame_id}/{total_frames} frames processed...")

cap.release()

# Select X evenly spaced high-confidence frames
if len(frame_meta) < NUM_FRAMES_TO_SAVE:
    print(f"⚠️ Only found {len(frame_meta)} frames with confident detections.")
    selected = frame_meta
else:
    frame_meta.sort(key=lambda x: x[0])  # sort by time
    indices = np.linspace(0, len(frame_meta) - 1, NUM_FRAMES_TO_SAVE, dtype=int)
    selected = [frame_meta[i] for i in indices]

# Re-open video and extract only selected frames
print("Re-opening video to extract selected frames...")
cap = cv2.VideoCapture(video_path)
frame_map = {meta[0]: (meta[2]) for meta in selected}  # frame_id -> box_info
saved = 0
frame_id = 0

while cap.isOpened() and saved < len(frame_map):
    success, frame = cap.read()
    if not success:
        break

    if frame_id in frame_map:
        boxes = frame_map[frame_id]
        image_name = f"frame_{frame_id:06d}.jpg"
        label_name = image_name.replace(".jpg", ".txt")

        cv2.imwrite(str(output_dir / image_name), frame)

        with open(output_dir / label_name, "w") as f:
            for (cls, xc, yc, w, h) in boxes:
                f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        saved += 1

    frame_id += 1

cap.release()
print(f"✅ Saved {saved} selected annotated frames to {output_dir}")
