from ultralytics import YOLO
import cv2
import numpy as np
from scipy.interpolate import splprep, splev

VIDEO_PATH = "basketball.mp4"
MODEL_PATH = "best2.pt"

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

positions = []

print(model.model.names)


print("Starting video processing...")

for frame_idx in range(300):  # however many frames you're testing
    ret, frame = cap.read()
    if not ret:
        print("Video ended at frame", frame_idx)
        break

    # Run YOLO
    results = model(frame)

    # Debug print
    print(f"\n=== FRAME {frame_idx} ===")
    print("YOLO detected:", len(results[0].boxes))

    if len(results[0].boxes) == 0:
        continue

    # loop over predictions
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        if cls_id != 0:  # 0 = basketball in your model
            continue

        xyxy = box.xyxy[0].cpu().numpy()  # move to CPU
        x1, y1, x2, y2 = xyxy

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        positions.append([cx, cy])
        print("Added ball center:", (cx, cy))

positions = np.array(positions)

print("\nFinal positions shape:", positions.shape)
print("Final positions:", positions)


frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    results = model(frame, verbose=False)

    # No detections at all
    if results[0].boxes is None or len(results[0].boxes) == 0:
        positions.append([np.nan, np.nan])
        continue

    # Pick the highest confidence basketball
    boxes = results[0].boxes
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()

    # Filter class 0: basketball
    ball_idxs = np.where(cls == 0)[0]

    if len(ball_idxs) == 0:
        positions.append([np.nan, np.nan])
        continue

    # Select best ball
    best_idx = ball_idxs[np.argmax(conf[ball_idxs])]

    xyxy = boxes.xyxy[best_idx].cpu().numpy()
    x1, y1, x2, y2 = xyxy
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    positions.append([cx, cy])

    if positions.size == 0:
        print("\nNo ball detections recorded. Trajectory cannot be drawn.")
    exit()

# Now it's definitely non-empty and 2D
valid = ~np.isnan(positions[:, 0])
positions = positions[valid]

print("Filtered positions shape:", positions.shape)


cap.release()

positions = np.array(positions)
print("Final positions shape:", positions.shape)

# ======= TRAJECTORY FITTING =======

# Need at least 4 valid points for splines
valid = ~np.isnan(positions[:, 0])

if valid.sum() < 4:
    print("Not enough points for trajectory. Only", valid.sum(), "valid points.")
    exit()

valid_positions = positions[valid]

tck, u = splprep([valid_positions[:, 0], valid_positions[:, 1]], s=10)
u_new = np.linspace(0, 1, len(positions))
smooth_x, smooth_y = splev(u_new, tck)

# ======= DRAW TRAJECTORY ON VIDEO =======

cap = cv2.VideoCapture(VIDEO_PATH)
out = cv2.VideoWriter("trajectory_output.mp4",
                      cv2.VideoWriter_fourcc(*"mp4v"),
                      30,
                      (int(cap.get(3)), int(cap.get(4))))

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # draw smoothed trajectory point
    if not np.isnan(smooth_x[frame_idx]):
        cv2.circle(frame,
                   (int(smooth_x[frame_idx]), int(smooth_y[frame_idx])),
                   5,
                   (0, 0, 255),
                   -1)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()

print("Done! Output saved as trajectory_output.mp4")
