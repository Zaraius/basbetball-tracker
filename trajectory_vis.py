import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from yolox.tracker.byte_tracker import BYTETracker

##########################################
# Kalman Filter (2D constant velocity)
##########################################
def make_kf(dt=1/30):
    kf = KalmanFilter(dim_x=4, dim_z=2)

    kf.F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1,  0],
                     [0, 0, 0,  1]])

    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])

    kf.P *= 1000.0          # initial uncertainty
    kf.R = np.eye(2) * 10   # measurement noise
    kf.Q = np.eye(4) * 0.05 # process noise

    kf.x = np.zeros((4, 1)) # [x, y, vx, vy]
    return kf

##########################################
# Predict n future steps (for trajectory)
##########################################
def predict_future(kf, steps=10, dt=1/30):
    preds = []
    x_backup = kf.x.copy()
    P_backup = kf.P.copy()

    for _ in range(steps):
        kf.predict()
        preds.append(kf.x[:2].flatten())

    kf.x = x_backup
    kf.P = P_backup
    return preds

##########################################
# Main
##########################################
def main(detection_file, canvas_size=(720, 1280)):
    # Tracker
    tracker = BYTETracker(track_thresh=0.25,
                          match_thresh=0.8,
                          track_buffer=30,
                          frame_rate=30)

    kalman_filters = {}

    # Read all detections
    # Expected format per line: frame_id, x1, y1, x2, y2, score, class
    detections = {}
    with open(detection_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 7:
                continue
            frame_id = int(parts[0])
            x1, y1, x2, y2 = map(float, parts[1:5])
            score = float(parts[5])
            cls = int(parts[6])
            if frame_id not in detections:
                detections[frame_id] = []
            detections[frame_id].append([x1, y1, x2, y2, score, cls])

    max_frame = max(detections.keys())
    H, W = canvas_size

    # Process each frame
    for frame_id in range(1, max_frame + 1):
        frame = np.zeros((H, W, 3), dtype=np.uint8)  # blank canvas

        dets = np.array(detections.get(frame_id, []))
        if dets.size == 0:
            dets = np.empty((0, 6))

        # ByteTrack wants: [x1, y1, x2, y2, score]
        tracks = tracker.update(dets[:, :5], [H, W], [H, W])

        for t in tracks:
            track_id = int(t.track_id)
            x1, y1, x2, y2 = t.tlbr

            # Compute center point
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            meas = np.array([cx, cy])

            # Create Kalman filter if needed
            if track_id not in kalman_filters:
                kf = make_kf()
                kf.x[:2] = meas.reshape((2,1))
                kalman_filters[track_id] = kf

            kf = kalman_filters[track_id]

            # Kalman prediction + update
            kf.predict()
            kf.update(meas)
            x, y, vx, vy = kf.x.flatten()

            # Draw bounding box
            cv2.rectangle(frame,
                          (int(x1), int(y1)),
                          (int(x2), int(y2)),
                          (0, 255, 0), 2)

            # Draw smoothed center
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

            # Draw velocity vector
            cv2.arrowedLine(frame,
                            (int(x), int(y)),
                            (int(x + vx * 5), int(y + vy * 5)),
                            (255, 0, 0), 2)

            # Predict future trajectory
            preds = predict_future(kf, steps=10)
            for px, py in preds:
                cv2.circle(frame, (int(px), int(py)), 3, (0, 255, 255), -1)

            # Label
            cv2.putText(frame, f"ID {track_id}",
                        (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

        # Display
        cv2.imshow("Tracking from combined.txt", frame)
        if cv2.waitKey(1) == 27:  # ESC to exit
            break

    cv2.destroyAllWindows()

##########################################
# Run
##########################################
if __name__ == "__main__":
    main("combined.txt")  # ← path to your detection file
