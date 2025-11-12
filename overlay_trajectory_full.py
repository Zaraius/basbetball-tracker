#!/usr/bin/env python3
"""
overlay_trajectory_moving_avg.py

Detect ball with YOLOv8 (best2.pt), smooth trajectory, compute velocity & acceleration,
overlay arrows every N frames, save annotated video, and show moving average speed.

Usage:
    python overlay_trajectory_moving_avg.py
"""

import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO

# Optional smoothing
try:
    from scipy.interpolate import PchipInterpolator
    from scipy.signal import savgol_filter
    SCIPY = True
except Exception:
    PchipInterpolator = None
    savgol_filter = None
    SCIPY = False

# ---------------- CONFIG ----------------
VIDEO_PATH = "side_to_side.mp4"       # your video filename
YOLO_MODEL_PATH = "best2.pt"          # your yolov8 model
OUTPUT_PATH = "basketball_traj_out_mavg.mp4"

BALL_DIAMETER_INCHES = 9.4
BALL_DIAMETER_PIXELS = 30

FPS_OVERRIDE = None        # None to use video fps
FRAME_STEP = 10            # draw arrows every N frames
ARROW_VEL_SCALE = 0.04
ARROW_ACC_SCALE = 0.0015
BALL_CLASS = 0
CONF_THRESHOLD = 0.2
WINDOW_AVG = 20            # frames for moving average
MIN_VALID_DETECTIONS = 3
# ----------------------------------------

BALL_DIAMETER_METERS = BALL_DIAMETER_INCHES * 0.0254
M_PER_PX = BALL_DIAMETER_METERS / BALL_DIAMETER_PIXELS

def safe_boxes_from_result(res):
    if not hasattr(res, "boxes") or len(res.boxes) == 0:
        return None, None, None
    try:
        xyxy = res.boxes.xyxy.detach().cpu().numpy()
        conf = res.boxes.conf.detach().cpu().numpy()
        cls = res.boxes.cls.detach().cpu().numpy()
        return xyxy, conf, cls
    except Exception:
        return None, None, None

def central_diff_positions_to_velocity(pos_x, pos_y, dt):
    n = len(pos_x)
    v = np.zeros((n,2), dtype=float)
    if n < 2:
        return v
    v[0,0] = (pos_x[1] - pos_x[0]) / dt
    v[0,1] = (pos_y[1] - pos_y[0]) / dt
    for i in range(1, n-1):
        v[i,0] = (pos_x[i+1] - pos_x[i-1]) / (2*dt)
        v[i,1] = (pos_y[i+1] - pos_y[i-1]) / (2*dt)
    v[-1,0] = (pos_x[-1] - pos_x[-2]) / dt
    v[-1,1] = (pos_y[-1] - pos_y[-2]) / dt
    return v

def central_diff_velocity_to_acceleration(v, dt):
    n = len(v)
    a = np.zeros_like(v)
    if n < 2:
        return a
    a[0] = (v[1] - v[0]) / dt
    for i in range(1, n-1):
        a[i] = (v[i+1] - v[i-1]) / (2*dt)
    a[-1] = (v[-1] - v[-2]) / dt
    return a

def interp_and_smooth(frames, px, py):
    n = len(frames)
    idx = np.arange(n)
    valid = np.isfinite(px) & np.isfinite(py)
    if valid.sum() == 0:
        raise RuntimeError("No valid detections to interpolate.")
    px_filled = np.interp(idx, idx[valid], px[valid])
    py_filled = np.interp(idx, idx[valid], py[valid])
    if SCIPY and valid.sum() >= MIN_VALID_DETECTIONS:
        try:
            csx = PchipInterpolator(idx[valid], px[valid])
            csy = PchipInterpolator(idx[valid], py[valid])
            sx = csx(idx)
            sy = csy(idx)
            if savgol_filter is not None and n >= 7:
                win = 7 if n >= 7 else (n//2*2+1)
                sx = savgol_filter(sx, win, 3)
                sy = savgol_filter(sy, win, 3)
            return sx, sy
        except Exception:
            pass
    kernel = np.ones(5)/5.0
    sx = np.convolve(px_filled, kernel, mode='same')
    sy = np.convolve(py_filled, kernel, mode='same')
    return sx, sy

def moving_average(arr, window):
    n = len(arr)
    avg = np.zeros_like(arr)
    for i in range(n):
        start = max(0, i - window + 1)
        avg[i] = np.nanmean(arr[start:i+1])
    return avg

def main():
    if not os.path.exists(VIDEO_PATH):
        print("ERROR: video not found:", VIDEO_PATH); sys.exit(1)
    if not os.path.exists(YOLO_MODEL_PATH):
        print("ERROR: model not found:", YOLO_MODEL_PATH); sys.exit(1)

    model = YOLO(YOLO_MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("ERROR: cannot open video:", VIDEO_PATH); sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = FPS_OVERRIDE if FPS_OVERRIDE else (video_fps if video_fps>0 else 60.0)
    dt = 1.0 / fps

    positions = []
    frame_idx = 0
    print("Detection pass...")
    while True:
        ret, frame = cap.read()
        if not ret: break
        results = model(frame)
        res = results[0]
        boxes, confs, classes = safe_boxes_from_result(res)
        chosen = (np.nan, np.nan)
        best_conf = 0.0
        if boxes is not None and len(boxes) > 0:
            for i, c in enumerate(classes):
                if int(c) == BALL_CLASS and confs[i]>=CONF_THRESHOLD:
                    if confs[i]>best_conf:
                        x1,y1,x2,y2 = boxes[i]
                        chosen = ((x1+x2)/2.0, (y1+y2)/2.0)
                        best_conf = confs[i]
        positions.append(chosen)
        frame_idx +=1
    cap.release()

    positions = np.array(positions, dtype=float)
    frames = np.arange(len(positions))
    raw_x = positions[:,0]; raw_y = positions[:,1]
    smooth_x, smooth_y = interp_and_smooth(frames, raw_x, raw_y)

    vel_px = central_diff_positions_to_velocity(smooth_x, smooth_y, dt)
    acc_px = central_diff_velocity_to_acceleration(vel_px, dt)
    vel_m = vel_px * M_PER_PX
    acc_m = acc_px * M_PER_PX
    speeds_m_s = np.linalg.norm(vel_m, axis=1)
    avg_speeds = moving_average(speeds_m_s, WINDOW_AVG)

    # write overlay video
    cap = cv2.VideoCapture(VIDEO_PATH)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    n_frames = len(frames)
    print("Writing overlay video...")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break

        last_draw = frame_idx if frame_idx<n_frames else n_frames-1
        for i in range(1,last_draw+1):
            p0 = (int(round(smooth_x[i-1])), int(round(smooth_y[i-1])))
            p1 = (int(round(smooth_x[i])), int(round(smooth_y[i])))
            if np.isfinite(p0).all() and np.isfinite(p1).all():
                cv2.line(frame, p0, p1, (200,40,40),2)

        if frame_idx<n_frames and frame_idx%FRAME_STEP==0 and frame_idx>0:
            px, py = smooth_x[frame_idx], smooth_y[frame_idx]
            if np.isfinite(px) and np.isfinite(py):
                vx, vy = vel_px[frame_idx]; ax, ay = acc_px[frame_idx]
                end_v = (int(round(px + vx*ARROW_VEL_SCALE)), int(round(py + vy*ARROW_VEL_SCALE)))
                end_a = (int(round(px + ax*ARROW_ACC_SCALE)), int(round(py + ay*ARROW_ACC_SCALE)))
                cv2.arrowedLine(frame, (int(round(px)), int(round(py))), end_v, (0,255,0),3, tipLength=0.3)
                cv2.arrowedLine(frame, (int(round(px)), int(round(py))), end_a, (0,165,255),3, tipLength=0.3)

        avg_speed = avg_speeds[frame_idx] if frame_idx<n_frames else 0.0
        cv2.putText(frame, f"Avg speed ({WINDOW_AVG} frames): {avg_speed:.3f} m/s", (30,40),
                    cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,255),2)
        cv2.putText(frame, f"Frame: {frame_idx}/{n_frames}", (30,80),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(220,220,220),2)

        out.write(frame)
        frame_idx+=1

    cap.release()
    out.release()
    print("Done. Saved video:", OUTPUT_PATH)

if __name__=="__main__":
    main()
