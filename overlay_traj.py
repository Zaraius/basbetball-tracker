# overlay_traj_3d.py
import os
import cv2
import numpy as np

# Optional smoothing
try:
    from scipy.interpolate import PchipInterpolator
    from scipy.signal import savgol_filter
    SCIPY = True
except Exception:
    PchipInterpolator = None
    savgol_filter = None
    SCIPY = False

class BasketballTrajectory3D:
    def __init__(self,
                 video_path,
                 labels_folder,
                 img_width,
                 img_height,
                 distances=None,
                 ball_class=0,
                 ball_diameter_inches=9.4,
                 ball_diameter_pixels=30,
                 frame_step=10,
                 arrow_vel_scale=0.04,
                 arrow_acc_scale=0.0015,
                 window_avg=20,
                 min_valid_detections=3,
                 fps_override=None):
        self.video_path = video_path
        self.labels_folder = labels_folder
        self.img_width = img_width
        self.img_height = img_height
        self.distances = distances  # should be same length as number of frames
        self.ball_class = ball_class
        self.frame_step = frame_step
        self.arrow_vel_scale = arrow_vel_scale
        self.arrow_acc_scale = arrow_acc_scale
        self.window_avg = window_avg
        self.min_valid_detections = min_valid_detections
        self.fps_override = fps_override

        # Conversion from pixels to meters
        self.BALL_DIAMETER_METERS = ball_diameter_inches * 0.0254
        self.M_PER_PX = self.BALL_DIAMETER_METERS / ball_diameter_pixels

        # Results placeholders
        self.positions = None
        self.smooth_x = None
        self.smooth_y = None
        self.smooth_z = None
        self.vel_m = None
        self.acc_m = None
        self.avg_speeds = None
        self.frames = None
        self.dt = None

    # ------------------ Trajectory Utils ------------------
    @staticmethod
    def interp_and_smooth(frames, arr, min_valid_detections=3):
        n = len(frames)
        idx = np.arange(n)
        valid = np.isfinite(arr)
        if valid.sum() == 0:
            raise RuntimeError("No valid detections to interpolate.")
        arr_filled = np.interp(idx, idx[valid], arr[valid])
        if SCIPY and valid.sum() >= min_valid_detections:
            try:
                cs = PchipInterpolator(idx[valid], arr[valid])
                arr_smooth = cs(idx)
                if savgol_filter is not None and n >= 7:
                    win = 7 if n >= 7 else (n//2*2+1)
                    arr_smooth = savgol_filter(arr_smooth, win, 3)
                return arr_smooth
            except Exception:
                pass
        # fallback: simple moving average
        kernel = np.ones(5)/5.0
        return np.convolve(arr_filled, kernel, mode='same')

    @staticmethod
    def central_diff(arr, dt):
        n = len(arr)
        deriv = np.zeros_like(arr)
        if n < 2:
            return deriv
        deriv[0] = (arr[1] - arr[0]) / dt
        for i in range(1, n-1):
            deriv[i] = (arr[i+1] - arr[i-1]) / (2*dt)
        deriv[-1] = (arr[-1] - arr[-2]) / dt
        return deriv

    @staticmethod
    def moving_average(arr, window):
        n = len(arr)
        avg = np.zeros_like(arr)
        for i in range(n):
            start = max(0, i - window + 1)
            avg[i] = np.nanmean(arr[start:i+1])
        return avg

    # ------------------ Label Reading ------------------
    def read_labels(self):
        positions = []
        label_files = sorted(os.listdir(self.labels_folder))
        self.frames = np.arange(len(label_files))
        for i, label_file in enumerate(label_files):
            if not label_file.endswith(".txt"):
                positions.append((np.nan, np.nan))
                continue
            path = os.path.join(self.labels_folder, label_file)
            with open(path) as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            if len(lines) == 0:
                positions.append((np.nan, np.nan))
                continue
            cls, x, y, *_ = map(float, lines[0].split())
            if int(cls) != self.ball_class:
                positions.append((np.nan, np.nan))
                continue
            positions.append((x * self.img_width, y * self.img_height))
        self.positions = np.array(positions, dtype=float)
        return self.positions

    # ------------------ Compute Trajectory ------------------
    def compute_trajectory(self):
        if self.positions is None:
            self.read_labels()

        n_frames = len(self.positions)
        self.dt = 1.0 / (self.fps_override or 60.0)

        raw_x, raw_y = self.positions[:,0], self.positions[:,1]

        # Z positions from distances (or nan if missing)
        if self.distances is not None and len(self.distances) == n_frames:
            raw_z = np.array(self.distances)
        else:
            raw_z = np.full(n_frames, np.nan)

        self.smooth_x = self.interp_and_smooth(self.frames, raw_x, self.min_valid_detections)
        self.smooth_y = self.interp_and_smooth(self.frames, raw_y, self.min_valid_detections)
        self.smooth_z = self.interp_and_smooth(self.frames, raw_z, self.min_valid_detections)

        # velocities
        vx = self.central_diff(self.smooth_x, self.dt)
        vy = self.central_diff(self.smooth_y, self.dt)
        vz = self.central_diff(self.smooth_z, self.dt)
        self.vel_m = np.stack([vx, vy, vz], axis=1) * self.M_PER_PX

        # accelerations
        ax = self.central_diff(vx, self.dt)
        ay = self.central_diff(vy, self.dt)
        az = self.central_diff(vz, self.dt)
        self.acc_m = np.stack([ax, ay, az], axis=1) * self.M_PER_PX

        # average speeds
        speeds = np.linalg.norm(self.vel_m, axis=1)
        self.avg_speeds = self.moving_average(speeds, self.window_avg)

        return self.smooth_x, self.smooth_y, self.smooth_z, self.vel_m, self.acc_m, self.avg_speeds

    # ------------------ Overlay Video ------------------
    def overlay_video(self, output_path, show_z=True, show_vel=True, show_acc=True):
        if self.smooth_x is None or self.smooth_y is None or self.smooth_z is None:
            self.compute_trajectory()

        cap = cv2.VideoCapture(self.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.fps_override or cap.get(cv2.CAP_PROP_FPS) or 60.0

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        n_frames = len(self.frames)

        for frame_idx in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # draw trajectory
            for i in range(1, frame_idx+1):
                p0 = (int(round(self.smooth_x[i-1])), int(round(self.smooth_y[i-1])))
                p1 = (int(round(self.smooth_x[i])), int(round(self.smooth_y[i])))
                if np.isfinite(p0).all() and np.isfinite(p1).all():
                    cv2.line(frame, p0, p1, (200,40,40),2)

            # draw velocity & acceleration arrows
            if frame_idx % self.frame_step == 0 and frame_idx > 0:
                px, py, pz = self.smooth_x[frame_idx], self.smooth_y[frame_idx], self.smooth_z[frame_idx]
                vx, vy, vz = self.vel_m[frame_idx]
                ax, ay, az = self.acc_m[frame_idx]
                if show_vel:
                    cv2.arrowedLine(frame, (int(px), int(py)),
                                    (int(px + vx*self.arrow_vel_scale), int(py + vy*self.arrow_vel_scale)),
                                    (0,255,0), 3, tipLength=0.3)
                if show_acc:
                    cv2.arrowedLine(frame, (int(px), int(py)),
                                    (int(px + ax*self.arrow_acc_scale), int(py + ay*self.arrow_acc_scale)),
                                    (0,165,255), 3, tipLength=0.3)

            # overlay text: avg speed, Z, velocity, acceleration
            px, py, pz = self.smooth_x[frame_idx], self.smooth_y[frame_idx], self.smooth_z[frame_idx]
            vx, vy, vz = self.vel_m[frame_idx]
            ax, ay, az = self.acc_m[frame_idx]

            z_text = f" Z: {pz:.2f} m" if show_z else ""
            vel_text = f" | V: ({vx:.2f},{vy:.2f},{vz:.2f}) m/s" if show_vel else ""
            acc_text = f" | A: ({ax:.2f},{ay:.2f},{az:.2f}) m/s^2" if show_acc else ""

            cv2.putText(frame,
                        f"Avg speed: {self.avg_speeds[frame_idx]:.2f} m/s" + z_text + vel_text + acc_text,
                        (60,950), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 6, 6), 3)

            out.write(frame)

        cap.release()
        out.release()
        print("Saved video:", output_path)
