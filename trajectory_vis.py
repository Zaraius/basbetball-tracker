import numpy as np
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt

##########################################
# Kalman Filter (2D constant acceleration)
##########################################
def make_kf(dt=1.0):
    kf = KalmanFilter(dim_x=6, dim_z=2)

    kf.F = np.array([
        [1,0,dt,0,0.5*dt*dt,0],
        [0,1,0,dt,0,0.5*dt*dt],
        [0,0,1,0,dt,0],
        [0,0,0,1,0,dt],
        [0,0,0,0,1,0],
        [0,0,0,0,0,1]
    ])

    kf.H = np.array([
        [1,0,0,0,0,0],
        [0,1,0,0,0,0]
    ])

    kf.P *= 1000.0
    kf.R = np.eye(2) * 1.0
    kf.Q = np.eye(6) * 0.01
    kf.x = np.zeros((6,1))
    return kf

##########################################
# Main
##########################################
def main(txt_file):
    data = np.loadtxt(txt_file)

    print("=== DEBUG: Raw data shape ===")
    print(data.shape)

    print("=== DEBUG: First 5 rows ===")
    print(data[:5])

    if np.isnan(data).any():
        print("WARNING: NaNs detected in input file.")

    # Expect at least x, y in first two cols
    if data.shape[1] < 2:
        raise ValueError("Input does not have at least 2 columns for x,y")

    # Use first two columns as (x,y)
    positions = data[:, 1:3]

    x_positions = positions[:, 0]
    y_positions = positions[:, 1]

    print("=== DEBUG: Position stats ===")
    print("x min/max:", np.min(x_positions), np.max(x_positions))
    print("y min/max:", np.min(y_positions), np.max(y_positions))

    kf = make_kf(dt=1.0)

    trajectory = []
    velocities = []
    accelerations = []

    for frame_idx, pos in enumerate(positions):
        meas = pos.reshape(2,1)

        if frame_idx == 0:
            kf.x[:2] = meas  # set x,y
            kf.x[2:6] = 0    # set v,a = 0

        kf.predict()
        kf.update(meas)

        x, y, vx, vy, ax, ay = kf.x.flatten()

        trajectory.append([x, y])
        velocities.append([vx, vy])
        accelerations.append([ax, ay])

        print(f"Frame {frame_idx}: x={x:.2f}, y={y:.2f}, vx={vx:.2f}, vy={vy:.2f}, ax={ax:.2f}, ay={ay:.2f}")

    trajectory = np.array(trajectory)
    velocities = np.array(velocities)
    accelerations = np.array(accelerations)

    print("=== DEBUG: Final check ===")
    print("Frames:", len(trajectory))
    print("Positions:", len(positions))
    print("Velocities:", len(velocities))
    print("Accelerations:", len(accelerations))

    # Plot trajectory with velocity and acceleration vectors
    plt.figure(figsize=(8,6))
    plt.plot(trajectory[:,0], trajectory[:,1], 'k-o', label='Trajectory')

    # Velocity arrows
    plt.quiver(
        trajectory[:,0], trajectory[:,1],
        velocities[:,0], velocities[:,1],
        color='blue', scale_units='xy', angles='xy', scale=3
    )

    # Acceleration arrows
    plt.quiver(
        trajectory[:,0], trajectory[:,1],
        accelerations[:,0], accelerations[:,1],
        color='red', scale_units='xy', angles='xy', scale=20
    )

    plt.title("Trajectory with Velocity (blue) and Acceleration (red)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    main("/home/satchel/basbetball-tracker/output_frames/labels/combined.txt")