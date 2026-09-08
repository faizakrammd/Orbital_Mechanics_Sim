import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --------------------------
# Quaternion utilities
# --------------------------
def normalize_quat(q):
    return q / np.linalg.norm(q)

def quat_mult(q1, q2):
    w0, x0, y0, z0 = q1
    w1, x1, y1, z1 = q2
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1
    ])

def omega_matrix(omega):
    wx, wy, wz = omega
    return np.array([
        [0, -wx, -wy, -wz],
        [wx, 0, wz, -wy],
        [wy, -wz, 0, wx],
        [wz, wy, -wx, 0]
    ])

def quat_to_rotmat(q):
    q = normalize_quat(q)
    w,x,y,z = q
    return np.array([
        [1-2*(y**2+z**2), 2*(x*y - z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x**2+z**2), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x**2+y**2)]
    ])

# --------------------------
# Sim parameters
# --------------------------
dt = 0.05
t_final = 40
steps = int(t_final/dt)

# True state
q_true = normalize_quat(np.array([1, 0, 0, 0]))
omega_true = np.array([0.1, -0.05, 0.07])  # rad/s

# Gyro bias + noise
gyro_bias = np.array([0.01, -0.005, 0.002])
gyro_noise_std = 0.002
st_noise_std = 0.01

# EKF setup
q_est = normalize_quat(np.array([0.9, 0.1, 0, 0]))  # wrong initial guess
bias_est = np.zeros(3)
P = np.eye(6) * 0.1
Q = np.eye(6) * 1e-4
R = np.eye(4) * (st_noise_std**2)

q_log = []
q_est_log = []

# --------------------------
# Define spacecraft cube
# --------------------------
def cube_vertices():
    r = [-0.5, 0.5]
    verts = np.array([[x,y,z] for x in r for y in r for z in r])
    faces = [
        [verts[j] for j in [0,1,3,2]],
        [verts[j] for j in [4,5,7,6]],
        [verts[j] for j in [0,1,5,4]],
        [verts[j] for j in [2,3,7,6]],
        [verts[j] for j in [0,2,6,4]],
        [verts[j] for j in [1,3,7,5]],
    ]
    return faces

faces = cube_vertices()

def rotate_vertices(faces, q):
    R = quat_to_rotmat(q)
    new_faces = []
    for f in faces:
        new_faces.append([R@v for v in f])
    return new_faces

# --------------------------
# Simulation loop
# --------------------------
for k in range(steps):
    # --- True dynamics ---
    dq = 0.5 * omega_matrix(omega_true) @ q_true
    q_true = normalize_quat(q_true + dq*dt)

    # --- Gyro measurement ---
    gyro_meas = omega_true + gyro_bias + np.random.randn(3)*gyro_noise_std

    # --- Star tracker measurement ---
    q_meas = normalize_quat(q_true + np.random.randn(4)*st_noise_std)

    # --- EKF Prediction ---
    omega_hat = gyro_meas - bias_est
    F = np.eye(6)
    F[:3,:3] += -dt*np.eye(3)
    P = F@P@F.T + Q

    dq_est = 0.5 * omega_matrix(omega_hat) @ q_est
    q_est = normalize_quat(q_est + dq_est*dt)

    # --- EKF Update ---
    z = q_meas
    h = q_est
    y = z - h
    H = np.zeros((4,6))
    K = P@H.T @ np.linalg.inv(H@P@H.T + R)

    dx = K@y
    q_est = normalize_quat(q_est + np.hstack(([0], dx[:3])))
    bias_est += dx[3:]
    P = (np.eye(6) - K@H)@P

    q_log.append(q_true)
    q_est_log.append(q_est)

q_log = np.array(q_log)
q_est_log = np.array(q_est_log)
t = np.arange(steps)*dt
err_log = q_log - q_est_log

# --------------------------
# Combined Figure (Side by Side)
# --------------------------
fig = plt.figure(figsize=(12,6))

# Left: Error plot
ax1 = fig.add_subplot(1,2,1)
lines = ax1.plot([], [], [], [], [], [], [], [])
ax1.set_xlim(0, t[-1])
ax1.set_ylim(-0.3, 0.3)
ax1.set_title("Quaternion Estimation Error")
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Error")
ax1.legend(["w","x","y","z"])

# Right: 3D cube
ax2 = fig.add_subplot(1,2,2, projection='3d')
ax2.set_xlim([-1,1]); ax2.set_ylim([-1,1]); ax2.set_zlim([-1,1])
ax2.set_title("True (blue) vs EKF Estimate (red) Attitude")

poly_true = Poly3DCollection(rotate_vertices(faces, q_log[0]), 
                             facecolors='cyan', alpha=0.25, edgecolors='b')
poly_est = Poly3DCollection(rotate_vertices(faces, q_est_log[0]), 
                            facecolors='orange', alpha=0.25, edgecolors='r')
ax2.add_collection3d(poly_true)
ax2.add_collection3d(poly_est)

# --------------------------
# Animation (10x faster)
# --------------------------
frame_skip = 10  # skip frames to speed up playback
frames = range(0, steps, frame_skip)

def update(i):
    idx = i
    # update error lines
    for j,l in enumerate(lines):
        l.set_data(t[:idx], err_log[:idx,j])
    # update cube orientation
    poly_true.set_verts(rotate_vertices(faces, q_log[idx]))
    poly_est.set_verts(rotate_vertices(faces, q_est_log[idx]))
    return lines + [poly_true, poly_est]

ani = FuncAnimation(fig, update, frames=frames, interval=30, blit=False)
plt.tight_layout()
plt.show()
