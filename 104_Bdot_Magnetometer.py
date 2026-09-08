import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# =====================
# Simulation parameters
# =====================
dt = 0.1
t_final = 200
steps = int(t_final/dt)

I = np.diag([0.02, 0.02, 0.03])
I_inv = np.linalg.inv(I)

B0 = np.array([2e-5, 3e-5, -4e-5])  # Tesla
k = 1e4

w = np.array([0.2, -0.3, 0.5])  # initial angular velocity (rad/s)
q = np.array([1, 0, 0, 0])      # quaternion

time_arr, w_norm_arr = [], []
w_hist, q_hist, torque_hist, energy_hist = [], [], [], []

# Quaternion helpers
def quat_mult(q1, q2):
    w0, x0, y0, z0 = q1
    w1, x1, y1, z1 = q2
    return np.array([
        -x0*x1 - y0*y1 - z0*z1 + w0*w1,
         x0*w1 + y0*z1 - z0*y1 + w0*x1,
        -x0*z1 + y0*w1 + z0*x1 + w0*y1,
         x0*y1 - y0*x1 + z0*w1 + w0*z1
    ])

def normalize(q):
    return q / np.linalg.norm(q)

# =====================
# Dynamics loop
# =====================
for step in range(steps):
    t = step * dt

    dBdt = -np.cross(w, B0)
    m_cmd = -k * dBdt
    torque = np.cross(m_cmd, B0)

    w_dot = I_inv @ (torque - np.cross(w, I @ w))
    w = w + w_dot * dt

    w_quat = np.concatenate([[0], w])
    q_dot = 0.5 * quat_mult(q, w_quat)
    q = normalize(q + q_dot * dt)

    # Energy
    energy = 0.5 * w @ (I @ w)

    # Store
    time_arr.append(t)
    w_norm_arr.append(np.linalg.norm(w))
    w_hist.append(w.copy())
    q_hist.append(q.copy())
    torque_hist.append(torque.copy())
    energy_hist.append(energy)

w_hist = np.array(w_hist)
q_hist = np.array(q_hist)
torque_hist = np.array(torque_hist)
energy_hist = np.array(energy_hist)

# =====================
# Find detumble time
# =====================
threshold = 0.01  # rad/s
detumble_time = None
for t, wn in zip(time_arr, w_norm_arr):
    if wn < threshold:
        detumble_time = t
        break

# =====================
# Plotting setup
# =====================
fig = plt.figure(figsize=(12, 8))

ax3d = fig.add_subplot(221, projection='3d')
ax_norm = fig.add_subplot(222)
ax_comp = fig.add_subplot(223)
ax_energy = fig.add_subplot(224)

# Cube definition
cube_def = np.array([[-0.5, -0.5, -0.5],
                     [ 0.5, -0.5, -0.5],
                     [ 0.5,  0.5, -0.5],
                     [-0.5,  0.5, -0.5],
                     [-0.5, -0.5,  0.5],
                     [ 0.5, -0.5,  0.5],
                     [ 0.5,  0.5,  0.5],
                     [-0.5,  0.5,  0.5]])

faces = [[cube_def[j] for j in [0,1,2,3]],
         [cube_def[j] for j in [4,5,6,7]],
         [cube_def[j] for j in [0,1,5,4]],
         [cube_def[j] for j in [2,3,7,6]],
         [cube_def[j] for j in [1,2,6,5]],
         [cube_def[j] for j in [4,7,3,0]]]

cube = Poly3DCollection(faces, alpha=0.3, edgecolor='k')
ax3d.add_collection3d(cube)
ax3d.set_xlim([-1, 1])
ax3d.set_ylim([-1, 1])
ax3d.set_zlim([-1, 1])
ax3d.set_title("CubeSat Orientation + Torque Vector")

# Angular velocity norm
line_norm, = ax_norm.plot([], [], lw=2)
ax_norm.set_xlim(0, t_final)
ax_norm.set_ylim(0, max(w_norm_arr))
ax_norm.set_title("|ω| (Angular Velocity Norm)")
ax_norm.set_xlabel("Time [s]")
ax_norm.set_ylabel("[rad/s]")

if detumble_time:
    ax_norm.axhline(threshold, color="r", ls="--", lw=1)
    ax_norm.annotate(f"Detumbled at t={detumble_time:.1f}s",
                     xy=(detumble_time, threshold),
                     xytext=(detumble_time+20, threshold*3),
                     arrowprops=dict(arrowstyle="->", color="red"),
                     color="red")

# Angular velocity components
line_wx, = ax_comp.plot([], [], label="ωx")
line_wy, = ax_comp.plot([], [], label="ωy")
line_wz, = ax_comp.plot([], [], label="ωz")
ax_comp.legend()
ax_comp.set_xlim(0, t_final)
ax_comp.set_ylim(np.min(w_hist)*1.1, np.max(w_hist)*1.1)
ax_comp.set_title("Angular Velocity Components")
ax_comp.set_xlabel("Time [s]")
ax_comp.set_ylabel("[rad/s]")

# Energy
line_energy, = ax_energy.plot([], [], color="purple")
ax_energy.set_xlim(0, t_final)
ax_energy.set_ylim(0, max(energy_hist))
ax_energy.set_title("Rotational Kinetic Energy")
ax_energy.set_xlabel("Time [s]")
ax_energy.set_ylabel("[J]")

# Quaternion → DCM
def quat_to_dcm(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2+z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x**2+z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2+y**2)]
    ])

# =====================
# Animation
# =====================
def animate(i):
    idx = i * 5  # ~4x faster
    if idx >= steps:
        idx = steps - 1

    q = q_hist[idx]
    R = quat_to_dcm(q)
    rotated = (R @ cube_def.T).T
    new_faces = [[rotated[j] for j in [0,1,2,3]],
                 [rotated[j] for j in [4,5,6,7]],
                 [rotated[j] for j in [0,1,5,4]],
                 [rotated[j] for j in [2,3,7,6]],
                 [rotated[j] for j in [1,2,6,5]],
                 [rotated[j] for j in [4,7,3,0]]]
    cube.set_verts(new_faces)

    # Torque vector arrow
    torq = torque_hist[idx]
    ax3d.quiver(0, 0, 0, torq[0], torq[1], torq[2], color="red", lw=2)

    # Update time histories
    line_norm.set_data(time_arr[:idx], w_norm_arr[:idx])
    line_wx.set_data(time_arr[:idx], w_hist[:idx,0])
    line_wy.set_data(time_arr[:idx], w_hist[:idx,1])
    line_wz.set_data(time_arr[:idx], w_hist[:idx,2])
    line_energy.set_data(time_arr[:idx], energy_hist[:idx])

    return cube, line_norm, line_wx, line_wy, line_wz, line_energy

ani = FuncAnimation(fig, animate, frames=steps//5, interval=30, blit=False)
plt.tight_layout()
plt.show()
