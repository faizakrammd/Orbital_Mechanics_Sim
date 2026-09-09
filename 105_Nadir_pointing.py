"""
Nadir-pointing simulation (fixed/robust version)

- Circular 500 km orbit, 51.6° inclination
- Quaternion attitude kinematics + rotational dynamics (rigid body)
- PD quaternion feedback controller that drives body +Z to nadir
- 3D animation: Earth, orbit, satellite cube, body axes, control torque vector
- Left plot: attitude error (deg) vs time (animated)
- Safer numerical handling, no default parameters referencing outer-scope variables,
  and protection against NaNs / overflow.

Run with: python nadir_pointing_fixed.py
Requires: numpy, matplotlib
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# -----------------------------
# Environment & orbit params
# -----------------------------
mu = 3.986004418e14        # Earth's gravitational parameter, m^3/s^2
R_earth = 6371e3           # Earth radius (m)
h = 500e3                  # altitude (m)
a = R_earth + h            # circular orbit radius (m)
incl = np.deg2rad(51.6)    # inclination (rad)

# orbital mean motion
n = math.sqrt(mu / a**3)

# -----------------------------
# Simulation params
# -----------------------------
dt = 0.5                   # simulation timestep (s)
t_final = 1200.0           # total simulated seconds
steps = int(t_final / dt)

# controller / dynamics
I = np.diag([0.02, 0.02, 0.03])   # inertia (kg·m^2)
I_inv = np.linalg.inv(I)
Kp = 1.8       # proportional gain (reduced for stability)
Kd = 0.8       # derivative gain
max_torque = 5e-3  # torque saturation (N·m)

frame_skip = 5  # playback speedup factor

# -----------------------------
# Quaternion utilities
# -----------------------------
def normalize(q):
    q = np.asarray(q, dtype=float)
    nrm = np.linalg.norm(q)
    if nrm == 0 or not np.isfinite(nrm):
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / nrm

def quat_conj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def quat_mul(q, r):
    w0,x0,y0,z0 = q
    w1,x1,y1,z1 = r
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1
    ])

def quat_to_dcm(q):
    q = normalize(q)
    w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w),   1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w), 1-2*(x*x+y*y)]
    ])

def rotm_to_quat(R):
    # numerically stable rotation matrix -> quaternion
    R = np.array(R, dtype=float)
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 1e-8:
        S = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    else:
        # find largest diagonal
        if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            w = (R[2,1] - R[1,2]) / S
            x = 0.25 * S
            y = (R[0,1] + R[1,0]) / S
            z = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            w = (R[0,2] - R[2,0]) / S
            x = (R[0,1] + R[1,0]) / S
            y = 0.25 * S
            z = (R[1,2] + R[2,1]) / S
        else:
            S = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            w = (R[1,0] - R[0,1]) / S
            x = (R[0,2] + R[2,0]) / S
            y = (R[1,2] + R[2,1]) / S
            z = 0.25 * S
    return normalize(np.array([w,x,y,z]))

def quat_error(q_des, q_est):
    # q_e = q_des * conj(q_est)
    q_e = quat_mul(q_des, quat_conj(q_est))
    if q_e[0] < 0:
        q_e = -q_e
    # small-angle vector approx: phi ≈ 2 * vector(q_e)
    phi = 2.0 * q_e[1:]
    # angle magnitude safe
    w0 = max(-1.0, min(1.0, q_e[0]))
    angle = 2.0 * math.acos(w0)
    return phi, angle

# -----------------------------
# Orbit functions
# -----------------------------
def orbit_position_velocity(t):
    theta = n * t
    r_orb = np.array([a * math.cos(theta), a * math.sin(theta), 0.0])
    v_orb = np.array([-a * n * math.sin(theta), a * n * math.cos(theta), 0.0])
    C_i = np.array([[1,0,0],
                    [0, math.cos(incl), -math.sin(incl)],
                    [0, math.sin(incl),  math.cos(incl)]])
    r = C_i @ r_orb
    v = C_i @ v_orb
    return r, v

# -----------------------------
# Initial conditions
# -----------------------------
q = normalize(np.array([0.7071, 0.2, 0.1, -0.1]))  # some initial attitude
omega = np.array([0.01, -0.02, 0.015])            # rad/s

# logs
time_log = np.zeros(steps)
err_angle_log = np.zeros(steps)
torque_log = np.zeros((steps,3))
omega_log = np.zeros((steps,3))
q_log = np.zeros((steps,4))
r_log = np.zeros((steps,3))

# -----------------------------
# Simulation loop
# -----------------------------
for k in range(steps):
    t = k * dt

    r, v = orbit_position_velocity(t)
    r_hat = r / np.linalg.norm(r)
    v_hat = v / (np.linalg.norm(v) + 1e-16)

    # desired body axes (in inertial)
    z_des = -r_hat                                  # body +Z -> nadir
    y_temp = np.cross(z_des, v_hat)
    if np.linalg.norm(y_temp) < 1e-8:
        y_des = np.array([0.0, 1.0, 0.0])
    else:
        y_des = y_temp / np.linalg.norm(y_temp)
    x_des = np.cross(y_des, z_des)
    x_des /= (np.linalg.norm(x_des) + 1e-16)

    R_des = np.column_stack((x_des, y_des, z_des))  # body->inertial
    q_des = rotm_to_quat(R_des)

    # attitude error
    phi, angle = quat_error(q_des, q)
    err_angle_log[k] = np.degrees(angle)

    # PD control in body frame (phi in body approx)
    torque_cmd = Kp * phi - Kd * omega
    tmag = np.linalg.norm(torque_cmd)
    if tmag > max_torque and tmag > 0:
        torque_cmd = torque_cmd * (max_torque / tmag)

    # rotational dynamics: I*w_dot + w x (I w) = torque_cmd
    w_dot = I_inv @ (torque_cmd - np.cross(omega, I @ omega))
    omega = omega + w_dot * dt

    # quaternion kinematics
    Omega = np.array([
        [0.0, -omega[0], -omega[1], -omega[2]],
        [omega[0], 0.0, omega[2], -omega[1]],
        [omega[1], -omega[2], 0.0, omega[0]],
        [omega[2], omega[1], -omega[0], 0.0]
    ])
    q = q + 0.5 * (Omega @ q) * dt
    q = normalize(q)

    # log
    time_log[k] = t
    torque_log[k,:] = torque_cmd
    omega_log[k,:] = omega
    q_log[k,:] = q
    r_log[k,:] = r

# -----------------------------
# Visualization setup
# -----------------------------
times = time_log
err_deg = err_angle_log

# small cube model for satellite
cube_verts = np.array([[-0.5,-0.5,-0.2],[0.5,-0.5,-0.2],[0.5,0.5,-0.2],[-0.5,0.5,-0.2],
                       [-0.5,-0.5,0.2],[0.5,-0.5,0.2],[0.5,0.5,0.2],[-0.5,0.5,0.2]])
cube_faces_idx = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[1,2,6,5],[4,7,3,0]]

def rotated_faces(qi, verts):
    R = quat_to_dcm(qi)
    rotated = (R @ verts.T).T
    faces = [[rotated[idx] for idx in face] for face in cube_faces_idx]
    return faces

# plotting: 1) left error plot, 2) right 3D scene
fig = plt.figure(figsize=(12,6))
ax_plot = fig.add_subplot(1,2,1)
ax_3d = fig.add_subplot(1,2,2, projection='3d')

ax_plot.set_xlim(0, times[-1])
ax_plot.set_ylim(0, max(err_deg)*1.1 + 1.0)
ax_plot.set_xlabel("Time (s)")
ax_plot.set_ylabel("Attitude error (deg)")
ax_plot.set_title("Nadir-pointing attitude error")
error_line, = ax_plot.plot([], [], lw=2, label="error (deg)")
ax_plot.legend(loc='upper right')
time_text = ax_plot.text(0.02, 0.9, '', transform=ax_plot.transAxes)

# 3D scene settings
ax_3d.set_box_aspect((1,1,1))
lim = a * 1.05
ax_3d.set_xlim([-lim, lim])
ax_3d.set_ylim([-lim, lim])
ax_3d.set_zlim([-lim, lim])
ax_3d.set_title("Orbit + satellite attitude (body axes)")

# draw Earth surface (low-res) once
u = np.linspace(0, 2*np.pi, 40)
v_ang = np.linspace(0, np.pi, 20)
x_s = (R_earth) * np.outer(np.cos(u), np.sin(v_ang))
y_s = (R_earth) * np.outer(np.sin(u), np.sin(v_ang))
z_s = (R_earth) * np.outer(np.ones_like(u), np.cos(v_ang))
ax_3d.plot_surface(x_s, y_s, z_s, rstride=2, cstride=2, color='lightblue', alpha=0.6, linewidth=0)

# orbit line
orbit_line, = ax_3d.plot(r_log[:,0], r_log[:,1], r_log[:,2], color='gray', alpha=0.6, lw=0.8)

# satellite cube and point
sat_cube = Poly3DCollection(rotated_faces(q_log[0], cube_verts), facecolor='cyan', edgecolor='k', alpha=0.8)
ax_3d.add_collection3d(sat_cube)
sat_point, = ax_3d.plot([r_log[0,0]], [r_log[0,1]], [r_log[0,2]], marker='o', color='red')

# helpers to add/remove dynamic artists (axes quivers, torque)
dynamic_artists = []

def draw_body_axes(ax, origin, Rmat, length):
    # Draw x (red), y (green), z (blue) axes using quiver; return handles
    ex = Rmat @ np.array([1.0,0.0,0.0]) * length
    ey = Rmat @ np.array([0.0,1.0,0.0]) * length
    ez = Rmat @ np.array([0.0,0.0,1.0]) * length
    a1 = ax.quiver(origin[0], origin[1], origin[2], ex[0], ex[1], ex[2], color='r', length=1.0, normalize=False)
    a2 = ax.quiver(origin[0], origin[1], origin[2], ey[0], ey[1], ey[2], color='g', length=1.0, normalize=False)
    a3 = ax.quiver(origin[0], origin[1], origin[2], ez[0], ez[1], ez[2], color='b', length=1.0, normalize=False)
    return [a1,a2,a3]

# annotation on 3D
annot = ax_3d.text2D(0.02, 0.95, "", transform=ax_3d.transAxes)

# -----------------------------
# Animation update
# -----------------------------
n_frames = int(np.ceil(steps / frame_skip))

def animate(frame_idx):
    # remove previous dynamic artists
    global dynamic_artists
    for art in dynamic_artists:
        try:
            art.remove()
        except Exception:
            pass
    dynamic_artists = []

    sim_idx = min(frame_idx * frame_skip, steps-1)

    # update left plot
    t_seg = times[:sim_idx+1]
    e_seg = err_deg[:sim_idx+1]
    error_line.set_data(t_seg, e_seg)
    time_text.set_text(f"t = {times[sim_idx]:.1f} s")

    # update satellite cube
    q_now = q_log[sim_idx]
    r_now = r_log[sim_idx]
    faces_rot = rotated_faces(q_now, cube_verts)
    sat_cube.set_verts(faces_rot)

    sat_point.set_data([r_now[0]], [r_now[1]])
    sat_point.set_3d_properties([r_now[2]])

    # draw body axes (length scaled by orbit radius)
    axes_len = 0.04 * a
    axes = draw_body_axes(ax_3d, r_now, quat_to_dcm(q_now), axes_len)
    dynamic_artists.extend(axes)

    # draw torque vector: transform body torque -> inertial
    torq_body = torque_log[sim_idx]
    torq_inertial = quat_to_dcm(q_now) @ torq_body
    tq = ax_3d.quiver(r_now[0], r_now[1], r_now[2],
                      torq_inertial[0], torq_inertial[1], torq_inertial[2],
                      color='magenta', length=axes_len*0.6, normalize=False)
    dynamic_artists.append(tq)

    annot.set_text(f"err = {err_deg[sim_idx]:.2f} deg\n|torque| = {np.linalg.norm(torq_body):.2e} N·m")

    return (error_line, time_text, sat_cube, sat_point, annot)

# -----------------------------
# Run animation
# -----------------------------
ani = FuncAnimation(fig, animate, frames=n_frames, interval=40, blit=False)
plt.tight_layout()
plt.show()
