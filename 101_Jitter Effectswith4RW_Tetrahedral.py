"""
Reaction Wheel Imbalance / Jitter Effects with 4 Reaction Wheels (Tetrahedral)
Single-file Python simulation + animation.

Features:
- 3-axis rigid-body rotational dynamics (quaternion representation)
- 4 reaction wheels in tetrahedral orientation (non-orthogonal)
- PD controller (3-axis) -> desired body torque
- Allocation: least-squares (pseudo-inverse) to produce wheel torques (with saturation)
- Wheel dynamics and speed limits
- Wheel imbalance / jitter modeled as small sinusoidal torques about each wheel's spin axis, amplitude scales with wheel speed
- Optional external disturbance torque 
- 3D cube animation + plots of angles, wheel speeds, torques 

Author: ChatGPT
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time as pytime  # use pytime for timing so we don't shadow it

# ---------------- User-tunable parameters ----------------
t_final = 40.0        # total simulated seconds
dt = 0.005            # integration timestep (s) - small for dynamics stability
steps = int(t_final / dt)

# Spacecraft inertia (principal axes) [kg*m^2]
Ixx, Iyy, Izz = 0.02, 0.025, 0.018
I_sc = np.diag([Ixx, Iyy, Izz])

# Reaction wheel inertia (rotor) [kg*m^2] (scalar for each)
I_rw = 1e-4

# Reaction wheel motor torque limit [N*m]
rw_torque_limit = 1e-4

# Wheel angular speed limits [rad/s] (~6000 rpm)
rw_speed_rpm_limit = 6000.0
rw_speed_limit = rw_speed_rpm_limit * 2*np.pi/60.0

# PD controller gains (tune for stability)
Kp = 0.6   # proportional (N*m/rad)
Kd = 0.08  # derivative (N*m*s/rad)

# Tetrahedral wheel axis configuration (body-frame unit vectors)
# Columns are the 4 wheel spin axes in body coordinates
a = 1.0 / math.sqrt(3.0)
A = np.array([
    [  a,  a, -a, -a],   # x-components
    [  a, -a,  a, -a],   # y-components
    [  a,  a,  a,  a]    # z-components
], dtype=float)  # shape (3,4)
# Normalize columns
for i in range(4):
    A[:,i] /= np.linalg.norm(A[:,i])

# Wheel imbalance / jitter parameters
eps = np.array([2e-8, 1.8e-8, 2.2e-8, 1.9e-8])  # tune for visible jitter
phi = np.random.uniform(0, 2*np.pi, size=4)     # jitter phase per wheel

# External disturbance torque (small)
external_disturbance_scale = 5e-7  # N*m (RMS)

# Visualization parameters
frame_skip = 20   # draw every N simulation steps to speed animation
cube_size = 0.12  # m for plotting

# ---------------- Utility functions ----------------
def skew(v):
    return np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])

def quat_mult(q, r):
    w0,x0,y0,z0 = q
    w1,x1,y1,z1 = r
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1
    ])

def quat_conj(q):
    w,x,y,z = q
    return np.array([w,-x,-y,-z])

def quat_to_rotm(q):
    w,x,y,z = q
    R = np.array([
        [1-2*(y*y+z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w),   1-2*(x*x+z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w),   1-2*(x*x+y*y)]
    ])
    return R

def normalize_quat(q):
    return q / np.linalg.norm(q)

# ---------------- State initialization ----------------
q = np.array([1.0, 0.0, 0.0, 0.0])   # unit quaternion
q = normalize_quat(q.copy())
omega = np.array([0.0, 0.0, 0.0])   # body angular velocity (rad/s)
omega_rw = np.zeros(4)               # wheel speeds (rad/s) initial

# commanded attitude: small roll,pitch, yaw (use yaw=30 deg here)
cmd_euler = np.array([0.0, 0.0, math.radians(30.0)])  # roll,pitch,yaw
cy = math.cos(cmd_euler[2]*0.5); sy = math.sin(cmd_euler[2]*0.5)
cp = math.cos(cmd_euler[1]*0.5); sp = math.sin(cmd_euler[1]*0.5)
cr = math.cos(cmd_euler[0]*0.5); sr = math.sin(cmd_euler[0]*0.5)
q_cmd = np.array([
    cr*cp*cy + sr*sp*sy,
    sr*cp*cy - cr*sp*sy,
    cr*sp*cy + sr*cp*sy,
    cr*cp*sy - sr*sp*cy
])

# ---------------- Logging arrays ----------------
t_log = np.zeros(steps)
quat_log = np.zeros((steps,4))
omega_log = np.zeros((steps,3))
omega_rw_log = np.zeros((steps,4))
Tcmd_log = np.zeros((steps,3))
Talloc_log = np.zeros((steps,4))
Tjitter_log = np.zeros((steps,3))
Tout_log = np.zeros((steps,3))

# ---------------- Main simulation loop ----------------
print("Running simulation: t_final=%.1f s, dt=%.4f s, steps=%d" % (t_final, dt, steps))
start_wall = pytime.time()
for k in range(steps):
    t = k*dt
    t_log[k] = t

    # attitude error: q_err = q_cmd * conj(q)
    q_conj = quat_conj(q)
    q_err = quat_mult(q_cmd, q_conj)
    if q_err[0] < 0:
        q_err = -q_err
    # small-angle approx -> attitude error vector
    att_err = q_err[1:] * (2.0 / (np.linalg.norm(q_err) + 1e-30))

    # PD controller (body torque command)
    Tcmd = Kp * att_err - Kd * omega  # body torque (3-vector)

    # allocation: want A @ tau = -Tcmd  => tau = pinv(A) * (-Tcmd)
    tau_des = np.linalg.pinv(A) @ (-Tcmd)   # 4x1 desired wheel torques
    # clip to per-wheel torque limits
    tau_clipped = np.clip(tau_des, -rw_torque_limit, rw_torque_limit)
    # actual achieved spacecraft torque from wheels
    T_alloc_sc = -A @ tau_clipped

    # wheel dynamics (apply torques to wheels): torque_on_wheel = -tau_clipped
    domega_rw = (-tau_clipped) / I_rw
    omega_rw = omega_rw + domega_rw * dt
    # enforce wheel speed limits (clip)
    omega_rw = np.clip(omega_rw, -rw_speed_limit, rw_speed_limit)

    # wheel jitter / imbalance torque
    T_jitter = np.zeros(3)
    for i in range(4):
        mag_i = eps[i] * omega_rw[i] * math.sin(omega_rw[i] * t + phi[i])
        axis_i = A[:,i]
        T_jitter += mag_i * axis_i

    # optional external disturbance
    T_dist = external_disturbance_scale * np.random.randn(3)

    # total torque on spacecraft
    T_total = T_alloc_sc + T_jitter + T_dist

    # rigid-body rotational dynamics: I*omega_dot + omega x (I*omega) = T_total
    omega_dot = np.linalg.inv(I_sc) @ (T_total - np.cross(omega, I_sc @ omega))
    omega = omega + omega_dot * dt

    # quaternion kinematics
    wx,wy,wz = omega
    Omega = np.array([
        [0.0, -wx, -wy, -wz],
        [wx,  0.0,  wz, -wy],
        [wy, -wz,  0.0,  wx],
        [wz,  wy, -wx,  0.0]
    ])
    q = q + 0.5 * (Omega @ q) * dt
    q = normalize_quat(q)

    # logging
    quat_log[k,:] = q
    omega_log[k,:] = omega
    omega_rw_log[k,:] = omega_rw
    Tcmd_log[k,:] = Tcmd
    Talloc_log[k,:] = tau_clipped
    Tjitter_log[k,:] = T_jitter
    Tout_log[k,:] = T_total

wall_elapsed = pytime.time() - start_wall
print("Simulation completed in %.2f s (wall-clock)." % wall_elapsed)

# ---------------- Post-processing: Euler angles ----------------
def quat_to_euler_zyx(q):
    w,x,y,z = q
    siny_cosp = 2*(w*z + x*y)
    cosy_cosp = 1 - 2*(y*y + z*z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    sinp = 2*(w*y - z*x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi/2, sinp)
    else:
        pitch = math.asin(sinp)
    sinr_cosp = 2*(w*x + y*z)
    cosr_cosp = 1 - 2*(x*x + y*y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    return roll, pitch, yaw

euler_log = np.zeros((steps,3))
for k in range(steps):
    euler_log[k,:] = quat_to_euler_zyx(quat_log[k,:])

# ---------------- Visualization / Animation ----------------
fig = plt.figure(figsize=(13,9))
ax3d = fig.add_subplot(221, projection='3d')
ax_eul = fig.add_subplot(222)
ax_wheels = fig.add_subplot(212)

# Cube geometry
s = cube_size/2.0
verts0 = np.array([[-s,-s,-s],[ s,-s,-s],[ s, s,-s],[-s, s,-s],
                   [-s,-s, s],[ s,-s, s],[ s, s, s],[-s, s, s]])
faces = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[1,2,6,5],[4,7,3,0]]

t_arr = t_log
roll = euler_log[:,0]; pitch = euler_log[:,1]; yaw = euler_log[:,2]
jitter_norm = np.linalg.norm(Tjitter_log, axis=1)

# Pre-plot static elements
ax_eul.plot(t_arr, np.degrees(roll), label='roll (deg)', alpha=0.3)
ax_eul.plot(t_arr, np.degrees(pitch), label='pitch (deg)', alpha=0.3)
ax_eul.plot(t_arr, np.degrees(yaw), label='yaw (deg)', alpha=0.3)
ax_eul.axhline(np.degrees(cmd_euler[0]), color='k', ls='--', alpha=0.6)
ax_eul.axhline(np.degrees(cmd_euler[1]), color='k', ls='--', alpha=0.6)
ax_eul.axhline(np.degrees(cmd_euler[2]), color='k', ls='--', alpha=0.6)
ax_eul.set_xlabel('Time (s)'); ax_eul.set_ylabel('Angle (deg)')
ax_eul.legend(); ax_eul.grid(True)
ax_eul.set_title("Euler angles (full run shown faded)")

ax_wheels.set_xlabel('Time (s)')
ax_wheels.set_ylabel('omega_rw (rad/s)')

n_frames = max(1, steps // frame_skip)

def animate(i):
    idx = int(i * frame_skip)
    if idx >= steps: idx = steps - 1

    # rotation matrix (body->inertial)
    Rbi = quat_to_rotm(quat_log[idx,:])
    verts = verts0.dot(Rbi.T)  # transform cube vertices into inertial frame for plotting
    ax3d.cla()
    cube = Poly3DCollection([[verts[j] for j in f] for f in faces], alpha=0.6, facecolor='skyblue', edgecolor='k')
    ax3d.add_collection3d(cube)

    # body axes in inertial frame
    origin = np.array([0.0,0.0,0.0])
    ex = Rbi.dot(np.array([1.0,0.0,0.0]))*0.08
    ey = Rbi.dot(np.array([0.0,1.0,0.0]))*0.08
    ez = Rbi.dot(np.array([0.0,0.0,1.0]))*0.08
    ax3d.quiver(origin[0],origin[1],origin[2], ex[0],ex[1],ex[2], color='r', length=0.08)
    ax3d.quiver(origin[0],origin[1],origin[2], ey[0],ey[1],ey[2], color='g', length=0.08)
    ax3d.quiver(origin[0],origin[1],origin[2], ez[0],ez[1],ez[2], color='b', length=0.08)
    ax3d.set_xlim([-0.2,0.2]); ax3d.set_ylim([-0.2,0.2]); ax3d.set_zlim([-0.2,0.2])
    ax3d.set_title(f"Spacecraft Attitude @ t={t_arr[idx]:.2f}s")

    # Euler angles up to idx
    ax_eul.cla()
    ax_eul.plot(t_arr[:idx+1], np.degrees(roll[:idx+1]), label='roll (deg)')
    ax_eul.plot(t_arr[:idx+1], np.degrees(pitch[:idx+1]), label='pitch (deg)')
    ax_eul.plot(t_arr[:idx+1], np.degrees(yaw[:idx+1]), label='yaw (deg)')
    ax_eul.axhline(np.degrees(cmd_euler[0]), color='k', ls='--', alpha=0.6)
    ax_eul.axhline(np.degrees(cmd_euler[1]), color='k', ls='--', alpha=0.6)
    ax_eul.axhline(np.degrees(cmd_euler[2]), color='k', ls='--', alpha=0.6)
    ax_eul.set_xlabel('Time (s)'); ax_eul.set_ylabel('Angle (deg)')
    ax_eul.legend(loc='upper right'); ax_eul.grid(True)
    ax_eul.set_title("Euler angles")

    # wheel speeds and jitter norm
    ax_wheels.cla()
    for iw in range(4):
        ax_wheels.plot(t_arr[:idx+1], omega_rw_log[:idx+1, iw], label=f'wheel{iw+1}')
    ax_wheels.axhline(rw_speed_limit, color='r', ls=':')
    ax_wheels.axhline(-rw_speed_limit, color='r', ls=':')
    ax_wheels_twin = ax_wheels.twinx()
    ax_wheels_twin.plot(t_arr[:idx+1], jitter_norm[:idx+1], color='m', label='jitter norm')
    ax_wheels.set_xlabel('Time (s)'); ax_wheels.set_ylabel('wheel omega (rad/s)')
    ax_wheels_twin.set_ylabel('jitter torque norm (N*m)')
    l1, lab1 = ax_wheels.get_legend_handles_labels()
    l2, lab2 = ax_wheels_twin.get_legend_handles_labels()
    ax_wheels.legend(l1 + l2, lab1 + lab2, loc='upper right')
    ax_wheels.set_title("Wheel speeds & jitter")

    return cube,

ani = FuncAnimation(fig, animate, frames=n_frames, interval=40, blit=False)
plt.tight_layout()
plt.show()

# ---------------- Summary ----------------
print("\nSummary:")
print("Max |wheel speed| [RPM]: {:.1f}".format(np.max(np.abs(omega_rw_log))*60/(2*np.pi)))
print("RMS jitter torque (N*m): {:.3e}".format(np.sqrt(np.mean(np.sum(Tjitter_log**2,axis=1)))))
# final attitude error (approx)
q_final = quat_log[-1,:]
q_conj_final = quat_conj(q_final)
q_err_final = quat_mult(q_cmd, q_conj_final)
if q_err_final[0] < 0:
    q_err_final = -q_err_final
att_err_final = q_err_final[1:] * (2.0 / (np.linalg.norm(q_err_final)+1e-30))
print("Final attitude error magnitude (deg): {:.3f}".format(math.degrees(np.linalg.norm(att_err_final))))
