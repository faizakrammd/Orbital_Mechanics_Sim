"""
Reaction Wheel Saturation demo (Milestone 1)

- 1-axis yaw dynamics (spacecraft + reaction wheel)
- PD controller commanding wheel torque
- Finite wheel speed (RPM) limit -> torque is reduced when wheel would exceed speed limit
- Stochastic DSMC-inspired disturbance torque
- Offline simulation then animation (3D Cubesat + plots)

Usage: python rw_saturation_demo.py
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time

# ---------------- User-configurable parameters ----------------
# Environment / disturbance model
U_inf = 7600.0          # m/s (unused directly in simplified generator)
n_inf = 3.0e14
m_O = 2.656e-26

# Spacecraft geometry (for visualization only)
body_w = 0.10
body_h = 0.10
body_d = 0.10

# Dynamics / controller parameters
I_sc = 0.02         # spacecraft moment of inertia [kg*m^2]
I_rw = 1e-4         # reaction wheel inertia [kg*m^2]
max_rw_torque = 5e-5  # motor torque limit [N*m] (command limit)
rw_speed_rpm_limit = 6000.0  # wheel speed limit in RPM (±)
# convert to rad/s
rw_speed_limit = rw_speed_rpm_limit * 2.0 * math.pi / 60.0

Kp = 0.12           # PD gains
Kd = 0.018

# Simulation time settings
t_final = 60.0      # seconds
dt = 0.02           # integrator timestep (s)
steps = int(t_final / dt)

# Disturbance generator parameters (simple, stochastic)
DSMC_Np = 2000
dsmc_noise_scale = 0.3

# Yaw command initial / initial state
yaw_cmd_deg = 30.0
yaw_cmd = math.radians(yaw_cmd_deg)
state0 = (math.radians(5.0), 0.0, 0.0)  # theta (rad), omega_sc (rad/s), omega_rw (rad/s)

# Visualization speed-up
frame_skip = 100    # show every Nth simulated step (so animation appears faster)

# ---------------- Simple DSMC-inspired disturbance torque generator ----------------
def generate_aero_torque_simple(Np=DSMC_Np, AoA_deg=10.0, alpha=0.85):
    """
    Very simple stochastic torque generator:
     - Generates an expected mean torque proportional to small projected area and random lever arm.
     - Adds noise to mimic discrete impacts.
    The sign convention: positive torque -> rotates spacecraft CCW (z positive).
    """
    AoA = math.radians(AoA_deg)
    # tiny projected area (m^2)
    A_proj = body_w * body_h * abs(math.cos(AoA))
    # approximate mass flux and mean drag force magnitude (extremely small at 400 km)
    rho = n_inf * m_O
    mass_flux = rho * U_inf
    expected_force = mass_flux * U_inf * A_proj  # rough N
    per_particle_force = expected_force / (Np + 1e-30)
    # sample lever arms across half-height
    y_samples = np.random.uniform(-body_h/2, body_h/2, size=Np)
    torques = per_particle_force * y_samples
    torque_mean = np.sum(torques) * (0.8 + 0.4 * (1.0 - alpha))
    noise = np.random.normal(scale=abs(torque_mean) * dsmc_noise_scale + 1e-12)
    return torque_mean + noise

# ---------------- Dynamics & control with wheel saturation check ----------------
def compute_control_torque(theta, omega_sc):
    """
    PD controller computes desired torque on spacecraft (which is the torque the RW should apply).
    The controller output is the spacecraft-side torque (i.e., positive -> applied to spacecraft).
    """
    err = yaw_cmd - theta
    derr = -omega_sc
    T_des = Kp * err + Kd * derr
    # saturate torque by motor capability
    T_des = max(min(T_des, max_rw_torque), -max_rw_torque)
    return T_des

def integrate_step(state, T_dist, dt):
    """
    state: (theta, omega_sc, omega_rw)
    T_dist: external disturbance torque acting on spacecraft (N*m)
    Returns new_state, T_applied (torque actually applied by wheel onto spacecraft)
    Reaction wheel torque acts equal & opposite on wheel.
    Handles wheel speed limit by reducing applied torque so wheel doesn't exceed limit.
    """
    theta, omega_sc, omega_rw = state
    # controller computes desired torque applied to spacecraft
    T_des = compute_control_torque(theta, omega_sc)  # spacecraft torque (from wheel actuation)
    # Reaction wheel experiences torque -T_des (on wheel).
    # Wheel angular acceleration if full T_des applied:
    alpha_rw = (-T_des) / I_rw
    omega_rw_new_unclipped = omega_rw + alpha_rw * dt

    # If wheel would exceed speed limit, compute allowed torque so it hits exactly the limit
    if omega_rw_new_unclipped > rw_speed_limit:
        # compute torque that would bring wheel to rw_speed_limit in dt:
        omega_allowed_delta = rw_speed_limit - omega_rw
        T_allowed = - (omega_allowed_delta * I_rw) / dt  # T_rw = -I_rw * domega/dt
        # only allow torque in same sign as T_des (works if trying to accelerate further)
        # The sign convention: positive T_des -> wheel torque on wheel is -T_des
        # If T_des is negative (wheel accelerates +), T_allowed should be negative, etc.
        # So clamp T_applied to be between T_allowed and T_des depending on sign
        if T_des > 0:
            T_applied = min(T_des, T_allowed)
        else:
            # if desired torque is negative but wheel new exceeds +limit (rare), keep T_des
            T_applied = T_des
    elif omega_rw_new_unclipped < -rw_speed_limit:
        omega_allowed_delta = -rw_speed_limit - omega_rw
        T_allowed = - (omega_allowed_delta * I_rw) / dt
        if T_des < 0:
            T_applied = max(T_des, T_allowed)
        else:
            T_applied = T_des
    else:
        T_applied = T_des

    # Now integrate spacecraft and wheel with applied torque
    # Spacecraft: I_sc * domega_sc = T_dist + T_applied
    alpha_sc = (T_dist + T_applied) / I_sc
    omega_sc_new = omega_sc + alpha_sc * dt
    theta_new = theta + omega_sc_new * dt

    # Wheel: I_rw * domega_rw = -T_applied
    alpha_rw_actual = (-T_applied) / I_rw
    omega_rw_new = omega_rw + alpha_rw_actual * dt

    # Return new state and the applied torque (on spacecraft)
    return (theta_new, omega_sc_new, omega_rw_new), T_applied, T_des

# ---------------- Pre-run simulation (fast offline) ----------------
print("Simulating dynamics (offline)...")
t0 = time.time()
state = state0
history = {
    't': np.zeros(steps),
    'theta': np.zeros(steps),
    'omega_sc': np.zeros(steps),
    'omega_rw': np.zeros(steps),
    'T_dist': np.zeros(steps),
    'T_applied': np.zeros(steps),
    'T_des': np.zeros(steps),
    'yaw_err_deg': np.zeros(steps)
}

for i in range(steps):
    t = i * dt
    # optionally modulate AoA to make disturbance time-varying; small sine modulation:
    AoA = 10.0 * math.sin(0.01 * i)
    T_dist = generate_aero_torque_simple(AoA_deg=AoA)
    state, T_applied, T_des = integrate_step(state, T_dist, dt)

    history['t'][i] = t
    history['theta'][i] = state[0]
    history['omega_sc'][i] = state[1]
    history['omega_rw'][i] = state[2]
    history['T_dist'][i] = T_dist
    history['T_applied'][i] = T_applied
    history['T_des'][i] = T_des
    history['yaw_err_deg'][i] = math.degrees(yaw_cmd - state[0])

elapsed = time.time() - t0
print(f"Simulation finished in {elapsed:.2f} s (wall-clock). Steps: {steps}, dt={dt}")

# ---------------- Visualization / Animation ----------------
# 3D CubeSat box (yaw rotation only)
verts0 = np.array([
    [-body_w/2,-body_h/2,-body_d/2],
    [ body_w/2,-body_h/2,-body_d/2],
    [ body_w/2, body_h/2,-body_d/2],
    [-body_w/2, body_h/2,-body_d/2],
    [-body_w/2,-body_h/2, body_d/2],
    [ body_w/2,-body_h/2, body_d/2],
    [ body_w/2, body_h/2, body_d/2],
    [-body_w/2, body_h/2, body_d/2]
])
faces = [[0,1,2,3],[4,5,6,7],[0,1,5,4],
         [2,3,7,6],[1,2,6,5],[4,7,3,0]]

# figure layout
fig = plt.figure(figsize=(12,8))
ax3d = fig.add_subplot(221, projection='3d')
ax_tor = fig.add_subplot(222)
ax_rw = fig.add_subplot(212)

# data arrays for plotting
t_arr = history['t']
theta_arr = history['theta']
Tdist_arr = history['T_dist']
Tapp_arr = history['T_applied']
Tdes_arr = history['T_des']
omega_rw_arr = history['omega_rw']
yaw_err_arr = history['yaw_err_deg']

# set axis limits and labels
ax3d.set_xlim([-0.2,0.2]); ax3d.set_ylim([-0.2,0.2]); ax3d.set_zlim([-0.2,0.2])
ax3d.set_title("CubeSat attitude (yaw)")

ax_tor.set_title("Torques")
ax_tor.set_xlabel("Time (s)")
ax_tor.set_ylabel("Torque (N·m)")

ax_rw.set_title("Reaction wheel speed & yaw")
ax_rw.set_xlabel("Time (s)")

# animation function: draw every frame_skip step
n_frames = max(1, len(t_arr) // frame_skip)

def animate(frame_idx):
    idx = frame_idx * frame_skip
    if idx >= len(t_arr):
        idx = len(t_arr) - 1

    th = theta_arr[idx]
    # yaw rotation matrix
    R = np.array([[math.cos(th), -math.sin(th), 0],
                  [math.sin(th),  math.cos(th), 0],
                  [0, 0, 1]])
    verts = verts0.dot(R.T)

    # clear and draw 3D box and body axes
    ax3d.cla()
    cube = Poly3DCollection([[verts[j] for j in f] for f in faces],
                            alpha=0.5, facecolor='lightblue', edgecolor='k')
    ax3d.add_collection3d(cube)
    # body axes vectors (columns of R)
    ex, ey, ez = R[:,0], R[:,1], R[:,2]
    ax3d.quiver(0,0,0, ex[0], ex[1], ex[2], color='r', length=0.15)
    ax3d.quiver(0,0,0, ey[0], ey[1], ey[2], color='g', length=0.15)
    ax3d.quiver(0,0,0, ez[0], ez[1], ez[2], color='b', length=0.15)
    ax3d.set_xlim([-0.2,0.2]); ax3d.set_ylim([-0.2,0.2]); ax3d.set_zlim([-0.2,0.2])
    ax3d.set_title("CubeSat attitude (yaw)")

    # torques plot (disturbance + applied + desired)
    ax_tor.cla()
    ax_tor.plot(t_arr[:idx+1], Tdist_arr[:idx+1], label='T_dist')
    ax_tor.plot(t_arr[:idx+1], Tapp_arr[:idx+1], label='T_applied')
    ax_tor.plot(t_arr[:idx+1], Tdes_arr[:idx+1], label='T_des', alpha=0.5, ls='--')
    ax_tor.axhline(0, color='k', lw=0.3)
    ax_tor.set_xlabel("Time (s)")
    ax_tor.set_ylabel("Torque (N·m)")
    ax_tor.legend(loc='upper right')
    ax_tor.set_title("Disturbance vs RW applied torque")

    # reaction wheel speed & yaw error
    ax_rw.cla()
    ax_rw.plot(t_arr[:idx+1], omega_rw_arr[:idx+1], label='omega_rw (rad/s)')
    ax_rw.axhline(y=rw_speed_limit, color='r', ls=':', label='rw_speed_limit')
    ax_rw.axhline(y=-rw_speed_limit, color='r', ls=':')
    ax_rw_twin = ax_rw.twinx()
    ax_rw_twin.plot(t_arr[:idx+1], yaw_err_arr[:idx+1], color='m', label='yaw_err (deg)')
    ax_rw.set_xlabel("Time (s)")
    ax_rw.set_ylabel("wheel speed (rad/s)")
    ax_rw_twin.set_ylabel("yaw error (deg)")
    # legends combined
    lines1, labels1 = ax_rw.get_legend_handles_labels()
    lines2, labels2 = ax_rw_twin.get_legend_handles_labels()
    ax_rw.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax_rw.set_title("Reaction wheel speed & yaw error")

    return cube,

# create animation
ani = FuncAnimation(fig, animate, frames=n_frames, interval=40, blit=False)
plt.tight_layout()
plt.show()

# ---------------- Print summary ----------------
half = len(t_arr) // 2
mean_Tdist = np.mean(Tdist_arr[half:])
mean_Tapp = np.mean(Tapp_arr[half:])
max_rw_speed = np.max(np.abs(omega_rw_arr))
final_err_deg = yaw_err_arr[-1]

print("\nSummary (second half averages):")
print(f"mean disturbance torque = {mean_Tdist:.3e} N·m")
print(f"mean applied RW torque  = {mean_Tapp:.3e} N·m")
print(f"max |rw speed| observed = {max_rw_speed:.3f} rad/s   (= {max_rw_speed*60/(2*math.pi):.1f} RPM)")
print(f"final yaw error = {final_err_deg:.3f} deg")

# optional: save history to CSV
try:
    import csv
    with open('rw_saturation_history.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t','theta_rad','omega_sc','omega_rw','T_dist','T_applied','T_des','yaw_err_deg'])
        for i in range(len(t_arr)):
            w.writerow([t_arr[i], theta_arr[i], history['omega_sc'][i], omega_rw_arr[i],
                        Tdist_arr[i], Tapp_arr[i], Tdes_arr[i], yaw_err_arr[i]])
    print("History saved to rw_saturation_history.csv")
except Exception as e:
    print("Could not save CSV:", e)
