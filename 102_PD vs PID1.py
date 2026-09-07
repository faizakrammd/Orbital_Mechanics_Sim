import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------------- Parameters ----------------
t_final = 120.0
dt = 0.05
steps = int(t_final / dt)

I_sc = 0.02
I_rw = 1e-4
max_rw_torque = 5e-5
rw_speed_rpm_limit = 6000.0
rw_speed_limit = rw_speed_rpm_limit * 2.0 * math.pi / 60.0

Kp = 0.12
Kd = 0.05
Ki = 0.002

yaw_cmd = 0.0
theta0 = math.radians(5.0)
omega_sc0 = 0.0
omega_rw0 = 0.0

bias_torque = 2e-5
noise_std = 2e-7

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# ---------------- Controllers ----------------
def controller_PD(theta, omega_sc):
    err = yaw_cmd - theta
    derr = -omega_sc
    T_des = Kp * err + Kd * derr
    T_applied = clamp(T_des, -max_rw_torque, max_rw_torque)
    return T_applied

def controller_PID_naive(theta, omega_sc, integ_state):
    err = yaw_cmd - theta
    derr = -omega_sc
    integ_state_new = integ_state + err * dt
    T_des = Kp * err + Ki * integ_state_new + Kd * derr
    T_applied = clamp(T_des, -max_rw_torque, max_rw_torque)
    return T_applied, integ_state_new

def controller_PID_clamped(theta, omega_sc, integ_state):
    err = yaw_cmd - theta
    derr = -omega_sc
    T_des_proposed = Kp * err + Ki * integ_state + Kd * derr
    integ_candidate = integ_state + err * dt
    T_des_with_candidate = Kp * err + Ki * integ_candidate + Kd * derr

    if abs(T_des_with_candidate) <= max_rw_torque:
        integ_state_new = integ_candidate
        T_des = T_des_with_candidate
    else:
        integ_state_new = integ_state
        T_des = T_des_proposed

    T_applied = clamp(T_des, -max_rw_torque, max_rw_torque)
    return T_applied, integ_state_new

# ---------------- Simulator ----------------
def simulate_controller(controller_name):
    theta, omega_sc, omega_rw = theta0, omega_sc0, omega_rw0
    integ_state = 0.0

    theta_log, Tctrl_log, omega_rw_log = [], [], []

    for _ in range(steps):
        T_dist = bias_torque + np.random.randn() * noise_std

        if controller_name == 'PD':
            T_applied = controller_PD(theta, omega_sc)
        elif controller_name == 'PID_naive':
            T_applied, integ_state = controller_PID_naive(theta, omega_sc, integ_state)
        elif controller_name == 'PID_clamped':
            T_applied, integ_state = controller_PID_clamped(theta, omega_sc, integ_state)
        else:
            raise ValueError("Unknown controller")

        alpha_rw = (-T_applied) / I_rw
        omega_rw = clamp(omega_rw + alpha_rw * dt, -rw_speed_limit, rw_speed_limit)

        alpha_sc = (T_dist + T_applied) / I_sc
        omega_sc = omega_sc + alpha_sc * dt
        theta = theta + omega_sc * dt

        theta_log.append(theta)
        Tctrl_log.append(T_applied)
        omega_rw_log.append(omega_rw)

    return np.array(theta_log), np.array(Tctrl_log), np.array(omega_rw_log)

# ---------------- Run sims ----------------
res_PD = simulate_controller('PD')
res_PID_naive = simulate_controller('PID_naive')
res_PID_clamped = simulate_controller('PID_clamped')
time_arr = np.linspace(0, t_final, steps)

# ---------------- Animation ----------------
fig, ax = plt.subplots(figsize=(8,5))
ax.set_xlim(0, t_final)
ax.set_ylim(-10, 6)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Attitude (deg)")
ax.set_title("PD vs PID Attitude Control (1-axis)")
ax.grid(True)

line_PD, = ax.plot([], [], 'b-', lw=2, marker='o', markevery=40, label="PD")
line_PID_naive, = ax.plot([], [], 'orange', lw=1.5, label="PID naive")
line_PID_clamped, = ax.plot([], [], 'g--', lw=1.5, marker='*', markevery=50, label="PID clamped")
ax.legend()

text_PD = ax.text(0.02, 0.9, '', transform=ax.transAxes, color='b')
text_PID_naive = ax.text(0.02, 0.82, '', transform=ax.transAxes, color='orange')
text_PID_clamped = ax.text(0.02, 0.74, '', transform=ax.transAxes, color='g')

def init():
    line_PD.set_data([], [])
    line_PID_naive.set_data([], [])
    line_PID_clamped.set_data([], [])
    return line_PD, line_PID_naive, line_PID_clamped

# speed-up factor
stride = 10  

def animate(i):
    idx = i * stride
    if idx >= steps:
        idx = steps - 1
    t = time_arr[:idx]

    line_PD.set_data(t, np.degrees(res_PD[0][:idx]))
    line_PID_naive.set_data(t, np.degrees(res_PID_naive[0][:idx]))
    line_PID_clamped.set_data(t, np.degrees(res_PID_clamped[0][:idx]))

    text_PD.set_text(f"PD: Torque={res_PD[1][idx]:.2e} Nm, ω_rw={res_PD[2][idx]*60/(2*np.pi):.0f} RPM")
    text_PID_naive.set_text(f"PID naive: Torque={res_PID_naive[1][idx]:.2e} Nm, ω_rw={res_PID_naive[2][idx]*60/(2*np.pi):.0f} RPM")
    text_PID_clamped.set_text(f"PID clamped: Torque={res_PID_clamped[1][idx]:.2e} Nm, ω_rw={res_PID_clamped[2][idx]*60/(2*np.pi):.0f} RPM")

    return line_PD, line_PID_naive, line_PID_clamped, text_PD, text_PID_naive, text_PID_clamped

ani = FuncAnimation(fig, animate, frames=steps//stride, init_func=init, interval=30, blit=True)
plt.show()
