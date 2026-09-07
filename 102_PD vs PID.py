"""
PD vs PID (1-axis yaw) — Reaction Wheel + saturation demo
Updated so PD curve is clearly distinct.
"""
import numpy as np
import math
import matplotlib.pyplot as plt

# ---------------- User parameters ----------------
t_final = 200.0         # seconds
dt = 0.02               # timestep
steps = int(t_final / dt)

# Spacecraft and actuator
I_sc = 0.02             # spacecraft moment of inertia (kg*m^2)
I_rw = 1e-4             # reaction wheel inertia (kg*m^2)
max_rw_torque = 5e-5    # motor torque limit (N*m)
rw_speed_rpm_limit = 6000.0
rw_speed_limit = rw_speed_rpm_limit * 2.0 * math.pi / 60.0  # rad/s

# Control gains
Kp = 0.12
Kd = 0.018
Ki = 0.002          # integral gain (small) — adjust to observe windup

# Command & initial
yaw_cmd = math.radians(0.0)   # command angle (rad)
theta0 = math.radians(5.0)    # initial attitude error (rad)
omega_sc0 = 0.0
omega_rw0 = 0.0

# Disturbance: constant bias + small white noise
bias_torque = 2e-5    # ↑ bigger so PD shows steady-state error
noise_std = 2e-7      # stochastic noise amplitude

# -------------------------------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def controller_PD(theta, omega_sc):
    err = yaw_cmd - theta
    derr = -omega_sc
    T_des = Kp * err + Kd * derr
    T_applied = clamp(T_des, -max_rw_torque, max_rw_torque)
    return T_applied, {'T_des': T_des}

def controller_PID_naive(theta, omega_sc, integ_state):
    err = yaw_cmd - theta
    derr = -omega_sc
    integ_state_new = integ_state + err * dt
    T_des = Kp * err + Ki * integ_state_new + Kd * derr
    T_applied = clamp(T_des, -max_rw_torque, max_rw_torque)
    return T_applied, integ_state_new, {'T_des': T_des}

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
    return T_applied, integ_state_new, {'T_des': T_des}

# -------------------------------------------------
def simulate_controller(controller_name):
    theta = theta0
    omega_sc = omega_sc0
    omega_rw = omega_rw0
    integ_state = 0.0

    t_log, theta_log, omega_sc_log = np.zeros(steps), np.zeros(steps), np.zeros(steps)
    omega_rw_log, Tdist_log, Tctrl_log, Tdes_log, integ_log = \
        np.zeros(steps), np.zeros(steps), np.zeros(steps), np.zeros(steps), np.zeros(steps)

    for k in range(steps):
        t = k * dt
        T_dist = bias_torque + np.random.randn() * noise_std

        if controller_name == 'PD':
            T_applied, info = controller_PD(theta, omega_sc)
            T_des = info['T_des']
        elif controller_name == 'PID_naive':
            T_applied, integ_state, info = controller_PID_naive(theta, omega_sc, integ_state)
            T_des = info['T_des']
        elif controller_name == 'PID_clamped':
            T_applied, integ_state, info = controller_PID_clamped(theta, omega_sc, integ_state)
            T_des = info['T_des']
        else:
            raise ValueError("Unknown controller")

        # Dynamics
        alpha_rw = (-T_applied) / I_rw
        omega_rw_new = omega_rw + alpha_rw * dt
        omega_rw_new = clamp(omega_rw_new, -rw_speed_limit, rw_speed_limit)

        alpha_sc = (T_dist + T_applied) / I_sc
        omega_sc_new = omega_sc + alpha_sc * dt
        theta_new = theta + omega_sc_new * dt

        # Log
        t_log[k] = t
        theta_log[k] = theta_new
        omega_sc_log[k] = omega_sc_new
        omega_rw_log[k] = omega_rw_new
        Tdist_log[k] = T_dist
        Tctrl_log[k] = T_applied
        Tdes_log[k] = T_des
        integ_log[k] = integ_state

        theta, omega_sc, omega_rw = theta_new, omega_sc_new, omega_rw_new

    return {
        't': t_log, 'theta': theta_log, 'omega_sc': omega_sc_log,
        'omega_rw': omega_rw_log, 'Tdist': Tdist_log, 'Tctrl': Tctrl_log,
        'Tdes': Tdes_log, 'integ': integ_log
    }

# ---------------- Run ----------------
res_PD = simulate_controller('PD')
res_PID_naive = simulate_controller('PID_naive')
res_PID_clamped = simulate_controller('PID_clamped')

# ---------------- Plots ----------------
plt.rcParams.update({'font.size': 12})
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

# Attitude
axs[0].plot(res_PD['t'], np.degrees(res_PD['theta']), label='PD', color='C0', ls='--')
axs[0].plot(res_PID_naive['t'], np.degrees(res_PID_naive['theta']), label='PID naive', color='C1')
axs[0].plot(res_PID_clamped['t'], np.degrees(res_PID_clamped['theta']), label='PID clamped', color='C2')
axs[0].axhline(math.degrees(yaw_cmd), color='k', ls='--', label='cmd')
axs[0].set_ylabel('theta (deg)')
axs[0].set_title('Attitude response')
axs[0].legend(); axs[0].grid(True)

# Torque
axs[1].plot(res_PD['t'], res_PD['Tctrl'], label='PD', color='C0', ls='--')
axs[1].plot(res_PID_naive['t'], res_PID_naive['Tctrl'], label='PID naive', color='C1', alpha=0.8)
axs[1].plot(res_PID_clamped['t'], res_PID_clamped['Tctrl'], label='PID clamped', color='C2', alpha=0.8)
axs[1].set_ylabel('T_ctrl (N·m)')
axs[1].set_title('Applied control torque (after saturation)')
axs[1].legend(); axs[1].grid(True)

# Wheel speed
axs[2].plot(res_PD['t'], res_PD['omega_rw']*60/(2*np.pi), label='PD', color='C0', ls='--')
axs[2].plot(res_PID_naive['t'], res_PID_naive['omega_rw']*60/(2*np.pi), label='PID naive', color='C1')
axs[2].plot(res_PID_clamped['t'], res_PID_clamped['omega_rw']*60/(2*np.pi), label='PID clamped', color='C2')
axs[2].axhline(rw_speed_rpm_limit, color='r', ls=':', label='rw limit')
axs[2].axhline(-rw_speed_rpm_limit, color='r', ls=':')
axs[2].set_ylabel('wheel speed (RPM)')
axs[2].set_title('Reaction wheel speed')
axs[2].legend(); axs[2].grid(True)

# Integrator
axs[3].plot(res_PID_naive['t'], res_PID_naive['integ'], label='PID naive integrator', color='C1')
axs[3].plot(res_PID_clamped['t'], res_PID_clamped['integ'], label='PID clamped integrator', color='C2')
axs[3].set_ylabel('integrator state')
axs[3].set_xlabel('time (s)')
axs[3].set_title('Integral state (wind-up in naive PID)')
axs[3].legend(); axs[3].grid(True)

plt.tight_layout(); plt.show()
