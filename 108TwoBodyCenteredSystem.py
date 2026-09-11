# tethered_pair_sim.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

# ---------------------------
# Utilities (quaternions etc)
# ---------------------------
def q_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n == 0:
        return np.array([1.0,0,0,0], dtype=float)
    a = axis / n
    s = np.sin(0.5*angle)
    return np.array([np.cos(0.5*angle), *(a*s)])

def q_to_R(q):
    w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1-2*(x*x+z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1-2*(x*x+y*y)]
    ])

def qmul(q,p):
    w1,x1,y1,z1 = q
    w2,x2,y2,z2 = p
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def qconj(q):
    w,x,y,z = q
    return np.array([w,-x,-y,-z])

def quat_integrate(q, omega, dt):
    # omega: body angular rate vector
    # qdot = 0.5 * Omega(q) * q
    wx,wy,wz = omega
    Omega = np.array([
        [0.0, -wx, -wy, -wz],
        [wx,  0.0,  wz, -wy],
        [wy, -wz,  0.0,  wx],
        [wz,  wy, -wx,  0.0]
    ])
    qdot = 0.5 * Omega @ q
    q_new = q + qdot * dt
    q_new /= np.linalg.norm(q_new)
    # keep scalar positive
    if q_new[0] < 0: q_new = -q_new
    return q_new

# ---------------------------
# Environment
# ---------------------------
mu_E = 3.986004418e14  # Earth's mu
R_E  = 6371e3

def r_eci_of_theta(theta, a):
    return np.array([a*np.cos(theta), a*np.sin(theta), 0.0])

def orbital_speed(a):
    return np.sqrt(mu_E / a)

# ---------------------------
# Model parameters
# ---------------------------
# masses
m1 = 3.0    # kg (primary)
m2 = 1.0    # kg (secondary)
# inertias (small rigid bodies)
I1 = np.diag([0.03, 0.03, 0.04])
I2 = np.diag([0.01, 0.01, 0.02])

# tether (rigid) length
L = 10.0  # meters

# tether attachment offsets in body frames (where tether force is applied)
# measured from each body's center-of-mass toward the tether
rho1_b = np.array([0.0, 0.0, 1.0]) * 0.0   # attach at CM for simplicity (no moment) -> set nonzero to couple rotation
rho2_b = np.array([0.0, 0.0, -1.0]) * 0.0  # currently zero offsets -> no tether torque
# if you want rotational coupling, set these to nonzero e.g. [0,0,0.5] etc.

# small structural damping in rotation
c_rot = 1e-4

# orbit
h = 500e3
a = R_E + h
n = np.sqrt(mu_E / a**3)
T_orbit = 2*np.pi / n

# ---------------------------
# Initial conditions
# ---------------------------
# initial central position (theta = 0)
theta0 = 0.0
r0 = r_eci_of_theta(theta0, a)
# orbital velocity direction (tangential)
v_mag = orbital_speed(a)
v_tan = np.array([-np.sin(theta0), np.cos(theta0), 0.0]) * v_mag

# place the two masses along local vertical (nadir) separated by L
nadir = -r0 / np.linalg.norm(r0)
r1_0 = r0 + 0.5*L*nadir
r2_0 = r0 - 0.5*L*nadir
v1_0 = v_tan.copy()
v2_0 = v_tan.copy()

# small relative initial perturbation (spin about local axis)
v1_0 += np.array([0.0, 0.0, 0.0])
v2_0 += np.array([0.0, 0.0, 0.0])

# attitudes (identity) and small angular rates
q1_0 = q_from_axis_angle([1,0,0], np.deg2rad(2.0))  # small tilt
q2_0 = q_from_axis_angle([1,0,0], np.deg2rad(-3.0))
w1_0 = np.deg2rad([0.1,0.0,0.0])
w2_0 = np.deg2rad([-0.05,0.0,0.0])

# ---------------------------
# Dynamics helper functions
# ---------------------------
def gravity_accel(r):
    norm = np.linalg.norm(r)
    return -mu_E * r / norm**3

def gravity_gradient_torque(Ib, q, r_i):
    # body->inertial = R
    R = q_to_R(q)
    Rib = R.T
    r_b = Rib @ r_i
    rnorm = np.linalg.norm(r_b)
    if rnorm == 0:
        return np.zeros(3)
    s = r_b / rnorm
    tau = 3.0 * mu_E / (rnorm**3) * np.cross(s, Ib @ s)
    return tau

# ---------------------------
# State vector:
# y = [r1(3), v1(3), r2(3), v2(3), q1(4), w1(3), q2(4), w2(3)]
# ---------------------------
def pack_state(r1,v1,r2,v2,q1,w1,q2,w2):
    return np.hstack([r1,v1,r2,v2,q1,w1,q2,w2])

def unpack_state(y):
    r1 = y[0:3]; v1 = y[3:6]; r2 = y[6:9]; v2 = y[9:12]
    q1 = y[12:16]; w1 = y[16:19]; q2 = y[19:23]; w2 = y[23:26]
    return r1,v1,r2,v2,q1,w1,q2,w2

# ---------------------------
# Constraint: ||r2 - r1|| = L  (rigid massless tether)
# Solve tension lambda algebraically each evaluation
# ---------------------------
def compute_tension_and_accels(r1,v1,r2,v2, F1_no_tension, F2_no_tension):
    # r_rel = r2 - r1
    r_rel = r2 - r1
    v_rel = v2 - v1
    dist = np.linalg.norm(r_rel)
    if dist == 0:
        u = np.array([1.0,0,0])
        dist = 1.0
    else:
        u = r_rel / dist
    # constraint 2nd derivative: r_rel · (a2 - a1) + |v_rel|^2 = 0
    # a1 = F1/m1 + lambda * u / m1
    # a2 = F2/m2 - lambda * u / m2
    # a_rel = (F2/m2 - F1/m1) - lambda * u * (1/m2 + 1/m1)
    denom = (1.0/m1 + 1.0/m2) * dist
    if np.abs(denom) < 1e-12:
        lam = 0.0
    else:
        lam = (np.dot(r_rel, (F2_no_tension / m2 - F1_no_tension / m1)) + np.dot(v_rel, v_rel)) / denom
    # tension force on mass1: +lambda * u (on mass1 toward mass2)
    F_t1 = lam * u
    F_t2 = -F_t1
    a1 = F1_no_tension / m1 + F_t1 / m1
    a2 = F2_no_tension / m2 + F_t2 / m2
    return lam, a1, a2, u

# ---------------------------
# ODE right-hand side
# ---------------------------
def dydt(t, y):
    r1,v1,r2,v2,q1,w1,q2,w2 = unpack_state(y)

    # gravitational accelerations (central)
    g1 = gravity_accel(r1)
    g2 = gravity_accel(r2)

    # additional environmental forces could be added (drag, SRP, etc.)
    F1_no_tension = m1 * g1
    F2_no_tension = m2 * g2

    # compute tension and translational accelerations
    lam, a1, a2, u = compute_tension_and_accels(r1,v1,r2,v2, F1_no_tension, F2_no_tension)

    # rotational dynamics: torques due to tether (applied at attachment point) + gravity-gradient + small rotational damping
    R1 = q_to_R(q1)
    R2 = q_to_R(q2)
    # tether force in inertial applied at attachment points (body->inertial)
    # if rho_b is zero, no rotational coupling from tether
    r_attach1_i = r1 + R1 @ rho1_b
    r_attach2_i = r2 + R2 @ rho2_b
    # tether force on body1 (in inertial) is +F_t1 (we computed as +lam*u)
    F_t1 = lam * u
    F_t2 = -F_t1

    # convert tether forces to body frame to compute torque: tau = rho_b x F_body
    F_t1_b = R1.T @ F_t1
    F_t2_b = R2.T @ F_t2
    tau_t1 = np.cross(rho1_b, F_t1_b)
    tau_t2 = np.cross(rho2_b, F_t2_b)

    # gravity-gradient torques (in body frame)
    tau_gg1 = gravity_gradient_torque(I1, q1, r1)
    tau_gg2 = gravity_gradient_torque(I2, q2, r2)

    # total torques in body frames
    tau1_body = tau_t1 + tau_gg1 - c_rot * w1
    tau2_body = tau_t2 + tau_gg2 - c_rot * w2

    # rotational equations: I wdot + w x (I w) = tau
    wdot1 = np.linalg.inv(I1) @ (tau1_body - np.cross(w1, I1 @ w1))
    wdot2 = np.linalg.inv(I2) @ (tau2_body - np.cross(w2, I2 @ w2))

    # quaternion kinematics
    # qdot = 0.5 * Omega(w) * q
    def qdot_from_w(q, w):
        w0 = np.array([0.0, *w])
        return qmul(q, w0) * 0.5

    q1dot = qdot_from_w(q1, w1)
    q2dot = qdot_from_w(q2, w2)

    # pack derivatives
    dydt = np.hstack([v1, a1, v2, a2, q1dot, wdot1, q2dot, wdot2])
    return dydt, lam

# ---------------------------
# Integrator (simple RK4 stepping with tension evaluation inside)
# We'll integrate using small dt RK4; compute lam at each rhs evaluation.
# ---------------------------
def rk4_step(func, t, y, dt):
    # func returns (dydt, lam)
    k1, lam1 = func(t, y)
    k2, lam2 = func(t + 0.5*dt, y + 0.5*dt*k1)
    k3, lam3 = func(t + 0.5*dt, y + 0.5*dt*k2)
    k4, lam4 = func(t + dt, y + dt*k3)
    dy = (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return y + dy, (lam1, lam2, lam3, lam4)

# ---------------------------
# Simulation loop
# ---------------------------
# initial state
y0 = pack_state(r1_0, v1_0, r2_0, v2_0, q1_0, w1_0, q2_0, w2_0)

t_final = 0.5 * T_orbit   # simulate half an orbit initially
dt = 0.5                  # seconds (small enough for tether dynamics)
N = int(np.ceil(t_final / dt)) + 1

t_hist = np.zeros(N)
y_hist = np.zeros((N, len(y0)))
lam_hist = np.zeros(N)

t = 0.0
y = y0.copy()
idx = 0
t_hist[idx] = t
y_hist[idx,:] = y
_, lam0 = dydt(t, y)
lam_hist[idx] = lam0

for k in range(1, N):
    y, lam_tuple = rk4_step(dydt, t, y, dt)
    # renormalize quaternions after step & keep scalar positive
    # q1 at indices 12:16, q2 at 19:23
    q1 = y[12:16]; q2 = y[19:23]
    q1 /= np.linalg.norm(q1); q2 /= np.linalg.norm(q2)
    if q1[0] < 0: q1 = -q1
    if q2[0] < 0: q2 = -q2
    y[12:16] = q1; y[19:23] = q2

    # store
    t += dt
    t_hist[k] = t
    y_hist[k,:] = y
    # pick the midpoint lambda as representative
    lam_hist[k] = lam_tuple[1] if isinstance(lam_tuple, tuple) else lam_tuple

# ---------------------------
# Post-processing & plots
# ---------------------------
r1s = y_hist[:,0:3]
r2s = y_hist[:,6:9]

# tether relative distance / angle
rel_vecs = r2s - r1s
dists = np.linalg.norm(rel_vecs, axis=1)
# angle between tether vector and local nadir direction
angles_deg = np.zeros_like(dists)
for i in range(len(dists)):
    ri = 0.5*(r1s[i] + r2s[i])
    nadir = -ri / np.linalg.norm(ri)
    u = rel_vecs[i] / (np.linalg.norm(rel_vecs[i]) + 1e-12)
    angles_deg[i] = np.degrees(np.arccos(np.clip(np.dot(u, nadir), -1, 1)))

# simple 3D animation of the pair (positions + body axes for body1)
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-15,15); ax.set_ylim(-15,15); ax.set_zlim(-15,15)
ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
ax.set_title('Tethered pair in LEO (in meters, local frame centered near initial orbit point)')

p1, = ax.plot([],[],[],'o-', lw=2, label='Mass1 (m1)')
p2, = ax.plot([],[],[],'o-', lw=2, label='Mass2 (m2)')
line_tether, = ax.plot([],[],[],'--', lw=1, label='Tether')
axis_x, = ax.plot([],[],[],'r-')
axis_y, = ax.plot([],[],[],'g-')
axis_z, = ax.plot([],[],[],'b-')
ax.legend()

# for clarity, work in a local translate so motion near origin: subtract initial mean position
mean0 = 0.5*(r1s[0] + r2s[0])
def init_anim():
    p1.set_data([],[]); p1.set_3d_properties([])
    p2.set_data([],[]); p2.set_3d_properties([])
    line_tether.set_data([],[]); line_tether.set_3d_properties([])
    axis_x.set_data([],[]); axis_x.set_3d_properties([])
    axis_y.set_data([],[]); axis_y.set_3d_properties([])
    axis_z.set_data([],[]); axis_z.set_3d_properties([])
    return p1,p2,line_tether,axis_x,axis_y,axis_z

def animate(i):
    r1 = r1s[i] - mean0
    r2 = r2s[i] - mean0
    p1.set_data([r1[0]],[r1[1]]); p1.set_3d_properties([r1[2]])
    p2.set_data([r2[0]],[r2[1]]); p2.set_3d_properties([r2[2]])
    line_tether.set_data([r1[0], r2[0]],[r1[1], r2[1]]); line_tether.set_3d_properties([r1[2], r2[2]])
    # body1 axes (approx using q1 stored in history)
    q1 = y_hist[i,12:16]; R1 = q_to_R(q1)
    origin = r1
    L_axes = 2.0
    x_end = origin + (R1 @ np.array([L_axes,0,0]))
    y_end = origin + (R1 @ np.array([0,L_axes,0]))
    z_end = origin + (R1 @ np.array([0,0,L_axes]))
    axis_x.set_data([origin[0], x_end[0]], [origin[1], x_end[1]]); axis_x.set_3d_properties([origin[2], x_end[2]])
    axis_y.set_data([origin[0], y_end[0]], [origin[1], y_end[1]]); axis_y.set_3d_properties([origin[2], y_end[2]])
    axis_z.set_data([origin[0], z_end[0]], [origin[1], z_end[1]]); axis_z.set_3d_properties([origin[2], z_end[2]])
    return p1,p2,line_tether,axis_x,axis_y,axis_z

ani = animation.FuncAnimation(fig, animate, frames=len(t_hist), init_func=init_anim, interval=30, blit=False)
plt.show()

# Plots: tether tension, distance error, and angle to nadir
fig2, axarr = plt.subplots(3,1, figsize=(8,9))
axarr[0].plot(t_hist/60.0, lam_hist); axarr[0].set_ylabel('Tension λ [N]'); axarr[0].grid(True)
axarr[1].plot(t_hist/60.0, dists - L); axarr[1].set_ylabel('Distance error [m]'); axarr[1].grid(True)
axarr[2].plot(t_hist/60.0, angles_deg); axarr[2].set_ylabel('Tether angle to nadir [deg]'); axarr[2].set_xlabel('Time [min]'); axarr[2].grid(True)
plt.tight_layout()
plt.show()
