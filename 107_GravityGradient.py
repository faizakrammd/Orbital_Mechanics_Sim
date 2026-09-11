import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# === Simulation Parameters ===
mu = 398600.0          # Earth's gravitational parameter [km^3/s^2]
Re = 6378.0            # Earth radius [km]
a = Re + 500.0         # Orbit altitude 500 km
n = np.sqrt(mu / a**3) # Mean motion [rad/s]
dt = 0.5
t_final = 2000
steps = int(t_final/dt)

# Two CubeSat inertia matrices
I1 = np.diag([2.0, 1.0, 0.5])
I2 = np.diag([1.5, 1.0, 0.8])

# Initial angular velocity
omega0 = np.array([0.01, 0.02, 0.015])

# === Quaternion utilities ===
def quat_dot(q, w):
    wx, wy, wz = w
    return 0.5*np.array([
        -q[1]*wx - q[2]*wy - q[3]*wz,
         q[0]*wx + q[2]*wz - q[3]*wy,
         q[0]*wy - q[1]*wz + q[3]*wx,
         q[0]*wz + q[1]*wy - q[2]*wx])

def quat_to_dcm(q):
    q0,q1,q2,q3 = q
    return np.array([
        [1-2*(q2**2+q3**2),   2*(q1*q2-q0*q3),     2*(q1*q3+q0*q2)],
        [2*(q1*q2+q0*q3),     1-2*(q1**2+q3**2),   2*(q2*q3-q0*q1)],
        [2*(q1*q3-q0*q2),     2*(q2*q3+q0*q1),     1-2*(q1**2+q2**2)]
    ])

def normalize(q):
    return q/np.linalg.norm(q)

# === Dynamics ===
def gravity_gradient_torque(I, R_bi, r_vec):
    r_body = R_bi @ r_vec
    r_norm = np.linalg.norm(r_body)
    return (3*mu/(r_norm**5)) * np.cross(r_body, (I @ r_body))

def simulate(I, torque_scale=3e-3):
    q = np.array([1,0,0,0], dtype=float)
    w = omega0.copy()
    q_log, w_log = [], []
    for k in range(steps):
        theta = n*k*dt
        r_vec = np.array([a*np.cos(theta), a*np.sin(theta), 0.0])
        R_bi = quat_to_dcm(q).T
        T = gravity_gradient_torque(I, R_bi, r_vec)

        # Sinusoidal disturbance torque
        disturbance = torque_scale * np.array([
            np.sin(0.01*k*dt),
            np.cos(0.02*k*dt),
            np.sin(0.03*k*dt)
        ])
        T += disturbance

        wdot = np.linalg.inv(I) @ (T - np.cross(w, I@w))
        w = w + wdot*dt
        dq = quat_dot(q, w)*dt
        q = normalize(q + dq)

        q_log.append(q)
        w_log.append(w)
    return np.array(q_log), np.array(w_log)

q_log1, w_log1 = simulate(I1, torque_scale=3e-3)
q_log2, w_log2 = simulate(I2, torque_scale=5e-3)

# === Visualization ===
cube_verts = np.array([[-1,-1,-1],
                       [ 1,-1,-1],
                       [ 1, 1,-1],
                       [-1, 1,-1],
                       [-1,-1, 1],
                       [ 1,-1, 1],
                       [ 1, 1, 1],
                       [-1, 1, 1]])*0.5

faces_idx = [[0,1,2,3],
             [4,5,6,7],
             [0,1,5,4],
             [2,3,7,6],
             [1,2,6,5],
             [4,7,3,0]]

def rotated_faces(q, verts, offset=np.zeros(3)):
    R = quat_to_dcm(q)
    rv = verts @ R.T + offset
    return [[rv[j] for j in f] for f in faces_idx]

# === Layout: 3D on left, two plots stacked on right ===
fig = plt.figure(figsize=(14,6))
ax3d = fig.add_subplot(121, projection='3d')
axw1 = fig.add_subplot(222)
axw2 = fig.add_subplot(224)

# Angular velocity subplots
for ax, title in zip([axw1, axw2], ["CubeSat 1", "CubeSat 2"]):
    ax.set_title(f"Angular Velocity - {title}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("ω [rad/s]")
    ax.set_xlim(0, t_final)

lines1 = [axw1.plot([],[],label=lab)[0] for lab in ['ωx','ωy','ωz']]
lines2 = [axw2.plot([],[],label=lab)[0] for lab in ['ωx','ωy','ωz']]
axw1.legend(); axw2.legend()

# Two CubeSats in 3D
sat1 = Poly3DCollection(rotated_faces(q_log1[0], cube_verts, offset=[-1.2,0,0]), 
                        facecolors='cyan', edgecolors='k', alpha=0.6)
sat2 = Poly3DCollection(rotated_faces(q_log2[0], cube_verts, offset=[ 1.2,0,0]), 
                        facecolors='orange', edgecolors='k', alpha=0.6)
ax3d.add_collection3d(sat1)
ax3d.add_collection3d(sat2)
ax3d.set_xlim([-2,2]); ax3d.set_ylim([-2,2]); ax3d.set_zlim([-2,2])
ax3d.set_title("Two CubeSats Attitude")

def init():
    for ln in lines1+lines2: ln.set_data([],[])
    return lines1+lines2+[sat1,sat2]

def animate(i):
    idx = i*10
    if idx >= steps: idx = steps-1

    # Update CubeSat orientation
    sat1.set_verts(rotated_faces(q_log1[idx], cube_verts, offset=[-1.2,0,0]))
    sat2.set_verts(rotated_faces(q_log2[idx], cube_verts, offset=[ 1.2,0,0]))

    # Update ω plots
    tvals = np.arange(idx+1)*dt
    for j in range(3):
        lines1[j].set_data(tvals, w_log1[:idx+1,j])
        lines2[j].set_data(tvals, w_log2[:idx+1,j])

    axw1.set_ylim(np.min(w_log1[:idx+1])-0.01, np.max(w_log1[:idx+1])+0.01)
    axw2.set_ylim(np.min(w_log2[:idx+1])-0.01, np.max(w_log2[:idx+1])+0.01)

    return lines1+lines2+[sat1,sat2]

ani = FuncAnimation(fig, animate, frames=steps//10, init_func=init,
                    interval=50, blit=False)
plt.tight_layout()
plt.show()
