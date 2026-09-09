import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation

# -------- Simulation Parameters --------
dt = 0.05
t_final = 60
steps = int(t_final / dt)

# CubeSat parameters
I = np.diag([0.02, 0.025, 0.03])  # inertia matrix (kg*m^2)
k = 0.05  # control gain for momentum dumping

# Initial angular velocity (rad/s)
w = np.array([0.5, 0.3, 0.4])

# Quaternion attitude [q0, q1, q2, q3]
q = np.array([1, 0, 0, 0])

# Cube definition
cube_verts = np.array([[-0.5, -0.5, -0.5],
                       [ 0.5, -0.5, -0.5],
                       [ 0.5,  0.5, -0.5],
                       [-0.5,  0.5, -0.5],
                       [-0.5, -0.5,  0.5],
                       [ 0.5, -0.5,  0.5],
                       [ 0.5,  0.5,  0.5],
                       [-0.5,  0.5,  0.5]])

faces = [[0,1,2,3], [4,5,6,7],
         [0,1,5,4], [2,3,7,6],
         [1,2,6,5], [4,7,3,0]]

# -------- Utility Functions --------
def quat_mult(q, r):
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = r
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1
    ])

def quat_deriv(q, w):
    w_quat = np.array([0, *w])
    return 0.5 * quat_mult(q, w_quat)

def normalize_quat(q):
    return q / np.linalg.norm(q)

def quat_to_rotmat(q):
    q0, q1, q2, q3 = q
    return np.array([
        [1-2*(q2**2+q3**2),   2*(q1*q2-q0*q3),     2*(q1*q3+q0*q2)],
        [2*(q1*q2+q0*q3),     1-2*(q1**2+q3**2),   2*(q2*q3-q0*q1)],
        [2*(q1*q3-q0*q2),     2*(q2*q3+q0*q1),     1-2*(q1**2+q2**2)]
    ])

def rotated_faces(q, verts):
    R = quat_to_rotmat(q)
    v_rot = verts @ R.T
    return [[v_rot[j] for j in face] for face in faces]

# -------- Simulation Logs --------
q_log = []
w_log = []
t_log = []

for i in range(steps):
    # Control torque = -k * angular momentum
    h = I @ w
    torque = -k * h

    # Angular velocity dynamics
    w_dot = np.linalg.inv(I) @ (torque - np.cross(w, I @ w))
    w = w + w_dot * dt

    # Quaternion dynamics
    q_dot = quat_deriv(q, w)
    q = q + q_dot * dt
    q = normalize_quat(q)

    # Logging
    q_log.append(q.copy())
    w_log.append(np.linalg.norm(w))
    t_log.append(i*dt)

q_log = np.array(q_log)
w_log = np.array(w_log)
t_log = np.array(t_log)

# -------- Visualization --------
fig = plt.figure(figsize=(10,5))
ax_cube = fig.add_subplot(121, projection='3d')
ax_graph = fig.add_subplot(122)

# CubeSat view
ax_cube.set_xlim([-1, 1])
ax_cube.set_ylim([-1, 1])
ax_cube.set_zlim([-1, 1])
ax_cube.set_title("CubeSat Momentum Dumping")

# Angular velocity graph
ax_graph.set_xlim([0, t_final])
ax_graph.set_ylim([0, max(w_log)*1.1])
ax_graph.set_title("Angular Velocity Magnitude")
ax_graph.set_xlabel("Time [s]")
ax_graph.set_ylabel("|ω| [rad/s]")
(line,) = ax_graph.plot([], [], 'r-', lw=2)

# Initialize artists
cube_art = [ax_cube.add_collection3d(Poly3DCollection(rotated_faces(q_log[0], cube_verts), 
                                                      facecolors='cyan', linewidths=1, edgecolors='k', alpha=0.6))]
arrow = ax_cube.quiver(0,0,0, w[0], w[1], w[2], color='r', length=0.5)

def animate(i):
    global cube_art, arrow
    # Remove old
    for art in cube_art:
        art.remove()
    arrow.remove()

    # Draw new cube
    cube_art = [ax_cube.add_collection3d(Poly3DCollection(rotated_faces(q_log[i], cube_verts),
                                                          facecolors='cyan', linewidths=1, edgecolors='k', alpha=0.6))]
    # Angular velocity arrow
    arrow = ax_cube.quiver(0,0,0, q_log[i][1], q_log[i][2], q_log[i][3], color='r', length=0.7)

    # Update graph
    line.set_data(t_log[:i], w_log[:i])

    return cube_art + [arrow, line]

ani = FuncAnimation(fig, animate, frames=len(t_log), interval=50, blit=False)
plt.tight_layout()
plt.show()
