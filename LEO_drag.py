import math, random, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----------------------- Parameters -----------------------
# Physical / freestream
U_inf = 7600.0            # m/s (satellite velocity)
T_inf = 1000.0            # K
n_inf = 3.0e14            # m^-3 (number density at ~400 km)
m_O = 2.656e-26           # kg (atomic oxygen)
kB = 1.380649e-23

# Domain (meters)
Lx, Ly = 1.2, 0.6
depth = 0.10              # assumed out-of-plane depth

# Body (10x10 cm CubeSat face)
body_w, body_h = 0.10, 0.10
body_x, body_y = 0.5, 0.0

# Simulation params
Np = 3000                 # number of macro-particles
particle_weight = (n_inf * (Lx * Ly * depth)) / Np
dt = 2.0e-6
steps = 1500

# Derived
rho_inf = n_inf * m_O
q_inf = 0.5 * rho_inf * U_inf**2
area_ref = 0.01           # reference area = 10x10 cm

# Init particle state
np.random.seed(0)
particles = {}
particles['x'] = np.random.uniform(0.0, Lx*0.6, size=Np)
particles['y'] = np.random.uniform(-Ly/2, Ly/2, size=Np)

v_th = math.sqrt(2.0 * kB * T_inf / m_O)
particles['vx'] = np.random.normal(loc=U_inf, scale=v_th/math.sqrt(2.0), size=Np)
particles['vy'] = np.random.normal(loc=0.0, scale=v_th/math.sqrt(2.0), size=Np)

# Body bounds
bxmin, bxmax = body_x - body_w/2, body_x + body_w/2
bymin, bymax = body_y - body_h/2, body_y + body_h/2

def is_inside_body(x, y):
    return (x >= bxmin) & (x <= bxmax) & (y >= bymin) & (y <= bymax)

# Force storage
force_hist = []

# ---------------- Simulation Step ----------------
def step():
    global particles

    # Move
    particles['x'] += particles['vx'] * dt
    particles['y'] += particles['vy'] * dt

    # Inflow (left boundary) & reinitialize
    left = particles['x'] < 0.0
    if left.any():
        n = left.sum()
        particles['x'][left] = np.random.uniform(0.0, 0.02, size=n)
        particles['y'][left] = np.random.uniform(-Ly/2, Ly/2, size=n)
        particles['vx'][left] = np.random.normal(loc=U_inf, scale=v_th/math.sqrt(2.0), size=n)
        particles['vy'][left] = np.random.normal(loc=0.0, scale=v_th/math.sqrt(2.0), size=n)

    # Outflow (right boundary) -> recycle as inflow
    right = particles['x'] > Lx
    if right.any():
        n = right.sum()
        particles['x'][right] = np.random.uniform(0.0, 0.02, size=n)
        particles['y'][right] = np.random.uniform(-Ly/2, Ly/2, size=n)
        particles['vx'][right] = np.random.normal(loc=U_inf, scale=v_th/math.sqrt(2.0), size=n)
        particles['vy'][right] = np.random.normal(loc=0.0, scale=v_th/math.sqrt(2.0), size=n)

    # Periodic top/bottom
    particles['y'][particles['y'] > Ly/2] -= Ly
    particles['y'][particles['y'] < -Ly/2] += Ly

    # Collisions with body (diffuse reflection)
    inside = is_inside_body(particles['x'], particles['y'])
    Fx = 0.0
    if inside.any():
        idxs = np.where(inside)[0]
        for i in idxs:
            p_in = m_O * particle_weight * particles['vx'][i]

            # Previous position
            prev_x = particles['x'][i] - particles['vx'][i] * dt
            prev_y = particles['y'][i] - particles['vy'][i] * dt
            if prev_x < bxmin:   # front face
                nx, ny = -1.0, 0.0
                particles['x'][i] = bxmin - 1e-8
            elif prev_x > bxmax:
                nx, ny = 1.0, 0.0
                particles['x'][i] = bxmax + 1e-8
            elif prev_y < bymin:
                nx, ny = 0.0, -1.0
                particles['y'][i] = bymin - 1e-8
            else:
                nx, ny = 0.0, 1.0
                particles['y'][i] = bymax + 1e-8

            # Diffuse re-emission at 300K wall
            v_mag = abs(np.random.normal(0.0, math.sqrt(kB*300/m_O)))
            vt = np.random.normal(scale=0.5*v_mag)
            vx_new = v_mag*nx + vt*(-ny)
            vy_new = v_mag*ny + vt*(nx)

            p_out = m_O * particle_weight * vx_new
            Fx += (p_out - p_in)

            particles['vx'][i] = vx_new
            particles['vy'][i] = vy_new

    F_step = Fx / dt * depth
    force_hist.append(F_step)
    return F_step

# ---------------- Animation ----------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7,8))

# particle plot
scat = ax1.scatter([], [], s=1, c="blue")
rect = plt.Rectangle((bxmin, bymin), body_w, body_h, fc="red", alpha=0.4)
ax1.add_patch(rect)
ax1.set_xlim(0, Lx)
ax1.set_ylim(-Ly/2, Ly/2)
ax1.set_title("Particles around CubeSat face")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")

# force plot
line, = ax2.plot([], [], lw=1.2)
ax2.set_xlim(0, steps*dt)
ax2.set_ylim(-1e-5, 1e-5)
ax2.set_title("Drag force vs time")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Force (N)")

def init():
    scat.set_offsets(np.empty((0, 2)))   # ✅ fixed
    line.set_data([], [])
    return scat, line

def update(frame):
    F = step()
    # update scatter (subsample for speed)
    idx = np.random.choice(Np, size=800, replace=False)
    scat.set_offsets(np.c_[particles['x'][idx], particles['y'][idx]])
    # update force line
    t = np.arange(len(force_hist))*dt
    line.set_data(t, force_hist)
    return scat, line

ani = FuncAnimation(fig, update, frames=steps, init_func=init,
                    blit=False, interval=20, repeat=False)

plt.tight_layout()
plt.show()

# After animation, print mean Cd
if force_hist:
    F_mean = np.mean(force_hist[len(force_hist)//2:])
    Cd = F_mean / (q_inf*area_ref + 1e-30)
    print(f"Estimated mean drag force = {F_mean:.3e} N, Cd ≈ {Cd:.3f}")
