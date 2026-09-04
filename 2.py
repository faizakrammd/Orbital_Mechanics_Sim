import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# -----------------------
# Constants
# -----------------------
mu = 398600.4418  # Earth's gravitational parameter, km^3/s^2
Re = 6378.137     # Earth radius, km

# -----------------------
# Initial Orbit (500 km circular, 51.6° inclination)
# -----------------------
altitude = 500  # km
a = Re + altitude  # semi-major axis
e = 0.0           # eccentricity
i = np.radians(51.6)  # inclination
raan = 0.0
argp = 0.0
nu = 0.0          # true anomaly at epoch

# -----------------------
# Orbital mechanics helpers
# -----------------------
def kepler_to_cartesian(a, e, i, raan, argp, nu, mu):
    p = a * (1 - e**2)
    r_pqw = np.array([
        p * np.cos(nu) / (1 + e * np.cos(nu)),
        p * np.sin(nu) / (1 + e * np.cos(nu)),
        0.0
    ])
    v_pqw = np.array([
        -np.sqrt(mu/p) * np.sin(nu),
        np.sqrt(mu/p) * (e + np.cos(nu)),
        0.0
    ])
    # rotation matrix PQW → ECI
    R3_W = np.array([[np.cos(-raan), -np.sin(-raan), 0],
                     [np.sin(-raan), np.cos(-raan), 0],
                     [0, 0, 1]])
    R1_i = np.array([[1, 0, 0],
                     [0, np.cos(-i), -np.sin(-i)],
                     [0, np.sin(-i), np.cos(-i)]])
    R3_w = np.array([[np.cos(-argp), -np.sin(-argp), 0],
                     [np.sin(-argp), np.cos(-argp), 0],
                     [0, 0, 1]])
    Q_pqw_to_eci = R3_W @ R1_i @ R3_w
    r_eci = Q_pqw_to_eci @ r_pqw
    v_eci = Q_pqw_to_eci @ v_pqw
    return r_eci, v_eci

def two_body_rhs(t, state, mu):
    r = state[:3]
    v = state[3:]
    r_norm = np.linalg.norm(r)
    a = -mu * r / r_norm**3
    return np.hstack((v, a))

def rk4_step(fun, t, y, dt, mu):
    k1 = fun(t, y, mu)
    k2 = fun(t + dt/2, y + dt/2*k1, mu)
    k3 = fun(t + dt/2, y + dt/2*k2, mu)
    k4 = fun(t + dt, y + dt*k3, mu)
    return y + dt/6*(k1 + 2*k2 + 2*k3 + k4)

# -----------------------
# Propagate orbit
# -----------------------
r0, v0 = kepler_to_cartesian(a, e, i, raan, argp, nu, mu)
state0 = np.hstack((r0, v0))

T = 2 * np.pi * np.sqrt(a**3 / mu)  # orbital period
num_orbits = 2
dt = 10.0
steps = int(T*num_orbits/dt)

rs = np.zeros((steps,3))
ts = np.zeros(steps)
state = state0.copy()
t = 0.0
for k in range(steps):
    rs[k] = state[:3]
    ts[k] = t
    state = rk4_step(two_body_rhs, t, state, dt, mu)
    t += dt

# -----------------------
# Ground track conversion
# -----------------------
omega_earth = 7.2921150e-5  # rad/s
lats = np.zeros(steps)
lons = np.zeros(steps)
for k in range(steps):
    theta = omega_earth * ts[k]
    R3 = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1]])
    r_ecef = R3 @ rs[k]
    x, y, z = r_ecef
    lats[k] = np.degrees(np.arcsin(z/np.linalg.norm(r_ecef)))
    lons[k] = np.degrees(np.arctan2(y, x))
    if lons[k] > 180: lons[k] -= 360
    if lons[k] < -180: lons[k] += 360

# -----------------------
# Plot + Animate
# -----------------------
fig = plt.figure(figsize=(12,6))

# Left: 3D orbit
ax1 = fig.add_subplot(121, projection="3d")
ax1.set_title("Satellite Orbit in ECI Frame")
ax1.set_xlim([-8000,8000]); ax1.set_ylim([-8000,8000]); ax1.set_zlim([-8000,8000])
# Earth wireframe
u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
x = Re*np.cos(u)*np.sin(v)
y = Re*np.sin(u)*np.sin(v)
z = Re*np.cos(v)
ax1.plot_wireframe(x,y,z,color="b", alpha=0.3)
# orbit line + satellite marker
orbit_line, = ax1.plot([], [], [], "r", lw=1)
satellite, = ax1.plot([], [], [], "ro", markersize=5)

def update_3d(frame):
    orbit_line.set_data(rs[:frame,0], rs[:frame,1])
    orbit_line.set_3d_properties(rs[:frame,2])
    satellite.set_data([rs[frame,0]], [rs[frame,1]])   # fixed
    satellite.set_3d_properties([rs[frame,2]])         # fixed
    return orbit_line, satellite

ani1 = FuncAnimation(fig, update_3d, frames=steps, interval=30, blit=True)

# Right: Ground track
ax2 = fig.add_subplot(122)
ax2.set_title("Ground Track")
ax2.set_xlim([-180,180]); ax2.set_ylim([-90,90])
ax2.set_xlabel("Longitude [deg]"); ax2.set_ylabel("Latitude [deg]")
ground_line, = ax2.plot([], [], "r", lw=1)
ground_point, = ax2.plot([], [], "ro", markersize=3)

def update_ground(frame):
    ground_line.set_data(lons[:frame], lats[:frame])
    ground_point.set_data([lons[frame]], [lats[frame]])
    return ground_line, ground_point

ani2 = FuncAnimation(fig, update_ground, frames=steps, interval=30, blit=True)

plt.tight_layout()
plt.show()
