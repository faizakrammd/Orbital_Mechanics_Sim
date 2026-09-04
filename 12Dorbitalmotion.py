#!/usr/bin/env python3
"""
two_body_groundtrack.py

Two-body (Kepler) orbit propagation with RK4 and ground-track plotting.

Saves three figures:
 - orbit_eci.png        : 3D inertial (ECI) orbit around Earth wireframe
 - ground_track.png     : longitude vs latitude ground track
 - lon_vs_time.png      : longitude vs time (shows Earth's rotation effect)

Modify the initial orbital elements in the CONFIG section below.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
import os

# -------------------------- CONFIG / initial orbital elements ---------------------------
# Example: near-circular LEO
ALTITUDE = 500e3        # meters above mean Earth radius
ECC = 0.001             # eccentricity
INC_DEG = 51.6          # inclination in degrees
RAAN_DEG = 0.0          # right ascension of ascending node, degrees
ARGP_DEG = 0.0          # argument of perigee, degrees
NU_DEG = 0.0            # true anomaly at epoch, degrees

NUM_ORBITS = 2.0        # how many orbits to propagate (float)
DT = 10.0               # fixed RK4 timestep (seconds)

# Output directory for saved images
OUT_DIR = "output_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------- PHYSICAL CONSTANTS ----------------------------------------
MU_EARTH = 398600.4418e9      # Earth's gravitational parameter, m^3/s^2
R_EARTH = 6378136.3           # Earth's mean radius, m
OMEGA_EARTH = 7.2921150e-5    # Earth's rotation rate, rad/s

# -------------------------- HELPER / MATH FUNCTIONS -----------------------------------
def rot_x(angle_rad):
    c = np.cos(angle_rad); s = np.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c, -s],
                     [0.0, s,  c]])

def rot_z(angle_rad):
    c = np.cos(angle_rad); s = np.sin(angle_rad)
    return np.array([[ c, -s, 0.0],
                     [ s,  c, 0.0],
                     [0.0, 0.0, 1.0]])

def oe_to_rv(a, e, i_deg, raan_deg, argp_deg, nu_deg, mu=MU_EARTH):
    """
    Convert classical orbital elements to ECI position and velocity (meters, m/s).
    Inputs:
      a         : semi-major axis (m)
      e         : eccentricity
      i_deg     : inclination, degrees
      raan_deg  : RAAN, degrees
      argp_deg  : argument of perigee, degrees
      nu_deg    : true anomaly, degrees
    Returns:
      r_eci (3,), v_eci (3,)
    """
    i = np.deg2rad(i_deg)
    raan = np.deg2rad(raan_deg)
    argp = np.deg2rad(argp_deg)
    nu = np.deg2rad(nu_deg)

    p = a * (1.0 - e**2)
    r_pqw = (p / (1.0 + e * np.cos(nu))) * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_pqw = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    # Rotation from PQW (perifocal) to ECI
    Q_pX = rot_z(raan) @ rot_x(i) @ rot_z(argp)
    r_eci = Q_pX @ r_pqw
    v_eci = Q_pX @ v_pqw
    return r_eci, v_eci

def accel_two_body(r, mu=MU_EARTH):
    rnorm = np.linalg.norm(r)
    if rnorm == 0.0:
        return np.zeros(3)
    return -mu * r / (rnorm**3)

def rk4_step(r, v, dt):
    """
    Single RK4 step for second-order ODE (converted to first-order).
    state = [r, v]
    """
    def deriv(state):
        r_ = state[:3]
        v_ = state[3:]
        a_ = accel_two_body(r_)
        return np.concatenate((v_, a_))

    state = np.concatenate((r, v))
    k1 = deriv(state)
    k2 = deriv(state + 0.5 * dt * k1)
    k3 = deriv(state + 0.5 * dt * k2)
    k4 = deriv(state + dt * k3)
    state_next = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return state_next[:3], state_next[3:]

def eci_to_ecef(r_eci, t, omega=OMEGA_EARTH):
    """
    Convert ECI vector to ECEF by rotating about Z axis by theta = omega * t.
    NOTE: This uses a simplified convention assuming epoch GMST = 0 at t=0.
    For real precise conversions use proper Earth orientation parameters.
    """
    theta = omega * t
    return rot_z(-theta) @ r_eci

def latlon_from_ecef(r_ecef):
    x, y, z = r_ecef
    rnorm = np.linalg.norm(r_ecef)
    lat = np.arcsin(z / rnorm)
    lon = np.arctan2(y, x)
    return np.rad2deg(lat), np.rad2deg(lon)

# -------------------------- MAIN SIMULATION -------------------------------------------
def run_simulation():
    # initial orbital parameters
    a = R_EARTH + ALTITUDE
    e = ECC
    i_deg = INC_DEG
    raan_deg = RAAN_DEG
    argp_deg = ARGP_DEG
    nu_deg = NU_DEG

    # initial state in ECI
    r0, v0 = oe_to_rv(a, e, i_deg, raan_deg, argp_deg, nu_deg)

    # orbital period (Keplerian)
    period = 2.0 * np.pi * np.sqrt(a**3 / MU_EARTH)
    print(f"[INFO] Semi-major axis: {a/1000.0:.3f} km  |  Period: {period/60.0:.2f} min")

    t_final = NUM_ORBITS * period
    n_steps = int(np.ceil(t_final / DT)) + 1

    times = np.linspace(0.0, t_final, n_steps)
    rs = np.zeros((n_steps, 3))
    vs = np.zeros((n_steps, 3))
    rs[0] = r0
    vs[0] = v0

    # propagate using fixed-step RK4
    for k in range(1, n_steps):
        r_prev = rs[k-1]
        v_prev = vs[k-1]
        r_next, v_next = rk4_step(r_prev, v_prev, DT)
        rs[k] = r_next
        vs[k] = v_next

    # compute lat/lon ground track (convert ECI to ECEF at each time)
    lats = np.zeros(n_steps)
    lons = np.zeros(n_steps)
    for k in range(n_steps):
        r_ecef = eci_to_ecef(rs[k], times[k])
        lat, lon = latlon_from_ecef(r_ecef)
        # normalize lon to [-180, 180]
        lon = ((lon + 180.0) % 360.0) - 180.0
        lats[k] = lat
        lons[k] = lon

    # -------------------- Plot 1: 3D ECI orbit (with Earth wireframe) --------------------
    fig1 = plt.figure(figsize=(9, 6))
    ax = fig1.add_subplot(111, projection='3d')
    # Earth wireframe
    u = np.linspace(0, 2.0 * np.pi, 60)
    v = np.linspace(-np.pi/2.0, np.pi/2.0, 30)
    x_s = R_EARTH * np.outer(np.cos(u), np.cos(v))
    y_s = R_EARTH * np.outer(np.sin(u), np.cos(v))
    z_s = R_EARTH * np.outer(np.ones_like(u), np.sin(v))
    ax.plot_wireframe(x_s, y_s, z_s, linewidth=0.4, rstride=3, cstride=3)
    # Orbit
    ax.plot(rs[:, 0], rs[:, 1], rs[:, 2], linewidth=1.2)
    ax.scatter([rs[0, 0]], [rs[0, 1]], [rs[0, 2]], color='red', s=40, label='start')
    ax.set_title("3D Inertial Orbit (ECI) - Two-body propagation")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_box_aspect([1, 1, 1])
    max_r = np.max(np.linalg.norm(rs, axis=1))
    lim = max_r * 1.1
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.legend()
    fig1.tight_layout()
    fig1_path = os.path.join(OUT_DIR, "orbit_eci.png")
    fig1.savefig(fig1_path, dpi=200)
    print(f"[INFO] Saved 3D orbit plot to: {fig1_path}")

    # -------------------- Plot 2: Ground track (Longitude vs Latitude) --------------------
    fig2 = plt.figure(figsize=(10, 4))
    ax2 = fig2.add_subplot(111)
    ax2.plot(lons, lats, marker='.', markersize=2, linestyle='-')
    ax2.set_xlabel("Longitude (deg)"); ax2.set_ylabel("Latitude (deg)")
    ax2.set_title("Ground track (Longitude vs Latitude) - Earth rotation included")
    ax2.set_xlim(-180, 180); ax2.set_ylim(-90, 90)
    ax2.grid(True)
    fig2.tight_layout()
    fig2_path = os.path.join(OUT_DIR, "ground_track.png")
    fig2.savefig(fig2_path, dpi=200)
    print(f"[INFO] Saved ground track plot to: {fig2_path}")

    # -------------------- Plot 3: Longitude vs Time -------------------------------------
    fig3 = plt.figure(figsize=(10, 3))
    ax3 = fig3.add_subplot(111)
    ax3.plot(times / 3600.0, lons, linewidth=0.9)
    ax3.set_xlabel("Time (hours)"); ax3.set_ylabel("Longitude (deg)")
    ax3.set_title("Longitude vs Time (shows Earth's rotation effect)")
    ax3.grid(True)
    fig3.tight_layout()
    fig3_path = os.path.join(OUT_DIR, "lon_vs_time.png")
    fig3.savefig(fig3_path, dpi=200)
    print(f"[INFO] Saved longitude vs time plot to: {fig3_path}")

    # Show plots interactively (comment out plt.show() if running on headless server)
    plt.show()

if __name__ == "__main__":
    run_simulation()
