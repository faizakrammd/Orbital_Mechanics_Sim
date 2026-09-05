import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -------------------- Constants --------------------
R_E = 6378.137e3
MU  = 3.986004418e14
g0  = 9.80665
RHO0 = 1.225
H_SCALE = 7200.0

m   = 9000.0
S   = 15.0
CD  = 1.1
CL  = 0.35

K_Q = 1.83e-4
QDOT_MAX = 1.2e7
N_MAX    = 4.5

h_entry   = 120e3
v_entry   = 7500.0
gamma_deg = -1.2
theta0    = 0.0

T_END = 1600.0
DT    = 1.0
N_STEPS = int(T_END/DT) + 1
t_grid  = np.linspace(0.0, T_END, N_STEPS)

# -------------------- Models --------------------
def atmosphere_rho(h):
    return RHO0 * np.exp(-max(h, 0)/H_SCALE)

def aero_forces(v, h, sigma):
    rho = atmosphere_rho(h)
    q = 0.5 * rho * v*v
    D = q * S * CD
    L = q * S * CL * np.cos(sigma)
    return D, L, rho

def heat_rate(v, h):
    rho = atmosphere_rho(h)
    return K_Q * np.sqrt(rho) * (v**3)

def load_factor(D, L):
    return np.sqrt(D*D + L*L) / (m * g0)

def dynamics_step(state, sigma, dt):
    h, v, gam, th = state

    def f(st):
        hh, vv, gg, tt = st
        rr = R_E + hh
        Dk, Lk, _ = aero_forces(vv, hh, sigma)
        gk = MU / (rr*rr)
        hh_dot = vv * np.sin(gg)
        vv_dot = -Dk/m - gk * np.sin(gg)
        gg_dot = (Lk/(m*max(vv,1e-3))) + (vv/rr - gk/max(vv,1e-3)) * np.cos(gg)
        tt_dot = vv * np.cos(gg) / rr
        return np.array([hh_dot, vv_dot, gg_dot, tt_dot])

    k1 = f(state)
    k2 = f(state + 0.5*dt*k1)
    k3 = f(state + 0.5*dt*k2)
    k4 = f(state + dt*k3)
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# -------------------- Simple rollout --------------------
def rollout():
    sigma = np.deg2rad(45.0)  # fixed bank angle for demo
    state = np.array([h_entry, v_entry, np.radians(gamma_deg), theta0], dtype=float)

    H=[]; V=[]; TH=[]; QDOT=[]; NLOAD=[]
    for t in t_grid:
        H.append(state[0])
        V.append(state[1])
        TH.append(state[3])
        D,L,_=aero_forces(state[1], state[0], sigma)
        QDOT.append(heat_rate(state[1], state[0]))
        NLOAD.append(load_factor(D,L))

        if state[0] < 0 or state[1] < 50:
            break
        state = dynamics_step(state, sigma, DT)

    return {
        "t": np.array(t_grid[:len(H)]),
        "h": np.array(H)/1000.0,
        "v": np.array(V),
        "theta": np.array(TH)*R_E/1000.0,
        "qdot": np.array(QDOT)/1e6,
        "nload": np.array(NLOAD)
    }

# -------------------- Run trajectory --------------------
out = rollout()
t = out["t"]
h = out["h"]
rng = out["theta"]
v = out["v"]
qdot = out["qdot"]
nload = out["nload"]

# -------------------- Set up animation --------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

# Left: trajectory
ax1.plot(rng, h, 'k--', alpha=0.6)
traj_point, = ax1.plot([], [], 'ro', markersize=6)
ax1.set_xlim(0, np.max(rng)*1.05)
ax1.set_ylim(0, np.max(h)*1.1)
ax1.set_xlabel("Downrange [km]")
ax1.set_ylabel("Altitude [km]")
ax1.set_title("Re-entry trajectory")

# Right: heat & g
ln_q, = ax2.plot(t, qdot, 'r-', label="Heat-rate [MW/m²]")
ln_n, = ax2.plot(t, nload, 'b-', label="Load factor [g]")
dot_q, = ax2.plot([], [], 'ro')
dot_n, = ax2.plot([], [], 'bo')
ax2.legend()
ax2.set_xlim(0, np.max(t))
ax2.set_ylim(0, max(np.max(qdot), np.max(nload))*1.2)
ax2.set_xlabel("Time [s]")
ax2.set_title("Path constraints")

# Dynamic annotation text box
text_box = ax2.text(0.05, 0.95, "", transform=ax2.transAxes,
                    fontsize=10, va="top", ha="left",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="black"))

# -------------------- Animation update --------------------
def update(frame):
    traj_point.set_data([rng[frame]], [h[frame]])
    dot_q.set_data([t[frame]], [qdot[frame]])
    dot_n.set_data([t[frame]], [nload[frame]])

    # Update text box with live values
    text_box.set_text(
        f"t   = {t[frame]:.0f} s\n"
        f"h   = {h[frame]:.1f} km\n"
        f"v   = {v[frame]/1000:.2f} km/s\n"
        f"q̇   = {qdot[frame]:.2f} MW/m²\n"
        f"n   = {nload[frame]:.2f} g"
    )

    return traj_point, dot_q, dot_n, text_box

ani = FuncAnimation(fig, update, frames=len(t), interval=40, blit=True)

plt.tight_layout()
plt.show()
