"""
Q2C - Kinetic Simulation of Metabolic Pathway
UAS Bioteknologi BISB211605 - Systems Biology
Syarifathunnisa Iryanti | 23/518191/BI/11280

Pathway:
  X --v1--> A --v2--> B --v3--> P
                 |
                v4
                 |
            Byproduct

P exerts allosteric (non-competitive) inhibition on v1.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ── Parameters ──────────────────────────────────────────────
V1max = 5.0   # Max rate of V1
Km1   = 2.0   # Michaelis constant for V1
Ki    = 3.0   # Inhibition constant
X     = 10.0  # External substrate (constant)
k2    = 1.0   # First-order rate constant A -> B
k3    = 0.8   # First-order rate constant B -> P
k4    = 0.3   # First-order rate constant A -> Byproduct

# ── ODE System ───────────────────────────────────────────────
def metabolic_system(t, y):
    A, B, P = y

    # v1: Michaelis-Menten with non-competitive inhibition by P
    v1 = (V1max * X) / ((Km1 + X) * (1 + P / Ki))
    v2 = k2 * A          # first-order
    v3 = k3 * B          # first-order
    v4 = k4 * A          # first-order

    dA_dt = v1 - v2 - v4
    dB_dt = v2 - v3
    dP_dt = v3

    return [dA_dt, dB_dt, dP_dt]

# ── Initial conditions & time span ───────────────────────────
y0     = [0.0, 0.0, 0.0]          # A, B, P all start at 0
t_span = (0, 48)                   # 48-hour fermentation
t_eval = np.linspace(0, 48, 1000)

# ── Solve ODEs ───────────────────────────────────────────────
sol = solve_ivp(metabolic_system, t_span, y0,
                t_eval=t_eval, method='RK45', rtol=1e-8)

t      = sol.t
A, B, P = sol.y

# ── Compute flux rates ───────────────────────────────────────
v1_t = (V1max * X) / ((Km1 + X) * (1 + P / Ki))
v2_t = k2 * A
v3_t = k3 * B
v4_t = k4 * A

# ── Plot ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle('Kinetic Simulation of Metabolic Pathway\n(with Allosteric Inhibition of v₁ by Product P)',
             fontsize=13, fontweight='bold')

ax1 = axes[0]
ax1.plot(t, A, color='#2196F3', linewidth=2.2, label='Metabolite A')
ax1.plot(t, B, color='#F44336', linewidth=2.2, label='Metabolite B (toxic!)')
ax1.plot(t, P, color='#4CAF50', linewidth=2.2, label='Product P')
ax1.set_ylabel('Concentration (mM)', fontsize=11)
ax1.set_title('Metabolite Concentrations Over Time', fontsize=11)
ax1.legend(fontsize=10, loc='center right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(bottom=0)
b_peak_idx = np.argmax(B)
ax1.axvline(x=t[b_peak_idx], color='#F44336', linestyle='--', alpha=0.5)
ax1.annotate(f'B peak ({t[b_peak_idx]:.1f}h, {B[b_peak_idx]:.2f} mM)',
             xy=(t[b_peak_idx], B[b_peak_idx]),
             xytext=(t[b_peak_idx]+3, B[b_peak_idx]+0.05),
             fontsize=8.5, color='#F44336',
             arrowprops=dict(arrowstyle='->', color='#F44336'))

ax2 = axes[1]
ax2.plot(t, v1_t, color='#9C27B0', linewidth=2.2, label='v₁ (X→A, inhibited by P)')
ax2.plot(t, v2_t, color='#2196F3', linewidth=2.2, label='v₂ (A→B)')
ax2.plot(t, v3_t, color='#4CAF50', linewidth=2.2, label='v₃ (B→P)')
ax2.plot(t, v4_t, color='#FF9800', linewidth=2.2, linestyle='--', label='v₄ (A→Byproduct)')
ax2.set_ylabel('Flux Rate (mM/h)', fontsize=11)
ax2.set_xlabel('Time (hours)', fontsize=11)
ax2.set_title('Reaction Flux Rates Over Time', fontsize=11)
ax2.legend(fontsize=10, loc='center right')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig('simulation_figure.png', dpi=150, bbox_inches='tight')
plt.show()
print("Simulation complete.")
