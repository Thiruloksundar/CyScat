import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

n_trials = 200

# Load all results
all_tau_sq = []
all_rho_sq = []
T_vals     = []
R_vals     = []
missing    = []

for i in range(n_trials):
    path = f'svd_results/trial_{i:04d}.npz'
    if not os.path.exists(path):
        missing.append(i)
        continue
    d = np.load(path)
    all_tau_sq.append(d['tau_sq'])
    all_rho_sq.append(d['rho_sq'])
    T_vals.append(float(d['T']))
    R_vals.append(float(d['R']))

print(f"Loaded {n_trials - len(missing)}/{n_trials} trials")
if missing:
    print(f"Missing trials: {missing}")

all_tau_sq = np.concatenate(all_tau_sq)   # (n_trials × num_modes,)
all_rho_sq = np.concatenate(all_rho_sq)

print("tau range:", np.min(all_tau_sq), np.max(all_tau_sq))
print("rho range:", np.min(all_rho_sq), np.max(all_rho_sq))
print("tau std:", np.std(all_tau_sq))
print("rho std:", np.std(all_rho_sq))
# === Plot ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f'Singular Value Distribution — 200 trials, 1600 cylinders', fontsize=13)

axes[0].hist(all_tau_sq, bins=50, density=True,
             color='green', edgecolor='white', alpha=0.85)
axes[0].set_xlabel(r'$\tau^2$ (transmission eigenvalue)')
axes[0].set_ylabel(r'$\rho(\tau^2)$')
axes[0].set_title('Transmission eigenvalue distribution')
# axes[0].set_xlim(0, 1)
# axes[0].set_ylim(0, 1)
axes[0].set_xlim(np.min(all_tau_sq), np.max(all_tau_sq))
axes[0].grid(True, alpha=0.3)

axes[1].hist(all_rho_sq, bins=50, density=True,
             color='blue', edgecolor='white', alpha=0.85)
axes[1].set_xlabel(r'$\rho^2$ (reflection eigenvalue)')
axes[1].set_ylabel(r'$\rho(\rho^2)$')
axes[1].set_title('Reflection eigenvalue distribution')
# axes[1].set_xlim(0, 1)
# axes[1].set_ylim(0, 1)
axes[1].set_xlim(np.min(all_rho_sq), np.max(all_rho_sq))
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('svd_distribution_1600cyl_200trials.png', dpi=150)
print("Plot saved to svd_distribution_1600cyl_200trials.png")

print(f"\nMean T = {np.mean(T_vals):.4f} ± {np.std(T_vals):.4f}")
print(f"Mean R = {np.mean(R_vals):.4f} ± {np.std(R_vals):.4f}")
