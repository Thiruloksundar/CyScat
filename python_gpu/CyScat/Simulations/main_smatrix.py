"""
S-Matrix Computation with Unitarity Validation
Original: MainSMatrix.m by Curtis Jin (University of Michigan)

Description:
    Computes the full scattering matrix for a periodic cylinder configuration,
    validates unitarity via DOU (Degree of Unitarity), and extracts
    transmission eigenvalues from SVD of the S21 block.

    DOU = sum(|S'*S|) / length(S)
    For a unitary matrix, DOU = 1.

Usage:
    python main_smatrix.py

Parameters (default):
    - 1 cylinder at [40, 2]
    - epsilon=2, mu=1, radius=0.1
    - Period = 42.5, wavelength = 0.93
    - Incident angle = pi/2
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import matplotlib.pyplot as plt
from Scattering_Code.smatrix import smatrix
from Scattering_Code.smatrix_parameters import smatrix_parameters

# ===== Parameter Settings (matching MainSMatrix.m active code) =====
clocs = np.array([[40, 1]])
cepmus = np.array([[2, 1]])
crads = np.array([0.1])
cmmaxs = np.array([5])

# Adjust positions (MATLAB: clocs(:,2) = clocs(:,2) + max(abs(clocs(:,2)))+1)
clocs[:, 1] = clocs[:, 1] + np.max(np.abs(clocs[:, 1])) + 1
period = np.max(np.abs(clocs[:, 0])) + 2.5
wavelength = 0.93
nmax = int(np.floor(period / wavelength))
d = 2 * np.max(np.abs(clocs[:, 1])) + 1

phiinc = np.pi / 2
epsseries = 1e-11
epsloc = 1e-4
nrepeatSpatial = 5
nrepeatSpectral = 3
jmax = 1000
kshanksSpatial = 3
kshanksSpectral = -1
spectral = 1
spectralCond = period / 240

sp = smatrix_parameters(wavelength, period, phiinc, epsseries, epsloc,
                        nrepeatSpatial, nrepeatSpectral, jmax,
                        kshanksSpatial, kshanksSpectral, spectral, spectralCond)

# ===== S-Matrix Computation =====
print(f"Computing S-matrix: {len(crads)} cylinder(s), period={period:.2f}, nmax={nmax}")
print(f"  Wavelength={wavelength}, phiinc={phiinc:.4f}, d={d:.2f}")
S, STP = smatrix(clocs, cmmaxs, cepmus, crads, period, wavelength, nmax, d, sp, 'On')
print(f"  S-matrix shape: {S.shape}")
print(f"  Computation time: {STP['TST']:.1f}s")

# ===== Unitarity Check =====
test = np.abs(S.conj().T @ S)
DOU = np.sum(test) / len(test)
print(f"\n  DOU (Degree of Unitarity) = {DOU:.6f}")

# ===== Transmission Eigenvalues (SVD of S21) =====
S21 = S[2*nmax+1:, :2*nmax+1]
e = np.linalg.svd(S21, compute_uv=False) ** 2
print(f"  S21 transmission eigenvalues (first 5): {e[:5]}")

# ===== Full S Singular Values =====
svals = np.linalg.svd(S, compute_uv=False)
print(f"  Max singular value: {np.max(svals):.6f}")
print(f"  Min singular value: {np.min(svals):.6f}")

# ===== Plots =====
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Unitarity check |S'*S|
im = axes[0].imshow(test, cmap='hot', aspect='auto')
axes[0].set_title(f"|S'S| (DOU = {DOU:.4f})")
plt.colorbar(im, ax=axes[0])

# Plot 2: Transmission eigenvalue distribution
axes[1].hist(e, bins=30, color='blue', edgecolor='white', density=True)
axes[1].set_title('Transmission Coefficient Distribution')
axes[1].set_xlabel(r'$\tau^2$ (from SVD of S21)')
axes[1].set_ylabel('Density')
axes[1].grid(True, alpha=0.3)

# Plot 3: Singular values of full S
axes[2].plot(range(1, len(svals)+1), svals, 'b-o', markersize=3)
axes[2].axhline(1.0, color='r', ls='--', alpha=0.5)
axes[2].set_title('S Singular Values')
axes[2].set_xlabel('Index')
axes[2].set_ylabel('Singular Value')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('smatrix_analysis.png', dpi=150)
plt.show()
print("Done.")
