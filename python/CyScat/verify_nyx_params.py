"""
Quick verification: run cascade with Nyx parameters (period=183.31, radius=0.1, cmmax=3)
to confirm max tau^2 stays near 1 for dielectric cylinders.
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, './Scattering_Code')

from Scattering_Code.smatrix_parameters import smatrix_parameters
from Scattering_Code.smatrix import smatrix
from Scattering_Code.cascadertwo import cascadertwo
from get_partition import smat_to_s21

# === Nyx parameters ===
wavelength = 0.93
Width = 180
period = Width + 3.31  # = 183.31
Radius = 0.1
IOR = 1.3
epsilon = IOR ** 2  # 1.69
NoCylinderModes = 3
phiinc = np.pi / 2
cyl_per_slab = 10
n_cascade = 20  # 200 cylinders total (quick test)

# Evanescent mode calculation (matching MainNyx.m)
CascadeTol = 10 ** (-1.1)
Bufferlength = Radius * 2
NoPropagatingModes = int(np.floor(period / wavelength))
NoEvaMode = int(np.floor(
    period / (2 * np.pi) * np.sqrt(
        (np.log(CascadeTol) / Bufferlength) ** 2 + (2 * np.pi / wavelength) ** 2
    )
)) - NoPropagatingModes
NoEvaMode = max(NoEvaMode, 0)
nmax = NoPropagatingModes + NoEvaMode

# Thickness for sparse slab of 10 cylinders
Thickness = 10  # small slab
d = Thickness

print("=" * 60)
print("Verify Nyx params: dielectric cascade")
print(f"  period={period}, wavelength={wavelength}")
print(f"  radius={Radius}, epsilon={epsilon}, cmmax={NoCylinderModes}")
print(f"  nmax={nmax}, NoEvaMode={NoEvaMode}, NoPropModes={NoPropagatingModes}")
print(f"  S-matrix size: {2*(2*nmax+1)} x {2*(2*nmax+1)}")
print(f"  {cyl_per_slab} cyl/slab x {n_cascade} cascades = {cyl_per_slab * n_cascade} total")
print("=" * 60)

sp = smatrix_parameters(wavelength, period, phiinc,
                        1e-11, 1e-4, 5, 3, 1000, 3, -1, 1, period / 120)

np.random.seed(42)

Scas = None
dcas = 0.0

for i in range(n_cascade):
    t0 = time.time()

    # Generate random cylinder positions (sparse, within Width x Thickness)
    margin = Radius * 2
    min_sep = Radius * 10  # matching MinInterDistance from MainNyx
    clocs = np.zeros((cyl_per_slab, 2))
    for j in range(cyl_per_slab):
        for attempt in range(5000):
            x = np.random.uniform(margin, Width - margin)
            y = np.random.uniform(margin, Thickness - margin)
            if j == 0:
                clocs[j] = [x, y]
                break
            dists = np.sqrt((x - clocs[:j, 0])**2 + (y - clocs[:j, 1])**2)
            if np.all(dists > min_sep):
                clocs[j] = [x, y]
                break

    cmmaxs = NoCylinderModes * np.ones(cyl_per_slab, dtype=int)
    cepmus = np.column_stack([epsilon * np.ones(cyl_per_slab),
                               np.ones(cyl_per_slab)])
    crads = Radius * np.ones(cyl_per_slab)

    S_new, _ = smatrix(clocs, cmmaxs, cepmus, crads, period, wavelength,
                       nmax, d, sp, 'On')

    # Truncate evanescent modes before cascading
    if NoEvaMode > 0:
        half = len(S_new) // 2
        s11 = S_new[:half, :half][NoEvaMode:-NoEvaMode, NoEvaMode:-NoEvaMode]
        s12 = S_new[:half, half:][NoEvaMode:-NoEvaMode, NoEvaMode:-NoEvaMode]
        s21 = S_new[half:, :half][NoEvaMode:-NoEvaMode, NoEvaMode:-NoEvaMode]
        s22 = S_new[half:, half:][NoEvaMode:-NoEvaMode, NoEvaMode:-NoEvaMode]
        S_new = np.block([[s11, s12], [s21, s22]])

    if Scas is None:
        Scas = S_new
        dcas = d
    else:
        Scas, dcas = cascadertwo(Scas, dcas, S_new, d)

    S21 = smat_to_s21(Scas)
    tau = np.linalg.svd(S21, compute_uv=False)
    tau_sq = tau ** 2

    test = np.abs(Scas.conj().T @ Scas)
    dou = np.sum(test) / len(Scas)

    n_total = (i + 1) * cyl_per_slab
    elapsed = time.time() - t0
    print(f"  Cascade {i+1:3d}/{n_cascade} ({n_total:4d} cyl): "
          f"max tau^2={np.max(tau_sq):.6f}, min tau^2={np.min(tau_sq):.6e}, "
          f"DOU={dou:.6f}, time={elapsed:.1f}s")

print(f"\nFinal: {n_cascade * cyl_per_slab} cylinders, max tau^2 = {np.max(tau_sq):.6f}")
