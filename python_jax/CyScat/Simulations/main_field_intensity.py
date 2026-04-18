"""
Main Field Intensity Simulation
Original: Main.m by Curtis Jin (University of Michigan)

Description:
    Computes scattering coefficients for a set of cylinders and calculates
    the total electric field intensity on a 2D grid. Produces a pcolor-style
    intensity plot showing the scattered field pattern.

Usage:
    python main_field_intensity.py

Parameters (default):
    - 2 dielectric cylinders at [-0.11, 0] and [0.11, 0]
    - epsilon=2, mu=1, radius=0.1
    - Aperiodic (period < 0)
    - Wavelength = 0.9, incident angle = pi/2
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import matplotlib.pyplot as plt
from Scattering_Code.smatrix_parameters import smatrix_parameters
from Scattering_Code.coefficients import coefficients
from Scattering_Code.efieldtot import efieldtot

# ===== Parameter Settings =====
clocs = np.array([[-0.11, 0], [0.11, 0]])
cepmus = np.array([[2, 1], [2, 1]])
crads = np.array([0.1, 0.1])
cmmaxs = np.array([10, 10])
period = -10.5      # Negative = aperiodic
wavelength = 0.9
phiinc = np.pi / 2  # Normal incidence

# SMatrix parameters (for the spatial sums)
epsseries = 1e-13
epsloc = 1e-4
nrepeat = 15
jmax = 3000
kshanksSpatial = 10
kshanksSpectral = 3
spectral = 1
sp = smatrix_parameters(epsseries, epsloc, nrepeat, jmax, kshanksSpatial, kshanksSpectral, spectral)

# ===== Coefficient Computation =====
print("Computing scattering coefficients...")
up_down = 1  # Going up
coeff = coefficients(clocs, cmmaxs, cepmus, crads, period, wavelength, phiinc, up_down, sp)
print(f"  Coefficients computed. Shape: {coeff.shape}")

# ===== Near Field Intensity Computation =====
print("Computing intensity field on 100x100 grid...")
N = 100
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
Intensity = np.zeros((N, N))

for xidx in range(N):
    for yidx in range(N):
        temp = efieldtot(clocs, crads, cmmaxs, wavelength, coeff,
                         -1, np.array([x[xidx], y[yidx]]), phiinc, up_down)
        Intensity[yidx, xidx] = np.abs(temp) ** 2
    if (xidx + 1) % 20 == 0:
        print(f"  Row {xidx + 1}/{N} done")

# ===== Plot =====
print("Plotting...")
fig, ax = plt.subplots(figsize=(8, 6))
pcm = ax.pcolormesh(x, y, Intensity, shading='gouraud', cmap='jet')
plt.colorbar(pcm, ax=ax)
ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_title(f'{len(crads)} cylinders, Incident Angle = {int(phiinc * 180 / np.pi)} degrees')
plt.tight_layout()
plt.savefig('field_intensity.png', dpi=150)
plt.show()
print("Done.")
