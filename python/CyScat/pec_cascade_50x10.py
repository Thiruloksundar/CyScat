"""
PEC Cascade: Compute 50-cylinder PEC slab, cascade 10 times → 500 cylinders.
"""
import sys
import time
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
sys.path.insert(0, './Scattering_Code')

from Scattering_Code.smatrix_parameters import smatrix_parameters
from Scattering_Code.smatrix import smatrix
from Scattering_Code.cascadertwo import cascadertwo
from get_partition import smat_to_s11, smat_to_s12, smat_to_s21, smat_to_s22


def main():
    # === Parameters ===
    num_cyl = 50
    n_cascade = 10
    wavelength = 0.93
    period = 12.81
    radius = 0.25
    epsilon = -1  # PEC
    mu = 1.0
    cmmax = 5
    phiinc = np.pi / 2
    seed = 42

    # Thickness for 50 cylinders
    spacing = 2.5 * radius
    cyls_per_row = int(period / spacing)
    rows_needed = num_cyl / cyls_per_row + 2
    thickness = max(0.5, rows_needed * spacing * 1.5)
    thickness = round(thickness, 1)

    # No evanescent buffer for PEC
    NoPropagatingModes = int(np.floor(period / wavelength))
    NoEvaMode = 0
    nmax = NoPropagatingModes
    d = thickness

    print("=" * 60)
    print(f"PEC Cascade: {num_cyl} cylinders x {n_cascade} = {num_cyl * n_cascade} effective")
    print(f"  period={period}, wavelength={wavelength}, thickness={thickness}")
    print(f"  radius={radius}, nmax={nmax} (no evanescent buffer)")
    print("=" * 60)

    # === Generate positions ===
    np.random.seed(seed)
    margin = radius * 1.5
    clocs = np.zeros((num_cyl, 2))
    placed = 0
    for i in range(num_cyl):
        for attempt in range(5000):
            x = np.random.uniform(margin, period - margin)
            y = np.random.uniform(margin, thickness - margin)
            if i == 0:
                clocs[i] = [x, y]
                placed += 1
                break
            dists = np.sqrt((x - clocs[:i, 0])**2 + (y - clocs[:i, 1])**2)
            if np.all(dists > 2.5 * radius):
                clocs[i] = [x, y]
                placed += 1
                break
    print(f"Placed {placed}/{num_cyl} cylinders")

    cmmaxs = cmmax * np.ones(num_cyl, dtype=int)
    cepmus = np.column_stack([epsilon * np.ones(num_cyl), np.ones(num_cyl)])
    crads = radius * np.ones(num_cyl)

    sp = smatrix_parameters(wavelength, period, phiinc,
                            1e-11, 1e-4, 5, 3, 1000, 3, 5, 1, period / 120)

    # === Step 1: Compute single 50-cylinder S-matrix ===
    print(f"\nStep 1: Computing S-matrix for {num_cyl} PEC cylinders...")
    t0 = time.time()
    S, STP = smatrix(clocs, cmmaxs, cepmus, crads, period, wavelength,
                     nmax, d, sp, 'On')
    t_smatrix = time.time() - t0
    print(f"  Done in {t_smatrix:.1f}s")

    # Check single slab unitarity
    test = np.abs(S.conj().T @ S)
    DOU_single = np.sum(test) / len(test)
    print(f"  Single slab DOU = {DOU_single:.10f}")

    # === Step 2: Cascade 10 times ===
    print(f"\nStep 2: Cascading {n_cascade} times...")
    t0 = time.time()
    Scas = S.copy()
    dcas = d
    for i in range(1, n_cascade):
        Scas, dcas = cascadertwo(Scas, dcas, S, d)
        if (i + 1) % 2 == 0:
            test_i = np.abs(Scas.conj().T @ Scas)
            dou_i = np.sum(test_i) / len(test_i)
            print(f"  After {i + 1} cascades: DOU = {dou_i:.10f}, thickness = {dcas:.1f}")
    t_cascade = time.time() - t0
    print(f"  Cascade done in {t_cascade:.1f}s")

    # === Diagnostics ===
    nm = 2 * nmax + 1
    S11 = Scas[:nm, :nm]
    S12 = Scas[:nm, nm:]
    S21 = Scas[nm:, :nm]
    S22 = Scas[nm:, nm:]

    svals_full = np.linalg.svd(Scas, compute_uv=False)
    tau = np.linalg.svd(S21, compute_uv=False)
    tau_sq = tau**2
    rho = np.linalg.svd(S11, compute_uv=False)
    rho_sq = rho**2

    test = np.abs(Scas.conj().T @ Scas)
    DOU = np.sum(test) / len(test)

    center = NoPropagatingModes
    R = np.sum(np.abs(S11[:, center])**2)
    T = np.sum(np.abs(S21[:, center])**2)

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {num_cyl} x {n_cascade} = {num_cyl * n_cascade} effective PEC cylinders")
    print(f"  Total thickness: {dcas:.1f}")
    print(f"  S-matrix size: {Scas.shape[0]}x{Scas.shape[1]}")
    print(f"  Max singular value: {np.max(svals_full):.10f}")
    print(f"  DOU = {DOU:.10f}")
    print(f"  R={R:.6f}, T={T:.6f}, R+T={R+T:.6f}")
    print(f"  S21 SVs: min={np.min(tau):.6f}, max={np.max(tau):.6f}, mean={np.mean(tau):.6f}")
    print(f"  Time: smatrix={t_smatrix:.1f}s + cascade={t_cascade:.1f}s = {t_smatrix + t_cascade:.1f}s total")
    print(f"{'=' * 60}")

    # Save
    np.savez(f'results_pec_{num_cyl}x{n_cascade}.npz',
             svals=svals_full, tau=tau, tau_sq=tau_sq, rho=rho, rho_sq=rho_sq,
             DOU=DOU, R=R, T=T, clocs=clocs, Scas=Scas, dcas=dcas,
             num_cyl=num_cyl, n_cascade=n_cascade, thickness=thickness,
             wavelength=wavelength, period=period, nmax=nmax)
    print(f"Results saved to results_pec_{num_cyl}x{n_cascade}.npz")

    # === Plot ===
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f'PEC: {num_cyl} cylinders x {n_cascade} cascade = {num_cyl * n_cascade} effective\n'
                 f'Period={period} | λ={wavelength} | Total thickness={dcas:.1f}',
                 fontsize=14)

    axes[0, 0].plot(range(1, len(svals_full) + 1), svals_full,
                    'b-o', markersize=3, linewidth=1.5)
    axes[0, 0].axhline(1.0, color='r', ls='--', alpha=0.5)
    axes[0, 0].set_title(f'S Singular Values (DOU = {DOU:.6f})')
    axes[0, 0].set_xlabel('Index')
    axes[0, 0].set_ylabel('Singular Value')
    axes[0, 0].set_ylim(bottom=0, top=max(1.05, np.max(svals_full) * 1.05))
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(clocs[:, 1], clocs[:, 0], s=10, c='blue')
    axes[0, 1].axvline(0, color='magenta', linewidth=3)
    axes[0, 1].axvline(thickness, color='magenta', linewidth=3)
    axes[0, 1].set_title(f'Single Slab Geometry ({num_cyl} cyl)')
    axes[0, 1].set_xlabel('y')
    axes[0, 1].set_ylabel('x')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].hist(rho_sq, bins=30, color='blue', edgecolor='black', density=True)
    axes[1, 0].set_title('Reflection Eigenvalue Distribution')
    axes[1, 0].set_xlabel('ρ²')
    axes[1, 0].set_ylabel('ρ(ρ²)')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(tau_sq, bins=30, color='green', edgecolor='black', density=True)
    axes[1, 1].set_title('Transmission Eigenvalue Distribution')
    axes[1, 1].set_xlabel('τ²')
    axes[1, 1].set_ylabel('ρ(τ²)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    outfile = f'result_pec_{num_cyl}x{n_cascade}.png'
    plt.savefig(outfile, dpi=150)
    print(f"Plot saved to {outfile}")


if __name__ == '__main__':
    main()
