"""
Compute S21 singular value distribution for N cylinders.
Saves cylinder positions to CSV so MATLAB can use the exact same ones.

Usage:
    python compute_ncyl.py 500
    python compute_ncyl.py 5000
    python compute_ncyl.py 500 --load positions_500.csv   # reuse saved positions
    python compute_ncyl.py 5000 --gpus 4                  # use 4 GPUs in parallel

Multi-GPU note: when --gpus > 1, the driver process uses CPU only (to avoid
locking GPU 0).  Each worker process claims its own GPU via CUDA_VISIBLE_DEVICES
before CuPy is initialized.
"""
import sys
import os
import time
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Headless backend for server
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
sys.path.insert(0, './Scattering_Code')

# get_partition has no GPU dependency — safe to import at module level
from get_partition import smat_to_s11, smat_to_s12, smat_to_s21, smat_to_s22

# NOTE: smatrix_parameters, smatrix_cascade, gpu_backend are imported inside
# main() so we can set CYSCAT_FORCE_CPU=1 BEFORE CuPy is imported in the
# parent process (multi-GPU mode needs parent to stay CPU so it doesn't lock
# any GPU before workers start).


def main():
    # === Parse arguments ===
    parser = argparse.ArgumentParser()
    parser.add_argument('num_cyl', type=int, default=500, nargs='?')
    parser.add_argument('--load', type=str, default=None, help='Load positions from CSV')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs (default: 1). >1 enables multi-GPU batch mode.')
    args = parser.parse_args()

    # ── Critical: set FORCE_CPU BEFORE importing anything that touches CuPy ──
    # Parent process must not initialize CUDA; worker processes unset this flag.
    if args.gpus > 1:
        os.environ['CYSCAT_FORCE_CPU'] = '1'

    # Now safe to import GPU-dependent modules
    from Scattering_Code.smatrix_parameters import smatrix_parameters
    from Scattering_Code.smatrix_cascade import smatrix_cascade
    from Scattering_Code.gpu_backend import get_info as gpu_info

    # === Parameters (matching MATLAB GUI defaults) ===
    num_cyl = 500 if args.num_cyl is None else args.num_cyl
    wavelength = 0.93
    period = 12.81
    radius = 0.25
    n_cylinder = 1.3
    #eps = n_cylinder**2
    eps = -1
    mu = 1.0
    cmmax = 5
    phiinc = np.pi / 2
    seed = args.seed

    # Thickness scales with cylinder count
    spacing = 2.5 * radius
    cyls_per_row = int(period / spacing)
    rows_needed = num_cyl / cyls_per_row + 2
    thickness = max(0.5, rows_needed * spacing * 1.5)
    thickness = round(thickness, 1)

    # Evanescent mode truncation (matching MATLAB)
    EvanescentModeTol = 1e-2
    NoPropagatingModes = int(np.floor(period / wavelength))
    NoEvaMode = int(np.floor(
        period / (2 * np.pi) * np.sqrt(
            (np.log(EvanescentModeTol) / (2 * radius))**2 + (2 * np.pi / wavelength)**2
        )
    )) - NoPropagatingModes
    NoEvaMode = max(NoEvaMode, 0)
    if eps < 0:  # PEC: skip evanescent buffer (causes numerical instability)
        NoEvaMode = 0
    nmax = NoPropagatingModes + NoEvaMode
    d = thickness

    if args.gpus > 1:
        backend_str = f"Multi-GPU mode — {args.gpus} GPUs (driver: CPU only)"
    else:
        backend_str = gpu_info()

    print(f"{'='*60}")
    print(f"CyScat: {num_cyl} cylinders")
    print(f"  Backend: {backend_str}")
    print(f"  GPUs requested: {args.gpus}")
    print(f"  period={period}, wavelength={wavelength}, thickness={thickness}")
    print(f"  radius={radius}, n_cylinder={n_cylinder}, cmmax={cmmax}")
    print(f"  phiinc=pi/2, EvanescentModeTol={EvanescentModeTol}")
    print(f"  NoPropModes={NoPropagatingModes}, NoEvaMode={NoEvaMode}, nmax={nmax}")
    print(f"  S-matrix: {2*(2*nmax+1)}x{2*(2*nmax+1)} -> truncated "
          f"{2*(2*NoPropagatingModes+1)}x{2*(2*NoPropagatingModes+1)}")
    print(f"{'='*60}")

    # === Generate or load cylinder positions ===
    if args.load and os.path.exists(args.load):
        print(f"Loading positions from {args.load}")
        clocs = np.loadtxt(args.load, delimiter=',')
        if len(clocs) != num_cyl:
            print(f"  WARNING: file has {len(clocs)} rows but num_cyl={num_cyl}")
            num_cyl = len(clocs)
    else:
        print(f"Generating {num_cyl} positions (seed={seed})...")
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
            else:
                print(f"  Warning: Could not place cylinder {i+1} after 5000 attempts")
        print(f"  Placed {placed}/{num_cyl} cylinders")

        csv_file = f'positions_{num_cyl}.csv'
        np.savetxt(csv_file, clocs, delimiter=',', fmt='%.15e')
        print(f"  Saved positions to {csv_file}")

    clocs_original = clocs.copy()

    cmmaxs = np.array([cmmax] * num_cyl)
    cepmus = np.array([[eps, mu]] * num_cyl)
    crads = np.array([radius] * num_cyl)

    # === Compute S-matrix ===
    sp = smatrix_parameters(wavelength, period, phiinc,
                            1e-11, 1e-4, 5, 3, 1000, 3, 5, 1, period / 120)

    print(f"\nComputing S-matrix...")
    t0 = time.time()
    S, STP = smatrix_cascade(clocs, cmmaxs, cepmus, crads, period, wavelength,
                              nmax, d, sp, 'On',
                              cylinders_per_group=50, num_gpus=args.gpus)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # === Truncate evanescent modes ===
    S11 = smat_to_s11(S)
    S12 = smat_to_s12(S)
    S21 = smat_to_s21(S)
    S22 = smat_to_s22(S)

    if NoEvaMode > 0:
        S11T = S11[NoEvaMode:-NoEvaMode, NoEvaMode:-NoEvaMode]
        S12T = S12[NoEvaMode:-NoEvaMode, NoEvaMode:-NoEvaMode]
        S21T = S21[NoEvaMode:-NoEvaMode, NoEvaMode:-NoEvaMode]
        S22T = S22[NoEvaMode:-NoEvaMode, NoEvaMode:-NoEvaMode]
        STruncated = np.block([[S11T, S12T], [S21T, S22T]])
    else:
        S11T, S12T, S21T, S22T = S11, S12, S21, S22
        STruncated = S

    # === Diagnostics ===
    svals_full = np.linalg.svd(STruncated, compute_uv=False)
    tau = np.linalg.svd(S21T, compute_uv=False)
    tau_sq = tau**2
    rho = np.linalg.svd(S11T, compute_uv=False)
    rho_sq = rho**2

    test = np.abs(STruncated.conj().T @ STruncated)
    DOU = np.sum(test) / len(STruncated)

    center = NoPropagatingModes
    R = np.sum(np.abs(S11T[:, center])**2)
    T = np.sum(np.abs(S21T[:, center])**2)

    print(f"\n{'='*60}")
    print(f"RESULTS: {num_cyl} cylinders, thickness={thickness}")
    print(f"  Truncated S: {STruncated.shape[0]}x{STruncated.shape[1]}")
    print(f"  Max singular value: {np.max(svals_full):.6f}")
    print(f"  DOU = {DOU:.6f}")
    print(f"  R={R:.6f}, T={T:.6f}, R+T={R+T:.6f}")
    print(f"  S21 SVs: min={np.min(tau):.6f}, max={np.max(tau):.6f}, mean={np.mean(tau):.6f}")
    print(f"{'='*60}")

    # Save results
    np.savez(f'results_{num_cyl}cyl.npz',
             svals=svals_full, tau=tau, tau_sq=tau_sq, rho=rho, rho_sq=rho_sq,
             DOU=DOU, R=R, T=T, clocs=clocs_original,
             num_cyl=num_cyl, thickness=thickness, wavelength=wavelength, period=period)
    print(f"Results saved to results_{num_cyl}cyl.npz")

    # === Plot ===
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f'CyScat: {num_cyl} Cylinders | Period={period} | λ={wavelength} | '
                 f'Thickness={thickness}', fontsize=14, color='white')

    axes[0, 0].plot(range(1, len(svals_full) + 1), svals_full,
                    'b-o', markersize=2, linewidth=1.5)
    axes[0, 0].set_title(f'S Singular Values (DOU = {DOU:.4f})')
    axes[0, 0].set_xlabel('Index')
    axes[0, 0].set_ylabel('Singular Value')
    axes[0, 0].axhline(1.0, color='r', ls='--', alpha=0.5)
    axes[0, 0].set_ylim(bottom=0, top=max(1.05, np.max(svals_full) * 1.05))
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].axvline(0, color='magenta', linewidth=3)
    axes[0, 1].axvline(thickness, color='magenta', linewidth=3)
    axes[0, 1].scatter(clocs_original[:, 1], clocs_original[:, 0],
                       s=max(1, 50 // max(1, num_cyl // 100)),
                       c='blue', edgecolors='blue', linewidths=0.3)
    axes[0, 1].set_title('Geometry')
    axes[0, 1].set_xlabel('y (unit)')
    axes[0, 1].set_ylabel('x (unit)')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].hist(rho_sq, bins=30, color='blue', edgecolor='white', density=True)
    axes[1, 0].set_title('Reflection Coefficient Distribution')
    axes[1, 0].set_xlabel(r'$\rho^2$ (SVD)')
    axes[1, 0].set_ylabel(r'$\rho(\rho^2)$')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(tau_sq, bins=30, color='green', edgecolor='white', density=True)
    axes[1, 1].set_title('Transmission Coefficient Distribution')
    axes[1, 1].set_xlabel(r'$\tau^2$ (SVD)')
    axes[1, 1].set_ylabel(r'$\rho(\tau^2)$')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    outfile = f'result_{num_cyl}cyl.png'
    plt.savefig(outfile, facecolor=fig.get_facecolor(), dpi=150)
    print(f"Plot saved to {outfile}")
    plt.style.use('default')


if __name__ == '__main__':
    main()
