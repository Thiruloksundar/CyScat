"""
MainAbsorption - Energy conservation (DOU) test
Translated from MainAbsorption.m (Curtis Jin, 2011)

Tests unitarity of the S-matrix for a fixed arrangement of cylinders.
For lossless cylinders, DOU should equal 1.0 (energy conservation).

Usage:
    python main_absorption.py
    python main_absorption.py --num_cyl 16
"""
import sys
import time
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
sys.path.insert(0, './Scattering_Code')

from Scattering_Code.smatrix_parameters import smatrix_parameters
from Scattering_Code.smatrix import smatrix


def main():
    parser = argparse.ArgumentParser(description='Energy conservation test')
    parser.add_argument('--num_cyl', type=int, default=16)
    args = parser.parse_args()

    # Fixed arrangement: cylinders at x=10, evenly spaced in y
    NoCyl = args.num_cyl
    y_positions = list(range(10, 10 * NoCyl + 1, 10))
    # Skip y=110,120 to match MATLAB (gap in the arrangement)
    y_positions = [y for y in y_positions if y not in [110, 120]]
    y_positions = y_positions[:NoCyl]

    clocs = np.column_stack([10.0 * np.ones(NoCyl), np.array(y_positions, dtype=float)])
    cepmus = np.column_stack([2.0 * np.ones(NoCyl), np.ones(NoCyl)])
    crads = 0.1 * np.ones(NoCyl)
    cmmaxs = 5 * np.ones(NoCyl, dtype=int)

    period = 12.5
    wavelength = 0.93
    nmax = int(np.floor(period / wavelength))
    d = 13.0
    phiinc = np.pi / 2

    print("=" * 60)
    print(f"Absorption Test: {NoCyl} cylinders")
    print(f"  period={period}, wavelength={wavelength}, d={d}")
    print(f"  nmax={nmax}")
    print("=" * 60)

    sp = smatrix_parameters(wavelength, period, phiinc,
                            1e-11, 1e-4, 5, 3, 1000, 3, -1, 1, period / 240)

    print("\nComputing S-matrix...")
    t0 = time.time()
    S, STP = smatrix(clocs, cmmaxs, cepmus, crads, period, wavelength,
                     nmax, d, sp, 'On')
    print(f"Done in {time.time() - t0:.1f}s")

    # Unitarity check
    test = np.abs(S.conj().T @ S)
    DOU = np.sum(test) / len(test)

    # Find small entries (diagnostic)
    small_mask = np.abs(S) < 0.9
    max_small = np.max(np.abs(S[small_mask])) if np.any(small_mask) else 0

    # Transmission eigenvalues
    nm = 2 * nmax + 1
    S21 = S[nm:, :nm]
    e = np.linalg.svd(S21, compute_uv=False)**2

    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"  DOU = {DOU:.10f}")
    print(f"  Max |S| entry below 0.9: {max_small:.6f}")
    print(f"  Transmission eigenvalues: min={np.min(e):.6f}, max={np.max(e):.6f}")
    print(f"  Mean transmission: {np.mean(e):.6f}")
    print(f"{'=' * 60}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(test, aspect='auto', cmap='viridis')
    axes[0].set_title(f'|S^H S| (DOU = {DOU:.6f})')
    axes[0].colorbar = plt.colorbar(axes[0].images[0], ax=axes[0])

    axes[1].hist(e, bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_title(f'Transmission Eigenvalue Distribution ({NoCyl} cyl)')
    axes[1].set_xlabel('τ²')
    axes[1].set_ylabel('Count')

    plt.tight_layout()
    plt.savefig('result_absorption.png', dpi=150)
    print("Plot saved to result_absorption.png")


if __name__ == '__main__':
    main()
