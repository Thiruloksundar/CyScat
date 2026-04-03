"""
MainProperNoEvanescentModes - Convergence study for evanescent mode count
Translated from MainProperNoEvanescentModes.m (Curtis Jin, 2011)

Tests how many evanescent modes are needed for accurate cascade results.
Sweeps over different evanescent mode tolerances, computes S-matrix with
evanescent buffer + cascade + truncation, and compares against direct
computation. Reports Frobenius norm error and DOU for each.

Usage:
    python main_evanescent_modes.py
    python main_evanescent_modes.py --num_cyl 15
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
from Scattering_Code.cascadertwo import cascadertwo


def main():
    parser = argparse.ArgumentParser(description='Evanescent mode convergence study')
    parser.add_argument('--num_cyl', type=int, default=15)
    args = parser.parse_args()

    NoCyl = args.num_cyl
    wavelength = 0.93
    Width = NoCyl + 2
    period = Width + 3.31
    radius = 0.1
    epsilon = 2.0
    mu = 1.0
    cmmax = 3
    d = 0.4
    phiinc = np.pi / 2

    # Cylinder positions
    xpos = np.arange(1, NoCyl + 1, dtype=float)
    clocs = np.column_stack([xpos, 0.2 * np.ones(NoCyl)])
    cepmus = np.column_stack([epsilon * np.ones(NoCyl), np.ones(NoCyl)])
    crads = radius * np.ones(NoCyl)
    cmmaxs = cmmax * np.ones(NoCyl, dtype=int)

    NoPropagatingModes = int(np.floor(period / wavelength))
    Bufferlength = d - np.max(np.abs(clocs[:, 1])) + 0.2

    # Sweep over tolerances
    tolerances = [1e-1, 1e-2, 1e-3, 1e-5, 1e-8, 1e-11]
    results = []

    # First compute the "ground truth": direct computation of doubled slab
    print("Computing ground truth (direct doubled slab, no evanescent)...")
    sp0 = smatrix_parameters(wavelength, period, phiinc,
                             1e-11, 1e-4, 5, 3, 1000, 3, -1, 1, period / 240)
    offset = np.column_stack([np.zeros(NoCyl), d * np.ones(NoCyl)])
    clocs2 = np.vstack([clocs, clocs + offset])
    cepmus2 = np.vstack([cepmus, cepmus])
    crads2 = np.concatenate([crads, crads])
    cmmaxs2 = np.concatenate([cmmaxs, cmmaxs])
    nmax_base = NoPropagatingModes

    S_truth, _ = smatrix(clocs2, cmmaxs2, cepmus2, crads2,
                         period, wavelength, nmax_base, 2 * d, sp0, 'On')
    nm_base = 2 * nmax_base + 1
    e_truth = np.linalg.svd(S_truth[nm_base:, :nm_base], compute_uv=False)**2
    DOU_truth = np.sum(np.abs(S_truth.conj().T @ S_truth)) / len(S_truth)
    print(f"  DOU_truth = {DOU_truth:.10f}")

    print(f"\n{'=' * 60}")
    print(f"Evanescent Mode Convergence Study: {NoCyl} cylinders")
    print(f"  period={period:.2f}, d={d}, Bufferlength={Bufferlength:.3f}")
    print(f"  NoPropagatingModes={NoPropagatingModes}")
    print(f"{'=' * 60}\n")

    for tol in tolerances:
        NoEvaMode = int(np.floor(
            period / (2 * np.pi) * np.sqrt(
                (np.log(tol) / Bufferlength)**2 + (2 * np.pi / wavelength)**2
            )
        )) - NoPropagatingModes
        NoEvaMode = max(NoEvaMode, 0)
        nmax = NoPropagatingModes + NoEvaMode

        print(f"tol={tol:.0e}: NoEvaMode={NoEvaMode}, nmax={nmax}")

        sp = smatrix_parameters(wavelength, period, phiinc,
                                1e-11, 1e-4, 5, 3, 1000, 3, -1, 1, period / 240)

        t0 = time.time()
        S, _ = smatrix(clocs, cmmaxs, cepmus, crads, period, wavelength,
                       nmax, d, sp, 'On')

        # Cascade with itself
        Scas, dcas = cascadertwo(S, d, S, d)

        # Truncate evanescent modes
        nm = 2 * nmax + 1
        if NoEvaMode > 0:
            Scas11 = Scas[:nm, :nm][NoEvaMode:-NoEvaMode, NoEvaMode:-NoEvaMode]
            Scas12 = Scas[:nm, nm:][NoEvaMode:-NoEvaMode, NoEvaMode:-NoEvaMode]
            Scas21 = Scas[nm:, :nm][NoEvaMode:-NoEvaMode, NoEvaMode:-NoEvaMode]
            Scas22 = Scas[nm:, nm:][NoEvaMode:-NoEvaMode, NoEvaMode:-NoEvaMode]
            ScasTruncated = np.block([[Scas11, Scas12], [Scas21, Scas22]])
        else:
            ScasTruncated = Scas

        elapsed = time.time() - t0

        DOU_cas = np.sum(np.abs(ScasTruncated.conj().T @ ScasTruncated)) / len(ScasTruncated)
        nm_trunc = ScasTruncated.shape[0] // 2
        e_cas = np.linalg.svd(ScasTruncated[nm_trunc:, :nm_trunc], compute_uv=False)**2

        frob_error = np.linalg.norm(S_truth - ScasTruncated, 'fro')
        trans_error = np.linalg.norm(e_truth - e_cas)

        results.append({
            'tol': tol, 'NoEvaMode': NoEvaMode, 'nmax': nmax,
            'DOU': DOU_cas, 'frob_error': frob_error,
            'trans_error': trans_error, 'time': elapsed
        })

        print(f"  DOU={DOU_cas:.10f}, FrobError={frob_error:.6e}, "
              f"TransError={trans_error:.6e}, time={elapsed:.1f}s")

    # Summary table
    print(f"\n{'=' * 80}")
    print(f"{'Tol':>10s} {'NoEva':>6s} {'nmax':>5s} {'DOU':>14s} "
          f"{'FrobError':>12s} {'TransError':>12s} {'Time':>6s}")
    print("-" * 80)
    for r in results:
        print(f"{r['tol']:>10.0e} {r['NoEvaMode']:>6d} {r['nmax']:>5d} "
              f"{r['DOU']:>14.10f} {r['frob_error']:>12.6e} "
              f"{r['trans_error']:>12.6e} {r['time']:>6.1f}s")
    print("=" * 80)

    # Plot convergence
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    tols = [r['tol'] for r in results]
    frobs = [r['frob_error'] for r in results]
    trans = [r['trans_error'] for r in results]

    axes[0].semilogy(range(len(tols)), frobs, 'b-o')
    axes[0].set_xticks(range(len(tols)))
    axes[0].set_xticklabels([f'{t:.0e}' for t in tols], rotation=45)
    axes[0].set_xlabel('Evanescent Mode Tolerance')
    axes[0].set_ylabel('Frobenius Norm Error')
    axes[0].set_title('Cascade vs Direct: Frobenius Error')
    axes[0].grid(True)

    axes[1].semilogy(range(len(tols)), trans, 'r-o')
    axes[1].set_xticks(range(len(tols)))
    axes[1].set_xticklabels([f'{t:.0e}' for t in tols], rotation=45)
    axes[1].set_xlabel('Evanescent Mode Tolerance')
    axes[1].set_ylabel('Trans Coeff Error')
    axes[1].set_title('Cascade vs Direct: Transmission Error')
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('result_evanescent_convergence.png', dpi=150)
    print("Plot saved to result_evanescent_convergence.png")


if __name__ == '__main__':
    main()
