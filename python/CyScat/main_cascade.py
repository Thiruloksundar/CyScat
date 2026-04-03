"""
MainCascade - Test cascade accuracy
Translated from MainCascade.m (Curtis Jin, 2011)

Tests that cascading two identical S-matrices gives the same result as
computing the S-matrix for a doubled slab directly.
Compares: Frobenius norm error, transmission coefficient error, DOU.

Usage:
    python main_cascade.py
    python main_cascade.py --num_cyl 30 --width 30
"""
import sys
import os
import time
import argparse
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, './Scattering_Code')

from Scattering_Code.smatrix_parameters import smatrix_parameters
from Scattering_Code.smatrix import smatrix
from Scattering_Code.cascadertwo import cascadertwo


def main():
    parser = argparse.ArgumentParser(description='Test cascade accuracy')
    parser.add_argument('--num_cyl', type=int, default=15)
    parser.add_argument('--width', type=float, default=None,
                        help='Period width (default: num_cyl + 2 + 3.31)')
    parser.add_argument('--eva_tol', type=float, default=1e-11,
                        help='Evanescent mode tolerance')
    args = parser.parse_args()

    NoCyl = args.num_cyl
    wavelength = 0.93
    period = args.width if args.width else NoCyl + 2 + 3.31
    radius = 0.1
    epsilon = 2.0
    mu = 1.0
    cmmax = 3
    phiinc = np.pi / 2

    # Cylinder positions: evenly spaced in x, all at y=0.2
    xpos = np.arange(1, NoCyl + 1, dtype=float)
    clocs = np.column_stack([xpos, 0.2 * np.ones(NoCyl)])
    cepmus = np.column_stack([epsilon * np.ones(NoCyl), np.ones(NoCyl)])
    crads = radius * np.ones(NoCyl)
    cmmaxs = cmmax * np.ones(NoCyl, dtype=int)

    d = 0.4  # slab thickness
    NoPropagatingModes = int(np.floor(period / wavelength))

    # Evanescent mode calculation
    tol = args.eva_tol
    Bufferlength = d - np.max(np.abs(clocs[:, 1])) + 0.2
    NoEvaMode = int(np.floor(
        period / (2 * np.pi) * np.sqrt(
            (np.log(tol) / Bufferlength)**2 + (2 * np.pi / wavelength)**2
        )
    )) - NoPropagatingModes
    NoEvaMode = max(NoEvaMode, 0)
    nmax = NoPropagatingModes + NoEvaMode

    print("=" * 60)
    print(f"Cascade Test: {NoCyl} cylinders")
    print(f"  period={period:.2f}, wavelength={wavelength}")
    print(f"  NoPropModes={NoPropagatingModes}, NoEvaMode={NoEvaMode}, nmax={nmax}")
    print(f"  thickness d={d}")
    print("=" * 60)

    sp = smatrix_parameters(wavelength, period, phiinc,
                            1e-11, 1e-4, 5, 3, 1000, 3, -1, 1, period / 240)

    # === Compute S-matrix for single slab ===
    print("\n1) Computing S-matrix for single slab...")
    t0 = time.time()
    S, STP = smatrix(clocs, cmmaxs, cepmus, crads, period, wavelength,
                     nmax, d, sp, 'On')
    print(f"   Done in {time.time() - t0:.1f}s")

    # === Cascade S with itself (two identical slabs) ===
    print("\n2) Cascading S with itself...")
    Scas, dcas = cascadertwo(S, d, S, d)

    # Truncate evanescent modes from cascaded result
    nm = 2 * nmax + 1
    if NoEvaMode > 0:
        Scas11 = Scas[:nm, :nm][NoEvaMode:-NoEvaMode, NoEvaMode:-NoEvaMode]
        Scas12 = Scas[:nm, nm:][NoEvaMode:-NoEvaMode, NoEvaMode:-NoEvaMode]
        Scas21 = Scas[nm:, :nm][NoEvaMode:-NoEvaMode, NoEvaMode:-NoEvaMode]
        Scas22 = Scas[nm:, nm:][NoEvaMode:-NoEvaMode, NoEvaMode:-NoEvaMode]
        ScasTruncated = np.block([[Scas11, Scas12], [Scas21, Scas22]])
    else:
        ScasTruncated = Scas

    # === Compute S-matrix for doubled slab directly ===
    print("\n3) Computing S-matrix for doubled slab directly...")
    offset = np.column_stack([np.zeros(NoCyl), d * np.ones(NoCyl)])
    clocs2 = np.vstack([clocs, clocs + offset])
    cepmus2 = np.vstack([cepmus, cepmus])
    crads2 = np.concatenate([crads, crads])
    cmmaxs2 = np.concatenate([cmmaxs, cmmaxs])
    d2 = d * 2
    nmax_single = NoPropagatingModes  # no evanescent for direct comparison

    t0 = time.time()
    S_direct, _ = smatrix(clocs2, cmmaxs2, cepmus2, crads2, period, wavelength,
                          nmax_single, dcas, sp, 'On')
    print(f"   Done in {time.time() - t0:.1f}s")

    # === Compare ===
    checker1 = np.abs(S_direct.conj().T @ S_direct)
    DOU_single = np.sum(checker1) / len(S_direct)
    e_single = np.linalg.svd(S_direct[nm:, :nm] if NoEvaMode == 0
                             else S_direct[NoPropagatingModes + 1:, :NoPropagatingModes + 1],
                             compute_uv=False)**2

    # For direct S-matrix (no evanescent modes), extract S21
    nm_direct = 2 * nmax_single + 1
    e_single = np.linalg.svd(S_direct[nm_direct:, :nm_direct], compute_uv=False)**2

    checker2 = np.abs(ScasTruncated.conj().T @ ScasTruncated)
    DOU_cas = np.sum(checker2) / len(ScasTruncated)
    nm_trunc = ScasTruncated.shape[0] // 2
    e_cas = np.linalg.svd(ScasTruncated[nm_trunc:, :nm_trunc], compute_uv=False)**2

    frob_error = np.linalg.norm(S_direct - ScasTruncated, 'fro')
    trans_error = np.linalg.norm(e_single - e_cas)

    print("\n" + "=" * 60)
    print("RESULTS")
    print(f"  DOU (direct):   {DOU_single:.10f}")
    print(f"  DOU (cascaded): {DOU_cas:.10f}")
    print(f"  Frobenius Norm Error: {frob_error:.6e}")
    print(f"  Trans Coeff Error:    {trans_error:.6e}")
    print("=" * 60)


if __name__ == '__main__':
    main()
