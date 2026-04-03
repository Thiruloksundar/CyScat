"""
MainIncreasingBuffer - Test buffer extension accuracy
Translated from MainIncreasingBuffer.m (Curtis Jin, 2011)

Tests the IncreaseBuffer function which artificially extends the slab
buffer region by cascading with an empty slab. Compares the extended
S-matrix against a direct recomputation with the larger buffer.

Usage:
    python main_increasing_buffer.py
"""
import sys
import time
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, './Scattering_Code')

from Scattering_Code.smatrix_parameters import smatrix_parameters
from Scattering_Code.smatrix import smatrix
from Scattering_Code.cascadertwo import cascadertwo
from Scattering_Code.ky import ky


def increase_buffer(S, d, d_increase, wavelength, period, nmax):
    """
    Increase the buffer (slab thickness) by cascading with an empty slab.

    Translated from IncreaseBuffer.m:
    Creates an identity-like S-matrix for free space of thickness d_increase,
    then cascades it with the original S-matrix.

    Parameters
    ----------
    S : (2*nm, 2*nm) complex — original S-matrix
    d : float — original thickness
    d_increase : float — additional thickness
    wavelength : float
    period : float
    nmax : int

    Returns
    -------
    S_mod : modified S-matrix with increased buffer
    d_mod : new total thickness
    """
    k = 2 * np.pi / wavelength
    nm = 2 * nmax + 1

    # Build free-space S-matrix for empty slab of thickness d_increase
    # S11 = S22 = 0 (no reflection), S12 = S21 = phase propagation
    S_free = np.zeros((2 * nm, 2 * nm), dtype=complex)

    m_indices = np.arange(-nmax, nmax + 1)
    kxs = 2 * np.pi / period * m_indices
    kys_val = ky(k, kxs)

    # Transmission: exp(-i * ky * d_increase) for each mode
    phase = np.exp(-1j * kys_val * d_increase)

    # Normalize like smatrix does: P2 @ S @ P1
    nor2 = np.sqrt(kys_val / k)
    nor1 = np.sqrt(k / kys_val)

    # S21 and S12 blocks: diagonal with phase * normalization
    for idx in range(nm):
        # S21: lower-left block
        S_free[nm + idx, idx] = phase[idx] * nor2[idx] * nor1[idx]
        # S12: upper-right block
        S_free[idx, nm + idx] = phase[idx] * nor2[idx] * nor1[idx]

    # For properly normalized S-matrix, S12=S21=diag(phase), S11=S22=0
    # The normalization factors cancel: nor2 * nor1 = sqrt(ky/k) * sqrt(k/ky) = 1
    S_free = np.zeros((2 * nm, 2 * nm), dtype=complex)
    for idx in range(nm):
        S_free[nm + idx, idx] = phase[idx]  # S21
        S_free[idx, nm + idx] = phase[idx]  # S12

    S_mod, d_mod = cascadertwo(S, d, S_free, d_increase)
    return S_mod, d_mod


def main():
    # Parameters matching MATLAB MainIncreasingBuffer.m
    NoCylinders = 16
    wavelength = 0.93
    Width = 18
    period = Width + 3.31
    nmax = int(np.floor(period / wavelength))
    phiinc = np.pi / 2

    # Generate simple test positions
    np.random.seed(42)
    clocs = np.column_stack([
        np.random.uniform(1, period - 1, NoCylinders),
        np.random.uniform(0.5, 9.5, NoCylinders)
    ])
    cepmus = np.column_stack([2.0 * np.ones(NoCylinders), np.ones(NoCylinders)])
    crads = 0.1 * np.ones(NoCylinders)
    cmmaxs = 3 * np.ones(NoCylinders, dtype=int)

    CascadeTol = 8
    d = 10 + CascadeTol * np.log(10) / np.sqrt(
        (2 * np.pi / period * (nmax + 1))**2 - (2 * np.pi / wavelength)**2
    )

    print("=" * 60)
    print(f"Buffer Extension Test: {NoCylinders} cylinders")
    print(f"  period={period:.2f}, nmax={nmax}, d={d:.3f}")
    print("=" * 60)

    sp = smatrix_parameters(wavelength, period, phiinc,
                            1e-11, 1e-4, 5, 3, 1000, 3, -1, 1, period / 240)

    # Compute original S-matrix
    print("\n1) Computing S-matrix (original buffer)...")
    t0 = time.time()
    S, _ = smatrix(clocs, cmmaxs, cepmus, crads, period, wavelength,
                   nmax, d, sp, 'On')
    print(f"   Done in {time.time() - t0:.1f}s")

    # Increase buffer
    d_increase = 5.0
    print(f"\n2) Increasing buffer by {d_increase}...")
    S_mod, d_mod = increase_buffer(S, d, d_increase, wavelength, period, nmax)

    # Compute S-matrix directly with the larger buffer
    print(f"\n3) Computing S-matrix directly (d={d_mod:.3f})...")
    t0 = time.time()
    S_actual, _ = smatrix(clocs, cmmaxs, cepmus, crads, period, wavelength,
                          nmax, d_mod, sp, 'On')
    print(f"   Done in {time.time() - t0:.1f}s")

    # Compare
    checker1 = np.abs(S_actual.conj().T @ S_actual)
    DOU_actual = np.sum(checker1) / len(S_actual)
    nm = 2 * nmax + 1
    e_actual = np.linalg.svd(S_actual[nm:, :nm], compute_uv=False)**2

    checker2 = np.abs(S_mod.conj().T @ S_mod)
    DOU_mod = np.sum(checker2) / len(S_mod)
    e_mod = np.linalg.svd(S_mod[nm:, :nm], compute_uv=False)**2

    frob_error = np.linalg.norm(S_actual - S_mod, 'fro')
    trans_error = np.linalg.norm(e_actual - e_mod)

    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"  DOU (direct):         {DOU_actual:.10f}")
    print(f"  DOU (buffer extend):  {DOU_mod:.10f}")
    print(f"  Frobenius Norm Error: {frob_error:.6e}")
    print(f"  Trans Coeff Error:    {trans_error:.6e}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
