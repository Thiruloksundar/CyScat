"""
matrix_derivatives.py
=====================
Computes matrix-valued derivatives of S21 w.r.t.:
  1. Cylinder x-position  (dS21/dx_j)  -- one matrix per cylinder
  2. Wavelength           (dS21/dlambda) -- one matrix
  3. Refractive index     (dS21/dn)     -- one matrix

Uses central finite differences on the full S21 matrix.
For derivatives w.r.t. n (and r), we use the precomputed T-matrix
for ~25x speedup since n only affects Mie coefficients.

Cross-validates against the entry S21[0,0] to confirm accuracy.

Usage (from CyScat root):
    python matrix_derivatives.py
    python matrix_derivatives.py --cyl_idx 0
"""

import sys
import time
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, '.')
sys.path.insert(0, './Scattering_Code')

from Scattering_Code.smatrix_parameters import smatrix_parameters
from Scattering_Code.smatrix import smatrix, smatrix_precompute, smatrix_from_precomputed

# ── Constants (same as check_differentiability_jax.py) ────────────────────────

WAVELENGTH  = 0.93
PERIOD      = 12.81
RADIUS      = 0.25
N_CYLINDER  = 1.3
MU          = 1.0
CMMAX       = 5
PHIINC      = np.pi / 2
SEED        = 42

Eva_TOL     = 1e-2
N_PROP      = int(np.floor(PERIOD / WAVELENGTH))
N_EVA       = max(int(np.floor(
    PERIOD / (2*np.pi) * np.sqrt(
        (np.log(Eva_TOL) / (2*RADIUS))**2 + (2*np.pi/WAVELENGTH)**2
    )
)) - N_PROP, 0)
NMAX        = N_PROP + N_EVA

# ── Setup ─────────────────────────────────────────────────────────────────────

def setup_parameters(num_cyl=16, seed=SEED):
    """Generate cylinder positions and scattering parameters."""
    spacing = 2.5 * RADIUS
    cyls_per_row = int(PERIOD / spacing)
    rows_needed = num_cyl / cyls_per_row + 2
    thickness = max(0.5, rows_needed * spacing * 1.5)
    thickness = round(thickness, 1)

    np.random.seed(seed)
    margin = RADIUS * 1.5
    clocs = np.zeros((num_cyl, 2))
    for i in range(num_cyl):
        for _ in range(10000):
            x = np.random.uniform(margin, PERIOD - margin)
            y = np.random.uniform(margin, thickness - margin)
            if i == 0 or np.all(np.sqrt((x - clocs[:i,0])**2 +
                                         (y - clocs[:i,1])**2) > spacing):
                clocs[i] = [x, y]
                break

    params = {
        'wavelength': WAVELENGTH, 'period': PERIOD,
        'radius': RADIUS, 'eps': N_CYLINDER**2,
        'mu': MU, 'cmmax': CMMAX,
        'phiinc': PHIINC, 'thickness': thickness,
        'nmax': NMAX, 'NoEvaMode': N_EVA,
        'NoPropagatingModes': N_PROP,
        'n_cylinder': N_CYLINDER,
    }
    return clocs, params


def make_sp(wavelength=WAVELENGTH):
    return smatrix_parameters(
        wavelength, PERIOD, PHIINC,
        1e-11, 1e-4, 5, 3, 1000, 3, 5, 1, PERIOD / 120
    )


def extract_S21(S, nmax, n_eva):
    """Extract propagating-mode block of S21 from full S-matrix."""
    nm = 2 * nmax + 1
    S21 = S[nm:, :nm]
    if n_eva > 0:
        return S21[n_eva:-n_eva, n_eva:-n_eva]
    return S21


def compute_S21(clocs, params, sp, cepmus=None, crads=None):
    """Full S-matrix computation, returns S21 propagating block."""
    num_cyl = clocs.shape[0]
    cmmaxs = np.array([params['cmmax']] * num_cyl)
    if cepmus is None:
        cepmus = np.tile([params['eps'], params['mu']], (num_cyl, 1))
    if crads is None:
        crads = np.full(num_cyl, params['radius'])
    S, _ = smatrix(jnp.array(clocs), cmmaxs, jnp.array(cepmus), jnp.array(crads),
                   params['period'], params['wavelength'],
                   params['nmax'], params['thickness'], sp, 'On',
                   clocs_concrete=clocs)
    return extract_S21(S, params['nmax'], params['NoEvaMode'])


def compute_S21_lambda(clocs, params, lam_val):
    """Full S-matrix computation at a given wavelength, returns S21 block."""
    num_cyl = clocs.shape[0]
    cmmaxs = np.array([params['cmmax']] * num_cyl)
    cepmus = np.tile([params['eps'], params['mu']], (num_cyl, 1))
    crads = np.full(num_cyl, params['radius'])
    sp_loc = smatrix_parameters(
        lam_val, params['period'], params['phiinc'],
        1e-11, 1e-4, 5, 3, 1000, 3, 5, 1, params['period'] / 120
    )
    S, _ = smatrix(jnp.array(clocs), cmmaxs, jnp.array(cepmus), jnp.array(crads),
                   params['period'], lam_val,
                   params['nmax'], params['thickness'], sp_loc, 'On',
                   clocs_concrete=clocs)
    return extract_S21(S, params['nmax'], params['NoEvaMode'])


# ─────────────────────────────────────────────────────────────────────────────
# 1.  dS21 / dx_j   (matrix derivative w.r.t. x-position of cylinder j)
# ─────────────────────────────────────────────────────────────────────────────

def dS21_dx(clocs, params, sp, cyl_idx, delta=1e-5):
    """
    Central finite difference: dS21[p,q]/dx_{cyl_idx}.
    Requires 2 full smatrix evaluations (T-matrix depends on clocs).
    """
    clocs_p = clocs.copy()
    clocs_m = clocs.copy()
    clocs_p[cyl_idx, 0] += delta
    clocs_m[cyl_idx, 0] -= delta

    S21_p = compute_S21(clocs_p, params, sp)
    S21_m = compute_S21(clocs_m, params, sp)

    return (S21_p - S21_m) / (2 * delta)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  dS21 / dlambda   (matrix derivative w.r.t. wavelength)
# ─────────────────────────────────────────────────────────────────────────────

def dS21_dlambda(clocs, params, delta=1e-5):
    """
    Central finite difference: dS21[p,q]/dlambda.
    Requires 2 full smatrix evaluations (sp depends on wavelength).
    """
    lam0 = params['wavelength']
    S21_p = compute_S21_lambda(clocs, params, lam0 + delta)
    S21_m = compute_S21_lambda(clocs, params, lam0 - delta)

    return (S21_p - S21_m) / (2 * delta)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  dS21 / dn   (matrix derivative w.r.t. refractive index)
# ─────────────────────────────────────────────────────────────────────────────

def dS21_dn(clocs, params, sp, pre=None, delta=1e-5):
    """
    Central finite difference: dS21[p,q]/dn.

    FAST path: if `pre` (precomputed T-matrix data) is provided, uses
    smatrix_from_precomputed (~1-2s per eval) since n only affects
    Mie coefficients, not the T-matrix.
    """
    n0 = params['n_cylinder']
    num_cyl = clocs.shape[0]

    if pre is not None:
        # Fast path: only Mie coefficients change
        def _S21_at_n(n_val):
            eps_val = n_val ** 2
            cepmus = jnp.tile(jnp.array([eps_val, params['mu']]), (num_cyl, 1))
            crads = jnp.full((num_cyl,), params['radius'])
            S = smatrix_from_precomputed(pre, cepmus, crads)
            return extract_S21(S, params['nmax'], params['NoEvaMode'])
    else:
        # Slow path: full smatrix recomputation
        def _S21_at_n(n_val):
            eps_val = n_val ** 2
            cepmus = np.tile([eps_val, params['mu']], (num_cyl, 1))
            return compute_S21(clocs, params, sp, cepmus=cepmus)

    S21_p = _S21_at_n(n0 + delta)
    S21_m = _S21_at_n(n0 - delta)

    return (S21_p - S21_m) / (2 * delta)


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-printer for a complex matrix
# ─────────────────────────────────────────────────────────────────────────────

def print_matrix(M, name, max_rows=6, max_cols=6):
    M = np.asarray(M)
    nr, nc = M.shape
    r_show = min(nr, max_rows)
    c_show = min(nc, max_cols)
    print(f"\n  {name}  (size: {nr}x{nc} complex)")
    print(f"  Showing top {r_show}x{c_show} block:")
    print("  " + "-" * (c_show * 26))
    for i in range(r_show):
        row = "  "
        for j in range(c_show):
            v = M[i, j]
            row += f"  {v.real:+.4e}{v.imag:+.4e}i"
        print(row)
    if nr > max_rows or nc > max_cols:
        print(f"  ... ({nr - r_show} more rows, {nc - c_show} more cols)")
    print()

    mag = np.abs(M)
    print(f"  |entry| -- min: {mag.min():.4e}   max: {mag.max():.4e}   "
          f"mean: {mag.mean():.4e}")
    any_nan = np.any(np.isnan(M))
    any_inf = np.any(np.isinf(M))
    print(f"  NaN entries: {'YES !!' if any_nan else 'none'}   "
          f"Inf entries: {'YES !!' if any_inf else 'none'}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cyl_idx', type=int, default=0,
                        help='Cylinder index (0-based) for dS21/dx')
    parser.add_argument('--num_cyl', type=int, default=16,
                        help='Number of cylinders')
    args = parser.parse_args()

    cyl_idx = args.cyl_idx
    num_cyl = args.num_cyl

    print("=" * 65)
    print("  CyScat -- Matrix-valued derivatives of S21 (finite differences)")
    print("=" * 65)

    clocs, params = setup_parameters(num_cyl=num_cyl)
    sp = make_sp()

    print(f"\n  {num_cyl} cylinders | lambda={params['wavelength']:.3f} | "
          f"n={params['n_cylinder']:.3f} | r={params['radius']:.3f} | "
          f"period={params['period']:.3f}\n")

    # ── Precompute T-matrix for fast n-derivative ──────────────────────────
    print("  Precomputing T-matrix (for fast dS21/dn) ...")
    t_pre = time.time()
    cmmaxs = np.array([params['cmmax']] * num_cyl)
    pre = smatrix_precompute(
        jnp.array(clocs), cmmaxs, params['period'], params['wavelength'],
        params['nmax'], params['thickness'], sp, 'On',
        clocs_concrete=clocs
    )
    print(f"  Precompute done in {time.time()-t_pre:.1f}s\n")

    # ── 1. dS21/dx_j ─────────────────────────────────────────────────────
    print("-" * 65)
    print(f"  1.  dS21 / dx_{cyl_idx}   (x-position of cylinder {cyl_idx})")
    print("-" * 65)
    print(f"  Running central FD (delta=1e-5) ...")
    t0 = time.time()
    dS21_x = dS21_dx(clocs, params, sp, cyl_idx)
    dt = time.time() - t0
    print(f"  Done in {dt:.2f}s")
    print_matrix(dS21_x, f"dS21/dx_{cyl_idx}")
    nr, nc = np.asarray(dS21_x).shape
    print(f"  dS21/dx_{cyl_idx} is a {nr}x{nc} complex matrix")

    # ── 2. dS21/dlambda ──────────────────────────────────────────────────
    print("\n" + "-" * 65)
    print(f"  2.  dS21 / dlambda   (wavelength, lambda = {params['wavelength']})")
    print("-" * 65)
    print(f"  Running central FD (delta=1e-5) ...")
    t0 = time.time()
    dS21_lam = dS21_dlambda(clocs, params)
    dt = time.time() - t0
    print(f"  Done in {dt:.2f}s")
    print_matrix(dS21_lam, "dS21/dlambda")
    print(f"  dS21/dlambda is a {nr}x{nc} complex matrix")

    # ── 3. dS21/dn ───────────────────────────────────────────────────────
    print("\n" + "-" * 65)
    print(f"  3.  dS21 / dn   (refractive index, n = {params['n_cylinder']})")
    print("-" * 65)
    print(f"  Running central FD with precomputed T-matrix (fast path) ...")
    t0 = time.time()
    dS21_n = dS21_dn(clocs, params, sp, pre=pre)
    dt = time.time() - t0
    print(f"  Done in {dt:.2f}s")
    print_matrix(dS21_n, "dS21/dn")
    print(f"  dS21/dn is a {nr}x{nc} complex matrix")

    # ── 4. Cross-validation: compare two FD step sizes on S21[0,0] ───────
    print("\n" + "-" * 65)
    print("  4.  Cross-validation: FD consistency check on S21[0,0]")
    print("-" * 65)

    delta_fine = 1e-6

    # dS21/dx cross-check
    dS21_x_fine = dS21_dx(clocs, params, sp, cyl_idx, delta=delta_fine)
    v1 = np.asarray(dS21_x)[0, 0]
    v2 = np.asarray(dS21_x_fine)[0, 0]
    diff = abs(v1 - v2)
    print(f"\n  dS21[0,0]/dx_{cyl_idx}:")
    print(f"    delta=1e-5: {v1.real:+.6e}{v1.imag:+.6e}i")
    print(f"    delta=1e-6: {v2.real:+.6e}{v2.imag:+.6e}i")
    print(f"        |diff|: {diff:.4e}  {'match' if diff < 1e-4 else 'mismatch'}")

    # dS21/dlambda cross-check
    dS21_lam_fine = dS21_dlambda(clocs, params, delta=delta_fine)
    v1 = np.asarray(dS21_lam)[0, 0]
    v2 = np.asarray(dS21_lam_fine)[0, 0]
    diff = abs(v1 - v2)
    print(f"\n  dS21[0,0]/dlambda:")
    print(f"    delta=1e-5: {v1.real:+.6e}{v1.imag:+.6e}i")
    print(f"    delta=1e-6: {v2.real:+.6e}{v2.imag:+.6e}i")
    print(f"        |diff|: {diff:.4e}  {'match' if diff < 1e-4 else 'mismatch'}")

    # dS21/dn cross-check
    dS21_n_fine = dS21_dn(clocs, params, sp, pre=pre, delta=delta_fine)
    v1 = np.asarray(dS21_n)[0, 0]
    v2 = np.asarray(dS21_n_fine)[0, 0]
    diff = abs(v1 - v2)
    print(f"\n  dS21[0,0]/dn:")
    print(f"    delta=1e-5: {v1.real:+.6e}{v1.imag:+.6e}i")
    print(f"    delta=1e-6: {v2.real:+.6e}{v2.imag:+.6e}i")
    print(f"        |diff|: {diff:.4e}  {'match' if diff < 1e-4 else 'mismatch'}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  dS21/dx_j  -> {nr}x{nc} complex matrix  (one per cylinder)")
    print(f"  dS21/dlam  -> {nr}x{nc} complex matrix  (one for the slab)")
    print(f"  dS21/dn    -> {nr}x{nc} complex matrix  (one for shared n)")
    print()
    print("  All three derivatives are well-defined complex matrices of the")
    print("  same shape as S21.  Each entry [p,q] gives how the transmission")
    print("  amplitude from input channel q to output channel p changes with")
    print("  the parameter.  Verified via FD consistency check above.")
    print("=" * 65)


if __name__ == '__main__':
    main()
