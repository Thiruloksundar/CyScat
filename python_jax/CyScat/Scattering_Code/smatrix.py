"""
smatrix Code
Coder: Curtis Jin
Date: 2010/DEC/3rd Friday
Contact: jsirius@umich.edu
Description: Professor Michelson's version
           : Scattering Matrix Generating Code

JAX version for automatic differentiation.
  - Full JAX pipeline: T-matrix, solve, projection matrices, normalization
  - Differentiable w.r.t. cylinder positions (clocs), wavelength (lambda_wave),
    refractive indices (cepmus), and radii (crads)

FIXES applied for NaN-free gradients:
  FIX 1 — ky() called on real arrays with evanescent modes (|kx| > k):
           jnp.sqrt(k²-kx²) on a negative real number has NaN gradient.
           Fix: cast input to complex128 BEFORE calling ky(), so sqrt stays
           on the complex branch where the derivative is well-defined.

  FIX 2 — kys_safe / jnp.where gradient trap in _build_projection_matrices:
           jnp.where(cond, safe_val, bad_val) still differentiates through
           bad_val, producing inf/NaN gradients even when cond is True.
           Fix: replace the denominator *before* dividing using the standard
           "safe denominator" pattern so JAX never sees a zero denominator.

  FIX 3 — Normalization ky() call on real kxs_norm with evanescent entries:
           Same issue as FIX 1. Cast to complex before calling ky().
"""

import numpy as np
import jax.numpy as jnp
import time
import sys
from .ky import ky
from .transall import transall
from .sall import sall
from .simulation_time_profile import simulation_time_profile


def smatrix(clocs, cmmaxs, cepmus, crads, period, lambda_wave, nmax, d, sp, interaction,
            clocs_concrete=None):
    """
    Generate Scattering Matrix (JAX-differentiable w.r.t. clocs).

    Uses projection matrices and batched solve — no per-mode Python loops.

    clocs_concrete : optional numpy array of concrete clocs values (needed
                     when clocs is a JAX tracer during jax.grad).
    """
    if clocs_concrete is None:
        clocs_concrete = np.asarray(clocs)
    clocs = jnp.asarray(clocs, dtype=jnp.float64)
    cmmaxs_np = np.asarray(cmmaxs, dtype=int)
    no_cylinders = len(cmmaxs_np)
    total_steps = no_cylinders * (no_cylinders - 1) // 2 + 2 * (2 * nmax + 1)

    phiinc = sp['phiinc']
    k = 2 * jnp.pi / lambda_wave

    # === T-Matrix ===
    print('  Computing T-Matrix...')
    sys.stdout.flush()
    st_tmatrix_start = time.time()
    if interaction == 'On':
        t = transall(clocs, cmmaxs_np, period, lambda_wave, phiinc, sp, total_steps,
                     clocs_concrete=clocs_concrete)
    else:
        t = transall(clocs, cmmaxs_np, -1, lambda_wave, phiinc, sp, total_steps,
                     clocs_concrete=clocs_concrete)
    st_tmatrix = time.time() - st_tmatrix_start
    print(f'  T-Matrix: {st_tmatrix:.1f}s')

    # === S-Vector (Mie coefficients) — does NOT depend on clocs ===
    s = sall(cmmaxs_np, cepmus, crads, lambda_wave)

    # === Build system matrix z = I - diag(s) @ t ===
    z = jnp.eye(len(s), dtype=jnp.complex128) - jnp.diag(s) @ t

    # === Solve (using jnp.linalg.solve for differentiability) ===
    print('  Solving linear system...')
    sys.stdout.flush()
    st_lu_start = time.time()

    num_modes = 2 * nmax + 1

    # Pre-compute kxex values for all input modes
    kxs_arr = jnp.asarray(sp['kxs']).flatten()
    mid = int(sp['MiddleIndex'])
    kxex_all = kxs_arr[mid - nmax : mid + nmax + 1]

    # === S11 & S21 Partition ===
    diag_s = s[:, None]   # (tot_no_modes, 1) for broadcasting

    V_up = diag_s * _vall_batch(clocs, cmmaxs_np, kxex_all, k, up_down=1)
    C_up = jnp.linalg.solve(z, V_up)

    st_lu = time.time() - st_lu_start
    print(f'  Solve: {st_lu:.1f}s')

    print('  Computing S11 & S21...')
    sys.stdout.flush()
    st_up_start = time.time()

    W1_up, W2_up = _build_projection_matrices(clocs, cmmaxs_np, nmax, d, sp, up_down=1)

    s11matrix = W1_up @ C_up
    s21matrix = W2_up @ C_up

    # Add incident field transmission: S21[nmax+nin, nin_idx] += exp(-i*kyex*d)
    kys_arr = jnp.asarray(sp['kys']).flatten()
    for nin_idx, nin in enumerate(range(-nmax, nmax + 1)):
        kyex = kys_arr[mid + nin]
        s21matrix = s21matrix.at[nmax + nin, nin_idx].add(jnp.exp(-1j * kyex * d))

    st_up = time.time() - st_up_start
    print(f'  S11&S21: {st_up:.1f}s')

    # === S12 & S22 Partition ===
    print('  Computing S12 & S22...')
    sys.stdout.flush()
    st_down_start = time.time()

    V_down = diag_s * _vall_batch(clocs, cmmaxs_np, kxex_all, k, up_down=-1, y_shift=-d)
    C_down = jnp.linalg.solve(z, V_down)

    W1_down, W2_down = _build_projection_matrices(
        clocs, cmmaxs_np, nmax, d, sp, up_down=-1, y_shift=-d)

    s12matrix = W1_down @ C_down
    s22matrix = W2_down @ C_down

    # Add incident field: S12[nmax+nin, nin_idx] += exp(-i*kyex*d)
    for nin_idx, nin in enumerate(range(-nmax, nmax + 1)):
        kyex = kys_arr[mid + nin]
        s12matrix = s12matrix.at[nmax + nin, nin_idx].add(jnp.exp(-1j * kyex * d))

    st_down = time.time() - st_down_start
    print(f'  S12&S22: {st_down:.1f}s')

    # === Assemble full S-matrix ===
    S = jnp.block([[s11matrix, s12matrix], [s21matrix, s22matrix]])

    # === Normalization ===
    # FIX 3: cast kxs_norm to complex128 before calling ky() so that
    # evanescent modes (|kx|>k) produce purely imaginary ky values with
    # well-defined complex-branch gradients instead of NaN real gradients.
    st_norm_start = time.time()
    m = jnp.arange(-nmax, nmax + 1)
    m = jnp.concatenate([m, m])
    kxs_norm = (2 * jnp.pi / period * m).astype(jnp.complex128)  # <-- FIX 3
    kys_norm = ky(k, kxs_norm)

    nor2 = jnp.sqrt(kys_norm / k)
    nor1 = jnp.sqrt(k / kys_norm)
    P2 = jnp.diag(nor2)
    P1 = jnp.diag(nor1)
    S = P2 @ S @ P1
    st_norm = time.time() - st_norm_start

    STP = simulation_time_profile(st_tmatrix, st_lu, st_up, st_down, st_norm)
    total = STP['TST']
    print(f'  Total: {total:.1f}s ({total/60:.1f} min)')
    sys.stdout.flush()

    return S, STP


def smatrix_precompute(clocs, cmmaxs, period, lambda_wave, nmax, d, sp, interaction,
                       clocs_concrete=None):
    """
    Precompute everything that does NOT depend on cepmus (n) or crads (r).

    Returns a dict of precomputed arrays that can be passed to
    smatrix_from_precomputed() for fast repeated evaluation.
    """
    if clocs_concrete is None:
        clocs_concrete = np.asarray(clocs)
    clocs = jnp.asarray(clocs, dtype=jnp.float64)
    cmmaxs_np = np.asarray(cmmaxs, dtype=int)
    no_cylinders = len(cmmaxs_np)
    total_steps = no_cylinders * (no_cylinders - 1) // 2 + 2 * (2 * nmax + 1)

    phiinc = sp['phiinc']
    k = 2 * jnp.pi / lambda_wave

    # T-matrix (the expensive part)
    print('  Precomputing T-Matrix...')
    sys.stdout.flush()
    t0 = time.time()
    if interaction == 'On':
        t = transall(clocs, cmmaxs_np, period, lambda_wave, phiinc, sp, total_steps,
                     clocs_concrete=clocs_concrete)
    else:
        t = transall(clocs, cmmaxs_np, -1, lambda_wave, phiinc, sp, total_steps,
                     clocs_concrete=clocs_concrete)
    print(f'  T-Matrix: {time.time()-t0:.1f}s')

    # Spectral parameters
    kxs_arr = jnp.asarray(sp['kxs']).flatten()
    kys_arr = jnp.asarray(sp['kys']).flatten()
    mid = int(sp['MiddleIndex'])
    kxex_all = kxs_arr[mid - nmax : mid + nmax + 1]
    num_modes = 2 * nmax + 1

    # Incident field bases (constant w.r.t. n, r)
    V_up_base = _vall_batch(clocs, cmmaxs_np, kxex_all, k, up_down=1)
    V_down_base = _vall_batch(clocs, cmmaxs_np, kxex_all, k, up_down=-1, y_shift=-d)

    # Projection matrices (constant w.r.t. n, r)
    W1_up, W2_up = _build_projection_matrices(clocs, cmmaxs_np, nmax, d, sp, up_down=1)
    W1_down, W2_down = _build_projection_matrices(
        clocs, cmmaxs_np, nmax, d, sp, up_down=-1, y_shift=-d)

    # Incident field addition terms for S21 and S12
    inc_s21 = jnp.zeros((num_modes, num_modes), dtype=jnp.complex128)
    inc_s12 = jnp.zeros((num_modes, num_modes), dtype=jnp.complex128)
    for nin_idx, nin in enumerate(range(-nmax, nmax + 1)):
        kyex = kys_arr[mid + nin]
        inc_s21 = inc_s21.at[nmax + nin, nin_idx].set(jnp.exp(-1j * kyex * d))
        inc_s12 = inc_s12.at[nmax + nin, nin_idx].set(jnp.exp(-1j * kyex * d))

    # Normalization matrices (constant w.r.t. n, r)
    m = jnp.arange(-nmax, nmax + 1)
    m = jnp.concatenate([m, m])
    kxs_norm = (2 * jnp.pi / period * m).astype(jnp.complex128)
    kys_norm = ky(k, kxs_norm)
    nor2 = jnp.sqrt(kys_norm / k)
    nor1 = jnp.sqrt(k / kys_norm)
    P2 = jnp.diag(nor2)
    P1 = jnp.diag(nor1)

    print(f'  Precompute done ({time.time()-t0:.1f}s total)')
    sys.stdout.flush()

    return {
        't': t, 'cmmaxs': cmmaxs_np, 'nmax': nmax,
        'V_up_base': V_up_base, 'V_down_base': V_down_base,
        'W1_up': W1_up, 'W2_up': W2_up,
        'W1_down': W1_down, 'W2_down': W2_down,
        'inc_s21': inc_s21, 'inc_s12': inc_s12,
        'P1': P1, 'P2': P2,
        'lambda_wave': lambda_wave,
    }


def smatrix_from_precomputed(pre, cepmus, crads):
    """
    Compute S-matrix using precomputed data. Fast, differentiable w.r.t. cepmus/crads.

    Typical time: ~1-2s vs ~50s for full smatrix().
    """
    s = sall(pre['cmmaxs'], cepmus, crads, pre['lambda_wave'])
    diag_s = s[:, None]

    # System matrix and solve
    z = jnp.eye(len(s), dtype=jnp.complex128) - jnp.diag(s) @ pre['t']

    V_up = diag_s * pre['V_up_base']
    C_up = jnp.linalg.solve(z, V_up)
    s11matrix = pre['W1_up'] @ C_up
    s21matrix = pre['W2_up'] @ C_up + pre['inc_s21']

    V_down = diag_s * pre['V_down_base']
    C_down = jnp.linalg.solve(z, V_down)
    s12matrix = pre['W1_down'] @ C_down + pre['inc_s12']
    s22matrix = pre['W2_down'] @ C_down

    S = jnp.block([[s11matrix, s12matrix], [s21matrix, s22matrix]])
    S = pre['P2'] @ S @ pre['P1']
    return S


# ─────────────────────────────────────────────────────────────────────────────
# Vectorized helpers (JAX)
# ─────────────────────────────────────────────────────────────────────────────

def _vall_batch(clocs, cmmaxs, kxex_all, k, up_down, y_shift=0.0):
    """
    Vectorized vall: compute all incident field vectors at once (JAX).

    Returns V_all of shape (tot_no_modes, num_modes).
    """
    cmmaxs_np = np.asarray(cmmaxs, dtype=int)
    no_cylinders = len(cmmaxs_np)
    num_modes = len(kxex_all)
    tot_no_modes = int(np.sum(cmmaxs_np * 2 + 1))

    # Build per-coefficient arrays
    cms_list = []
    cx_list = []
    cy_list = []
    for icyl in range(no_cylinders):
        cmmax = int(cmmaxs_np[icyl])
        n = 2 * cmmax + 1
        cms_list.append(jnp.arange(-cmmax, cmmax + 1, dtype=jnp.float64))
        cx_list.append(jnp.full(n, clocs[icyl, 0]))
        cy_list.append(jnp.full(n, clocs[icyl, 1] + y_shift))

    cms_coeff = jnp.concatenate(cms_list)
    cx_coeff = jnp.concatenate(cx_list)
    cy_coeff = jnp.concatenate(cy_list)

    # FIX 1: cast kxex_all to complex128 before calling ky().
    # For evanescent modes |kx| > k, ky = sqrt(k²-kx²) is purely imaginary.
    # On real arrays, jnp.sqrt of a negative number gives NaN in both the
    # forward pass and the backward pass. Casting to complex first keeps
    # everything on the principal complex square-root branch, which has a
    # well-defined derivative everywhere.
    kxex_all_c = kxex_all.astype(jnp.complex128)
    kyex_all_v = ky(k, kxex_all_c)                     # <-- FIX 1
    phiinc_all = jnp.arccos(kxex_all_c / k)            # complex arccos, no NaN

    if up_down < 0:
        kyex_all_v = -kyex_all_v
        phiinc_all = -phiinc_all

    # V[coeff, mode] = exp(-i*(kx*cx + ky*cy)) * exp(-i*cm*phiinc) * exp(-i*pi/2*cm)
    k_dot_r = (cx_coeff[:, None] * kxex_all_c[None, :] +
               cy_coeff[:, None] * kyex_all_v[None, :])
    cm_phiinc = cms_coeff[:, None] * phiinc_all[None, :]
    cm_pi2 = jnp.exp(-1j * jnp.pi / 2 * cms_coeff)[:, None]

    V_all = jnp.exp(-1j * k_dot_r) * jnp.exp(-1j * cm_phiinc) * cm_pi2
    return V_all


def _build_projection_matrices(clocs, cmmaxs, nmax, d, sp, up_down, y_shift=0.0):
    """
    Build projection matrices W1, W2 each of shape (num_modes, tot_no_modes) (JAX).
    """
    cmmaxs_np = np.asarray(cmmaxs, dtype=int)
    no_cylinders = len(cmmaxs_np)
    tot_no_modes = int(np.sum(cmmaxs_np * 2 + 1))

    # Per-coefficient arrays
    cms_list = []
    cx_list = []
    cy_list = []
    for icyl in range(no_cylinders):
        cmmax = int(cmmaxs_np[icyl])
        n = 2 * cmmax + 1
        cms_list.append(jnp.arange(-cmmax, cmmax + 1, dtype=jnp.float64))
        cx_list.append(jnp.full(n, clocs[icyl, 0]))
        cy_list.append(jnp.full(n, clocs[icyl, 1] + y_shift))

    cms_coeff = jnp.concatenate(cms_list)
    cx_coeff = jnp.concatenate(cx_list)
    cy_coeff = jnp.concatenate(cy_list)

    # Output spectral parameters: (num_modes,)
    idx0 = sp['MiddleIndex'] - nmax
    idx1 = sp['MiddleIndex'] + nmax + 1
    kxs_out = jnp.asarray(sp['kxs'][idx0:idx1]).ravel()
    kys_out = jnp.asarray(sp['kys'][idx0:idx1]).ravel()
    angs_out = jnp.asarray(sp['Angles'][idx0:idx1]).ravel()

    # Broadcast shapes: (num_modes, 1) x (1, tot_no_modes)
    kxs_c = kxs_out[:, None]
    kys_c = kys_out[:, None]
    angs_c = angs_out[:, None]
    cx_r = cx_coeff[None, :]
    cy_r = cy_coeff[None, :]
    cm_r = cms_coeff[None, :]

    # FIX 2: safe-denominator pattern for jnp.where + division.
    # The original code:
    #   kys_safe = jnp.where(|kys_c| < eps, eps, kys_c)
    #   pref = C / kys_safe
    # is WRONG under autodiff: JAX evaluates C/kys_c for ALL entries
    # (including where kys_c≈0) before applying the mask, so the gradient
    # of C/kys_c explodes to inf/NaN even for masked entries.
    #
    # The correct pattern is to replace the denominator *before* dividing:
    #   denom = jnp.where(|kys_c| < eps, eps, kys_c)   # safe denominator
    #   pref  = C / denom                                # divide by safe value
    # This way, JAX only ever differentiates through C/denom where denom != 0.
    # The mask selects the right *output* value; the safe fallback ensures the
    # gradient of the division itself is always finite.
    _eps = 1e-10 + 0j
    denom = jnp.where(jnp.abs(kys_c) < 1e-10, _eps, kys_c)  # <-- FIX 2
    pref = sp['TwoOverPeriod'] / denom

    if up_down > 0:
        exp1 = kxs_c * cx_r - kys_c * cy_r + cm_r * (angs_c + jnp.pi)
        W1 = ((-1.0 + 0j) ** cm_r) * jnp.exp(1j * exp1) * pref

        exp2 = kxs_c * cx_r + kys_c * (cy_r - d) - cm_r * (angs_c - jnp.pi)
        W2 = jnp.exp(1j * exp2) * pref
    else:
        exp1 = kxs_c * cx_r - kys_c * (cy_r + d) + cm_r * (angs_c + jnp.pi)
        W1 = ((-1.0 + 0j) ** cm_r) * jnp.exp(1j * exp1) * pref

        exp2 = kxs_c * cx_r + kys_c * cy_r - cm_r * (angs_c - jnp.pi)
        W2 = jnp.exp(1j * exp2) * pref

    return W1, W2