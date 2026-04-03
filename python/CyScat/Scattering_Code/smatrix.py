"""
smatrix Code
Coder: Curtis Jin
Date: 2010/DEC/3rd Friday
Contact: jsirius@umich.edu
Description: Professor Michelson's version
           : Scattering Matrix Generating Code

GPU-accelerated version:
  - Batched LU solve for all spectral modes simultaneously
  - Projection matrix replaces 939-iteration scatteringcoefficientsall loop
  - _vall_batch: all incident field vectors at once (no per-mode loop)
"""

import numpy as np
import time
import sys
from .ky import ky
from .transall import transall
from .sall import sall
from .simulation_time_profile import simulation_time_profile
from . import gpu_backend as gb


def smatrix(clocs, cmmaxs, cepmus, crads, period, lambda_wave, nmax, d, sp, interaction):
    """
    Generate Scattering Matrix.
    Uses projection matrices and batched LU solve — no per-mode Python loops.
    """
    no_cylinders = len(cmmaxs)
    total_steps = no_cylinders * (no_cylinders - 1) // 2 + 2 * (2 * nmax + 1)

    k = 2 * np.pi / lambda_wave
    phiinc = sp['phiinc']

    # === T-Matrix ===
    print('  Computing T-Matrix...')
    sys.stdout.flush()
    st_tmatrix_start = time.time()
    if interaction == 'On':
        t = transall(clocs, cmmaxs, period, lambda_wave, phiinc, sp, total_steps)
    else:
        t = transall(clocs, cmmaxs, -1, lambda_wave, phiinc, sp, total_steps)
    st_tmatrix = time.time() - st_tmatrix_start
    print(f'  T-Matrix: {st_tmatrix:.1f}s')

    # === S-Vector (Mie coefficients) ===
    s = sall(cmmaxs, cepmus, crads, lambda_wave)
    diag_s = s[:, None]   # (tot_no_modes, 1) for broadcasting

    # === Build system matrix z = I - diag(s) @ t ===
    z = np.eye(len(s)) - np.diag(s) @ t

    # === LU Decomposition (GPU or CPU) ===
    print('  LU Decomposition...')
    sys.stdout.flush()
    st_lu_start = time.time()
    lu_and_piv = gb.lu_factor(z)
    st_lu = time.time() - st_lu_start
    print(f'  LU: {st_lu:.1f}s' + (' [GPU]' if gb.GPU_AVAILABLE else ''))
    del z

    num_modes = 2 * nmax + 1

    # Pre-compute kxex values for all input modes
    kxex_all = np.array([
        _extract_scalar(sp['kxs'][sp['MiddleIndex'] + nin])
        for nin in range(-nmax, nmax + 1)
    ])   # (num_modes,)

    # === S11 & S21 Partition ===
    print('  Computing S11 & S21...')
    sys.stdout.flush()
    st_up_start = time.time()

    # All incident field vectors at once: (tot_no_modes, num_modes)
    V_up = diag_s * _vall_batch(clocs, cmmaxs, kxex_all, k, up_down=1)
    C_up = gb.to_cpu(gb.lu_solve(lu_and_piv, gb.to_gpu(V_up) if gb.GPU_AVAILABLE else V_up))
    assert C_up.ndim == 2 and C_up.shape[1] == 2*nmax+1, \
        f"lu_solve shape mismatch: got {C_up.shape}, expected ({len(s)}, {2*nmax+1})"
    V_up = diag_s * _vall_batch(clocs, cmmaxs, kxex_all, k, up_down=1)
    C_up = gb.to_cpu(gb.lu_solve(lu_and_piv, gb.to_gpu(V_up) if gb.GPU_AVAILABLE else V_up))
    print(f"  DEBUG: C_up.shape={C_up.shape}, nmax={nmax}, GPU={gb.GPU_AVAILABLE}", flush=True)  # ADD THIS

    W1_up, W2_up = _build_projection_matrices(clocs, cmmaxs, nmax, d, sp, up_down=1)
    print(f"  DEBUG: W2_up.shape={W2_up.shape}", flush=True)  # ADD THIS

    s11matrix = W1_up @ C_up
    s21matrix = W2_up @ C_up
    print(f"  DEBUG: s21matrix.shape={s21matrix.shape}", flush=True)  # ADD THIS

    del V_up

    # Projection matrices: (num_modes, tot_no_modes) each
    W1_up, W2_up = _build_projection_matrices(clocs, cmmaxs, nmax, d, sp, up_down=1)

    s11matrix = W1_up @ C_up   # (num_modes, num_modes)
    s21matrix = W2_up @ C_up   # (num_modes, num_modes)
    del W1_up, W2_up, C_up

    # Add incident field transmission: S21[nmax+nin, nin] += exp(-i*kyex*d)
    for nin_idx, nin in enumerate(range(-nmax, nmax + 1)):
        kyex = _extract_scalar(sp['kys'][sp['MiddleIndex'] + nin])
        s21matrix[nmax + nin, nin_idx] += np.exp(-1j * kyex * d)

    st_up = time.time() - st_up_start
    print(f'  S11&S21: {st_up:.1f}s')

    # === S12 & S22 Partition ===
    print('  Computing S12 & S22...')
    sys.stdout.flush()
    st_down_start = time.time()

    # Down direction: y-positions shifted by -d (pass y_shift=-d, don't mutate clocs)
    V_down = diag_s * _vall_batch(clocs, cmmaxs, kxex_all, k, up_down=-1, y_shift=-d)
    C_down = gb.to_cpu(gb.lu_solve(lu_and_piv, gb.to_gpu(V_down) if gb.GPU_AVAILABLE else V_down))
    assert C_down.ndim == 2 and C_down.shape[1] == 2*nmax+1, \
        f"lu_solve shape mismatch: got {C_down.shape}, expected ({len(s)}, {2*nmax+1})"
    
    del V_down

    W1_down, W2_down = _build_projection_matrices(clocs, cmmaxs, nmax, d, sp, up_down=-1, y_shift=-d)

    s12matrix = W1_down @ C_down   # (num_modes, num_modes)
    s22matrix = W2_down @ C_down   # (num_modes, num_modes)
    del W1_down, W2_down, C_down

    # Add incident field: S12[nmax+nin, nin] += exp(-i*kyex*d)
    for nin_idx, nin in enumerate(range(-nmax, nmax + 1)):
        kyex = _extract_scalar(sp['kys'][sp['MiddleIndex'] + nin])
        s12matrix[nmax + nin, nin_idx] += np.exp(-1j * kyex * d)

    st_down = time.time() - st_down_start
    print(f'  S12&S22: {st_down:.1f}s')

    # === Assemble full S-matrix ===
    S = np.block([[s11matrix, s12matrix], [s21matrix, s22matrix]])

    # === Normalization ===
    st_norm_start = time.time()
    m = np.arange(-nmax, nmax + 1)
    m = np.concatenate([m, m])
    kxs = 2 * np.pi / period * m
    kys_norm = ky(k, kxs)

    nor2 = np.sqrt(kys_norm / k)
    nor1 = np.sqrt(k / kys_norm)
    P2 = np.diag(nor2)
    P1 = np.diag(nor1)
    S = P2 @ S @ P1
    st_norm = time.time() - st_norm_start

    STP = simulation_time_profile(st_tmatrix, st_lu, st_up, st_down, st_norm)
    total = STP['TST']
    print(f'  Total: {total:.1f}s ({total/60:.1f} min)')
    sys.stdout.flush()

    return S, STP


# ─────────────────────────────────────────────────────────────────────────────
# Vectorized helpers
# ─────────────────────────────────────────────────────────────────────────────

def _vall_batch(clocs, cmmaxs, kxex_all, k, up_down, y_shift=0.0):
    """
    Vectorized vall: compute all incident field vectors at once.

    Returns V_all of shape (tot_no_modes, num_modes) where
    V_all[:, i] = vall(clocs, cmmaxs, lambda, kxex_all[i], up_down)
    but with y positions shifted by y_shift.
    """
    no_cylinders = len(cmmaxs)
    num_modes = len(kxex_all)
    tot_no_modes = int(np.sum(cmmaxs * 2 + 1))

    # Build per-coefficient arrays
    cms_coeff = np.empty(tot_no_modes)
    cx_coeff = np.empty(tot_no_modes)
    cy_coeff = np.empty(tot_no_modes)
    j = 0
    for icyl in range(no_cylinders):
        cmmax = int(cmmaxs[icyl])
        n = 2 * cmmax + 1
        cms_coeff[j:j + n] = np.arange(-cmmax, cmmax + 1)
        cx_coeff[j:j + n] = clocs[icyl, 0]
        cy_coeff[j:j + n] = clocs[icyl, 1] + y_shift
        j += n

    # kyex and phiinc per input mode
    kyex_all_v = ky(k, kxex_all)          # (num_modes,) — handles evanescent (complex)
    phiinc_all = np.arccos(kxex_all / k + 0j)   # (num_modes,) complex for evanescent
    if up_down < 0:
        kyex_all_v = -kyex_all_v
        phiinc_all = -phiinc_all

    # V[coeff, mode] = exp(-i*(kx*cx + ky*cy)) * exp(-i*cm*phiinc) * exp(-i*pi/2*cm)
    # Broadcast: (tot, 1) × (1, num)
    k_dot_r = (cx_coeff[:, None] * kxex_all[None, :] +
               cy_coeff[:, None] * kyex_all_v[None, :])   # (tot, num)
    cm_phiinc = cms_coeff[:, None] * phiinc_all[None, :]  # (tot, num)
    cm_pi2 = np.exp(-1j * np.pi / 2 * cms_coeff)[:, None]  # (tot, 1)

    V_all = np.exp(-1j * k_dot_r) * np.exp(-1j * cm_phiinc) * cm_pi2
    return V_all


def _build_projection_matrices(clocs, cmmaxs, nmax, d, sp, up_down, y_shift=0.0):
    """
    Build projection matrices W1, W2 each of shape (num_modes, tot_no_modes).

    W1 @ C gives the S11 (or S12) column; W2 @ C gives S21 (or S22).
    y_shift: applied to cylinder y-positions (0 for up, -d for down).
    """
    no_cylinders = len(cmmaxs)
    tot_no_modes = int(np.sum(cmmaxs * 2 + 1))

    # Per-coefficient arrays
    cms_coeff = np.empty(tot_no_modes)
    cx_coeff = np.empty(tot_no_modes)
    cy_coeff = np.empty(tot_no_modes)
    j = 0
    for icyl in range(no_cylinders):
        cmmax = int(cmmaxs[icyl])
        n = 2 * cmmax + 1
        cms_coeff[j:j + n] = np.arange(-cmmax, cmmax + 1)
        cx_coeff[j:j + n] = clocs[icyl, 0]
        cy_coeff[j:j + n] = clocs[icyl, 1] + y_shift
        j += n

    # Output spectral parameters: (num_modes,)
    idx0 = sp['MiddleIndex'] - nmax
    idx1 = sp['MiddleIndex'] + nmax + 1
    kxs_out = np.asarray(sp['kxs'][idx0:idx1]).ravel()
    kys_out = np.asarray(sp['kys'][idx0:idx1]).ravel()
    angs_out = np.asarray(sp['Angles'][idx0:idx1]).ravel()

    # Broadcast shapes: (num_modes, 1) × (1, tot_no_modes)
    kxs_c = kxs_out[:, None]
    kys_c = kys_out[:, None]
    angs_c = angs_out[:, None]
    cx_r = cx_coeff[None, :]
    cy_r = cy_coeff[None, :]
    cm_r = cms_coeff[None, :]

    kys_safe = np.where(np.abs(kys_c) < 1e-10, 1e-10 + 0j, kys_c)
    pref = sp['TwoOverPeriod'] / kys_safe   # (num_modes, tot_no_modes)

    if up_down > 0:
        # W1 → S11: exponent = kxs*x - kys*y + cm*(angles + pi)
        exp1 = kxs_c * cx_r - kys_c * cy_r + cm_r * (angs_c + np.pi)
        W1 = ((-1.0 + 0j) ** cm_r) * np.exp(1j * exp1) * pref

        # W2 → S21: exponent = kxs*x + kys*(y - d) - cm*(angles - pi)
        #   cy_r is already y_original here (y_shift=0)
        exp2 = kxs_c * cx_r + kys_c * (cy_r - d) - cm_r * (angs_c - np.pi)
        W2 = np.exp(1j * exp2) * pref
    else:
        # cy_r = y_original + y_shift = y_original - d  (y_shift = -d)
        #
        # W1 → S12: derived from scatteringcoefficients_m_matrix with kys_clocy_plus_d
        #   kys_clocy_plus_d = kys * (cy_r + d) = kys * y_original
        #   exponent = kxs*x - kys*(cy_r + d) + cm*(angles + pi)
        exp1 = kxs_c * cx_r - kys_c * (cy_r + d) + cm_r * (angs_c + np.pi)
        W1 = ((-1.0 + 0j) ** cm_r) * np.exp(1j * exp1) * pref

        # W2 → S22: exponent = kxs*x + kys*cy_r - cm*(angles - pi)
        #   = kxs*x + kys*(y_original - d) - cm*(angles - pi)
        exp2 = kxs_c * cx_r + kys_c * cy_r - cm_r * (angs_c - np.pi)
        W2 = np.exp(1j * exp2) * pref

    return W1, W2


def _extract_scalar(val):
    """Extract scalar from numpy array or scalar."""
    if hasattr(val, 'item'):
        return val.item()
    if isinstance(val, np.ndarray):
        return float(val.flatten()[0])
    return val
