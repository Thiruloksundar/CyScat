"""
transall Code
Coder: Curtis Jin
Date: 2011/FEB/13th Sunday
Contact: jsirius@umich.edu
Description: Professor Michelson's version
           : translational matrix generating code
             Using Modified Shanks Transformation
             Exploiting symmetric structure of T-Matrix

JAX version for automatic differentiation.
  - Self-sum: fast numpy by default; JAX spatial path only when λ is traced
  - Spectral pairs: JAX vectorized exponentials
  - Spatial pairs: JAX fori_loop with bessel_jax Hankel functions

FIXES applied for NaN-free gradients:
  FIX 1 — jnp.where + division (spectral: kys_safe) — same trap as smatrix.py:
           Replace denominator BEFORE dividing so JAX never sees 1/0.

  FIX 2 — jnp.arctan2 gradient at origin:
           arctan2(0, 0) is undefined; its gradient is NaN.
           Guard r < epsloc entries by replacing rv with a safe fallback
           before the atan2 call, then masking the result to 0 afterward.

  FIX 3 — jnp.where(isfinite(h), h, 0) gradient trap:
           isfinite() is not differentiable; using it inside jnp.where
           masks the gradient for entries where h IS finite.
           Fix: replace with a safe-argument pattern — clamp the Hankel
           argument away from zero so the function stays finite, then mask
           only the r < epsloc case (which is a concrete Python bool based
           on clocs_concrete, not a JAX-traced condition).

  FIX 4 — _shanks_batch_jax: 1/(a - b) when a == b == inf → NaN gradient.
           Guard denominators in the Shanks update with the safe-denom pattern.
"""

import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import hankel2 as scipy_hankel2
from scipy.linalg import toeplitz as scipy_toeplitz
from .bessel_jax import hankel2 as jax_hankel2
import sys


def transall(clocs, cmmaxs, period, lambda_wave, phiinc, sp, total_steps, clocs_concrete=None):
    """
    Generate translational matrix for all cylinders (JAX-differentiable).

    Self-sum is precomputed (constant w.r.t. clocs).
    Cross-pair entries are computed with JAX operations for autodiff.

    clocs_concrete : optional numpy array of concrete clocs values for
                     pair classification (needed when clocs is a JAX tracer).
    """
    clocs = jnp.asarray(clocs, dtype=jnp.float64)
    if clocs_concrete is None:
        try:
            clocs_concrete = np.asarray(clocs)
        except jax.errors.TracerArrayConversionError:
            raise ValueError("transall: clocs is a JAX tracer but clocs_concrete was not provided")
    cmmaxs_np = np.asarray(cmmaxs, dtype=int)
    no_cylinders = len(cmmaxs_np)
    tot_no_modes = int(np.sum(cmmaxs_np * 2 + 1))
    max_order = int(np.max(cmmaxs_np))

    # ── Self-sum ───────────────────────────────────────────────────────
    self_sum_vector = _compute_self_sum_numpy(max_order, period, lambda_wave, phiinc, sp)
    self_sum_vector = jnp.asarray(self_sum_vector)

    t = jnp.zeros((tot_no_modes, tot_no_modes), dtype=jnp.complex128)

    # Fill diagonal blocks with self-sum toeplitz
    istart = 0
    for icyl in range(no_cylinders):
        cmmaxob = int(cmmaxs_np[icyl])
        offset = 2 * max_order
        c = self_sum_vector[offset:offset + 2 * cmmaxob + 1]
        r = jnp.flip(self_sum_vector[offset - 2 * cmmaxob:offset + 1])
        t11 = _toeplitz_jax(c, r)
        t = t.at[istart:istart + 2 * cmmaxob + 1,
                 istart:istart + 2 * cmmaxob + 1].set(t11)
        istart += 2 * cmmaxob + 1

    # ── Classify cross-pairs ──────────────────────────────────────────────
    epsloc = sp['epsloc']
    spectral_cond = sp['spectralCond']

    pairs = []
    istart = 0
    for icyl in range(no_cylinders):
        cmmaxob = int(cmmaxs_np[icyl])
        jstart = istart + 2 * cmmaxob + 1
        for jcyl in range(icyl + 1, no_cylinders):
            cmmaxso = int(cmmaxs_np[jcyl])
            rv = clocs_concrete[icyl] - clocs_concrete[jcyl]
            r = np.linalg.norm(rv)
            y = rv[1]
            is_spectral = not (r < epsloc or abs(y) < spectral_cond or period < 0)
            pairs.append((icyl, jcyl, istart, jstart, cmmaxob, cmmaxso, is_spectral))
            jstart += 2 * cmmaxso + 1
        istart += 2 * cmmaxob + 1

    spectral_pairs = [p for p in pairs if p[6]]
    spatial_pairs  = [p for p in pairs if not p[6]]

    # ── Spectral pairs (JAX) ──────────────────────────────────────────────
    if spectral_pairs:
        t = _compute_spectral_jax(t, clocs, spectral_pairs, sp)

    # ── Spatial pairs (JAX with fori_loop) ────────────────────────────────
    if spatial_pairs:
        t = _compute_spatial_jax(t, clocs, spatial_pairs, period, lambda_wave, phiinc, sp)

    n_total    = len(pairs)
    n_spectral = len(spectral_pairs)
    n_spatial  = len(spatial_pairs)
    print(f"  transall: {n_total} pairs ({n_spectral} spectral, {n_spatial} spatial)")
    sys.stdout.flush()

    return t


# ─────────────────────────────────────────────────────────────────────────────
# Self-sum (numpy, precomputed — no changes needed here)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_self_sum_numpy(max_order, period, lambda_wave, phiinc, sp):
    cms_self = np.arange(-2 * max_order, 2 * max_order + 1)
    return _transvector_numpy(
        np.array([0.0, 0.0]), np.array([0.0, 0.0]),
        cms_self, period, lambda_wave, phiinc, sp)


def _transvector_numpy(clocob, clocso, cms, period, lambda_wave, phiinc, sp):
    rv = clocob - clocso
    r  = np.linalg.norm(rv)
    y  = rv[1]
    if r < sp['epsloc'] or abs(y) < sp['spectralCond'] or period < 0:
        max_cm = int(np.max(cms))
        min_cm = int(np.min(cms))
        array_size = max_cm - min_cm + 1
        temp = np.zeros(array_size, dtype=complex)
        for cm in cms:
            cm_int = int(cm)
            idx = cm_int - min_cm
            temp[idx] = _transper_numpy(clocob, cm_int, clocso, 0, period,
                                        lambda_wave, phiinc, sp)
        return temp
    else:
        kxs_all   = np.asarray(sp['kxs']).flatten()
        kys_all   = np.asarray(sp['kys']).flatten()
        angles_all = np.asarray(sp['Angles']).flatten()
        return _spectral_sum_single_numpy(
            rv[0], rv[1], cms, kxs_all, kys_all, angles_all, sp['TwoOverPeriod'])


def _spectral_sum_single_numpy(x, y, cms, kxs_all, kys_all, angles_all, two_over_period):
    cmgrid, kxsgrid   = np.meshgrid(cms, kxs_all)
    cmgrid, kysgrid   = np.meshgrid(cms, kys_all)
    cmgrid, anglesgrid = np.meshgrid(cms, angles_all)
    sign_y = np.sign(y) if y != 0 else 1.0
    exponent = (kxsgrid * x - kysgrid * np.abs(y) +
                cmgrid * (np.pi - anglesgrid * sign_y))
    odd_mask   = (np.abs(cmgrid.astype(int)) % 2 != 0)
    sign_power = np.where(sign_y < 0,
                          np.where(odd_mask, -1.0, 1.0),
                          1.0).astype(complex)
    exp_term    = np.exp(1j * exponent)
    kysgrid_safe = np.where(np.abs(kysgrid) < 1e-10, 1e-10 + 0j, kysgrid)
    t_matrix    = sign_power * exp_term / kysgrid_safe * two_over_period
    return np.sum(t_matrix, axis=0)


def _transper_numpy(clocob, cmob, clocso, cmso, period, lambda_wave, phiinc, sp):
    k          = sp['k0']
    epsseries  = sp['epsseries']
    epsloc     = sp['epsloc']
    jmax       = sp['jmax']
    spectral_internal = sp['spectral']
    if spectral_internal < 0:
        kshanks = sp['kshanksSpatial']
        nrepeat = sp['nrepeatSpatial']
    else:
        kshanks = sp['kshanksSpectral']
        nrepeat = sp['nrepeatSpectral']
    rv = clocob - clocso
    r  = np.linalg.norm(rv)
    x  = rv[0]
    y  = rv[1]
    if period < 0:
        if r < epsloc:
            return 0.0 + 0.0j
        return _trans_numpy(clocob, cmob, clocso, cmso, lambda_wave, sp)
    if period > 0:
        if abs(y) < sp['spectralCond']:
            spectral_internal = -1
            kshanks = sp['kshanksSpatial']
            nrepeat = sp['nrepeatSpatial']
        if kshanks > 0:
            ashankseps1 = np.zeros(kshanks + 2, dtype=complex)
            ashankseps2 = np.zeros(kshanks + 2, dtype=complex)
            ashankseps1[0] = np.inf
            ashankseps2[0] = np.inf
        if spectral_internal < 0:
            if r < epsloc:
                t = 0.0 + 0.0j
            else:
                t = _trans_numpy(clocob, cmob, clocso, cmso, lambda_wave, sp)
            if kshanks > 0:
                ashankseps1[1] = t
            ts = 1
        else:
            cm  = cmso - cmob
            mid = sp['MiddleIndex']
            kxs_arr   = np.asarray(sp['kxs']).flatten()
            kys_arr   = np.asarray(sp['kys']).flatten()
            angles_arr = np.asarray(sp['Angles']).flatten()
            exponent = (kxs_arr[mid] * x +
                        kys_arr[mid] * abs(y) +
                        cm * (np.sign(y) * angles_arr[mid] - np.pi))
            sign_y = np.sign(y) if y != 0 else 1.0
            t = (sign_y ** cm * np.exp(-1j * exponent) /
                 kys_arr[mid] * sp['TwoOverPeriod'])
            if kshanks > 0:
                ashankseps1[1] = t
            ts = 1
        j = 1
        irepeat = 0
        while irepeat < nrepeat and j < jmax:
            if spectral_internal < 0:
                add = (np.exp(-1j * k * j * period * np.cos(phiinc)) *
                       _trans_numpy(clocob, cmob, clocso + j * np.array([period, 0]),
                                    cmso, lambda_wave, sp) +
                       np.exp(-1j * k * (-j) * period * np.cos(phiinc)) *
                       _trans_numpy(clocob, cmob, clocso - j * np.array([period, 0]),
                                    cmso, lambda_wave, sp))
            else:
                mid = sp['MiddleIndex']
                kxs_arr    = np.asarray(sp['kxs']).flatten()
                kys_arr    = np.asarray(sp['kys']).flatten()
                angles_arr = np.asarray(sp['Angles']).flatten()
                kxp = kxs_arr[mid + j]
                kxm = kxs_arr[mid - j]
                exponent1 = (kxp * x + kys_arr[mid + j] * abs(y) +
                             cm * (np.sign(y) * angles_arr[mid + j] - np.pi))
                exponent2 = (kxm * x + kys_arr[mid - j] * abs(y) +
                             cm * (np.sign(y) * angles_arr[mid - j] - np.pi))
                sign_y = np.sign(y) if y != 0 else 1.0
                add = (sign_y ** cm * sp['TwoOverPeriod'] *
                       (np.exp(-1j * exponent1) / kys_arr[mid + j] +
                        np.exp(-1j * exponent2) / kys_arr[mid - j]))
            told = t
            t    = t + add
            if kshanks <= 0:
                rerror = abs(t - told)
            else:
                S = _shanks_numpy(t, ashankseps1, ashankseps2)
                ashankseps2 = ashankseps1.copy()
                ashankseps1 = S.copy()
                if j <= 2 * kshanks + 1:
                    rerror = 1
                else:
                    tsold  = ts
                    ts     = S[-1]
                    rerror = abs(ts - tsold)
            if rerror < epsseries:
                irepeat += 1
            else:
                irepeat = 0
            j += 1
        if j == jmax:
            print(f'WARNING: transper j reached jmax! cmob={cmob}, cmso={cmso}')
    if kshanks > 0 and period > 0:
        t = ts
    return t


def _trans_numpy(clocob, cmob, clocso, cmso, lambda_wave, sp):
    k      = sp['k0']
    rv     = clocob - clocso
    r      = np.linalg.norm(rv)
    epsloc = sp.get('epsloc', 1e-4)
    if r < epsloc:
        return 0.0 + 0.0j
    phip = np.arctan2(rv[1], rv[0])
    kr   = k * r
    n    = int(cmob - cmso)
    if kr < 1e-8:
        return 0.0 + 0.0j
    h = scipy_hankel2(n, kr)
    if not np.isfinite(h):
        return 0.0 + 0.0j
    t = h * np.exp(1j * (cmso - cmob) * (phip - np.pi))
    if not np.isfinite(t):
        return 0.0 + 0.0j
    return t


def _shanks_numpy(newentry, a1, a2):
    kplus2 = len(a1)
    S = np.zeros(kplus2, dtype=complex)
    S[0] = np.inf
    S[1] = newentry
    for idx in range(1, kplus2 - 1):
        with np.errstate(divide='ignore', invalid='ignore'):
            f1 = 1.0 / (a2[idx] - a1[idx])
            f2 = 1.0 / (S[idx] - a1[idx])
            f3 = 1.0 / (a2[idx - 1] - a1[idx])
        with np.errstate(divide='ignore', invalid='ignore'):
            candidate = 1.0 / (f1 + f2 - f3) + a1[idx]
        valid = (np.isfinite(f1) and np.isfinite(f2) and
                 np.isfinite(f3) and np.isfinite(candidate))
        S[idx + 1] = candidate if valid else a1[idx]
    return S


# ─────────────────────────────────────────────────────────────────────────────
# JAX helpers
# ─────────────────────────────────────────────────────────────────────────────

def _toeplitz_jax(c, r):
    nc   = len(c)
    nr   = len(r)
    vals = jnp.concatenate([jnp.flip(r[1:]), c])
    row_idx = jnp.arange(nc)[:, None]
    col_idx = jnp.arange(nr)[None, :]
    indices = row_idx - col_idx + (nr - 1)
    return vals[indices]


def _assign_pair_jax(t, tvector, istart, jstart, cmmaxob, cmmaxso):
    offset = cmmaxso + cmmaxob

    start_idx = cmmaxso - cmmaxob + offset
    c  = tvector[start_idx:]
    r  = jnp.flip(tvector[:start_idx + 1])
    t12 = _toeplitz_jax(c, r)

    start_idx2 = cmmaxob - cmmaxso + offset
    c2  = tvector[start_idx2:]
    r2  = jnp.flip(tvector[:start_idx2 + 1])
    t21 = _toeplitz_jax(c2, r2)

    c_mod = (-1.0) ** (jnp.arange(2 * cmmaxso + 1) + (cmmaxob - cmmaxso))
    r_mod = (-1.0) ** (jnp.arange(2 * cmmaxob + 1) + (cmmaxob - cmmaxso))
    modification_matrix = _toeplitz_jax(c_mod, r_mod)
    t21 = modification_matrix * t21

    t = t.at[istart:istart + 2 * cmmaxob + 1,
             jstart:jstart + 2 * cmmaxso + 1].set(t12)
    t = t.at[jstart:jstart + 2 * cmmaxso + 1,
             istart:istart + 2 * cmmaxob + 1].set(t21)
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Spectral pairs (JAX)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_spectral_jax(t, clocs, spectral_pairs, sp):
    groups = {}
    for p in spectral_pairs:
        key = (p[4], p[5])
        if key not in groups:
            groups[key] = []
        groups[key].append(p)

    for (cmmaxob, cmmaxso), group in groups.items():
        cms      = jnp.arange(-cmmaxob - cmmaxso, cmmaxob + cmmaxso + 1, dtype=jnp.float64)
        icyl_arr = jnp.array([p[0] for p in group])
        jcyl_arr = jnp.array([p[1] for p in group])
        x_arr    = clocs[icyl_arr, 0] - clocs[jcyl_arr, 0]
        y_arr    = clocs[icyl_arr, 1] - clocs[jcyl_arr, 1]
        t_all    = _spectral_sum_batch_jax(x_arr, y_arr, cms, sp)
        for i, p in enumerate(group):
            t = _assign_pair_jax(t, t_all[i], p[2], p[3], p[4], p[5])
    return t


def _spectral_sum_batch_jax(x_arr, y_arr, cms, sp):
    """Batch spectral sum using JAX. Returns (P, M) complex array."""
    kxs_all        = jnp.asarray(sp['kxs']).flatten()
    kys_all        = jnp.asarray(sp['kys']).flatten()
    angles_all     = jnp.asarray(sp['Angles']).flatten()
    two_over_period = sp['TwoOverPeriod']

    x      = x_arr[:, None, None]
    y      = y_arr[:, None, None]
    cms_g  = cms[None, None, :]
    kxs    = kxs_all[None, :, None]
    kys    = kys_all[None, :, None]
    angs   = angles_all[None, :, None]

    sign_y = jnp.sign(y)
    sign_y = jnp.where(sign_y == 0.0, 1.0, sign_y)
    abs_y  = jnp.abs(y)

    exponent = kxs * x - kys * abs_y + cms_g * (jnp.pi - angs * sign_y)

    odd_mask   = (jnp.abs(cms_g).astype(jnp.int32) % 2 != 0)
    sign_power = jnp.where(sign_y < 0,
                           jnp.where(odd_mask, -1.0 + 0j, 1.0 + 0j),
                           1.0 + 0j)

    exp_term = jnp.exp(1j * exponent)

    # FIX 1: safe-denominator pattern — replace before dividing, never divide by zero.
    # The old code used jnp.where(|kys|<eps, eps, kys) as the denominator, which
    # still lets JAX differentiate through 1/kys at kys=0 (both branches evaluated).
    # Replacing kys with denom first ensures the division always sees a nonzero value.
    denom    = jnp.where(jnp.abs(kys) < 1e-10, 1e-10 + 0j, kys)  # FIX 1
    t_matrix = sign_power * exp_term / denom * two_over_period
    t_batch  = jnp.sum(t_matrix, axis=1)  # sum over K spectral modes -> (P, M)
    return t_batch


# ─────────────────────────────────────────────────────────────────────────────
# Spatial pairs (JAX with fori_loop)
# ─────────────────────────────────────────────────────────────────────────────

def _hankel2_batch(n_flat, z_flat):
    return jax.vmap(jax_hankel2)(n_flat, z_flat)


def _compute_spatial_jax(t, clocs, spatial_pairs, period, lambda_wave, phiinc, sp):
    groups = {}
    for p in spatial_pairs:
        key = (p[4], p[5])
        if key not in groups:
            groups[key] = []
        groups[key].append(p)

    for (cmmaxob, cmmaxso), group in groups.items():
        cms_int  = jnp.arange(-cmmaxob - cmmaxso, cmmaxob + cmmaxso + 1, dtype=jnp.int32)
        icyl_arr = [p[0] for p in group]
        jcyl_arr = [p[1] for p in group]
        ob_locs  = jnp.stack([clocs[i] for i in icyl_arr])
        so_locs  = jnp.stack([clocs[j] for j in jcyl_arr])
        t_batch  = _spatial_sum_jax(ob_locs, so_locs, cms_int, sp, period, phiinc)
        for i, p in enumerate(group):
            t = _assign_pair_jax(t, t_batch[i], p[2], p[3], p[4], p[5])
    return t


def _spatial_sum_jax(ob_locs, so_locs, cms_int, sp, period, phiinc):
    """
    Spatial translation sum using JAX fori_loop with Shanks acceleration.
    """
    k         = sp['k0']
    epsseries = float(sp['epsseries'])
    epsloc    = float(sp['epsloc'])
    jmax      = min(int(sp['jmax']), 60)
    kshanks   = int(sp['kshanksSpatial'])
    nrepeat   = int(sp['nrepeatSpatial'])

    P = ob_locs.shape[0]
    M = cms_int.shape[0]

    rv       = ob_locs - so_locs        # (P, 2)
    r_vec    = jnp.linalg.norm(rv, axis=1)   # (P,)

    # FIX 2: safe arctan2 — replace rv with a nonzero fallback before calling
    # arctan2 so the gradient is never evaluated at (0, 0).
    # We use clamp-before-atan2: replace zero-norm entries with a unit vector.
    # The arctan2 result for those entries is masked to 0 below anyway.
    safe_rv  = jnp.where(r_vec[:, None] < epsloc,
                         jnp.array([[1.0, 0.0]]),   # safe fallback direction
                         rv)                          # FIX 2
    phip_vec = jnp.arctan2(safe_rv[:, 1], safe_rv[:, 0])   # never sees (0,0)

    r_2d    = r_vec[:, None]        # (P, 1)
    n_2d    = cms_int[None, :]      # (1, M)
    phip_2d = phip_vec[:, None]     # (P, 1)

    # FIX 3: remove jnp.where(isfinite(h), h, 0) — isfinite breaks gradients.
    # Instead, clamp the Hankel argument to a safe minimum so the function
    # never actually becomes non-finite in the first place.
    # Entries with r < epsloc are zeroed out by the r_2d mask below (concrete
    # comparison), which is fine since those are genuinely zero contributions.
    kr_safe = jnp.maximum(k * r_2d, 1e-8)          # never exactly 0   FIX 3
    n_full  = jnp.broadcast_to(n_2d, (P, M)).flatten()
    z_full  = jnp.broadcast_to(kr_safe, (P, M)).flatten()
    h0_flat = _hankel2_batch(n_full, z_full)
    h0      = h0_flat.reshape(P, M)
    # Zero out r < epsloc entries using a concrete (non-traced) mask
    valid_mask = (r_vec >= epsloc)[:, None]         # concrete Python bool array
    h0 = jnp.where(valid_mask, h0, 0.0 + 0j)       # safe: mask is not JAX-traced

    t0 = h0 * jnp.exp(1j * n_2d * (jnp.pi - phip_2d))   # (P, M)

    # ── Shanks initialization ─────────────────────────────────────────────
    K_shanks = kshanks + 2
    a1 = jnp.zeros((P, M, K_shanks), dtype=jnp.complex128)
    a2 = jnp.zeros((P, M, K_shanks), dtype=jnp.complex128)
    a1 = a1.at[:, :, 0].set(jnp.inf + 0j)
    a2 = a2.at[:, :, 0].set(jnp.inf + 0j)
    a1 = a1.at[:, :, 1].set(t0)

    ts         = t0.copy()
    irepeat    = jnp.zeros(P, dtype=jnp.int32)
    converged  = jnp.zeros(P, dtype=jnp.bool_)

    period_vec = jnp.array([period, 0.0])
    phase_base = jnp.exp(-1j * k * period * jnp.cos(phiinc))

    init_state = (t0, ts, a1, a2, irepeat, converged)

    def body_fn(j, state):
        t_curr, ts_curr, a1_curr, a2_curr, irep, conv = state

        j_f    = jnp.float64(j)
        so_pos = so_locs + j_f * period_vec   # (P, 2)
        so_neg = so_locs - j_f * period_vec   # (P, 2)

        rv_pos = ob_locs - so_pos             # (P, 2)
        rv_neg = ob_locs - so_neg

        r_pos  = jnp.linalg.norm(rv_pos, axis=1)[:, None]
        r_neg  = jnp.linalg.norm(rv_neg, axis=1)[:, None]

        # FIX 2 (repeated): safe arctan2 inside fori_loop
        safe_rv_pos = jnp.where(r_pos < epsloc, jnp.array([[1.0, 0.0]]), rv_pos)
        safe_rv_neg = jnp.where(r_neg < epsloc, jnp.array([[1.0, 0.0]]), rv_neg)
        phi_pos = jnp.arctan2(safe_rv_pos[:, 1], safe_rv_pos[:, 0])[:, None]
        phi_neg = jnp.arctan2(safe_rv_neg[:, 1], safe_rv_neg[:, 0])[:, None]

        # FIX 3 (repeated): clamp Hankel arguments away from zero
        n_bc        = jnp.broadcast_to(n_2d, (P, M)).flatten()
        z_pos_flat  = jnp.broadcast_to(jnp.maximum(k * r_pos, 1e-8), (P, M)).flatten()
        z_neg_flat  = jnp.broadcast_to(jnp.maximum(k * r_neg, 1e-8), (P, M)).flatten()

        h_pos = _hankel2_batch(n_bc, z_pos_flat).reshape(P, M)
        h_neg = _hankel2_batch(n_bc, z_neg_flat).reshape(P, M)

        valid_pos = (r_pos >= epsloc)
        valid_neg = (r_neg >= epsloc)
        h_pos = jnp.where(valid_pos, h_pos, 0.0 + 0j)
        h_neg = jnp.where(valid_neg, h_neg, 0.0 + 0j)

        add = (phase_base **  j_f * h_pos * jnp.exp(1j * n_2d * (jnp.pi - phi_pos)) +
               phase_base ** -j_f * h_neg * jnp.exp(1j * n_2d * (jnp.pi - phi_neg)))

        mask  = (~conv)[:, None]
        t_new = t_curr + add * mask

        new_S   = _shanks_batch_jax(t_new, a1_curr, a2_curr)
        a2_new  = a1_curr
        a1_new  = new_S

        warmup_done = (j > 2 * kshanks + 1)
        ts_new  = jnp.where(warmup_done, a1_new[:, :, -1], ts_curr)

        rerror  = jnp.where(
            warmup_done,
            jnp.max(jnp.abs(ts_new - ts_curr), axis=1),
            jnp.ones(P) * 1e10)

        new_irep = jnp.where(rerror < epsseries, irep + 1, 0)
        new_conv = conv | (new_irep >= nrepeat)

        return (t_new, ts_new, a1_new, a2_new, new_irep, new_conv)

    final_state = jax.lax.fori_loop(1, jmax, body_fn, init_state)
    t_final, ts_final, _, _, _, converged_final = final_state

    result = jnp.where(converged_final[:, None], ts_final, t_final)
    return result


def _shanks_batch_jax(newentry, a1, a2):
    """
    Vectorized Shanks transform for (P, M) batch.

    FIX 4: The original 1/(a2-a1) and 1/(S-a1) divisions produce NaN gradients
    when a2==a1 (both inf at initialisation) because d/dx(1/x) at x=0 = -inf²=NaN.
    Fix: use the safe-denominator pattern for every division in the Shanks update.
    """
    K = a1.shape[-1]
    S = jnp.zeros_like(a1)
    S = S.at[..., 0].set(jnp.inf + 0j)
    S = S.at[..., 1].set(newentry)

    for idx in range(1, K - 1):
        d1 = a2[..., idx]     - a1[..., idx]       # may be 0 at init (inf-inf=nan)
        d2 = S[..., idx]      - a1[..., idx]
        d3 = a2[..., idx - 1] - a1[..., idx]

        # FIX 4: replace zero/nan denominators before inverting
        _eps = 1e-30 + 0j
        safe_d1 = jnp.where(jnp.abs(d1) < 1e-30, _eps, d1)
        safe_d2 = jnp.where(jnp.abs(d2) < 1e-30, _eps, d2)
        safe_d3 = jnp.where(jnp.abs(d3) < 1e-30, _eps, d3)

        f1 = 1.0 / safe_d1
        f2 = 1.0 / safe_d2
        f3 = 1.0 / safe_d3

        denom_c   = f1 + f2 - f3
        safe_dc   = jnp.where(jnp.abs(denom_c) < 1e-30, _eps, denom_c)
        candidate = 1.0 / safe_dc + a1[..., idx]

        # Only accept candidate where all inputs were genuinely finite
        valid = (jnp.isfinite(f1) & jnp.isfinite(f2) &
                 jnp.isfinite(f3) & jnp.isfinite(candidate))
        S = S.at[..., idx + 1].set(jnp.where(valid, candidate, a1[..., idx]))

    return S