"""
transall Code
Coder: Curtis Jin
Date: 2011/FEB/13th Sunday
Contact: jsirius@umich.edu
Description: Professor Michelson's version
           : translational matrix generating code
             Using Modified Shanks Transformation
             Exploiting symmetric structure of T-Matrix

GPU-accelerated version:
  - Spectral pairs: batched GPU computation
  - Spatial pairs: vectorized active-mask convergence (all pairs x modes at once)
"""

import numpy as np
from scipy.special import hankel2
from scipy.linalg import toeplitz
from .gpu_backend import GPU_AVAILABLE, xp, to_gpu, to_cpu
import sys


def transall(clocs, cmmaxs, period, lambda_wave, phiinc, sp, total_steps):
    """
    Generate translational matrix for all cylinders.
    Spectral pairs batched on GPU; spatial pairs vectorized with active mask.
    """
    no_cylinders = len(cmmaxs)
    tot_no_modes = int(np.sum(cmmaxs * 2 + 1))
    t = np.zeros((tot_no_modes, tot_no_modes), dtype=complex)

    max_order = int(np.max(cmmaxs))

    # Self-sum (spatial path, computed on CPU via transper)
    cms_self = np.arange(-2 * max_order, 2 * max_order + 1)
    self_sum_vector = transvector(np.array([0.0, 0.0]), np.array([0.0, 0.0]),
                                  cms_self, period, lambda_wave, phiinc, sp)

    # Fill diagonal blocks with self-sum toeplitz
    istart = 0
    for icyl in range(no_cylinders):
        cmmaxob = int(cmmaxs[icyl])
        offset = 2 * max_order
        c = self_sum_vector[offset:offset + 2 * cmmaxob + 1]
        r = np.flipud(self_sum_vector[offset - 2 * cmmaxob:offset + 1])
        t11 = toeplitz(c, r)
        t[istart:istart + 2 * cmmaxob + 1,
          istart:istart + 2 * cmmaxob + 1] = t11
        istart += 2 * cmmaxob + 1

    # Classify all cross-pairs as spectral or spatial
    epsloc = sp['epsloc']
    spectral_cond = sp['spectralCond']

    pairs = []
    istart = 0
    for icyl in range(no_cylinders):
        cmmaxob = int(cmmaxs[icyl])
        jstart = istart + 2 * cmmaxob + 1
        for jcyl in range(icyl + 1, no_cylinders):
            cmmaxso = int(cmmaxs[jcyl])
            rv = clocs[icyl] - clocs[jcyl]
            r = np.linalg.norm(rv)
            y = rv[1]
            is_spectral = not (r < epsloc or abs(y) < spectral_cond or period < 0)
            pairs.append((icyl, jcyl, istart, jstart, cmmaxob, cmmaxso, is_spectral))
            jstart += 2 * cmmaxso + 1
        istart += 2 * cmmaxob + 1

    spectral_pairs = [p for p in pairs if p[6]]
    spatial_pairs = [p for p in pairs if not p[6]]

    # Batch compute spectral pairs on GPU
    if spectral_pairs:
        _compute_spectral_batch(t, clocs, spectral_pairs, sp)

    # Vectorized spatial pairs with active mask
    if spatial_pairs:
        _compute_spatial_vectorized(t, clocs, spatial_pairs, period, lambda_wave, phiinc, sp)

    n_total = len(pairs)
    n_spectral = len(spectral_pairs)
    n_spatial = len(spatial_pairs)
    print(f"  transall: {n_total} pairs ({n_spectral} spectral"
          f"{' [GPU]' if GPU_AVAILABLE and n_spectral > 0 else ''}"
          f", {n_spatial} spatial [vectorized])")
    sys.stdout.flush()

    return t


# ─────────────────────────────────────────────────────────────────────────────
# Spectral batch (GPU)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_spectral_batch(t, clocs, spectral_pairs, sp):
    """Batch compute spectral translation vectors on GPU and assign to T matrix."""
    groups = {}
    for p in spectral_pairs:
        key = (p[4], p[5])
        if key not in groups:
            groups[key] = []
        groups[key].append(p)

    for (cmmaxob, cmmaxso), group in groups.items():
        cms = np.arange(-cmmaxob - cmmaxso, cmmaxob + cmmaxso + 1)
        K = len(sp['kxs'].flatten())
        M = len(cms)

        x_arr = np.array([clocs[p[0], 0] - clocs[p[1], 0] for p in group])
        y_arr = np.array([clocs[p[0], 1] - clocs[p[1], 1] for p in group])
        P = len(x_arr)

        bytes_per_pair = K * M * 16 * 4
        chunk_size = max(1, int(2e9 / max(bytes_per_pair, 1)))

        t_all = np.zeros((P, M), dtype=complex)
        for start in range(0, P, chunk_size):
            end = min(start + chunk_size, P)
            t_all[start:end] = _spectral_sum_batch(
                x_arr[start:end], y_arr[start:end], cms, sp)

        for i, p in enumerate(group):
            _assign_pair(t, t_all[i], p[2], p[3], p[4], p[5])


def _spectral_sum_batch(x_arr, y_arr, cms, sp):
    """Batch spectral sum for multiple cylinder pairs (P,) → returns (P, M)."""
    kxs_all = sp['kxs'].flatten()
    kys_all = sp['kys'].flatten()
    angles_all = sp['Angles'].flatten()
    two_over_period = sp['TwoOverPeriod']

    if GPU_AVAILABLE:
        x = xp.asarray(x_arr, dtype=xp.float64)[:, None, None]
        y = xp.asarray(y_arr, dtype=xp.float64)[:, None, None]
        cms_g = xp.asarray(cms, dtype=xp.float64)[None, None, :]
        kxs = xp.asarray(kxs_all, dtype=xp.complex128)[None, :, None]
        kys = xp.asarray(kys_all, dtype=xp.complex128)[None, :, None]
        angs = xp.asarray(angles_all, dtype=xp.complex128)[None, :, None]

        sign_y = xp.sign(y)
        sign_y = xp.where(sign_y == 0.0, 1.0, sign_y)
        abs_y = xp.abs(y)

        exponent = kxs * x - kys * abs_y + cms_g * (np.pi - angs * sign_y)

        odd_mask = (xp.abs(cms_g.astype(xp.int64)) % 2 != 0)
        sign_power = xp.where(sign_y < 0,
                              xp.where(odd_mask, -1.0 + 0j, 1.0 + 0j),
                              1.0 + 0j)

        exp_term = xp.exp(1j * exponent)
        kys_safe = xp.where(xp.abs(kys) < 1e-10, 1e-10 + 0j, kys)
        t_matrix = sign_power * exp_term / kys_safe * two_over_period
        t_batch = xp.sum(t_matrix, axis=1)

        return to_cpu(t_batch)
    else:
        P = len(x_arr)
        t_batch = np.zeros((P, len(cms)), dtype=complex)
        for p in range(P):
            t_batch[p] = _spectral_sum_single(x_arr[p], y_arr[p], cms,
                                               kxs_all, kys_all, angles_all,
                                               two_over_period)
        return t_batch


def _spectral_sum_single(x, y, cms, kxs_all, kys_all, angles_all, two_over_period):
    """Single-pair spectral sum on CPU."""
    cmgrid, kxsgrid = np.meshgrid(cms, kxs_all)
    cmgrid, kysgrid = np.meshgrid(cms, kys_all)
    cmgrid, anglesgrid = np.meshgrid(cms, angles_all)

    sign_y = np.sign(y) if y != 0 else 1.0
    exponent = (kxsgrid * x - kysgrid * np.abs(y) +
                cmgrid * (np.pi - anglesgrid * sign_y))

    odd_mask = (np.abs(cmgrid.astype(int)) % 2 != 0)
    sign_power = np.where(sign_y < 0,
                          np.where(odd_mask, -1.0, 1.0),
                          1.0).astype(complex)

    exp_term = np.exp(1j * exponent)
    kysgrid_safe = np.where(np.abs(kysgrid) < 1e-10, 1e-10 + 0j, kysgrid)
    t_matrix = sign_power * exp_term / kysgrid_safe * two_over_period
    return np.sum(t_matrix, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Spatial vectorized (active mask)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_spatial_vectorized(t, clocs, spatial_pairs, period, lambda_wave, phiinc, sp):
    """Vectorized computation of all spatial pairs with active-mask convergence."""
    # Group by (cmmaxob, cmmaxso) so all pairs in a group share the same mode vector
    groups = {}
    for p in spatial_pairs:
        key = (p[4], p[5])
        if key not in groups:
            groups[key] = []
        groups[key].append(p)

    n_unconverged = 0
    for (cmmaxob, cmmaxso), group in groups.items():
        cms = np.arange(-cmmaxob - cmmaxso, cmmaxob + cmmaxso + 1)

        ob_locs = np.array([clocs[p[0]] for p in group])   # (P, 2)
        so_locs = np.array([clocs[p[1]] for p in group])   # (P, 2)

        t_batch, n_not_conv = _spatial_sum_vectorized(
            ob_locs, so_locs, cms, period, lambda_wave, phiinc, sp)
        n_unconverged += n_not_conv

        for i, p in enumerate(group):
            _assign_pair(t, t_batch[i], p[2], p[3], p[4], p[5])

    if n_unconverged > 0:
        print(f"  WARNING: {n_unconverged}/{len(spatial_pairs)} spatial pairs "
              f"did not converge (j={sp['jmax']})")


def _shanks_batch(newentry, a1, a2):
    """
    Vectorized modified epsilon Shanks transform.

    Parameters
    ----------
    newentry : (...) complex  — current partial sum (any batch shape)
    a1, a2   : (..., K) complex  — Shanks state arrays, K = kshanks+2

    Returns
    -------
    S : (..., K) complex  — updated state (becomes new a1)
    """
    K = a1.shape[-1]
    S = np.empty_like(a1)
    S[..., 0] = np.inf
    S[..., 1] = newentry

    for idx in range(1, K - 1):
        with np.errstate(divide='ignore', invalid='ignore'):
            f1 = 1.0 / (a2[..., idx] - a1[..., idx])
            f2 = 1.0 / (S[..., idx] - a1[..., idx])
            f3 = 1.0 / (a2[..., idx - 1] - a1[..., idx])
            candidate = 1.0 / (f1 + f2 - f3) + a1[..., idx]
        valid = (np.isfinite(f1) & np.isfinite(f2) &
                np.isfinite(f3) & np.isfinite(candidate))
        S[..., idx + 1] = np.where(valid, candidate, a1[..., idx])

    return S


def _spatial_sum_vectorized(ob_locs, so_locs, cms_f, period, lambda_wave, phiinc, sp):
    """
    Vectorized spatial (transper) sum for P pairs × M modes with active-mask
    convergence.  Replaces serial transper loop, giving ~10x speedup.

    Parameters
    ----------
    ob_locs : (P, 2) observer positions
    so_locs : (P, 2) source positions
    cms_f   : (M,) mode index array
    period, lambda_wave, phiinc : scalars
    sp : parameter dict

    Returns
    -------
    t_batch : (P, M) complex translation vectors
    n_not_converged : int  number of pairs that hit jmax without converging
    """
    k = sp['k0']
    epsseries = sp['epsseries']
    epsloc = sp['epsloc']
    jmax = sp['jmax']
    kshanks = sp['kshanksSpatial']
    nrepeat = sp['nrepeatSpatial']

    P = len(ob_locs)
    M = len(cms_f)
    cms_int = np.round(cms_f).astype(int)   # (M,) integer orders

    # ── j=0 term: trans(ob, cm, so) for each pair and mode ──────────────────
    rv = ob_locs - so_locs                       # (P, 2)
    r_vec = np.linalg.norm(rv, axis=1)           # (P,)
    phip_vec = np.arctan2(rv[:, 1], rv[:, 0])    # (P,)

    r_2d = r_vec[:, None]        # (P, 1)
    n_2d = cms_int[None, :]      # (1, M)
    phip_2d = phip_vec[:, None]  # (P, 1)

    with np.errstate(divide='ignore', invalid='ignore'):
        h0 = np.where(r_2d >= epsloc, hankel2(n_2d, k * r_2d), 0.0 + 0j)
    h0 = np.where(np.isfinite(h0), h0, 0.0 + 0j)
    t = h0 * np.exp(1j * n_2d * (np.pi - phip_2d))   # (P, M)

    # ── Shanks initialisation ────────────────────────────────────────────────
    if kshanks > 0:
        K = kshanks + 2
        # a1[p, m, 0] = inf, a1[p, m, 1] = t_initial, rest = 0
        a1 = np.zeros((P, M, K), dtype=complex)
        a2 = np.zeros((P, M, K), dtype=complex)
        a1[:, :, 0] = np.inf
        a2[:, :, 0] = np.inf
        a1[:, :, 1] = t

    ts = t.copy()               # Shanks estimate, updated once j > 2*kshanks+1
    active = np.ones(P, dtype=bool)
    irepeat = np.zeros(P, dtype=int)
    period_vec = np.array([period, 0.0])
    # Phase factor for j-th image: exp(-i*k*j*period*cos(phiinc))
    phase_base = np.exp(-1j * k * period * np.cos(phiinc))

    for j in range(1, jmax):
        if not np.any(active):
            break

        act_idx = np.where(active)[0]   # indices of active pairs
        Pa = len(act_idx)

        # Displaced source positions for +j and -j images
        so_pos = so_locs[act_idx] + j * period_vec   # (Pa, 2)
        so_neg = so_locs[act_idx] - j * period_vec   # (Pa, 2)

        rv_pos = ob_locs[act_idx] - so_pos           # (Pa, 2)
        rv_neg = ob_locs[act_idx] - so_neg           # (Pa, 2)

        r_pos = np.linalg.norm(rv_pos, axis=1)[:, None]          # (Pa, 1)
        r_neg = np.linalg.norm(rv_neg, axis=1)[:, None]          # (Pa, 1)
        phi_pos = np.arctan2(rv_pos[:, 1], rv_pos[:, 0])[:, None]
        phi_neg = np.arctan2(rv_neg[:, 1], rv_neg[:, 0])[:, None]

        n_2d_a = cms_int[None, :]   # (1, M)

        with np.errstate(divide='ignore', invalid='ignore'):
            h_pos = np.where(r_pos >= epsloc, hankel2(n_2d_a, k * r_pos), 0.0 + 0j)
            h_neg = np.where(r_neg >= epsloc, hankel2(n_2d_a, k * r_neg), 0.0 + 0j)
        h_pos = np.where(np.isfinite(h_pos), h_pos, 0.0 + 0j)
        h_neg = np.where(np.isfinite(h_neg), h_neg, 0.0 + 0j)

        add = (phase_base ** j  * h_pos * np.exp(1j * n_2d_a * (np.pi - phi_pos)) +
               phase_base ** (-j) * h_neg * np.exp(1j * n_2d_a * (np.pi - phi_neg)))

        told = t[act_idx].copy()
        t[act_idx] += add

        if kshanks > 0:
            new_S = _shanks_batch(t[act_idx], a1[act_idx], a2[act_idx])
            a2[act_idx] = a1[act_idx].copy()
            a1[act_idx] = new_S

            if j <= 2 * kshanks + 1:
                rerror = np.ones(Pa)   # not enough terms yet
            else:
                tsold_a = ts[act_idx].copy()
                ts[act_idx] = a1[act_idx, :, -1]
                rerror = np.max(np.abs(ts[act_idx] - tsold_a), axis=1)
        else:
            rerror = np.max(np.abs(t[act_idx] - told), axis=1)

        # Update active mask
        for pi_i, idx in enumerate(act_idx):
            if rerror[pi_i] < epsseries:
                irepeat[idx] += 1
                if irepeat[idx] >= nrepeat:
                    active[idx] = False
            else:
                irepeat[idx] = 0

    n_not_converged = int(np.sum(active))
    # Safety: replace any remaining NaNs/infs with the un-accelerated value
    if kshanks > 0:
        bad = ~np.isfinite(ts)
        if np.any(bad):
            ts = np.where(bad, t_before_shanks, ts)
        t = ts

    return t, n_not_converged


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _assign_pair(t, tvector, istart, jstart, cmmaxob, cmmaxso):
    """Build toeplitz matrices from translation vector and assign to T matrix."""
    offset = cmmaxso + cmmaxob

    start_idx = cmmaxso - cmmaxob + offset
    c = tvector[start_idx:]
    r = np.flipud(tvector[:start_idx + 1])
    t12 = toeplitz(c, r)

    start_idx = cmmaxob - cmmaxso + offset
    c = tvector[start_idx:]
    r = np.flipud(tvector[:start_idx + 1])
    t21 = toeplitz(c, r)

    c_mod = (-1) ** (np.arange(2 * cmmaxso + 1) + (cmmaxob - cmmaxso))
    r_mod = (-1) ** (np.arange(2 * cmmaxob + 1) + (cmmaxob - cmmaxso))
    modification_matrix = toeplitz(c_mod, r_mod)
    t21 = modification_matrix * t21

    t[istart:istart + 2 * cmmaxob + 1,
      jstart:jstart + 2 * cmmaxso + 1] = t12
    t[jstart:jstart + 2 * cmmaxso + 1,
      istart:istart + 2 * cmmaxob + 1] = t21


def transvector(clocob, clocso, cms, period, lambda_wave, phiinc, sp):
    """
    Calculate translation vector for a pair of cylinders.
    Used for spatial path (close cylinders) and self-sum.
    """
    epsloc = sp['epsloc']
    rv = clocob - clocso
    r = np.linalg.norm(rv)
    x = rv[0]
    y = rv[1]
    cms = np.atleast_1d(cms).flatten()

    if r < epsloc or abs(y) < sp['spectralCond'] or period < 0:
        max_cm = int(np.max(cms))
        min_cm = int(np.min(cms))
        array_size = max_cm - min_cm + 1
        temp = np.zeros(array_size, dtype=complex)
        for cm in cms:
            cm_int = int(cm)
            idx = cm_int - min_cm
            temp[idx] = transper(clocob, cm_int, clocso, 0, period,
                                 lambda_wave, phiinc, sp)
        return temp
    else:
        kxs_all = sp['kxs'].flatten()
        kys_all = sp['kys'].flatten()
        angles_all = sp['Angles'].flatten()
        return _spectral_sum_single(x, y, cms, kxs_all, kys_all,
                                    angles_all, sp['TwoOverPeriod'])


def transper(clocob, cmob, clocso, cmso, period, lambda_wave, phiinc, sp):
    """Calculate periodic translation coefficient (serial, used for self-sum)."""
    k = sp['k0']
    epsseries = sp['epsseries']
    epsloc = sp['epsloc']
    jmax = sp['jmax']

    spectral_internal = sp['spectral']
    if spectral_internal < 0:
        kshanks = sp['kshanksSpatial']
        nrepeat = sp['nrepeatSpatial']
    else:
        kshanks = sp['kshanksSpectral']
        nrepeat = sp['nrepeatSpectral']

    rv = clocob - clocso
    r = np.linalg.norm(rv)
    x = rv[0]
    y = rv[1]

    if period < 0:
        if r < epsloc:
            return 0.0 + 0.0j
        return trans(clocob, cmob, clocso, cmso, lambda_wave, sp)

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
                t = trans(clocob, cmob, clocso, cmso, lambda_wave, sp)
            if kshanks > 0:
                ashankseps1[1] = t
            ts = 1
        else:
            cm = cmso - cmob
            exponent = (sp['kxs'][sp['MiddleIndex']] * x +
                        sp['kys'][sp['MiddleIndex']] * abs(y) +
                        cm * (np.sign(y) * sp['Angles'][sp['MiddleIndex']] - np.pi))
            t = (np.sign(y)**cm * np.exp(-1j * exponent) /
                 sp['kys'][sp['MiddleIndex']] * sp['TwoOverPeriod'])
            if kshanks > 0:
                ashankseps1[1] = t
            ts = 1

        j = 1
        irepeat = 0

        while irepeat < nrepeat and j < jmax:
            if spectral_internal < 0:
                add = (np.exp(-1j * k * j * period * np.cos(phiinc)) *
                       trans(clocob, cmob, clocso + j * np.array([period, 0]), cmso, lambda_wave, sp) +
                       np.exp(-1j * k * (-j) * period * np.cos(phiinc)) *
                       trans(clocob, cmob, clocso - j * np.array([period, 0]), cmso, lambda_wave, sp))
            else:
                kxp = sp['kxs'][sp['MiddleIndex'] + j]
                kxm = sp['kxs'][sp['MiddleIndex'] - j]
                exponent1 = (kxp * x + sp['kys'][sp['MiddleIndex'] + j] * abs(y) +
                             cm * (np.sign(y) * sp['Angles'][sp['MiddleIndex'] + j] - np.pi))
                exponent2 = (kxm * x + sp['kys'][sp['MiddleIndex'] - j] * abs(y) +
                             cm * (np.sign(y) * sp['Angles'][sp['MiddleIndex'] - j] - np.pi))
                add = (np.sign(y)**cm * sp['TwoOverPeriod'] *
                       (np.exp(-1j * exponent1) / sp['kys'][sp['MiddleIndex'] + j] +
                        np.exp(-1j * exponent2) / sp['kys'][sp['MiddleIndex'] - j]))

            told = t
            t = t + add

            if kshanks <= 0:
                rerror = abs(t - told)
            else:
                from .modified_epsilon_shanks import modified_epsilon_shanks
                S = modified_epsilon_shanks(t, ashankseps1, ashankseps2)
                ashankseps2 = ashankseps1.copy()
                ashankseps1 = S.copy()
                if j <= 2 * kshanks + 1:
                    rerror = 1
                else:
                    tsold = ts
                    ts = S[-1]
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


def trans(clocob, cmob, clocso, cmso, lambda_wave, sp):
    """Calculate basic translation coefficient (non-periodic)."""
    k = sp['k0']
    rv = clocob - clocso
    r = np.linalg.norm(rv)
    epsloc = sp.get('epsloc', 1e-4)

    if r < epsloc:
        return 0.0 + 0.0j

    phip = np.arctan2(rv[1], rv[0])
    kr = k * r
    n = int(cmob - cmso)

    if kr < 1e-8:
        return 0.0 + 0.0j

    h = hankel2(n, kr)
    if not np.isfinite(h):
        return 0.0 + 0.0j

    t = h * np.exp(1j * (cmso - cmob) * (phip - np.pi))
    if not np.isfinite(t):
        return 0.0 + 0.0j

    return t
