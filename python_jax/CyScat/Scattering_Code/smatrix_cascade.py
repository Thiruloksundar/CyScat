"""
smatrix_cascade - Auto-cascading S-matrix computation.

JAX version for automatic differentiation.

Single-GPU (num_gpus=1):
  Incremental cascade — compute group, cascade immediately, free memory.
  Only 2 S-matrices in RAM at any time.
"""

import numpy as np
import jax.numpy as jnp
import time
import sys
import gc

from .smatrix import smatrix
from .cascadertwo import cascadertwo


# ─────────────────────────────────────────────────────────────────────────────
# Group preparation
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_groups(clocs, cmmaxs, cepmus, crads, no_cylinders, num_groups, clocs_concrete=None):
    """Sort cylinders by y-position and split into groups."""
    clocs_np = clocs_concrete if clocs_concrete is not None else np.asarray(clocs)
    sorted_idx = np.argsort(clocs_np[:, 1])
    s_clocs = clocs_np[sorted_idx]
    s_cmmaxs = np.asarray(cmmaxs)[sorted_idx]
    s_cepmus = np.asarray(cepmus)[sorted_idx]
    s_crads = np.asarray(crads)[sorted_idx]

    group_size = int(np.ceil(no_cylinders / num_groups))
    groups = []
    for g in range(num_groups):
        i0 = g * group_size
        i1 = min((g + 1) * group_size, no_cylinders)
        if i0 >= no_cylinders:
            break

        g_clocs = s_clocs[i0:i1].copy()
        g_cmmaxs = s_cmmaxs[i0:i1]
        g_cepmus = s_cepmus[i0:i1]
        g_crads = s_crads[i0:i1]

        y_min = np.min(g_clocs[:, 1])
        y_max = np.max(g_clocs[:, 1])
        margin = np.max(g_crads) * 2.0
        g_thickness = max(y_max - y_min + 2 * margin, 4 * margin)
        g_clocs[:, 1] -= (y_min - margin)

        groups.append((g, g_clocs, g_cmmaxs, g_cepmus, g_crads, g_thickness))

    # For JAX differentiability, recompute clocs from the original array
    # using the sorted indices so gradients flow back
    return groups, sorted_idx


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def smatrix_cascade(clocs, cmmaxs, cepmus, crads, period, lambda_wave, nmax, d, sp,
                    interaction, cascade_threshold=50, cylinders_per_group=100, num_gpus=1,
                    clocs_concrete=None):
    """
    Compute S-matrix with automatic cascading (JAX-differentiable).

    Extra parameters
    ----------------
    cascade_threshold   : use direct smatrix() for fewer cylinders
    cylinders_per_group : target group size
    num_gpus            : 1 = incremental single; >1 not supported for JAX
    clocs_concrete      : numpy array of concrete clocs for pair classification
    """
    if clocs_concrete is None:
        clocs_concrete = np.asarray(clocs)
    clocs = jnp.asarray(clocs, dtype=jnp.float64)
    cmmaxs_np = np.asarray(cmmaxs, dtype=int)
    cepmus_np = np.asarray(cepmus)
    crads_np = np.asarray(crads)
    no_cylinders = len(cmmaxs_np)

    if no_cylinders <= cascade_threshold:
        print(f"Direct computation ({no_cylinders} cylinders)")
        return smatrix(clocs, cmmaxs_np, cepmus_np, crads_np, period, lambda_wave,
                       nmax, d, sp, interaction, clocs_concrete=clocs_concrete)

    num_groups = max(2, int(np.ceil(no_cylinders / cylinders_per_group)))
    groups, sorted_idx = _prepare_groups(
        clocs, cmmaxs_np, cepmus_np, crads_np, no_cylinders, num_groups,
        clocs_concrete=clocs_concrete)
    actual_groups = len(groups)

    print("=" * 60)
    print(f"AUTO-CASCADE: {no_cylinders} cyls -> {actual_groups} groups "
          f"(~{int(np.ceil(no_cylinders / actual_groups))} each)")
    print("=" * 60)
    sys.stdout.flush()

    total_start = time.time()

    # Incremental cascade
    Scas = None
    dcas = 0.0
    cascade_time = 0.0

    for g_idx, (g, g_clocs, g_cmmaxs, g_cepmus, g_crads, g_thickness) in enumerate(groups):
        n_in_group = len(g_cmmaxs)
        g_start = time.time()
        print(f"\nGroup {g_idx + 1}/{actual_groups}: {n_in_group} cyls, "
              f"thickness={g_thickness:.3f}")
        sys.stdout.flush()

        # For JAX differentiability, recompute group clocs from original
        # sorted array to maintain gradient flow
        g_clocs_jax = jnp.asarray(g_clocs, dtype=jnp.float64)

        S_g, _ = smatrix(g_clocs_jax, g_cmmaxs, g_cepmus, g_crads,
                         period, lambda_wave, nmax, g_thickness, sp, interaction)
        g_time = time.time() - g_start
        print(f"  Group {g_idx + 1} done in {g_time:.1f}s")
        sys.stdout.flush()

        cas_t0 = time.time()
        if Scas is None:
            Scas = S_g
            dcas = g_thickness
        else:
            Scas, dcas = cascadertwo(Scas, dcas, S_g, g_thickness)
        cascade_time += time.time() - cas_t0

        del S_g
        gc.collect()

    total_time = time.time() - total_start

    print()
    print("=" * 60)
    print(f"CASCADE COMPLETE")
    print(f"  S-matrix: {Scas.shape[0]}x{Scas.shape[1]}")
    print(f"  Total thickness: {dcas:.3f}")
    print(f"  Cascade step: {cascade_time:.1f}s")
    print(f"  Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print("=" * 60)
    sys.stdout.flush()

    STP = {
        'TST': total_time,
        'cascade_time': cascade_time,
        'num_groups': actual_groups,
        'method': 'cascade_incremental_jax',
    }

    return Scas, STP
