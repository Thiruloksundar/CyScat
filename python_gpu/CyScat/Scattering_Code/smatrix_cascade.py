"""
smatrix_cascade - Auto-cascading S-matrix computation with optional multi-GPU.

Single-GPU (num_gpus=1):
  Incremental cascade — compute group, cascade immediately, free memory.
  Only 2 S-matrices in RAM at any time.

Multi-GPU (num_gpus>1):
  num_gpus worker processes, one per GPU.  Each worker is assigned a BATCH
  of groups to process sequentially on its GPU (no GPU conflicts).
  Uses Pool initializer to set CUDA_VISIBLE_DEVICES before CuPy is imported
  in each worker, so the parent process never touches any GPU.
"""

import numpy as np
import time
import sys
import gc
import os

from .smatrix import smatrix
from .cascadertwo import cascadertwo
from . import gpu_backend as gb


# ─────────────────────────────────────────────────────────────────────────────
# Group preparation
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_groups(clocs, cmmaxs, cepmus, crads, no_cylinders, num_groups):
    """Sort cylinders by y-position and split into groups."""
    sorted_idx = np.argsort(clocs[:, 1])
    s_clocs = clocs[sorted_idx]
    s_cmmaxs = cmmaxs[sorted_idx]
    s_cepmus = cepmus[sorted_idx]
    s_crads = crads[sorted_idx]

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
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Multi-GPU worker helpers (top-level = picklable)
# ─────────────────────────────────────────────────────────────────────────────

def _worker_init_gpu(gpu_queue):
    """
    Pool initializer — runs ONCE per worker process, BEFORE any task module
    is imported.  Claims a unique GPU via CUDA_VISIBLE_DEVICES so each worker
    sees exactly one GPU (device 0 = its assigned physical GPU).
    Also removes CYSCAT_FORCE_CPU so the worker can use the GPU.
    """
    try:
        gpu_id = gpu_queue.get_nowait()
    except Exception:
        gpu_id = 0

    os.environ.pop('CYSCAT_FORCE_CPU', None)          # parent may have set this
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)  # worker sees only this GPU
    print(f"  [GPU worker] CUDA_VISIBLE_DEVICES={gpu_id}", flush=True)


def _process_gpu_batch(args):
    """
    Worker task: process a batch of groups SEQUENTIALLY on the GPU assigned
    by CUDA_VISIBLE_DEVICES (set by _worker_init_gpu).
    Returns list of (g_idx, S_matrix, thickness).
    """
    group_batch, period, lambda_wave, nmax, sp, interaction, total_groups = args

    # Import here so gpu_backend is initialized after CUDA_VISIBLE_DEVICES is set
    from .smatrix import smatrix as _smat

    results = []
    for (g_idx, g_clocs, g_cmmaxs, g_cepmus, g_crads, g_thickness) in group_batch:
        t0 = time.time()
        n_in = len(g_cmmaxs)
        print(f"  Group {g_idx + 1}/{total_groups}: {n_in} cyls "
              f"[GPU {os.environ.get('CUDA_VISIBLE_DEVICES', '?')}]", flush=True)

        S_g, _ = _smat(g_clocs, g_cmmaxs, g_cepmus, g_crads,
                       period, lambda_wave, nmax, g_thickness, sp, interaction)

        results.append((g_idx, S_g, g_thickness))
        print(f"  Group {g_idx + 1} done in {time.time() - t0:.1f}s", flush=True)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def smatrix_cascade(clocs, cmmaxs, cepmus, crads, period, lambda_wave, nmax, d, sp,
                    interaction, cascade_threshold=50, cylinders_per_group=100, num_gpus=1):
    """
    Compute S-matrix with automatic cascading.

    Extra parameters
    ----------------
    cascade_threshold   : use direct smatrix() for fewer cylinders
    cylinders_per_group : target group size
    num_gpus            : 1 = incremental single-GPU; >1 = parallel batch multi-GPU
    """
    no_cylinders = len(cmmaxs)

    if no_cylinders <= cascade_threshold:
        print(f"Direct computation ({no_cylinders} cylinders)")
        return smatrix(clocs, cmmaxs, cepmus, crads, period, lambda_wave,
                       nmax, d, sp, interaction)

    num_groups = max(2, int(np.ceil(no_cylinders / cylinders_per_group)))
    groups = _prepare_groups(clocs, cmmaxs, cepmus, crads, no_cylinders, num_groups)
    actual_groups = len(groups)

    print("=" * 60)
    print(f"AUTO-CASCADE: {no_cylinders} cyls → {actual_groups} groups "
          f"(~{int(np.ceil(no_cylinders / actual_groups))} each)")
    print(f"Backend: {gb.get_info()} | GPUs: {num_gpus}")
    print("=" * 60)
    sys.stdout.flush()

    total_start = time.time()

    if num_gpus > 1:
        # ── Multi-GPU: one worker process per GPU ───────────────────────────
        import multiprocessing
        ctx = multiprocessing.get_context('spawn')

        # Each worker pops a unique GPU ID from the queue in its initializer
        gpu_id_queue = ctx.Queue()
        for i in range(num_gpus):
            gpu_id_queue.put(i)

        # Assign groups to GPUs in round-robin, build per-GPU batches
        batches = [[] for _ in range(num_gpus)]
        for g_idx, (g, g_clocs, g_cmmaxs, g_cepmus, g_crads, g_thickness) in enumerate(groups):
            batches[g_idx % num_gpus].append(
                (g_idx, g_clocs, g_cmmaxs, g_cepmus, g_crads, g_thickness))

        worker_args = [
            (batch, period, lambda_wave, nmax, sp, interaction, actual_groups)
            for batch in batches if batch
        ]
        n_workers = len(worker_args)

        print(f"\nLaunching {n_workers} GPU workers "
              f"(~{len(batches[0])} groups/GPU, sequential per GPU)...")
        sys.stdout.flush()

        with ctx.Pool(processes=n_workers,
                      initializer=_worker_init_gpu,
                      initargs=(gpu_id_queue,)) as pool:
            batch_results = pool.map(_process_gpu_batch, worker_args)

        # Flatten all results and sort by g_idx (spatial order for cascade)
        all_results = []
        for batch in batch_results:
            all_results.extend(batch)
        all_results.sort(key=lambda x: x[0])

        print(f"\nCascading {len(all_results)} groups sequentially...")
        sys.stdout.flush()
        cascade_start = time.time()

        Scas, dcas = all_results[0][1], all_results[0][2]
        for _, S_i, d_i in all_results[1:]:
            Scas, dcas = cascadertwo(Scas, dcas, S_i, d_i)

        cascade_time = time.time() - cascade_start

    else:
        # ── Single GPU: incremental cascade (minimal memory) ────────────────
        print(f"\nIncremental cascade (single GPU)...")
        sys.stdout.flush()

        Scas = None
        dcas = 0.0
        cascade_time = 0.0

        for g_idx, (g, g_clocs, g_cmmaxs, g_cepmus, g_crads, g_thickness) in enumerate(groups):
            n_in_group = len(g_cmmaxs)
            g_start = time.time()
            print(f"\nGroup {g_idx + 1}/{actual_groups}: {n_in_group} cyls, "
                  f"thickness={g_thickness:.3f}")
            sys.stdout.flush()

            S_g, _ = smatrix(g_clocs, g_cmmaxs, g_cepmus, g_crads,
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
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

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
        'method': 'cascade_multiGPU' if num_gpus > 1 else 'cascade_incremental',
    }

    return Scas, STP
