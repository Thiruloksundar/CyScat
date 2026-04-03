"""
Multi-node version of compute_ncyl.py using MPI.
Each MPI rank runs on one node and independently computes an S-matrix
using all 8 GPUs on its local node.

Test: 2 nodes = 16 GPUs running simultaneously.
"""
import sys
import os
import time
import argparse
import numpy as np

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()   # which node am I? (0 or 1)
size = comm.Get_size()   # total nodes (2)

sys.path.insert(0, '.')
sys.path.insert(0, './Scattering_Code')

from get_partition import smat_to_s11, smat_to_s12, smat_to_s21, smat_to_s22


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_cyl', type=int, default=500)
    parser.add_argument('--gpus', type=int, default=8)
    args = parser.parse_args()

    # Critical: set FORCE_CPU before CuPy import on every rank
    if args.gpus > 1:
        os.environ['CYSCAT_FORCE_CPU'] = '1'

    from Scattering_Code.smatrix_parameters import smatrix_parameters
    from Scattering_Code.smatrix_cascade import smatrix_cascade

    print(f"[Rank {rank}/{size}] Starting — seed={rank}, gpus={args.gpus}", flush=True)

    # === Physics parameters ===
    num_cyl    = args.n_cyl
    wavelength = 0.93
    period     = 12.81
    radius     = 0.01
    eps        = 1.3**2
    mu         = 1.0
    cmmax      = 5
    phiinc     = np.pi / 2

    spacing      = 2.5 * radius
    cyls_per_row = int(period / spacing)
    rows_needed  = num_cyl / cyls_per_row + 2
    thickness    = round(max(0.5, rows_needed * spacing * 1.5), 1)

    EvanescentModeTol  = 1e-2
    NoPropagatingModes = int(np.floor(period / wavelength))
    NoEvaMode = max(0, int(np.floor(
        period / (2*np.pi) * np.sqrt(
            (np.log(EvanescentModeTol) / (2*radius))**2 + (2*np.pi/wavelength)**2
        )
    )) - NoPropagatingModes)
    nmax = NoPropagatingModes + NoEvaMode
    d    = thickness

    # === Random cylinder placement — each rank uses its rank as seed ===
    rng    = np.random.default_rng(rank)
    margin = radius * 1.5
    clocs  = np.zeros((num_cyl, 2))
    placed = 0
    for i in range(num_cyl):
        for _ in range(5000):
            x = rng.uniform(margin, period - margin)
            y = rng.uniform(margin, thickness - margin)
            if i == 0 or np.all(
                np.sqrt((x - clocs[:i,0])**2 + (y - clocs[:i,1])**2) > 2.5*radius
            ):
                clocs[i] = [x, y]
                placed += 1
                break
    print(f"[Rank {rank}] Placed {placed}/{num_cyl} cylinders", flush=True)

    cmmaxs = np.full(num_cyl, cmmax)
    cepmus = np.tile([eps, mu], (num_cyl, 1))
    crads  = np.full(num_cyl, radius)

    sp = smatrix_parameters(wavelength, period, phiinc,
                            1e-11, 1e-4, 5, 3, 1000, 3, 5, 1, period/120)

    # === Compute S-matrix — uses all 8 GPUs on this node ===
    print(f"[Rank {rank}] Computing S-matrix...", flush=True)
    t0 = time.time()
    S, STP = smatrix_cascade(clocs, cmmaxs, cepmus, crads, period, wavelength,
                             nmax, d, sp, 'On', num_gpus=args.gpus)
    elapsed = time.time() - t0
    print(f"[Rank {rank}] Done in {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)

    # === Results ===
    S11 = smat_to_s11(S);  S12 = smat_to_s12(S)
    S21 = smat_to_s21(S);  S22 = smat_to_s22(S)

    if NoEvaMode > 0:
        S11T = S11[NoEvaMode:-NoEvaMode, NoEvaMode:-NoEvaMode]
        S12T = S12[NoEvaMode:-NoEvaMode, NoEvaMode:-NoEvaMode]
        S21T = S21[NoEvaMode:-NoEvaMode, NoEvaMode:-NoEvaMode]
        S22T = S22[NoEvaMode:-NoEvaMode, NoEvaMode:-NoEvaMode]
        STruncated = np.block([[S11T, S12T], [S21T, S22T]])
    else:
        S11T, S12T, S21T, S22T = S11, S12, S21, S22
        STruncated = S

    svals = np.linalg.svd(STruncated, compute_uv=False)
    tau   = np.linalg.svd(S21T, compute_uv=False)
    rho   = np.linalg.svd(S11T, compute_uv=False)
    center = NoPropagatingModes
    T   = np.sum(np.abs(S21T[:, center])**2)
    R   = np.sum(np.abs(S11T[:, center])**2)
    DOU = np.sum(np.abs(STruncated.conj().T @ STruncated)) / len(STruncated)

    print(f"[Rank {rank}] R={R:.4f}, T={T:.4f}, R+T={R+T:.4f}, "
          f"DOU={DOU:.6f}, max_sv={np.max(svals):.4f}", flush=True)

    np.savez(f'results_rank{rank}_{num_cyl}cyl.npz',
             tau=tau, tau_sq=tau**2, rho=rho, rho_sq=rho**2,
             svals=svals, DOU=DOU, R=R, T=T,
             num_cyl=num_cyl, seed=rank, elapsed=elapsed)
    print(f"[Rank {rank}] Saved results_rank{rank}_{num_cyl}cyl.npz", flush=True)

    # Wait for all ranks then print summary from rank 0
    comm.Barrier()
    if rank == 0:
        print(f"\n{'='*60}", flush=True)
        print(f"All {size} nodes completed successfully!", flush=True)
        print(f"{'='*60}", flush=True)


if __name__ == '__main__':
    main()
