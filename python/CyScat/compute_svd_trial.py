"""
Single trial for SVD distribution study.
Usage: python compute_svd_trial.py <batch_id> [--n_cyl 2000] [--gpus 8] [--trials_per_job 40]
"""
import sys
import os
import time
import argparse
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, './Scattering_Code')

from get_partition import smat_to_s11, smat_to_s21


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('trial_id', type=int)
    parser.add_argument('--n_cyl', type=int, default=2000)
    parser.add_argument('--gpus', type=int, default=8)
    parser.add_argument('--trials_per_job', type=int, default=1)
    args = parser.parse_args()

    if args.gpus > 1:
        os.environ['CYSCAT_FORCE_CPU'] = '1'

    from Scattering_Code.smatrix_parameters import smatrix_parameters
    from Scattering_Code.smatrix_cascade import smatrix_cascade

    # === Fixed physics parameters ===
    num_cyl     = args.n_cyl
    wavelength  = 0.93
    period      = 12.81
    radius      = 0.01
    eps         = 1.3**2
    mu          = 1.0
    cmmax       = 5
    phiinc      = np.pi / 2

    spacing      = 2.5 * radius
    cyls_per_row = int(period / spacing)
    rows_needed  = num_cyl / cyls_per_row + 2
    thickness    = round(max(0.5, rows_needed * spacing * 1.5), 1)

    EvanescentModeTol  = 1e-2
    NoPropagatingModes = int(np.floor(period / wavelength))
    NoEvaMode = max(0, int(np.floor(
        period / (2 * np.pi) * np.sqrt(
            (np.log(EvanescentModeTol) / (2 * radius))**2
            + (2 * np.pi / wavelength)**2
        )
    )) - NoPropagatingModes)
    nmax = NoPropagatingModes + NoEvaMode
    d    = thickness

    # === These don't change between trials — compute once ===
    cmmaxs = np.full(num_cyl, cmmax)
    cepmus = np.tile([eps, mu], (num_cyl, 1))
    crads  = np.full(num_cyl, radius)
    sp     = smatrix_parameters(wavelength, period, phiinc,
                                1e-11, 1e-4, 5, 3, 1000, 3, 5, 1, period / 120)

    os.makedirs('svd_results', exist_ok=True)

    # === Loop over trials in this batch ===
    batch_start = args.trial_id * args.trials_per_job

    for offset in range(args.trials_per_job):
        trial_id = batch_start + offset

        outpath = f'svd_results/trial_{trial_id:04d}.npz'
        if os.path.exists(outpath):
            print(f"Trial {trial_id} already exists, skipping.", flush=True)
            continue

        print(f"\n--- Trial {trial_id} ({offset+1}/{args.trials_per_job}) ---", flush=True)
        print(f"  {num_cyl} cyl, thickness={thickness}, nmax={nmax}", flush=True)

        # === Random cylinder placement (seed = trial_id) ===
        rng    = np.random.default_rng(trial_id)
        margin = radius * 1.5
        clocs  = np.zeros((num_cyl, 2))
        placed = 0
        for i in range(num_cyl):
            for _ in range(10000):
                x = rng.uniform(margin, period - margin)
                y = rng.uniform(margin, thickness - margin)
                if i == 0 or np.all(
                    np.sqrt((x - clocs[:i, 0])**2 + (y - clocs[:i, 1])**2) > 2.5 * radius
                ):
                    clocs[i] = [x, y]
                    placed += 1
                    break
        print(f"  Placed {placed}/{num_cyl} cylinders", flush=True)

        # === Compute S-matrix ===
        t0 = time.time()
        S, STP = smatrix_cascade(clocs, cmmaxs, cepmus, crads, period, wavelength,
                                 nmax, d, sp, 'On', num_gpus=args.gpus)
        elapsed = time.time() - t0
        print(f"  S-matrix done in {elapsed:.1f}s", flush=True)

        # === Extract singular values ===
        S11 = smat_to_s11(S)
        S21 = smat_to_s21(S)
        if NoEvaMode > 0:
            S11T = S11[NoEvaMode:-NoEvaMode, NoEvaMode:-NoEvaMode]
            S21T = S21[NoEvaMode:-NoEvaMode, NoEvaMode:-NoEvaMode]
        else:
            S11T, S21T = S11, S21

        tau    = np.linalg.svd(S21T, compute_uv=False)
        rho    = np.linalg.svd(S11T, compute_uv=False)
        center = NoPropagatingModes
        T = np.sum(np.abs(S21T[:, center])**2)
        R = np.sum(np.abs(S11T[:, center])**2)
        print(f"  R={R:.4f}, T={T:.4f}, R+T={R+T:.4f}", flush=True)

        # === Save ===
        np.savez(outpath,
                 tau=tau, tau_sq=tau**2, rho=rho, rho_sq=rho**2,
                 R=R, T=T, elapsed=elapsed,
                 trial_id=trial_id, num_cyl=num_cyl)
        print(f"  Saved {outpath}", flush=True)


if __name__ == '__main__':
    main()
