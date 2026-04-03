"""
MainGMRES - Find optimal input wavefront maximizing transmission
Translated from MainGMRES.m (Curtis Jin, 2011)

Uses iterative GMRES to find the input wavefront that maximizes
transmission through the scattering medium. Compares optimized
transmission with normal incidence transmission.

Usage:
    python main_gmres.py results_500cyl.npz
    python main_gmres.py --generate --num_cyl 100
"""
import sys
import os
import time
import argparse
import numpy as np
from scipy.sparse.linalg import gmres

sys.path.insert(0, '.')
sys.path.insert(0, './Scattering_Code')


def find_max_transmission(R, T, tol=1e-7, max_iter=2000, convergence_tol=0.5):
    """
    Iteratively apply GMRES to R (reflection matrix) to find the input
    wavefront b that minimizes reflection, thus maximizing transmission.

    Parameters
    ----------
    R : (m, m) complex array — reflection sub-matrix (S11)
    T : (m, m) complex array — transmission sub-matrix (S21)
    tol : float — GMRES tolerance
    max_iter : int — max GMRES iterations per step
    convergence_tol : float — convergence criterion for outer loop

    Returns
    -------
    b_opt : (m,) complex — optimal input vector
    trans_opt : float — optimized transmission coefficient
    trans_normal : float — normal incidence transmission coefficient
    """
    m = R.shape[0]

    # Initial random input
    b = np.random.randn(m) + 1j * np.random.randn(m)
    b = b / np.linalg.norm(b)

    RES = 100
    oldb = np.ones(m) * 100
    idx = 0

    print(f"  Iterating GMRES (tol={tol}, convergence_tol={convergence_tol})...")
    while RES > convergence_tol and idx < 200:
        # GMRES: solve R @ x = b (find x that minimizes ||R x - b||)
        b_new, info = gmres(R, b, tol=tol, maxiter=max_iter)
        b_new = b_new / np.linalg.norm(b_new)

        RES = np.linalg.norm(b_new - oldb)
        oldb = b_new.copy()
        b = b_new
        idx += 1

        if idx % 10 == 0:
            print(f"    iter {idx}: residual = {RES:.6e}")

    print(f"  Converged after {idx} iterations (residual = {RES:.6e})")

    # Normal incidence: center column
    center = m // 2
    normal_output = T[:, center]
    trans_normal = float(np.real(normal_output.conj() @ normal_output))

    # Optimized
    opt_output = T @ b
    trans_opt = float(np.real(opt_output.conj() @ opt_output))

    return b, trans_opt, trans_normal


def main():
    parser = argparse.ArgumentParser(description='GMRES transmission optimization')
    parser.add_argument('input_file', nargs='?', default=None,
                        help='Path to .npz file with S-matrix results')
    parser.add_argument('--generate', action='store_true',
                        help='Generate S-matrix instead of loading')
    parser.add_argument('--num_cyl', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.input_file and os.path.exists(args.input_file):
        print(f"Loading from {args.input_file}...")
        data = np.load(args.input_file)

        if 'tau' in data and 'rho' in data:
            # This is a results file from compute_ncyl.py
            # We need the actual S-matrix sub-blocks, not just SVDs
            print("ERROR: This .npz file only contains SVD results, not the full S-matrix.")
            print("Re-run compute_ncyl.py and save the full S-matrix, or use --generate.")
            return
        elif 'S' in data:
            S = data['S']
            nmax = int(data['nmax'])
            NoEvaMode = int(data.get('NoEvaMode', 0))
        else:
            print(f"ERROR: Unrecognized .npz format. Keys: {list(data.keys())}")
            return
    elif args.generate:
        from Scattering_Code.smatrix_parameters import smatrix_parameters
        from Scattering_Code.smatrix_cascade import smatrix_cascade

        wavelength = 0.93
        period = 12.81
        radius = 0.25
        epsilon = 2.0
        mu = 1.0
        cmmax = 5
        phiinc = np.pi / 2
        num_cyl = args.num_cyl

        spacing = 2.5 * radius
        cyls_per_row = int(period / spacing)
        rows_needed = num_cyl / cyls_per_row + 2
        thickness = max(0.5, rows_needed * spacing * 1.5)
        thickness = round(thickness, 1)

        NoPropagatingModes = int(np.floor(period / wavelength))
        nmax = NoPropagatingModes
        NoEvaMode = 0
        d = thickness

        np.random.seed(args.seed)
        margin = radius * 1.5
        clocs = np.zeros((num_cyl, 2))
        for i in range(num_cyl):
            for attempt in range(5000):
                x = np.random.uniform(margin, period - margin)
                y = np.random.uniform(margin, thickness - margin)
                if i == 0 or np.all(np.sqrt((x - clocs[:i, 0])**2 +
                                            (y - clocs[:i, 1])**2) > 2.5 * radius):
                    clocs[i] = [x, y]
                    break

        cmmaxs = cmmax * np.ones(num_cyl, dtype=int)
        cepmus = np.column_stack([epsilon * np.ones(num_cyl), np.ones(num_cyl)])
        crads = radius * np.ones(num_cyl)

        sp = smatrix_parameters(wavelength, period, phiinc,
                                1e-11, 1e-4, 5, 3, 1000, 3, 5, 1, period / 120)

        print(f"Computing S-matrix for {num_cyl} cylinders...")
        S, _ = smatrix_cascade(clocs, cmmaxs, cepmus, crads, period, wavelength,
                                nmax, d, sp, 'On')
    else:
        print("Usage: python main_gmres.py <results.npz>")
        print("   or: python main_gmres.py --generate --num_cyl 100")
        return

    # Extract R and T sub-matrices
    nm = 2 * nmax + 1
    S11 = S[:nm, :nm]
    S21 = S[nm:, :nm]

    if NoEvaMode > 0:
        R = S11[NoEvaMode:-NoEvaMode, NoEvaMode:-NoEvaMode]
        T = S21[NoEvaMode:-NoEvaMode, NoEvaMode:-NoEvaMode]
    else:
        R = S11
        T = S21

    m = R.shape[0]
    print(f"\nS-matrix loaded: R is {m}x{m}, T is {T.shape[0]}x{T.shape[1]}")

    # Run GMRES optimization
    b_opt, trans_opt, trans_normal = find_max_transmission(R, T)

    gain = trans_opt / trans_normal if trans_normal > 0 else float('inf')

    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"  Normal incidence transmission:  {trans_normal:.6f}")
    print(f"  Optimized transmission:         {trans_opt:.6f}")
    print(f"  Gain: {gain:.2f}x")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
