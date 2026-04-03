"""
MainCascadeResults - Load and cascade pre-computed S-matrices
Translated from MainCascadeResults.m (Curtis Jin, 2011)

Loads S-matrices from .mat or .npz files and cascades them together.
Saves intermediate results every N steps.

Usage:
    # Load .mat files from MATLAB Scattering Code folder
    python main_cascade_results.py --matdir "/path/to/Scattering Code" --count 10

    # Load .mat files from local nyx_matfiles directory
    python main_cascade_results.py --matdir nyx_matfiles --count 10

    # Load .npz files by pattern
    python main_cascade_results.py --pattern "results_*.npz"
"""
import sys
import os
import glob
import argparse
import numpy as np
import scipy.io as sio

sys.path.insert(0, '.')
sys.path.insert(0, './Scattering_Code')

from Scattering_Code.cascadertwo import cascadertwo
from get_partition import smat_to_s11, smat_to_s12, smat_to_s21, smat_to_s22


def load_smatrix_file(filepath):
    """Load S-matrix data from .mat or .npz file."""
    if filepath.endswith('.mat'):
        data = sio.loadmat(filepath)
        S = np.array(data['S'])
        d = float(np.squeeze(data['d']))
        nmax = int(np.squeeze(data['nmax']))
        NoEvaMode = int(np.squeeze(data.get('NoEvaMode', 0)))
    else:
        data = np.load(filepath)
        S = data['S']
        d = float(data['d'])
        nmax = int(data['nmax'])
        NoEvaMode = int(data.get('NoEvaMode', 0))
    return S, d, nmax, NoEvaMode


def truncate_and_check(S_full, nmax, NoEvaMode):
    """Truncate evanescent modes and compute DOU + transmission eigenvalues."""
    nm = 2 * nmax + 1
    S11 = S_full[:nm, :nm]
    S12 = S_full[:nm, nm:]
    S21 = S_full[nm:, :nm]
    S22 = S_full[nm:, nm:]

    if NoEvaMode > 0:
        S11T = S11[NoEvaMode:-NoEvaMode, NoEvaMode:-NoEvaMode]
        S12T = S12[NoEvaMode:-NoEvaMode, NoEvaMode:-NoEvaMode]
        S21T = S21[NoEvaMode:-NoEvaMode, NoEvaMode:-NoEvaMode]
        S22T = S22[NoEvaMode:-NoEvaMode, NoEvaMode:-NoEvaMode]
        ST = np.block([[S11T, S12T], [S21T, S22T]])
    else:
        ST = S_full

    test = np.abs(ST.conj().T @ ST)
    DOU = np.sum(test) / len(test)
    half = ST.shape[0] // 2
    e = np.linalg.svd(ST[half:, :half], compute_uv=False) ** 2
    return ST, DOU, e


def main():
    parser = argparse.ArgumentParser(description='Cascade pre-computed S-matrices')
    parser.add_argument('--matdir', type=str, default=None,
                        help='Directory containing .mat files '
                             '(e.g. "/path/to/Scattering Code" or "nyx_matfiles")')
    parser.add_argument('--prefix', type=str,
                        default='1440CylinderLatinWidth180Thickness1100PECIDX',
                        help='Filename prefix before IDX number')
    parser.add_argument('--pattern', type=str, default=None,
                        help='Glob pattern for .npz files (alternative to --matdir)')
    parser.add_argument('--count', type=int, default=10,
                        help='Number of files to cascade (IDX 1..count)')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save intermediate results every N steps')
    parser.add_argument('--outdir', type=str, default='.',
                        help='Output directory for cascaded results')
    args = parser.parse_args()

    # Find files
    if args.matdir:
        files = [os.path.join(args.matdir, f"{args.prefix}{i}.mat")
                 for i in range(1, args.count + 1)]
        files = [f for f in files if os.path.exists(f)]
    elif args.pattern:
        files = sorted(glob.glob(args.pattern))
    else:
        print("Usage: specify --matdir or --pattern")
        return

    if len(files) < 2:
        print(f"Need at least 2 files, found {len(files)}")
        if args.matdir:
            print(f"  Looked for: {args.prefix}*.mat in {args.matdir}")
        return

    print(f"Found {len(files)} files to cascade")
    os.makedirs(args.outdir, exist_ok=True)

    # Load first two and cascade
    S1, d1, nmax, NoEvaMode = load_smatrix_file(files[0])
    print(f"  File 1: {os.path.basename(files[0])} "
          f"(S: {S1.shape}, nmax={nmax}, NoEvaMode={NoEvaMode})")

    S2, d2, _, _ = load_smatrix_file(files[1])
    print(f"  File 2: {os.path.basename(files[1])}")

    Scas, dcas = cascadertwo(S1, d1, S2, d2)

    ST, DOU, e = truncate_and_check(Scas, nmax, NoEvaMode)
    e_sorted = np.sort(e)[::-1]
    print(f"  After 2 cascades: DOU = {DOU:.6f}, "
          f"max tau^2 = {e_sorted[0]:.6f}, min tau^2 = {e_sorted[-1]:.6e}")

    # Save initial result
    outpath = os.path.join(args.outdir, '2_cascaded.npz')
    np.savez(outpath, Scas=Scas, ScasTruncated=ST,
             DOU=DOU, e=e, dcas=dcas, nmax=nmax, NoEvaMode=NoEvaMode)

    # Continue cascading
    for idx in range(2, len(files)):
        S_i, d_i, _, _ = load_smatrix_file(files[idx])
        print(f"  Cascading file {idx + 1}: {os.path.basename(files[idx])}...")

        Scas, dcas = cascadertwo(Scas, dcas, S_i, d_i)

        if (idx + 1) % args.save_every == 0 or idx == len(files) - 1:
            ST, DOU, e = truncate_and_check(Scas, nmax, NoEvaMode)
            e_sorted = np.sort(e)[::-1]
            print(f"    Step {idx + 1}: DOU = {DOU:.6f}, dcas = {dcas:.1f}, "
                  f"max tau^2 = {e_sorted[0]:.6f}, min tau^2 = {e_sorted[-1]:.6e}")

            outpath = os.path.join(args.outdir, f'{idx + 1}_cascaded.npz')
            np.savez(outpath, Scas=Scas, ScasTruncated=ST,
                     DOU=DOU, e=e, dcas=dcas, nmax=nmax, NoEvaMode=NoEvaMode)

    print(f"\nDone. Final: DOU = {DOU:.6f}, dcas = {dcas:.1f}")
    print(f"Transmission eigenvalues ({len(e)} channels):")
    print(f"  max = {e_sorted[0]:.6f}, min = {e_sorted[-1]:.6e}")
    print(f"  Top 10: {', '.join(f'{v:.6f}' for v in e_sorted[:10])}")


if __name__ == '__main__':
    main()
