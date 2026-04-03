"""
Cascade Movie from pre-computed Nyx .mat files.

Loads 1440-cylinder S-matrix .mat files, cascades them one at a time,
and creates a movie showing the evolving τ² distribution.
Each frame = one more slab cascaded (1440 → 14400 cylinders).

Usage:
    python nyx_cascade_movie.py --matdir "/path/to/Scattering Code" --count 10
    python nyx_cascade_movie.py --matdir nyx_matfiles --count 10
    python nyx_cascade_movie.py --matdir nyx_matfiles --count 10 --fps 2
"""
import sys
import os
import argparse
import numpy as np
import scipy.io as sio

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

sys.path.insert(0, '.')
sys.path.insert(0, './Scattering_Code')

from Scattering_Code.cascadertwo import cascadertwo


def load_smatrix_file(filepath):
    """Load S-matrix data from .mat or .npz file."""
    if filepath.endswith('.mat'):
        data = sio.loadmat(filepath)
        S = np.array(data['S'])
        d = float(np.squeeze(data['d']))
        nmax = int(np.squeeze(data['nmax']))
        NoEvaMode = int(np.squeeze(data.get('NoEvaMode', 0)))
        NoCyl = int(np.squeeze(data.get('ModifiedNoCylinders',
                     data.get('NoCylinders', 0))))
    else:
        data = np.load(filepath)
        S = data['S']
        d = float(data['d'])
        nmax = int(data['nmax'])
        NoEvaMode = int(data.get('NoEvaMode', 0))
        NoCyl = int(data.get('ModifiedNoCylinders',
                     data.get('NoCylinders', 0)))
    return S, d, nmax, NoEvaMode, NoCyl


def truncate_and_analyze(S_full, nmax, NoEvaMode):
    """Truncate evanescent modes, compute DOU and tau^2."""
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
        S21T = S21

    test = np.abs(ST.conj().T @ ST)
    DOU = np.sum(test) / len(test)

    if NoEvaMode > 0:
        half = ST.shape[0] // 2
        tau_sq = np.linalg.svd(ST[half:, :half], compute_uv=False) ** 2
    else:
        half = S_full.shape[0] // 2
        tau_sq = np.linalg.svd(S_full[half:, :half], compute_uv=False) ** 2

    return ST, DOU, tau_sq


def main():
    parser = argparse.ArgumentParser(description='Nyx Cascade Movie')
    parser.add_argument('--matdir', type=str, required=True,
                        help='Directory containing .mat files')
    parser.add_argument('--prefix', type=str,
                        default='1440CylinderLatinWidth180Thickness1100PECIDX',
                        help='Filename prefix')
    parser.add_argument('--count', type=int, default=10,
                        help='Number of files to cascade')
    parser.add_argument('--fps', type=int, default=2,
                        help='Frames per second')
    parser.add_argument('--outdir', type=str, default='.',
                        help='Output directory')
    args = parser.parse_args()

    # Find files
    files = [os.path.join(args.matdir, f"{args.prefix}{i}.mat")
             for i in range(1, args.count + 1)]
    files = [f for f in files if os.path.exists(f)]

    if len(files) < 2:
        print(f"Need at least 2 files, found {len(files)}")
        return

    print(f"Found {len(files)} files to cascade")
    os.makedirs(args.outdir, exist_ok=True)

    # === Phase 1: Cascade all files and collect data ===
    all_tau_sq = []
    all_dou = []
    all_n_cyl = []
    cyl_per_slab = 0

    # Load first file as initial state
    S1, d1, nmax, NoEvaMode, n1 = load_smatrix_file(files[0])
    cyl_per_slab = n1
    print(f"  File 1: {os.path.basename(files[0])} "
          f"(S: {S1.shape}, nmax={nmax}, NoEvaMode={NoEvaMode}, cyl={n1})")

    ST, DOU, tau_sq = truncate_and_analyze(S1, nmax, NoEvaMode)
    all_tau_sq.append(tau_sq)
    all_dou.append(DOU)
    all_n_cyl.append(n1)
    print(f"    1 slab ({n1:5d} cyl): DOU={DOU:.6f}, "
          f"max tau^2={np.max(tau_sq):.6f}, min={np.min(tau_sq):.6e}")

    Scas, dcas = S1, d1

    for i in range(1, len(files)):
        S_i, d_i, _, _, n_i = load_smatrix_file(files[i])
        print(f"  Cascading file {i+1}: {os.path.basename(files[i])} ({n_i} cyl)...")

        Scas, dcas = cascadertwo(Scas, dcas, S_i, d_i)

        ST, DOU, tau_sq = truncate_and_analyze(Scas, nmax, NoEvaMode)
        total_cyl = all_n_cyl[-1] + n_i
        all_tau_sq.append(tau_sq)
        all_dou.append(DOU)
        all_n_cyl.append(total_cyl)
        print(f"    {i+1} slabs ({total_cyl:5d} cyl): DOU={DOU:.6f}, "
              f"max tau^2={np.max(tau_sq):.6f}, min={np.min(tau_sq):.6e}")

    n_frames = len(all_tau_sq)
    n_channels = len(all_tau_sq[0])

    # Save raw data
    datafile = os.path.join(args.outdir, 'nyx_cascade_movie_data.npz')
    np.savez(datafile,
             all_tau_sq=np.array(all_tau_sq, dtype=object),
             all_dou=np.array(all_dou),
             all_n_cyl=np.array(all_n_cyl))
    print(f"\nData saved to {datafile}")

    # === Phase 2: Create log-scale movie (two panels) ===
    print("\nCreating log-scale movie...")

    all_log_tau = []
    for tau_sq in all_tau_sq:
        all_log_tau.append(np.log10(np.clip(tau_sq, 1e-40, 1.0)))
    global_log_min = min(np.min(lt) for lt in all_log_tau)
    log_bins = np.linspace(global_log_min - 0.5, 0.5, 41)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    def animate_log(frame):
        ax1.clear()
        ax2.clear()

        tau_sq = all_tau_sq[frame]
        tau_sq_sorted = np.sort(tau_sq)[::-1]
        n_cyl = all_n_cyl[frame]
        dou = all_dou[frame]

        # Left: sorted eigenvalues
        ax1.semilogy(range(1, n_channels + 1), tau_sq_sorted,
                     'o-', color='steelblue', markersize=3, linewidth=1)
        ax1.set_xlim(0.5, n_channels + 0.5)
        ax1.set_ylim(1e-10, 10)
        ax1.set_xlabel('Channel index', fontsize=13)
        ax1.set_ylabel(r'$\tau^2$', fontsize=13)
        ax1.set_title('Transmission eigenvalues (sorted)', fontsize=12)
        ax1.axhline(1.0, color='red', ls='--', alpha=0.5, label=r'$\tau^2=1$')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, which='both')

        # Right: log histogram
        log_tau = np.log10(np.clip(tau_sq, 1e-40, 1.0))
        ax2.hist(log_tau, bins=log_bins, color='steelblue',
                 edgecolor='black', alpha=0.8)
        ax2.set_xlim(global_log_min - 0.5, 0.5)
        ax2.set_ylim(0, n_channels)
        ax2.set_xlabel(r'$\log_{10}(\tau^2)$', fontsize=13)
        ax2.set_ylabel('Count', fontsize=13)
        ax2.set_title(r'Distribution of $\log_{10}(\tau^2)$', fontsize=12)
        ax2.grid(True, alpha=0.3)

        fig.suptitle(f'{n_cyl} Dielectric Cylinders  '
                     f'({frame+1} x 1440 cyl/slab)   '
                     f'DOU = {dou:.4f}',
                     fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.93])

    anim = FuncAnimation(fig, animate_log, frames=n_frames,
                         interval=1000 // args.fps)
    try:
        writer = FFMpegWriter(fps=args.fps,
                              metadata={'title': 'Nyx Cascade S21 Evolution'})
        outfile = os.path.join(args.outdir, 'nyx_cascade_movie.mp4')
        anim.save(outfile, writer=writer, dpi=150)
        print(f"Log-scale movie saved to {outfile}")
    except Exception as e:
        print(f"MP4 failed ({e}), saving as GIF...")
        outfile = os.path.join(args.outdir, 'nyx_cascade_movie.gif')
        anim.save(outfile, writer='pillow', fps=args.fps, dpi=100)
        print(f"Log-scale movie saved to {outfile}")
    plt.close()

    # === Phase 3: Create linear-scale movie ===
    print("\nCreating linear-scale movie...")

    fig2, ax_lin = plt.subplots(figsize=(10, 6))
    linear_bins = np.linspace(0, 1, 51)

    y_max_lin = 0
    for tau_sq in all_tau_sq:
        counts, _ = np.histogram(tau_sq, bins=linear_bins, density=True)
        y_max_lin = max(y_max_lin, np.max(counts))
    y_max_lin *= 1.15

    def animate_linear(frame):
        ax_lin.clear()
        tau_sq = all_tau_sq[frame]
        n_cyl = all_n_cyl[frame]
        dou = all_dou[frame]

        ax_lin.hist(tau_sq, bins=linear_bins, density=True, color='steelblue',
                    edgecolor='black', alpha=0.8)
        ax_lin.set_xlim(0, 1)
        ax_lin.set_ylim(0, y_max_lin)
        ax_lin.set_xlabel(r'$\tau^2$ (transmission eigenvalue)', fontsize=14)
        ax_lin.set_ylabel(r'$p(\tau^2)$', fontsize=14)
        ax_lin.set_title(f'S21 Singular Value Distribution — '
                         f'{n_cyl} Dielectric Cylinders '
                         f'({frame+1} x 1440 cyl/slab)\n'
                         f'DOU = {dou:.4f}',
                         fontsize=13)
        ax_lin.grid(True, alpha=0.3)

    anim2 = FuncAnimation(fig2, animate_linear, frames=n_frames,
                          interval=1000 // args.fps)
    try:
        writer2 = FFMpegWriter(fps=args.fps,
                               metadata={'title': 'Nyx Cascade (linear)'})
        outfile2 = os.path.join(args.outdir, 'nyx_cascade_movie_linear.mp4')
        anim2.save(outfile2, writer=writer2, dpi=150)
        print(f"Linear movie saved to {outfile2}")
    except Exception as e:
        print(f"MP4 failed ({e}), saving as GIF...")
        outfile2 = os.path.join(args.outdir, 'nyx_cascade_movie_linear.gif')
        anim2.save(outfile2, writer='pillow', fps=args.fps, dpi=100)
        print(f"Linear movie saved to {outfile2}")
    plt.close()

    print("\nDone!")


if __name__ == '__main__':
    main()
