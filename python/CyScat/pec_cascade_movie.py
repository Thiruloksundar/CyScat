"""
Cascade Movie: Evolution of S21 singular values as a function of cascades.

Each cascade adds a NEW independent slab of random cylinders (PEC or dielectric).
Goes up to 100 cascades (1000 cylinders total by default).
Produces an MP4 animation showing the evolving PDF of transmission eigenvalues tau^2.

Usage:
    python pec_cascade_movie.py --pec                # PEC cylinders (eps = -1)
    python pec_cascade_movie.py                      # Dielectric (n=1.3, eps=1.69)
    python pec_cascade_movie.py --n_cascade 100 --cyl_per_slab 10
    python pec_cascade_movie.py --seed 42
"""
import sys
import os
import time
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

sys.path.insert(0, '.')
sys.path.insert(0, './Scattering_Code')

from Scattering_Code.smatrix_parameters import smatrix_parameters
from Scattering_Code.smatrix import smatrix
from Scattering_Code.cascadertwo import cascadertwo
from get_partition import smat_to_s11, smat_to_s21


def generate_cylinder_positions(num_cyl, period, thickness, radius):
    """Generate random cylinder positions with minimum separation constraint."""
    margin = radius * 1.5
    min_sep = 2.5 * radius
    clocs = np.zeros((num_cyl, 2))
    placed = 0
    for i in range(num_cyl):
        for attempt in range(5000):
            x = np.random.uniform(margin, period - margin)
            y = np.random.uniform(margin, thickness - margin)
            if i == 0:
                clocs[i] = [x, y]
                placed += 1
                break
            dists = np.sqrt((x - clocs[:i, 0])**2 + (y - clocs[:i, 1])**2)
            if np.all(dists > min_sep):
                clocs[i] = [x, y]
                placed += 1
                break
    if placed < num_cyl:
        print(f"  Warning: only placed {placed}/{num_cyl} cylinders")
    return clocs


def main():
    parser = argparse.ArgumentParser(description='PEC Cascade Movie')
    parser.add_argument('--n_cascade', type=int, default=100,
                        help='Number of cascades (default: 100)')
    parser.add_argument('--cyl_per_slab', type=int, default=10,
                        help='Cylinders per slab (default: 10)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fps', type=int, default=5,
                        help='Frames per second for movie (default: 5)')
    parser.add_argument('--pec', action='store_true',
                        help='Use PEC cylinders (eps=-1). Default: dielectric n=1.3')
    args = parser.parse_args()

    n_cascade = args.n_cascade
    cyl_per_slab = args.cyl_per_slab
    seed = args.seed

    # === Physical parameters ===
    wavelength = 0.93
    period = 12.81
    radius = 0.25
    n_cylinder = 1.3
    mu = 1.0
    cmmax = 5
    phiinc = np.pi / 2

    if args.pec:
        epsilon = -1
        material_str = 'PEC'
    else:
        epsilon = n_cylinder**2
        material_str = f'Dielectric (n={n_cylinder}, eps={epsilon:.2f})'

    # Thickness for each slab
    spacing = 2.5 * radius
    cyls_per_row = int(period / spacing)
    rows_needed = cyl_per_slab / cyls_per_row + 2
    thickness = max(0.5, rows_needed * spacing * 1.5)
    thickness = round(thickness, 1)

    # Evanescent mode truncation
    EvanescentModeTol = 1e-2
    NoPropagatingModes = int(np.floor(period / wavelength))
    if args.pec:
        NoEvaMode = 0  # PEC: skip evanescent buffer (numerical instability)
    else:
        NoEvaMode = int(np.floor(
            period / (2 * np.pi) * np.sqrt(
                (np.log(EvanescentModeTol) / (2 * radius))**2
                + (2 * np.pi / wavelength)**2
            )
        )) - NoPropagatingModes
        NoEvaMode = max(NoEvaMode, 0)
    nmax = NoPropagatingModes + NoEvaMode
    d = thickness

    print("=" * 60)
    print(f"Cascade Movie — {material_str}")
    print(f"  {cyl_per_slab} cylinders/slab x {n_cascade} cascades = {cyl_per_slab * n_cascade} total")
    print(f"  period={period}, wavelength={wavelength}, slab thickness={thickness}")
    print(f"  radius={radius}, nmax={nmax}, NoEvaMode={NoEvaMode}, seed={seed}")
    print("=" * 60)

    sp = smatrix_parameters(wavelength, period, phiinc,
                            1e-11, 1e-4, 5, 3, 1000, 3, 5, 1, period / 120)

    np.random.seed(seed)

    # Storage for all cascade steps
    all_tau_sq = []      # tau^2 at each cascade step
    all_dou = []         # DOU at each cascade step

    Scas = None
    dcas = 0.0

    t_total = time.time()

    for i in range(n_cascade):
        t0 = time.time()

        # Generate new random slab
        clocs = generate_cylinder_positions(cyl_per_slab, period, thickness, radius)
        cmmaxs = cmmax * np.ones(cyl_per_slab, dtype=int)
        cepmus = np.column_stack([epsilon * np.ones(cyl_per_slab),
                                   np.ones(cyl_per_slab)])
        crads = radius * np.ones(cyl_per_slab)

        # Compute S-matrix for this slab
        S_new, _ = smatrix(clocs, cmmaxs, cepmus, crads, period, wavelength,
                           nmax, d, sp, 'On')

        # Truncate evanescent modes BEFORE cascading to prevent
        # evanescent coupling between slabs
        if NoEvaMode > 0:
            half = len(S_new) // 2
            s11 = S_new[:half, :half][NoEvaMode:-NoEvaMode, NoEvaMode:-NoEvaMode]
            s12 = S_new[:half, half:][NoEvaMode:-NoEvaMode, NoEvaMode:-NoEvaMode]
            s21 = S_new[half:, :half][NoEvaMode:-NoEvaMode, NoEvaMode:-NoEvaMode]
            s22 = S_new[half:, half:][NoEvaMode:-NoEvaMode, NoEvaMode:-NoEvaMode]
            S_new = np.block([[s11, s12], [s21, s22]])

        # Cascade with accumulated result
        if Scas is None:
            Scas = S_new
            dcas = d
        else:
            Scas, dcas = cascadertwo(Scas, dcas, S_new, d)

        # Extract S21 and compute singular values
        S21 = smat_to_s21(Scas)
        tau = np.linalg.svd(S21, compute_uv=False)
        tau_sq = tau**2

        # DOU on propagating-only S-matrix
        test = np.abs(Scas.conj().T @ Scas)
        dou = np.sum(test) / len(Scas)

        all_tau_sq.append(tau_sq)
        all_dou.append(dou)

        elapsed = time.time() - t0
        n_total = (i + 1) * cyl_per_slab
        print(f"  Cascade {i+1:3d}/{n_cascade} ({n_total:4d} cyl): "
              f"tau^2 range=[{np.min(tau_sq):.6f}, {np.max(tau_sq):.6f}], "
              f"DOU={dou:.6f}, time={elapsed:.1f}s")

    t_total = time.time() - t_total
    print(f"\nTotal computation time: {t_total:.1f}s ({t_total/60:.1f} min)")

    # Save raw data
    tag = 'pec' if args.pec else 'dielectric'
    np.savez(f'{tag}_cascade_movie_data.npz',
             all_tau_sq=np.array(all_tau_sq, dtype=object),
             all_dou=np.array(all_dou),
             n_cascade=n_cascade, cyl_per_slab=cyl_per_slab,
             wavelength=wavelength, period=period, nmax=nmax)
    print(f"Data saved to {tag}_cascade_movie_data.npz")

    # === Create animation ===
    print("\nCreating animation...")

    n_channels = len(all_tau_sq[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Precompute global log-scale range for histogram
    all_log_tau = []
    for tau_sq in all_tau_sq:
        tau_sq_clipped = np.clip(tau_sq, 1e-40, 1.0)
        all_log_tau.append(np.log10(tau_sq_clipped))
    global_log_min = min(np.min(lt) for lt in all_log_tau)
    global_log_max = 0  # log10(1) = 0
    log_bins = np.linspace(global_log_min - 0.5, 0.5, 41)

    def animate(frame):
        ax1.clear()
        ax2.clear()

        tau_sq = all_tau_sq[frame]
        tau_sq_sorted = np.sort(tau_sq)[::-1]
        n_total = (frame + 1) * cyl_per_slab
        dou = all_dou[frame]

        # --- Left panel: sorted singular values (log scale) ---
        ax1.semilogy(range(1, n_channels + 1), tau_sq_sorted,
                     'o-', color='steelblue', markersize=5, linewidth=1.5)
        ax1.set_xlim(0.5, n_channels + 0.5)
        ax1.set_ylim(1e-40, 10)
        ax1.set_xlabel('Channel index', fontsize=13)
        ax1.set_ylabel(r'$\tau^2$', fontsize=13)
        ax1.set_title(f'Transmission eigenvalues (sorted)', fontsize=12)
        ax1.axhline(1.0, color='red', ls='--', alpha=0.5, label=r'$\tau^2=1$')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, which='both')

        # --- Right panel: histogram of log10(tau^2) ---
        log_tau = np.log10(np.clip(tau_sq, 1e-40, 1.0))
        ax2.hist(log_tau, bins=log_bins, color='steelblue',
                 edgecolor='black', alpha=0.8)
        ax2.set_xlim(global_log_min - 0.5, 0.5)
        ax2.set_ylim(0, n_channels)
        ax2.set_xlabel(r'$\log_{10}(\tau^2)$', fontsize=13)
        ax2.set_ylabel('Count', fontsize=13)
        ax2.set_title(r'Distribution of $\log_{10}(\tau^2)$', fontsize=12)
        ax2.grid(True, alpha=0.3)

        fig.suptitle(f'{n_total} {material_str} Cylinders  ({frame+1} cascades × {cyl_per_slab} cyl/slab)   '
                     f'DOU = {dou:.4f}',
                     fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.93])

    anim = FuncAnimation(fig, animate, frames=n_cascade, interval=1000//args.fps)

    # Try MP4 first, fall back to GIF
    try:
        writer = FFMpegWriter(fps=args.fps, metadata={'title': 'PEC Cascade S21 Evolution'})
        outfile = f'{tag}_cascade_movie.mp4'
        anim.save(outfile, writer=writer, dpi=150)
        print(f"Movie saved to {outfile}")
    except Exception as e:
        print(f"MP4 failed ({e}), saving as GIF...")
        outfile = f'{tag}_cascade_movie.gif'
        anim.save(outfile, writer='pillow', fps=args.fps, dpi=100)
        print(f"Movie saved to {outfile}")

    plt.close()

    # === Create linear-scale animation ===
    print("\nCreating linear-scale animation...")

    fig2, ax_lin = plt.subplots(figsize=(10, 6))
    linear_bins = np.linspace(0, 1, 51)

    # Find global y-limit for linear histogram
    y_max_lin = 0
    for tau_sq in all_tau_sq:
        counts, _ = np.histogram(tau_sq, bins=linear_bins, density=True)
        y_max_lin = max(y_max_lin, np.max(counts))
    y_max_lin *= 1.15

    def animate_linear(frame):
        ax_lin.clear()
        tau_sq = all_tau_sq[frame]
        n_total = (frame + 1) * cyl_per_slab
        dou = all_dou[frame]

        ax_lin.hist(tau_sq, bins=linear_bins, density=True, color='steelblue',
                    edgecolor='black', alpha=0.8)
        ax_lin.set_xlim(0, 1)
        ax_lin.set_ylim(0, y_max_lin)
        ax_lin.set_xlabel(r'$\tau^2$ (transmission eigenvalue)', fontsize=14)
        ax_lin.set_ylabel(r'$p(\tau^2)$', fontsize=14)
        ax_lin.set_title(f'S21 Singular Value Distribution — '
                         f'{n_total} {material_str} Cylinders '
                         f'({frame+1} cascades × {cyl_per_slab} cyl/slab)\n'
                         f'DOU = {dou:.4f}',
                         fontsize=13)
        ax_lin.grid(True, alpha=0.3)

    anim2 = FuncAnimation(fig2, animate_linear, frames=n_cascade,
                           interval=1000//args.fps)

    try:
        writer2 = FFMpegWriter(fps=args.fps,
                               metadata={'title': 'Cascade S21 Evolution (linear)'})
        outfile2 = f'{tag}_cascade_movie_linear.mp4'
        anim2.save(outfile2, writer=writer2, dpi=150)
        print(f"Movie saved to {outfile2}")
    except Exception as e:
        print(f"MP4 failed ({e}), saving as GIF...")
        outfile2 = f'{tag}_cascade_movie_linear.gif'
        anim2.save(outfile2, writer='pillow', fps=args.fps, dpi=100)
        print(f"Movie saved to {outfile2}")

    plt.close()
    print("Done!")


if __name__ == '__main__':
    main()
