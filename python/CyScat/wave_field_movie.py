"""
wave_field_movie.py
===================
Generates an animation of the EM wave field scattered by a slab of
cylinders, matching the MATLAB CyScat GUI visualization style (Curtis Jin).

The video shows two segments:
  1. Normal incidence — shows how much a plane wave transmits
  2. Optimal wavefront — SVD of S11, maximizes transmission

Field is computed in three regions stitched together:
  - Reflection side (y < 0): incident + backscattered Floquet waves
  - Slab interior (0 <= y <= d): interpolated forward/backward Floquet waves
  - Transmission side (y > d): transmitted Floquet waves

Usage (from CyScat root):
    python wave_field_movie.py --pec --num_cyl 50
    python wave_field_movie.py --mode normal
    python wave_field_movie.py --mode both  (default: normal + optimal)

Outputs:
    wave_field_movie.mp4 (or .gif), plus _tau_dist.png and _modal_coeffs.png
"""

import sys
import time
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, '.')
sys.path.insert(0, './Scattering_Code')

from Scattering_Code.smatrix_parameters import smatrix_parameters
from Scattering_Code.smatrix import smatrix
from Scattering_Code.ky import ky

# ── Constants ─────────────────────────────────────────────────────────────────

WAVELENGTH  = 0.93
PERIOD      = 12.81
RADIUS      = 0.25
N_CYLINDER  = 1.3
MU          = 1.0
CMMAX       = 5
PHIINC      = np.pi / 2
SEED        = 42
GRID_RES    = 200
PR_Y        = 7       # Plot region: extend 7 wavelengths on each side

Eva_TOL     = 1e-2
N_PROP      = int(np.floor(PERIOD / WAVELENGTH))
N_EVA_DIEL  = max(int(np.floor(
    PERIOD / (2*np.pi) * np.sqrt(
        (np.log(Eva_TOL) / (2*RADIUS))**2 + (2*np.pi/WAVELENGTH)**2
    )
)) - N_PROP, 0)


def get_nmax(pec=False):
    n_eva = 0 if pec else N_EVA_DIEL
    return N_PROP, n_eva, N_PROP + n_eva


# ── Setup ─────────────────────────────────────────────────────────────────────

def setup(num_cyl=16, seed=SEED, pec=False):
    """Generate cylinder positions and compute S-matrix."""
    spacing = 2.5 * RADIUS
    cyls_per_row = int(PERIOD / spacing)
    rows_needed = num_cyl / cyls_per_row + 2
    thickness = max(0.5, rows_needed * spacing * 1.5)
    thickness = round(thickness, 1)

    np.random.seed(seed)
    margin = RADIUS * 1.5
    clocs = np.zeros((num_cyl, 2))
    for i in range(num_cyl):
        for _ in range(10000):
            x = np.random.uniform(margin, PERIOD - margin)
            y = np.random.uniform(margin, thickness - margin)
            if i == 0 or np.all(np.sqrt((x - clocs[:i,0])**2 +
                                         (y - clocs[:i,1])**2) > spacing):
                clocs[i] = [x, y]
                break

    _, _, nmax = get_nmax(pec)
    sp = smatrix_parameters(
        WAVELENGTH, PERIOD, PHIINC,
        1e-11, 1e-4, 5, 3, 1000, 3, 5, 1, PERIOD / 120
    )

    cmmaxs = np.array([CMMAX] * num_cyl)
    eps_val = -1.0 if pec else N_CYLINDER ** 2
    material = 'PEC' if pec else f'Dielectric (n={N_CYLINDER})'
    cepmus = np.tile([eps_val, MU], (num_cyl, 1))
    crads = np.full(num_cyl, RADIUS)

    print(f"  Material: {material}")
    print(f"  Computing S-matrix ({num_cyl} cylinders, nmax={nmax}) ...")
    t0 = time.time()
    S, _ = smatrix(jnp.array(clocs), cmmaxs, jnp.array(cepmus), jnp.array(crads),
                   PERIOD, WAVELENGTH, nmax, thickness, sp, 'On',
                   clocs_concrete=clocs)
    S = np.asarray(S)
    print(f"  S-matrix done in {time.time()-t0:.1f}s")

    return clocs, thickness, S, material, nmax, crads


# ── Build input vector ────────────────────────────────────────────────────────

def make_input(S, nmax, n_eva, mode):
    """Build the incident mode superposition vector. Returns S11, S21, Input, label."""
    nm = 2 * nmax + 1
    S11 = S[:nm, :nm]
    S21 = S[nm:, :nm]

    if mode == 'opt_trans':
        # Optimize transmission via last right singular vector of S11 (min reflection)
        R_trunc = S11[n_eva:-n_eva, n_eva:-n_eva] if n_eva > 0 else S11.copy()
        T_trunc = S21[n_eva:-n_eva, n_eva:-n_eva] if n_eva > 0 else S21.copy()
        U, sigma, Vh = np.linalg.svd(R_trunc)
        v_opt = Vh[-1, :].conj()  # last row = smallest singular value
        Input = np.zeros(nm, dtype=complex)
        if n_eva > 0:
            Input[n_eva:-n_eva] = v_opt
        else:
            Input = v_opt
        tc = np.sum(np.abs(T_trunc @ v_opt)**2)
        label = f"Optimal Wavefront — {tc*100:.2f}% transmitted."

    elif mode == 'opt_reflect':
        R_trunc = S11[n_eva:-n_eva, n_eva:-n_eva] if n_eva > 0 else S11.copy()
        U, sigma, Vh = np.linalg.svd(R_trunc)
        v_opt = Vh[0, :].conj()  # first row = largest singular value
        Input = np.zeros(nm, dtype=complex)
        if n_eva > 0:
            Input[n_eva:-n_eva] = v_opt
        else:
            Input = v_opt
        label = f"Optimal Reflection (τ={sigma[0]:.4f})"

    else:  # normal
        Input = np.zeros(nm, dtype=complex)
        Input[nmax] = 1.0
        T_trunc = S21[n_eva:-n_eva, n_eva:-n_eva] if n_eva > 0 else S21.copy()
        center = nmax - n_eva if n_eva > 0 else nmax
        tc = np.sum(np.abs(T_trunc[:, center])**2)
        label = f"Normal Incidence — {tc*100:.2f}% transmitted."

    return S11, S21, Input, label


# ── Build field arrays ───────────────────────────────────────────────────────

def build_field_arrays(S, nmax, n_eva, thickness, mode, clocs, crads):
    """
    Build total field across three regions with y/lambda units.
    Returns FullField, x_grid, y_full, label.
    """
    nm = 2 * nmax + 1
    S11, S21, Input, label = make_input(S, nmax, n_eva, mode)

    k = 2 * np.pi / WAVELENGTH
    m = np.arange(-nmax, nmax + 1)
    kxs = 2 * np.pi / PERIOD * m
    kys = np.asarray(ky(k, jnp.array(kxs, dtype=jnp.complex128)))

    nor = np.sqrt(kys / k)
    P1 = np.diag(1.0 / nor)

    Incident_c = P1 @ Input
    Reflect_c  = P1 @ (S11 @ Input)
    Trans_c    = P1 @ (S21 @ Input)

    Nx = GRID_RES
    x_grid = np.linspace(0, PERIOD / WAVELENGTH, Nx)   # x/lambda
    x_phys = np.linspace(0, PERIOD, Nx)                 # physical x for Floquet

    # Propagating-mode subsets for incident field
    if n_eva > 0:
        sl = slice(n_eva, nm - n_eva)
    else:
        sl = slice(None)
    inc_c  = Incident_c[sl]
    kxs_p  = kxs[sl]
    kys_p  = kys[sl]

    # Three y-regions
    Ly      = PR_Y * WAVELENGTH
    d       = thickness
    Ny_side = GRID_RES
    Ny_slab = max(60, int(round(GRID_RES * d / (2 * Ly))))

    y_ref   = np.linspace(-Ly, 0, Ny_side)
    y_slab  = np.linspace(0, d, Ny_slab)
    y_trans = np.linspace(d, d + Ly, Ny_side)

    # ── Reflection field (y < 0) ──
    exp_x_p = np.exp(1j * np.outer(x_phys, -kxs_p))
    exp_y_i = np.exp(-1j * np.outer(kys_p, y_ref))
    IncField = exp_x_p @ np.diag(inc_c) @ exp_y_i

    exp_x_a = np.exp(1j * np.outer(x_phys, -kxs))
    exp_y_b = np.exp(1j * np.outer(kys, y_ref))
    BackField = exp_x_a @ np.diag(Reflect_c) @ exp_y_b

    RefField = IncField + BackField

    # ── Slab interior: propagating modes only ──
    prop_mask = np.abs(np.imag(kys)) < 1e-10
    prop_idx  = np.where(prop_mask)[0]

    kxs_slab = kxs[prop_idx]
    kys_slab = np.real(kys[prop_idx])
    inc_slab = Incident_c[prop_idx]
    ref_slab = Reflect_c[prop_idx]
    tra_slab = Trans_c[prop_idx]
    exp_x_slab = np.exp(1j * np.outer(x_phys, -kxs_slab))

    SlabField = np.zeros((Nx, Ny_slab), dtype=complex)

    for jy in range(Ny_slab):
        y  = y_slab[jy]
        alpha = y / d if d > 0 else 0.0

        fwd_amp = inc_slab * (1 - alpha) + tra_slab * alpha
        bwd_amp = ref_slab * (1 - alpha)

        fwd_phase = np.exp(-1j * kys_slab * y)
        bwd_phase = np.exp( 1j * kys_slab * y)

        field_modes = fwd_amp * fwd_phase + bwd_amp * bwd_phase
        SlabField[:, jy] = exp_x_slab @ field_modes

    # Zero out inside cylinders
    for jy in range(Ny_slab):
        for ix in range(Nx):
            px, py = x_phys[ix], y_slab[jy]
            for ic in range(len(crads)):
                if (px - clocs[ic, 0])**2 + (py - clocs[ic, 1])**2 < crads[ic]**2:
                    SlabField[ix, jy] = 0.0
                    break

    # ── Transmission field (y > d) ──
    exp_y_t = np.exp(-1j * np.outer(kys, y_trans - d))
    TransField = exp_x_a @ np.diag(Trans_c) @ exp_y_t

    # ── Stitch and convert y to y/lambda ──
    y_full    = np.concatenate([y_ref, y_slab, y_trans]) / WAVELENGTH
    FullField = np.hstack([RefField, SlabField, TransField])

    return FullField, x_grid, y_full, label


# ── Plot: Transmission eigenvalue distribution ───────────────────────────────

def plot_transmission_distribution(S, nmax, n_eva, num_cyl, material,
                                   outfile='transmission_distribution'):
    """Histogram of transmission singular values f(tau)."""
    nm = 2 * nmax + 1
    S21 = S[nm:, :nm]
    if n_eva > 0:
        S21T = S21[n_eva:-n_eva, n_eva:-n_eva]
    else:
        S21T = S21

    tau = np.linalg.svd(S21T, compute_uv=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(tau, bins=np.linspace(0, 1.05, 50), density=True,
            color='navy', edgecolor='navy')
    ax.set_xlabel('τ', fontsize=14)
    ax.set_ylabel('f(τ)', fontsize=14)
    ax.set_xlim(0, 1.05)
    ax.set_title(f'{num_cyl} {material} Cylinders', fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{outfile}.png', dpi=150)
    plt.close(fig)
    print(f"  Saved -> {outfile}.png")


# ── Plot: Modal coefficients of optimal wavefront ────────────────────────────

def plot_modal_coefficients(S, nmax, n_eva, outfile='modal_coefficients'):
    """Magnitude and phase of the optimal wavefront modal coefficients."""
    nm = 2 * nmax + 1
    S11 = S[:nm, :nm]
    k = 2 * np.pi / WAVELENGTH
    m = np.arange(-nmax, nmax + 1)
    kxs = 2 * np.pi / PERIOD * m
    kys = np.asarray(ky(k, jnp.array(kxs, dtype=jnp.complex128)))
    nor = np.sqrt(kys / k)
    P1 = np.diag(1.0 / nor)

    if n_eva > 0:
        S11T = S11[n_eva:-n_eva, n_eva:-n_eva]
        kxs_prop = kxs[n_eva:-n_eva]
    else:
        S11T = S11
        kxs_prop = kxs

    # Last right singular vector of S11 = min reflection = max transmission
    U, sigma, Vh = np.linalg.svd(S11T)
    v_opt = Vh[-1, :].conj()

    Input_full = np.zeros(nm, dtype=complex)
    if n_eva > 0:
        Input_full[n_eva:-n_eva] = v_opt
    else:
        Input_full = v_opt
    modal_amp = P1 @ Input_full

    if n_eva > 0:
        modal_prop = modal_amp[n_eva:-n_eva]
    else:
        modal_prop = modal_amp

    angles = np.degrees(np.arcsin(np.real(kxs_prop) / k))

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color_mag = 'blue'
    color_phase = 'green'

    ax1.plot(angles, np.abs(modal_prop), '-o', color=color_mag, lw=2,
             markersize=3, label='Magnitude')
    ax1.set_xlabel('Angles (°)', fontsize=12)
    ax1.set_ylabel('Magnitude', color=color_mag, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color_mag)
    ax1.legend(loc='upper right', fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(angles, np.degrees(np.angle(modal_prop)), '--D', color=color_phase,
             lw=2, markersize=3, label='Phase (°)')
    ax2.set_ylabel('Phase (°)', color=color_phase, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color_phase)
    ax2.legend(loc='upper left', fontsize=9)

    fig.suptitle('Modal coefficients of the Optimal Wavefront', fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{outfile}.png', dpi=150)
    plt.close(fig)
    print(f"  Saved -> {outfile}.png")


# ── Animation (Curtis-style single panel) ────────────────────────────────────

def make_movie(fields_list, x_grid, y_full, thickness, clocs, crads,
               num_cyl, material, n_frames=80, fps=20,
               outfile='wave_field_movie'):
    """Single-panel animation with jet colormap, magenta slab boundaries."""
    T_period = 4.0
    omega = 2 * np.pi / T_period
    dt = T_period / n_frames
    d_lam = thickness / WAVELENGTH

    # Global color range across all segments
    vmax = max(np.max(np.abs(np.real(F))) for F, _ in fields_list)
    vmax = max(vmax, 1e-10)

    Y, X = np.meshgrid(y_full, x_grid)

    # Cylinder outlines in y/lambda, x/lambda
    cyl_patches_y = [cyl[1] / WAVELENGTH for cyl in clocs]
    cyl_patches_x = [cyl[0] / WAVELENGTH for cyl in clocs]
    cyl_r_lam = RADIUS / WAVELENGTH

    total_frames = len(fields_list) * n_frames

    fig, ax = plt.subplots(figsize=(12, 6))

    # Initialize with first field
    Field0, label0 = fields_list[0]
    im = ax.pcolormesh(Y, X, np.real(Field0), cmap='jet',
                       vmin=-vmax, vmax=vmax, shading='gouraud')
    fig.colorbar(im, ax=ax, shrink=0.85)
    ax.axvline(0,     color='magenta', lw=2.5)
    ax.axvline(d_lam, color='magenta', lw=2.5)

    # Draw cylinder outlines
    for cy, cx in zip(cyl_patches_y, cyl_patches_x):
        ax.add_patch(plt.Circle((cy, cx), cyl_r_lam,
                     fill=False, ec='blue', lw=0.8, zorder=5))

    ax.set_xlabel('y/λ', fontsize=12)
    ax.set_ylabel('x/λ', fontsize=12)
    title_text = ax.set_title(label0, fontsize=12)
    fig.tight_layout()

    def update(global_frame):
        seg_idx = global_frame // n_frames
        frame = global_frame % n_frames
        t = frame * dt
        phase = np.exp(1j * omega * t)

        Field, label = fields_list[seg_idx]
        im.set_array(np.real(Field * phase).ravel())
        title_text.set_text(label)
        return [im, title_text]

    anim = animation.FuncAnimation(fig, update, frames=total_frames,
                                   interval=1000 // fps, blit=True)

    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(f'{outfile}.mp4', writer=writer, dpi=120)
        print(f"\n  Saved -> {outfile}.mp4")
    except Exception as e:
        print(f"  FFmpeg not available ({e}), saving as GIF...")
        anim.save(f'{outfile}.gif', writer='pillow', fps=fps, dpi=100)
        print(f"\n  Saved -> {outfile}.gif")

    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Wave field movie generator')
    parser.add_argument('--mode', choices=['normal', 'opt_trans', 'opt_reflect', 'both'],
                        default='both', help='Incident wave mode (default: both)')
    parser.add_argument('--num_cyl', type=int, default=50)
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--frames', type=int, default=80)
    parser.add_argument('-o', '--output', default='wave_field_movie')
    parser.add_argument('--pec', action='store_true',
                        help='Use PEC cylinders (eps=-1). Default: dielectric n=1.3')
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    _, n_eva, nmax = get_nmax(args.pec)

    print("=" * 65)
    print("  CyScat -- Wave Field Movie Generator")
    print("=" * 65)
    print(f"\n  Mode: {args.mode}  |  Cylinders: {args.num_cyl}  |  PEC: {args.pec}")
    print(f"  lambda={WAVELENGTH}  period={PERIOD}  r={RADIUS}  nmax={nmax}")

    clocs, thickness, S, material, nmax, crads = \
        setup(num_cyl=args.num_cyl, seed=args.seed, pec=args.pec)

    # Build field arrays for each mode
    if args.mode == 'both':
        modes_to_run = ['normal', 'opt_trans']
    else:
        modes_to_run = [args.mode]

    fields_list = []
    x_grid = y_full = None
    for m in modes_to_run:
        print(f"\n  Building field for mode={m} ...")
        F, xg, yf, label = build_field_arrays(S, nmax, n_eva, thickness, m, clocs, crads)
        print(f"  vmax={np.abs(np.real(F)).max():.4f}  label: {label}")
        fields_list.append((F, label))
        if x_grid is None:
            x_grid, y_full = xg, yf

    print(f"\n  Generating animation ({args.frames} frames/segment, {args.fps} fps) ...")
    make_movie(fields_list, x_grid, y_full, thickness, clocs, crads,
               args.num_cyl, material,
               n_frames=args.frames, fps=args.fps, outfile=args.output)

    # Generate standalone plots
    print("\n  Generating standalone plots ...")
    plot_transmission_distribution(S, nmax, n_eva, args.num_cyl, material,
                                   outfile=args.output + '_tau_dist')
    plot_modal_coefficients(S, nmax, n_eva,
                            outfile=args.output + '_modal_coeffs')

    print("=" * 65)


if __name__ == '__main__':
    main()
