"""
Check differentiability of the CyScat scattering pipeline using JAX autodiff.

Computes gradients via jax.grad and compares with finite difference results
from check_differentiability.py (fd_gradient_16cyl.npz).

Usage:
    python check_differentiability_jax.py
    python check_differentiability_jax.py --num_cyl 16
"""
import sys
import os
import time
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# JAX setup
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, '.')
sys.path.insert(0, './Scattering_Code')

from Scattering_Code.smatrix_parameters import smatrix_parameters
from Scattering_Code.smatrix import smatrix
from Scattering_Code.ky import ky


def setup_parameters(num_cyl, seed=42):
    """Set up scattering parameters (same as check_differentiability.py)."""
    wavelength = 0.93
    period = 12.81
    radius = 0.25
    n_cylinder = 1.3
    eps = n_cylinder ** 2
    mu = 1.0
    cmmax = 5
    phiinc = np.pi / 2

    spacing = 2.5 * radius
    cyls_per_row = int(period / spacing)
    rows_needed = num_cyl / cyls_per_row + 2
    thickness = max(0.5, rows_needed * spacing * 1.5)
    thickness = round(thickness, 1)

    EvanescentModeTol = 1e-2
    NoPropagatingModes = int(np.floor(period / wavelength))
    NoEvaMode = int(np.floor(
        period / (2 * np.pi) * np.sqrt(
            (np.log(EvanescentModeTol) / (2 * radius)) ** 2 +
            (2 * np.pi / wavelength) ** 2
        )
    )) - NoPropagatingModes
    NoEvaMode = max(NoEvaMode, 0)
    nmax = NoPropagatingModes + NoEvaMode

    # Generate positions (same as FD code)
    np.random.seed(seed)
    margin = radius * 1.5
    clocs = np.zeros((num_cyl, 2))
    for i in range(num_cyl):
        for attempt in range(5000):
            x = np.random.uniform(margin, period - margin)
            y = np.random.uniform(margin, thickness - margin)
            if i == 0:
                clocs[i] = [x, y]
                break
            dists = np.sqrt((x - clocs[:i, 0]) ** 2 + (y - clocs[:i, 1]) ** 2)
            if np.all(dists > 2.5 * radius):
                clocs[i] = [x, y]
                break

    params = {
        'wavelength': wavelength,
        'period': period,
        'radius': radius,
        'eps': eps,
        'mu': mu,
        'cmmax': cmmax,
        'phiinc': phiinc,
        'thickness': thickness,
        'nmax': nmax,
        'NoEvaMode': NoEvaMode,
        'NoPropagatingModes': NoPropagatingModes,
    }
    return clocs, params


def compute_smatrix_from_clocs(clocs_jax, params, sp, cmmaxs, cepmus, crads, clocs_concrete=None):
    """Compute S-matrix from cylinder positions (JAX-traceable)."""
    S, _ = smatrix(
        clocs_jax, cmmaxs, cepmus, crads,
        params['period'], params['wavelength'],
        params['nmax'], params['thickness'], sp, 'On',
        clocs_concrete=clocs_concrete)
    return S


def total_transmission_objective(clocs_jax, params, sp, cmmaxs, cepmus, crads, clocs_concrete=None):
    """
    Compute total transmission (sum of tau^2) from cylinder positions.
    This is the scalar objective for gradient computation.
    """
    S = compute_smatrix_from_clocs(clocs_jax, params, sp, cmmaxs, cepmus, crads,
                                    clocs_concrete=clocs_concrete)

    nm = 2 * params['nmax'] + 1
    S21 = S[nm:, :nm]
    neva = params['NoEvaMode']
    if neva > 0:
        S21T = S21[neva:-neva, neva:-neva]
    else:
        S21T = S21

    # tau_sq = jnp.linalg.svd(S21T, compute_uv=False) ** 2
    # return jnp.sum(tau_sq)
    return jnp.sum(jnp.abs(S21T) ** 2)


def main():
    parser = argparse.ArgumentParser(
        description='Check CyScat differentiability using JAX autodiff')
    parser.add_argument('--num_cyl', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--outdir', type=str, default='.')
    parser.add_argument('--fd_file', type=str, default=None,
                        help='Path to FD gradient file for comparison')
    args = parser.parse_args()

    print(f"{'=' * 60}")
    print(f"CyScat JAX Autodiff Gradient Check")
    print(f"  Cylinders: {args.num_cyl}")
    print(f"  JAX backend: {jax.default_backend()}")
    print(f"{'=' * 60}")

    clocs, params = setup_parameters(args.num_cyl, args.seed)
    print(f"  Period={params['period']}, wavelength={params['wavelength']}")
    print(f"  Thickness={params['thickness']}, nmax={params['nmax']}")
    print(f"  NoEvaMode={params['NoEvaMode']}, NoPropModes={params['NoPropagatingModes']}")
    print(f"  Placed {len(clocs)} cylinders\n")

    # Setup parameters
    sp = smatrix_parameters(
        params['wavelength'], params['period'], params['phiinc'],
        1e-11, 1e-4, 5, 3, 1000, 3, 5, 1, params['period'] / 120
    )

    num_cyl = len(clocs)
    cmmaxs = np.array([params['cmmax']] * num_cyl)
    cepmus = np.array([[params['eps'], params['mu']]] * num_cyl)
    crads = np.array([params['radius']] * num_cyl)

    clocs_jax = jnp.array(clocs)
    clocs_concrete = np.array(clocs)  # concrete copy for pair classification

    # === Forward pass ===
    print("=== Forward Pass ===")
    t0 = time.time()
    f0 = total_transmission_objective(clocs_jax, params, sp, cmmaxs, cepmus, crads,
                                       clocs_concrete=clocs_concrete)
    forward_time = time.time() - t0
    print(f"  total_transmission = {float(f0):.10f}")
    print(f"  Forward time: {forward_time:.1f}s\n")

    # === JAX gradient ===
    print("=== Computing JAX Gradient (jax.grad) ===")
    # Use closure to pass clocs_concrete without it being differentiated
    def objective_for_grad(clocs_jax):
        return total_transmission_objective(
            clocs_jax, params, sp, cmmaxs, cepmus, crads,
            clocs_concrete=clocs_concrete)

    grad_fn = jax.grad(objective_for_grad)

    t0 = time.time()
    jax_grad = grad_fn(clocs_jax)
    grad_time = time.time() - t0

    jax_grad_np = np.asarray(jax_grad)
    grad_x = jax_grad_np[:, 0]
    grad_y = jax_grad_np[:, 1]
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    print(f"  Gradient computation time: {grad_time:.1f}s")
    print(f"  df/dx: min={np.min(grad_x):+.6e}, max={np.max(grad_x):+.6e}, "
          f"mean={np.mean(np.abs(grad_x)):.6e}")
    print(f"  df/dy: min={np.min(grad_y):+.6e}, max={np.max(grad_y):+.6e}, "
          f"mean={np.mean(np.abs(grad_y)):.6e}")
    print(f"  |grad|: min={np.min(grad_mag):.6e}, max={np.max(grad_mag):.6e}")

    # Check for issues
    n_nan = np.sum(np.isnan(jax_grad_np))
    n_inf = np.sum(np.isinf(jax_grad_np))
    print(f"  NaN: {n_nan}, Inf: {n_inf}")

    os.makedirs(args.outdir, exist_ok=True)

    # Save JAX gradient
    outfile = os.path.join(args.outdir, f'jax_gradient_{args.num_cyl}cyl.npz')
    np.savez(outfile, grad=jax_grad_np, grad_x=grad_x, grad_y=grad_y,
             grad_mag=grad_mag, f0=float(f0), clocs=clocs,
             objective='total_transmission', **params)
    print(f"  JAX gradient saved to {outfile}")

    # === Compare with finite differences ===
    fd_file = args.fd_file
    if fd_file is None:
        fd_file = os.path.join(args.outdir, f'fd_gradient_{args.num_cyl}cyl.npz')

    if os.path.exists(fd_file):
        print(f"\n{'=' * 60}")
        print(f"COMPARISON: JAX autodiff vs Finite Differences")
        print(f"  FD file: {fd_file}")

        fd_data = np.load(fd_file)
        fd_grad = fd_data['grad']
        fd_f0 = float(fd_data['f0'])
        fd_delta = float(fd_data['delta'])

        print(f"  FD objective = {fd_f0:.10f}")
        print(f"  JAX objective = {float(f0):.10f}")
        print(f"  Objective difference: {abs(float(f0) - fd_f0):.6e}")
        print(f"  FD delta: {fd_delta}")

        # Per-component comparison
        abs_diff = np.abs(jax_grad_np - fd_grad)
        fd_mag = np.abs(fd_grad)
        rel_diff = np.where(fd_mag > 1e-10, abs_diff / fd_mag, abs_diff)

        print(f"\n  Per-component comparison:")
        print(f"  {'cyl':>4s} {'coord':>5s} {'FD gradient':>16s} {'JAX gradient':>16s} "
              f"{'abs diff':>12s} {'rel diff':>12s}")
        print(f"  {'-' * 70}")
        for i in range(num_cyl):
            for c, cname in enumerate(['x', 'y']):
                print(f"  {i:4d} {cname:>5s} {fd_grad[i, c]:+16.8e} "
                      f"{jax_grad_np[i, c]:+16.8e} {abs_diff[i, c]:12.4e} "
                      f"{rel_diff[i, c]:12.4e}")

        print(f"\n  Summary:")
        print(f"  Max absolute difference: {np.max(abs_diff):.6e}")
        print(f"  Mean absolute difference: {np.mean(abs_diff):.6e}")
        print(f"  Max relative difference: {np.max(rel_diff):.6e}")
        print(f"  Mean relative difference: {np.mean(rel_diff):.6e}")

        # Correlation
        corr_x = np.corrcoef(fd_grad[:, 0], jax_grad_np[:, 0])[0, 1]
        corr_y = np.corrcoef(fd_grad[:, 1], jax_grad_np[:, 1])[0, 1]
        print(f"  Correlation (x): {corr_x:.8f}")
        print(f"  Correlation (y): {corr_y:.8f}")

        # Threshold check
        match_threshold = 0.1  # 10% relative error is acceptable for FD comparison
        good = np.sum(rel_diff < match_threshold)
        total_comps = rel_diff.size
        print(f"\n  Components within {match_threshold*100:.0f}% relative error: "
              f"{good}/{total_comps} ({100*good/total_comps:.0f}%)")

        if np.mean(rel_diff) < 0.05 and corr_x > 0.99 and corr_y > 0.99:
            print(f"\n  RESULT: JAX gradients MATCH finite differences!")
        elif corr_x > 0.95 and corr_y > 0.95:
            print(f"\n  RESULT: JAX gradients APPROXIMATELY match FD "
                  f"(correlation > 0.95)")
        else:
            print(f"\n  RESULT: JAX gradients DO NOT match FD well. "
                  f"Investigation needed.")

        # === Comparison plots ===
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        fig.suptitle(f'JAX Autodiff vs Finite Differences — {args.num_cyl} Cylinders\n'
                     f'FD delta={fd_delta}',
                     fontsize=13, fontweight='bold')

        # Top-left: JAX vs FD scatter for x-gradients
        axes[0, 0].scatter(fd_grad[:, 0], jax_grad_np[:, 0],
                           c='steelblue', s=60, edgecolors='black', linewidths=0.5)
        lim = max(np.max(np.abs(fd_grad[:, 0])), np.max(np.abs(jax_grad_np[:, 0]))) * 1.1
        axes[0, 0].plot([-lim, lim], [-lim, lim], 'r--', alpha=0.5, label='y=x')
        axes[0, 0].set_xlabel(r'FD $\partial f/\partial x$')
        axes[0, 0].set_ylabel(r'JAX $\partial f/\partial x$')
        axes[0, 0].set_title(f'x-gradients (corr={corr_x:.4f})')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_aspect('equal')

        # Top-right: JAX vs FD scatter for y-gradients
        axes[0, 1].scatter(fd_grad[:, 1], jax_grad_np[:, 1],
                           c='coral', s=60, edgecolors='black', linewidths=0.5)
        lim = max(np.max(np.abs(fd_grad[:, 1])), np.max(np.abs(jax_grad_np[:, 1]))) * 1.1
        axes[0, 1].plot([-lim, lim], [-lim, lim], 'r--', alpha=0.5, label='y=x')
        axes[0, 1].set_xlabel(r'FD $\partial f/\partial y$')
        axes[0, 1].set_ylabel(r'JAX $\partial f/\partial y$')
        axes[0, 1].set_title(f'y-gradients (corr={corr_y:.4f})')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_aspect('equal')

        # Bottom-left: bar chart comparison
        x_pos = np.arange(num_cyl)
        width = 0.35
        axes[1, 0].bar(x_pos - width/2, fd_grad[:, 0], width, label='FD dx',
                        color='steelblue', alpha=0.7)
        axes[1, 0].bar(x_pos + width/2, jax_grad_np[:, 0], width, label='JAX dx',
                        color='coral', alpha=0.7)
        axes[1, 0].set_xlabel('Cylinder index')
        axes[1, 0].set_ylabel(r'$\partial f/\partial x$')
        axes[1, 0].set_title('x-gradient per cylinder')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Bottom-right: relative error histogram
        rel_diff_all = rel_diff.flatten()
        axes[1, 1].hist(rel_diff_all, bins=20, color='mediumpurple',
                        edgecolor='black', alpha=0.8)
        axes[1, 1].axvline(0.05, color='red', ls='--', alpha=0.7, label='5% threshold')
        axes[1, 1].set_xlabel('Relative error')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Distribution of relative errors')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plotfile = os.path.join(args.outdir,
                                f'jax_vs_fd_{args.num_cyl}cyl.png')
        plt.savefig(plotfile, dpi=150)
        print(f"\n  Comparison plot saved to {plotfile}")
        plt.close()

    else:
        print(f"\n  No FD file found at {fd_file}")
        print(f"  Run check_differentiability.py first to generate FD gradients")

    print(f"\n{'=' * 60}")


if __name__ == '__main__':
    main()
