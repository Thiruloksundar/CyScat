"""
Check differentiability of the CyScat scattering pipeline via finite differences.

Computes ∂f/∂x_i and ∂f/∂y_i for each cylinder position using central differences:
    df/dp ≈ (f(p + δ) - f(p - δ)) / (2δ)

where f is a scalar objective derived from the S-matrix (e.g., total transmission,
max τ², or mean τ²).

Usage:
    python check_differentiability.py                    # 16 cylinders, CPU
    python check_differentiability.py --num_cyl 16       # explicit
    python check_differentiability.py --num_cyl 32 --gpus 2
    python check_differentiability.py --delta 1e-5       # custom perturbation size
    python check_differentiability.py --convergence       # run convergence test
"""
import sys
import os
import time
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
sys.path.insert(0, './Scattering_Code')

from get_partition import smat_to_s21


def setup_parameters(num_cyl, seed=42):
    """Set up scattering parameters for a small test case."""
    wavelength = 0.93
    period = 12.81
    radius = 0.25
    n_cylinder = 1.3
    eps = n_cylinder ** 2  # dielectric (not PEC) so evanescent modes work
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

    # Generate positions
    np.random.seed(seed)
    margin = radius * 1.5
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
            dists = np.sqrt((x - clocs[:i, 0]) ** 2 + (y - clocs[:i, 1]) ** 2)
            if np.all(dists > 2.5 * radius):
                clocs[i] = [x, y]
                placed += 1
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


def compute_objective(clocs, params, num_gpus=1):
    """
    Compute scalar objectives from the S-matrix for given cylinder positions.

    Returns dict with multiple objectives so we can check differentiability of each.
    """
    from Scattering_Code.smatrix_parameters import smatrix_parameters
    from Scattering_Code.smatrix_cascade import smatrix_cascade

    num_cyl = len(clocs)
    cmmaxs = np.array([params['cmmax']] * num_cyl)
    cepmus = np.array([[params['eps'], params['mu']]] * num_cyl)
    crads = np.array([params['radius']] * num_cyl)

    sp = smatrix_parameters(
        params['wavelength'], params['period'], params['phiinc'],
        1e-11, 1e-4, 5, 3, 1000, 3, 5, 1, params['period'] / 120
    )

    S, _ = smatrix_cascade(
        clocs, cmmaxs, cepmus, crads,
        params['period'], params['wavelength'],
        params['nmax'], params['thickness'], sp, 'On',
        cylinders_per_group=50, num_gpus=num_gpus
    )

    # Extract S21 and truncate evanescent modes
    nm = 2 * params['nmax'] + 1
    S21 = S[nm:, :nm]
    neva = params['NoEvaMode']
    if neva > 0:
        S21T = S21[neva:-neva, neva:-neva]
    else:
        S21T = S21

    tau_sq = np.linalg.svd(S21T, compute_uv=False) ** 2

    return {
        'total_transmission': np.sum(tau_sq),
        'max_tau_sq': np.max(tau_sq),
        'mean_tau_sq': np.mean(tau_sq),
        'log_total_transmission': np.log(np.sum(tau_sq) + 1e-30),
        'tau_sq': tau_sq,  # full vector for reference
    }


def finite_difference_gradient(clocs, params, objective_name, delta, num_gpus=1):
    """
    Compute gradient of a scalar objective w.r.t. all cylinder positions
    using central finite differences.

    Returns gradient array of shape (num_cyl, 2) — [∂f/∂x_i, ∂f/∂y_i].
    """
    num_cyl = len(clocs)
    grad = np.zeros((num_cyl, 2))

    # Baseline (for logging, not needed for central differences)
    base_result = compute_objective(clocs, params, num_gpus)
    f0 = base_result[objective_name]
    print(f"  Baseline {objective_name} = {f0:.10f}")

    total_evals = 2 * num_cyl * 2  # 2 coords per cylinder, 2 evals per coord
    eval_count = 0

    for i in range(num_cyl):
        for coord in range(2):  # 0=x, 1=y
            # Forward perturbation
            clocs_plus = clocs.copy()
            clocs_plus[i, coord] += delta

            # Backward perturbation
            clocs_minus = clocs.copy()
            clocs_minus[i, coord] -= delta

            f_plus = compute_objective(clocs_plus, params, num_gpus)[objective_name]
            f_minus = compute_objective(clocs_minus, params, num_gpus)[objective_name]

            grad[i, coord] = (f_plus - f_minus) / (2 * delta)
            eval_count += 2

            coord_name = 'x' if coord == 0 else 'y'
            print(f"  cyl {i:3d} {coord_name}: "
                  f"f+ = {f_plus:.10f}, f- = {f_minus:.10f}, "
                  f"∂f/∂{coord_name} = {grad[i, coord]:+.8e}  "
                  f"[{eval_count}/{total_evals}]")

    return grad, f0


def convergence_test(clocs, params, objective_name, num_gpus=1):
    """
    Test convergence of finite differences by varying delta.
    If the function is smooth, the gradient estimate should converge
    as delta decreases, then diverge at very small delta due to floating point.
    """
    deltas = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    # Only perturb the first cylinder's x-coordinate for speed
    cyl_idx, coord_idx = 0, 0

    print(f"\nConvergence test: ∂({objective_name})/∂(cyl[0].x)")
    print(f"{'delta':>12s}  {'gradient':>16s}  {'rel_change':>12s}")
    print("-" * 44)

    grads = []
    for delta in deltas:
        clocs_plus = clocs.copy()
        clocs_plus[cyl_idx, coord_idx] += delta
        clocs_minus = clocs.copy()
        clocs_minus[cyl_idx, coord_idx] -= delta

        f_plus = compute_objective(clocs_plus, params, num_gpus)[objective_name]
        f_minus = compute_objective(clocs_minus, params, num_gpus)[objective_name]
        g = (f_plus - f_minus) / (2 * delta)
        grads.append(g)

        if len(grads) > 1:
            rel_change = abs(g - grads[-2]) / (abs(grads[-2]) + 1e-30)
            print(f"  {delta:.0e}    {g:+.10e}    {rel_change:.4e}")
        else:
            print(f"  {delta:.0e}    {g:+.10e}    {'---':>12s}")

    return deltas, grads


def main():
    parser = argparse.ArgumentParser(
        description='Check differentiability of CyScat via finite differences')
    parser.add_argument('--num_cyl', type=int, default=16,
                        help='Number of cylinders (default: 16)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--delta', type=float, default=1e-4,
                        help='Perturbation size for finite differences')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--objective', type=str, default='total_transmission',
                        choices=['total_transmission', 'max_tau_sq',
                                 'mean_tau_sq', 'log_total_transmission'],
                        help='Scalar objective to differentiate')
    parser.add_argument('--convergence', action='store_true',
                        help='Run convergence test instead of full gradient')
    parser.add_argument('--outdir', type=str, default='.')
    args = parser.parse_args()

    if args.gpus > 1:
        os.environ['CYSCAT_FORCE_CPU'] = '1'

    print(f"{'=' * 60}")
    print(f"CyScat Differentiability Check")
    print(f"  Cylinders: {args.num_cyl}")
    print(f"  Objective: {args.objective}")
    print(f"  Delta: {args.delta}")
    print(f"  GPUs: {args.gpus}")
    print(f"{'=' * 60}")

    clocs, params = setup_parameters(args.num_cyl, args.seed)
    print(f"  Period={params['period']}, wavelength={params['wavelength']}")
    print(f"  Thickness={params['thickness']}, nmax={params['nmax']}")
    print(f"  NoEvaMode={params['NoEvaMode']}, NoPropModes={params['NoPropagatingModes']}")
    print(f"  Placed {len(clocs)} cylinders\n")

    os.makedirs(args.outdir, exist_ok=True)

    if args.convergence:
        print("=== Convergence Test ===")
        t0 = time.time()
        deltas, grads = convergence_test(
            clocs, params, args.objective, args.gpus)
        elapsed = time.time() - t0
        print(f"\nConvergence test done in {elapsed:.1f}s")

        # Plot convergence
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogx(deltas, grads, 'o-', color='steelblue', markersize=6)
        ax.set_xlabel(r'$\delta$ (perturbation size)', fontsize=13)
        ax.set_ylabel(r'$\partial f / \partial x_0$', fontsize=13)
        ax.set_title(f'Finite Difference Convergence\n'
                     f'Objective: {args.objective}, {args.num_cyl} cylinders',
                     fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        plt.tight_layout()
        outfile = os.path.join(args.outdir, 'fd_convergence.png')
        plt.savefig(outfile, dpi=150)
        print(f"Plot saved to {outfile}")
        plt.close()

        np.savez(os.path.join(args.outdir, 'fd_convergence.npz'),
                 deltas=deltas, grads=grads,
                 objective=args.objective, num_cyl=args.num_cyl)
        return

    # === Full gradient computation ===
    print(f"=== Computing full gradient ({args.num_cyl} cyl x 2 coords "
          f"= {args.num_cyl * 2} parameters) ===")
    print(f"    Requires {args.num_cyl * 2 * 2 + 1} S-matrix evaluations\n")

    t0 = time.time()
    grad, f0 = finite_difference_gradient(
        clocs, params, args.objective, args.delta, args.gpus)
    elapsed = time.time() - t0

    grad_x = grad[:, 0]
    grad_y = grad[:, 1]
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    print(f"\n{'=' * 60}")
    print(f"GRADIENT RESULTS ({elapsed:.1f}s total)")
    print(f"  Objective: {args.objective} = {f0:.10f}")
    print(f"  Delta: {args.delta}")
    print(f"  ∂f/∂x: min={np.min(grad_x):+.6e}, max={np.max(grad_x):+.6e}, "
          f"mean={np.mean(np.abs(grad_x)):.6e}")
    print(f"  ∂f/∂y: min={np.min(grad_y):+.6e}, max={np.max(grad_y):+.6e}, "
          f"mean={np.mean(np.abs(grad_y)):.6e}")
    print(f"  |∇f|:  min={np.min(grad_mag):.6e}, max={np.max(grad_mag):.6e}, "
          f"mean={np.mean(grad_mag):.6e}")

    # Check for issues
    n_nan = np.sum(np.isnan(grad))
    n_inf = np.sum(np.isinf(grad))
    n_zero = np.sum(grad_mag < 1e-15)
    print(f"\n  NaN gradients: {n_nan}")
    print(f"  Inf gradients: {n_inf}")
    print(f"  Near-zero gradients (|∇| < 1e-15): {n_zero}")

    if n_nan == 0 and n_inf == 0:
        print(f"\n  RESULT: Function appears differentiable "
              f"(all {args.num_cyl * 2} gradients are finite)")
    else:
        print(f"\n  WARNING: Found {n_nan + n_inf} non-finite gradients — "
              f"potential differentiability issues")
    print(f"{'=' * 60}")

    # Save
    outfile = os.path.join(args.outdir,
                           f'fd_gradient_{args.num_cyl}cyl.npz')
    np.savez(outfile, grad=grad, grad_x=grad_x, grad_y=grad_y,
             grad_mag=grad_mag, f0=f0, clocs=clocs, delta=args.delta,
             objective=args.objective, **params)
    print(f"Data saved to {outfile}")

    # === Plot ===
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f'Finite Difference Gradients — {args.num_cyl} Cylinders\n'
                 f'Objective: {args.objective} = {f0:.6f}, δ = {args.delta}',
                 fontsize=13, fontweight='bold')

    # Top-left: cylinder positions colored by gradient magnitude
    sc = axes[0, 0].scatter(clocs[:, 0], clocs[:, 1], c=grad_mag,
                            cmap='hot', s=60, edgecolors='black', linewidths=0.5)
    plt.colorbar(sc, ax=axes[0, 0], label=r'$|\nabla f|$')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].set_title('Gradient magnitude at each cylinder')
    axes[0, 0].set_aspect('equal')
    axes[0, 0].grid(True, alpha=0.3)

    # Top-right: gradient quiver plot
    scale = np.max(grad_mag) * 10 if np.max(grad_mag) > 0 else 1
    axes[0, 1].quiver(clocs[:, 0], clocs[:, 1], grad_x, grad_y,
                      grad_mag, cmap='hot', scale=scale)
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    axes[0, 1].set_title('Gradient direction and magnitude')
    axes[0, 1].set_aspect('equal')
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom-left: histogram of gradient magnitudes
    axes[1, 0].hist(grad_mag, bins=20, color='steelblue',
                    edgecolor='black', alpha=0.8)
    axes[1, 0].set_xlabel(r'$|\nabla f|$')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Distribution of gradient magnitudes')
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom-right: ∂f/∂x vs ∂f/∂y scatter
    axes[1, 1].scatter(grad_x, grad_y, c='steelblue', s=40,
                       edgecolors='black', linewidths=0.5, alpha=0.8)
    axes[1, 1].axhline(0, color='gray', ls='--', alpha=0.5)
    axes[1, 1].axvline(0, color='gray', ls='--', alpha=0.5)
    axes[1, 1].set_xlabel(r'$\partial f / \partial x$')
    axes[1, 1].set_ylabel(r'$\partial f / \partial y$')
    axes[1, 1].set_title(r'$\partial f/\partial x$ vs $\partial f/\partial y$')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plotfile = os.path.join(args.outdir,
                            f'fd_gradient_{args.num_cyl}cyl.png')
    plt.savefig(plotfile, dpi=150)
    print(f"Plot saved to {plotfile}")
    plt.close()


if __name__ == '__main__':
    main()
