"""
optimize_n_r.py
===============
Fix cylinder positions, define f(param) = ||S21(param) @ x||
where x is a fixed random vector, then:

  A) Optimize refractive index n (r fixed) with Adam + jax.grad
  B) Optimize radius r (n fixed) with Adam + jax.grad

Key speedup: T-matrix (translation matrix) depends only on cylinder
positions, NOT on n or r. We precompute it once (~50s), then each
subsequent evaluation only recomputes Mie coefficients + linear algebra
(~1-2s). This gives ~25x speedup per evaluation.

Usage (from CyScat root):
    python optimize_n_r.py

Outputs:
    optimize_n_results.png
    optimize_r_results.png
"""

import sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, '.')
sys.path.insert(0, './Scattering_Code')

from Scattering_Code.smatrix_parameters import smatrix_parameters
from Scattering_Code.smatrix import smatrix_precompute, smatrix_from_precomputed

# ─── constants ───────────────────────────────────────────────────────────────

SEED_CLOCS  = 42
SEED_X      = 7
N_CYL       = 10
CMMAX       = 5
WAVELENGTH  = 0.93
PERIOD      = 12.81
PHIINC      = np.pi / 2
MU          = 1.0

Eva_TOL     = 1e-2
N_PROP      = int(np.floor(PERIOD / WAVELENGTH))
N_EVA       = max(int(np.floor(
    PERIOD / (2*np.pi) * np.sqrt(
        (np.log(Eva_TOL) / (2*0.25))**2 + (2*np.pi/WAVELENGTH)**2
    )
)) - N_PROP, 0)
NMAX        = N_PROP + N_EVA

# ─── generate fixed cylinder positions ───────────────────────────────────────

def make_clocs(num_cyl, radius=0.25, seed=SEED_CLOCS):
    np.random.seed(seed)
    margin    = radius * 1.5
    spacing   = 2.5 * radius
    rows      = num_cyl / int(PERIOD / spacing) + 2
    thickness = max(0.5, rows * spacing * 1.5)
    thickness = round(thickness, 1)
    clocs = np.zeros((num_cyl, 2))
    for i in range(num_cyl):
        for _ in range(10000):
            x = np.random.uniform(margin, PERIOD - margin)
            y = np.random.uniform(margin, thickness - margin)
            if i == 0 or np.all(np.sqrt((x - clocs[:i,0])**2 +
                                         (y - clocs[:i,1])**2) > spacing):
                clocs[i] = [x, y]
                break
    return clocs, thickness

def make_sp():
    return smatrix_parameters(
        WAVELENGTH, PERIOD, PHIINC,
        1e-11, 1e-4, 5, 3, 1000, 3, 5, 1, PERIOD / 120
    )

# ─── extract truncated S21 ──────────────────────────────────────────────────

def extract_S21T(S):
    nm = 2 * NMAX + 1
    S21 = S[nm:, :nm]
    if N_EVA > 0:
        return S21[N_EVA:-N_EVA, N_EVA:-N_EVA]
    return S21

# ─── objective functions (pure numpy, no JAX tracing) ────────────────────────

def _eval_objective(pre, cepmus, crads, x_vec):
    """Evaluate ||S21T @ x|| using precomputed data."""
    S = smatrix_from_precomputed(pre, cepmus, crads)
    S21T = extract_S21T(S)
    return float(jnp.linalg.norm(S21T @ x_vec))


def make_objective_n(pre, x_vec, r_fixed):
    """f(n) = ||S21(n) @ x||. Pure evaluation, no AD."""
    def obj(n_val):
        eps = n_val ** 2
        cepmus = jnp.tile(jnp.array([eps, MU]), (N_CYL, 1))
        crads = jnp.full((N_CYL,), r_fixed)
        return _eval_objective(pre, cepmus, crads, x_vec)
    return obj


def make_objective_r(pre, x_vec, n_fixed):
    """f(r) = ||S21(r) @ x||. Pure evaluation, no AD."""
    eps_fixed = n_fixed ** 2
    def obj(r_val):
        cepmus = jnp.tile(jnp.array([eps_fixed, MU]), (N_CYL, 1))
        crads = jnp.full((N_CYL,), r_val)
        return _eval_objective(pre, cepmus, crads, x_vec)
    return obj

# ─── Adam optimizer with finite-difference gradient ──────────────────────────

def adam_optimize(obj_fn, param0, maximize=False,
                  lr=0.1, n_steps=25, p_min=1e-6, p_max=1e6,
                  beta1=0.9, beta2=0.999, eps=1e-8,
                  fd_delta=1e-5, param_name='p'):
    """
    Adam optimizer in log-space using central finite differences.

    For a scalar parameter, FD costs 2 forward evals per step (~3s with
    precomputed T-matrix) — much faster than jax.grad which requires
    tracing/compiling the full backward pass.

    Returns (param_history, f_history) where the last entry is the best found.
    """
    sign = -1.0 if maximize else 1.0

    log_p = float(np.log(param0))
    m, v = 0.0, 0.0

    f0 = obj_fn(param0)
    p_hist = [param0]
    f_hist = [f0]
    best_f, best_p = f0, param0
    print(f"    step  0: {param_name}={param0:.4f}  f={f0:.6f}")

    for t in range(1, n_steps + 1):
        t0_time = time.time()
        p_curr = np.exp(log_p)

        # Central FD gradient: df/dp
        fp = obj_fn(p_curr + fd_delta)
        fm = obj_fn(p_curr - fd_delta)
        grad_p = (fp - fm) / (2 * fd_delta)

        # Chain rule to log-space: df/d(log_p) = df/dp * p
        grad_log = sign * grad_p * p_curr

        # Adam update in log-space
        m = beta1 * m + (1 - beta1) * grad_log
        v = beta2 * v + (1 - beta2) * grad_log**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        log_p = log_p - lr * m_hat / (np.sqrt(v_hat) + eps)
        log_p = float(np.clip(log_p, np.log(p_min), np.log(p_max)))

        # Use midpoint estimate as function value (free, already computed)
        fval = (fp + fm) / 2.0
        p_new = np.exp(log_p)
        p_hist.append(p_new)
        f_hist.append(fval)

        if (maximize and fval > best_f) or (not maximize and fval < best_f):
            best_f = fval
            best_p = p_curr

        print(f"    step {t:2d}: {param_name}={p_new:.4f}  f={fval:.6f}  "
              f"grad={grad_log:+.3e}  ({time.time()-t0_time:.1f}s)")

    # Append best as final entry
    p_hist.append(best_p)
    f_hist.append(best_f)
    print(f"    * Best: {param_name}={best_p:.4f}  f={best_f:.6f}")

    return np.array(p_hist), np.array(f_hist)

# ─── dense scan ──────────────────────────────────────────────────────────────

def scan_param(obj_fn, param_vals, param_name='p'):
    fvals = []
    for i, p in enumerate(param_vals):
        t0 = time.time()
        fval = obj_fn(p)
        fvals.append(fval)
        print(f"    [{i+1}/{len(param_vals)}]  {param_name}={p:.3f}  "
              f"f={fval:.6f}  ({time.time()-t0:.1f}s)")
    return np.array(fvals)

# ─── plotting ────────────────────────────────────────────────────────────────

def make_figure(param_name, unit, scan_vals, scan_f,
                traj_min, fmin, traj_max, fmax, outfile):
    # Last entry = best-so-far
    best_p_min, best_f_min = traj_min[-1], fmin[-1]
    best_p_max, best_f_max = traj_max[-1], fmax[-1]
    steps_min = range(len(fmin) - 1)
    steps_max = range(len(fmax) - 1)

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(
        f"Optimization of  f({param_name}) = ||S21({param_name}) * x||   "
        f"[{N_CYL} cylinders, Adam + jax.grad]",
        fontsize=13, fontweight='bold', y=1.01
    )
    gs = gridspec.GridSpec(1, 3, wspace=0.35)

    # col 1 — loss surface
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(scan_vals, scan_f, color='steelblue', lw=2, label='f scan')
    ax1.axvline(best_p_min, color='red',   lw=1.5, ls='--',
                label=f'best min {param_name}={best_p_min:.4f}')
    ax1.axvline(best_p_max, color='green', lw=1.5, ls='--',
                label=f'best max {param_name}={best_p_max:.4f}')
    ax1.scatter([best_p_min], [best_f_min], color='red',   zorder=5, s=60)
    ax1.scatter([best_p_max], [best_f_max], color='green', zorder=5, s=60)
    ax1.set_xlabel(f'{param_name}  ({unit})', fontsize=11)
    ax1.set_ylabel('f', fontsize=11)
    ax1.set_title('Loss surface (full scan)', fontsize=11)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # col 2 — minimization
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(list(steps_min), fmin[:-1], color='red', lw=1.8, label='Adam')
    ax2.axhline(scan_f.min(), color='grey', lw=1, ls=':', label='scan min')
    ax2.axhline(best_f_min, color='red', lw=1, ls='--', label='best')
    ax2.set_xlabel('Adam step', fontsize=11)
    ax2.set_ylabel('f', fontsize=11)
    ax2.set_title(f'Minimize  *  {param_name}={best_p_min:.4f}', fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2i = ax2.inset_axes([0.55, 0.55, 0.42, 0.38])
    ax2i.plot(list(steps_min), traj_min[:-1], color='red', lw=1.2)
    ax2i.axhline(best_p_min, color='grey', lw=1, ls=':')
    ax2i.set_xlabel('step', fontsize=7)
    ax2i.set_ylabel(param_name, fontsize=7)
    ax2i.tick_params(labelsize=6)

    # col 3 — maximization
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(list(steps_max), fmax[:-1], color='green', lw=1.8, label='Adam')
    ax3.axhline(scan_f.max(), color='grey', lw=1, ls=':', label='scan max')
    ax3.axhline(best_f_max, color='darkgreen', lw=1, ls='--', label='best')
    ax3.set_xlabel('Adam step', fontsize=11)
    ax3.set_ylabel('f', fontsize=11)
    ax3.set_title(f'Maximize  *  {param_name}={best_p_max:.4f}', fontsize=11)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3i = ax3.inset_axes([0.55, 0.1, 0.42, 0.38])
    ax3i.plot(list(steps_max), traj_max[:-1], color='green', lw=1.2)
    ax3i.axhline(best_p_max, color='grey', lw=1, ls=':')
    ax3i.set_xlabel('step', fontsize=7)
    ax3i.set_ylabel(param_name, fontsize=7)
    ax3i.tick_params(labelsize=6)

    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved -> {outfile}")

# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  CyScat -- n and r Optimization  (JAX + precomputed T-matrix)")
    print("=" * 62)

    R_FIXED = 0.25
    N_FIXED = 1.3

    clocs_np, thickness = make_clocs(N_CYL)
    clocs_jax = jnp.array(clocs_np)
    sp = make_sp()

    # Fixed random probe vector
    np.random.seed(SEED_X)
    x_np = np.random.randn(2*N_PROP+1) + 1j * np.random.randn(2*N_PROP+1)
    x_np /= np.linalg.norm(x_np)
    x_vec = jnp.array(x_np)

    print(f"\n  Cylinders  : {N_CYL}  (seed={SEED_CLOCS})")
    print(f"  lambda={WAVELENGTH}  period={PERIOD}  N_prop={N_PROP}  N_eva={N_EVA}")
    print(f"  S21_T size : {2*N_PROP+1} x {2*N_PROP+1}")

    # ── Precompute T-matrix and all position-dependent quantities ─────────
    print("\n  Precomputing (T-matrix, projection matrices, normalization)...")
    t_pre = time.time()
    cmmaxs = np.array([CMMAX] * N_CYL)
    pre = smatrix_precompute(
        clocs_jax, cmmaxs, PERIOD, WAVELENGTH, NMAX, thickness, sp, 'On',
        clocs_concrete=clocs_np
    )
    print(f"  Precompute total: {time.time()-t_pre:.1f}s")
    print(f"  (Subsequent evaluations will take ~1-2s each)\n")

    # ── PART 1: Optimize refractive index n ──────────────────────────────
    print("-" * 62)
    print(f"  PART 1: f(n) = ||S21(n) @ x||,   r = {R_FIXED} fixed")
    print("-" * 62)

    obj_n = make_objective_n(pre, x_vec, r_fixed=R_FIXED)

    print("\n  Scanning n in [1.0, 3.0] (10 points) ...")
    n_scan = np.linspace(1.0, 3.0, 10)
    t0 = time.time()
    f_n_scan = scan_param(obj_n, n_scan, param_name='n')
    print(f"  Scan done in {time.time()-t0:.0f}s")

    scan_min_n = n_scan[np.argmin(f_n_scan)]
    scan_max_n = n_scan[np.argmax(f_n_scan)]
    print(f"\n  Scan min: n={scan_min_n:.3f}  f={f_n_scan.min():.6f}")
    print(f"  Scan max: n={scan_max_n:.3f}  f={f_n_scan.max():.6f}")

    print("\n  Adam minimize n ...")
    t0 = time.time()
    traj_n_min, fmin_n = adam_optimize(obj_n, 1.5, maximize=False,
                                        lr=0.1, n_steps=25,
                                        p_min=1.0, p_max=3.5, param_name='n')
    print(f"  Done in {time.time()-t0:.0f}s")

    print("\n  Adam maximize n ...")
    t0 = time.time()
    traj_n_max, fmax_n = adam_optimize(obj_n, 1.5, maximize=True,
                                        lr=0.1, n_steps=25,
                                        p_min=1.0, p_max=3.5, param_name='n')
    print(f"  Done in {time.time()-t0:.0f}s")

    make_figure('n', 'refractive index', n_scan, f_n_scan,
                traj_n_min, fmin_n, traj_n_max, fmax_n,
                'optimize_n_results.png')

    # ── PART 2: Optimize radius r ────────────────────────────────────────
    print("\n" + "-" * 62)
    print(f"  PART 2: f(r) = ||S21(r) @ x||,   n = {N_FIXED} fixed")
    print("-" * 62)

    obj_r = make_objective_r(pre, x_vec, n_fixed=N_FIXED)

    print("\n  Scanning r in [0.15, 0.44] (8 points) ...")
    r_scan = np.linspace(0.15, 0.44, 8)
    t0 = time.time()
    f_r_scan = scan_param(obj_r, r_scan, param_name='r')
    print(f"  Scan done in {time.time()-t0:.0f}s")

    scan_min_r = r_scan[np.argmin(f_r_scan)]
    scan_max_r = r_scan[np.argmax(f_r_scan)]
    print(f"\n  Scan min: r={scan_min_r:.3f}  f={f_r_scan.min():.6f}")
    print(f"  Scan max: r={scan_max_r:.3f}  f={f_r_scan.max():.6f}")

    print("\n  Adam minimize r ...")
    t0 = time.time()
    traj_r_min, fmin_r = adam_optimize(obj_r, 0.25, maximize=False,
                                        lr=0.05, n_steps=25,
                                        p_min=0.15, p_max=0.44, param_name='r')
    print(f"  Done in {time.time()-t0:.0f}s")

    print("\n  Adam maximize r ...")
    t0 = time.time()
    traj_r_max, fmax_r = adam_optimize(obj_r, 0.25, maximize=True,
                                        lr=0.05, n_steps=25,
                                        p_min=0.15, p_max=0.44, param_name='r')
    print(f"  Done in {time.time()-t0:.0f}s")

    make_figure('r', 'cylinder radius', r_scan, f_r_scan,
                traj_r_min, fmin_r, traj_r_max, fmax_r,
                'optimize_r_results.png')

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  FINAL SUMMARY")
    print("=" * 62)
    print(f"\n  n  (r={R_FIXED} fixed):")
    print(f"    Scan min : n={scan_min_n:.3f}  f={f_n_scan.min():.6f}")
    print(f"    Adam min : n={traj_n_min[-1]:.4f}  f={fmin_n[-1]:.6f}  "
          f"{'ok' if abs(traj_n_min[-1]-scan_min_n)<0.25 else '? local min'}")
    print(f"    Scan max : n={scan_max_n:.3f}  f={f_n_scan.max():.6f}")
    print(f"    Adam max : n={traj_n_max[-1]:.4f}  f={fmax_n[-1]:.6f}  "
          f"{'ok' if abs(traj_n_max[-1]-scan_max_n)<0.25 else '? local max'}")
    print(f"\n  r  (n={N_FIXED} fixed):")
    print(f"    Scan min : r={scan_min_r:.3f}  f={f_r_scan.min():.6f}")
    print(f"    Adam min : r={traj_r_min[-1]:.4f}  f={fmin_r[-1]:.6f}  "
          f"{'ok' if abs(traj_r_min[-1]-scan_min_r)<0.06 else '? local min'}")
    print(f"    Scan max : r={scan_max_r:.3f}  f={f_r_scan.max():.6f}")
    print(f"    Adam max : r={traj_r_max[-1]:.4f}  f={fmax_r[-1]:.6f}  "
          f"{'ok' if abs(traj_r_max[-1]-scan_max_r)<0.06 else '? local max'}")
    print(f"\n  Plots: optimize_n_results.png  optimize_r_results.png")
    print("=" * 62)

if __name__ == '__main__':
    main()
