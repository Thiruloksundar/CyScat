"""
MainNyx - Large-scale batch S-matrix computation (HPC)
Translated from MainNyx.m (Curtis Jin, 2011)

Designed for HPC batch jobs. Takes an index IDX, generates random cylinder
positions using Latin hypercube sampling, computes the full S-matrix, and
saves everything to a .npz file.

Usage:
    python main_nyx.py 1
    python main_nyx.py 1 --num_cyl 720 --thickness 520
    python main_nyx.py 1 --num_cyl 1440 --thickness 1100 --gpus 4
"""
import sys
import os
import time
import argparse
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, './Scattering_Code')


def main():
    parser = argparse.ArgumentParser(description='HPC batch S-matrix computation')
    parser.add_argument('idx', type=int, help='Trial index (used for random seed)')
    parser.add_argument('--num_cyl', type=int, default=1440)
    parser.add_argument('--thickness', type=float, default=1100)
    parser.add_argument('--width', type=float, default=183.31,
                        help='Period/width (default: 180 + 3.31)')
    parser.add_argument('--radius', type=float, default=0.1)
    parser.add_argument('--ior', type=float, default=1.3,
                        help='Index of refraction (use 0 for PEC)')
    parser.add_argument('--gpus', type=int, default=1)
    args = parser.parse_args()

    IDX = args.idx
    NoCylinders = args.num_cyl
    Thickness = args.thickness
    period = args.width
    radius = args.radius
    wavelength = 0.93
    IOR = args.ior
    epsilon = -1 if IOR == 0 else IOR**2
    mu = 1.0
    NoCylinderModes = 3
    NoPropagatingModes = int(np.floor(period / wavelength))

    # Output filename
    cyl_type = 'PEC' if epsilon < 0 else f'IOR{IOR}'
    filename = (f'{NoCylinders}Cylinder_Width{period:.0f}_'
                f'Thickness{Thickness:.0f}_{cyl_type}_IDX{IDX}')
    npz_file = filename + '.npz'

    if os.path.exists(npz_file):
        print(f"{npz_file} already exists!")
        return

    # Force CPU for parent in multi-GPU mode
    if args.gpus > 1:
        os.environ['CYSCAT_FORCE_CPU'] = '1'

    from Scattering_Code.smatrix_parameters import smatrix_parameters
    from Scattering_Code.smatrix_cascade import smatrix_cascade

    # === Random seed ===
    seed = IDX * 1000 + int(time.time()) % 1000
    np.random.seed(seed)

    # === Generate random positions ===
    MinInterDistance = radius * 10
    margin = radius * 2
    clocs = np.zeros((NoCylinders, 2))
    placed = 0
    for i in range(NoCylinders):
        for attempt in range(10000):
            x = np.random.uniform(margin, period - margin)
            y = np.random.uniform(margin, Thickness - margin)
            if i == 0:
                clocs[i] = [x, y]
                placed += 1
                break
            dists = np.sqrt((x - clocs[:i, 0])**2 + (y - clocs[:i, 1])**2)
            if np.all(dists > MinInterDistance):
                clocs[i] = [x, y]
                placed += 1
                break

    ModifiedNoCylinders = placed
    clocs = clocs[:ModifiedNoCylinders]

    if NoCylinders - ModifiedNoCylinders > 7:
        print(f"WARNING: Only placed {ModifiedNoCylinders}/{NoCylinders} cylinders")

    cmmaxs = NoCylinderModes * np.ones(ModifiedNoCylinders, dtype=int)
    cepmus = np.column_stack([epsilon * np.ones(ModifiedNoCylinders),
                               np.ones(ModifiedNoCylinders)])
    crads = radius * np.ones(ModifiedNoCylinders)

    # === Evanescent mode truncation ===
    CascadeTol = 10**(-1.1)
    Bufferlength = radius * 2
    NoEvaMode = int(np.floor(
        period / (2 * np.pi) * np.sqrt(
            (np.log(CascadeTol) / Bufferlength)**2 + (2 * np.pi / wavelength)**2
        )
    )) - NoPropagatingModes
    NoEvaMode = max(NoEvaMode, 0)
    if epsilon < 0:  # PEC: skip evanescent buffer
        NoEvaMode = 0
    nmax = NoPropagatingModes + NoEvaMode
    d = Thickness

    print("=" * 60)
    print(f"MainNyx: {ModifiedNoCylinders} cylinders, IDX={IDX}")
    print(f"  period={period}, thickness={Thickness}")
    print(f"  NoProp={NoPropagatingModes}, NoEva={NoEvaMode}, nmax={nmax}")
    print(f"  GPUs: {args.gpus}")
    print("=" * 60)

    sp = smatrix_parameters(wavelength, period, phiinc=np.pi / 2,
                            epsseries=1e-11, epsloc=1e-4,
                            nrepeat_spatial=5, nrepeat_spectral=3,
                            jmax=1000, kshanks_spatial=3,
                            kshanks_spectral=-1, spectral=1,
                            spectral_cond=period / 120)

    # === Compute S-matrix ===
    print(f"\nComputing S-matrix...")
    t0 = time.time()
    S, STP = smatrix_cascade(clocs, cmmaxs, cepmus, crads, period, wavelength,
                              nmax, d, sp, 'On', num_gpus=args.gpus)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    # === Save ===
    np.savez(npz_file, S=S, d=d, nmax=nmax, NoEvaMode=NoEvaMode,
             clocs=clocs, cmmaxs=cmmaxs, cepmus=cepmus, crads=crads,
             NoPropagatingModes=NoPropagatingModes,
             ModifiedNoCylinders=ModifiedNoCylinders,
             period=period, wavelength=wavelength, thickness=Thickness,
             elapsed=elapsed, seed=seed, IDX=IDX)
    print(f"Saved to {npz_file}")


if __name__ == '__main__':
    main()
