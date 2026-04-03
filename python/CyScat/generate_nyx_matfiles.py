"""
Generate the .mat files equivalent to what MainNyx.m produced on the Nyx cluster.

Replicates MainNyx.m: for each IDX, generates 1440 random cylinder positions
(Latin hypercube), computes the full S-matrix, and saves everything to a .mat
file that MainCascadeResults.m can load.

Parameters are decoded from the original filename:
    1440CylinderLatinWidth180Thickness1100PECIDX{1..10}.mat

Usage (on Great Lakes with GPU):
    python generate_nyx_matfiles.py --idx 1          # single file
    python generate_nyx_matfiles.py --idx 1 10       # IDX 1 through 10
    python generate_nyx_matfiles.py --idx 1 10 --gpus 4  # multi-GPU
    python generate_nyx_matfiles.py --idx 1 10 --pec  # PEC cylinders (eps=-1)
"""
import sys
import os
import time
import argparse
import numpy as np
import scipy.io as sio

sys.path.insert(0, '.')
sys.path.insert(0, './Scattering_Code')

from position_generator import position_generator


def main():
    parser = argparse.ArgumentParser(
        description='Generate Nyx-style S-matrix .mat files')
    parser.add_argument('--idx', type=int, nargs='+', required=True,
                        help='IDX value(s). Single value or start end (e.g., --idx 1 10)')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs (default: 1)')
    parser.add_argument('--pec', action='store_true',
                        help='Use PEC cylinders (eps=-1). Default: dielectric (IOR=1.3)')
    parser.add_argument('--outdir', type=str, default='.',
                        help='Output directory for .mat files')
    args = parser.parse_args()

    # Parse IDX range
    if len(args.idx) == 1:
        idx_list = [args.idx[0]]
    elif len(args.idx) == 2:
        idx_list = list(range(args.idx[0], args.idx[1] + 1))
    else:
        idx_list = args.idx

    # Critical: set FORCE_CPU before importing GPU modules in multi-GPU mode
    if args.gpus > 1:
        os.environ['CYSCAT_FORCE_CPU'] = '1'

    # Now safe to import GPU-dependent modules
    from Scattering_Code.smatrix_parameters import smatrix_parameters
    from Scattering_Code.smatrix_cascade import smatrix_cascade

    # === Parameters matching MainNyx.m ===
    # Decoded from filename: 1440CylinderLatinWidth180Thickness1100PEC
    NoCylinders = 1440
    RandomSet = 'Latin'
    Width = 180          # The original Width before period adjustment
    Thickness = 1100
    Radius = 0.1
    lambda_wave = 0.93
    IOR = 1.3
    NoCylinderModes = 3
    RandFactor = 0.5
    MinInterDistance = Radius * 10  # = 1.0
    RemoveEdgeCylinders = 0

    # Period = Width + 3.31 (matching MainNyx.m original version)
    period = Width + 3.31

    if args.pec:
        epsilon = -1
        material_tag = 'PEC'
    else:
        epsilon = IOR ** 2  # = 1.69
        material_tag = 'PEC'  # filename always says PEC (matching original)

    NoPropagatingModes = int(np.floor(period / lambda_wave))

    # Evanescent mode calculation (matching MainNyx.m)
    CascadeTol = 10 ** (-1.1)
    Bufferlength = Radius * 2
    NoEvaMode = int(np.floor(
        period / (2 * np.pi) * np.sqrt(
            (np.log(CascadeTol) / Bufferlength) ** 2 + (2 * np.pi / lambda_wave) ** 2
        )
    )) - int(np.floor(period / lambda_wave))
    NoEvaMode = max(NoEvaMode, 0)

    if args.pec:
        NoEvaMode = 0

    nmax = int(np.floor(period / lambda_wave)) + NoEvaMode
    d = Thickness  # MainNyx.m: d = Thickness (no buffer added)

    # S-matrix parameters (matching MainNyx.m)
    phiinc = np.pi / 2
    sp = smatrix_parameters(lambda_wave, period, phiinc,
                            1e-11, 1e-4, 5, 3, 1000, 3, -1, 1, period / 120)

    print("=" * 70)
    print(f"Generate Nyx .mat files — {material_tag}")
    print(f"  NoCylinders={NoCylinders}, Width={Width}, Thickness={Thickness}")
    print(f"  period={period}, lambda={lambda_wave}, Radius={Radius}")
    print(f"  epsilon={epsilon}, NoCylinderModes={NoCylinderModes}")
    print(f"  nmax={nmax}, NoEvaMode={NoEvaMode}, NoPropModes={NoPropagatingModes}")
    print(f"  S-matrix size: {2*(2*nmax+1)} x {2*(2*nmax+1)}")
    print(f"  IDX range: {idx_list}")
    print(f"  GPUs: {args.gpus}")
    print("=" * 70)

    os.makedirs(args.outdir, exist_ok=True)

    for IDX in idx_list:
        filename = (f"{NoCylinders}Cylinder{RandomSet}Width{Width}"
                    f"Thickness{Thickness}{material_tag}IDX{IDX}.mat")
        filepath = os.path.join(args.outdir, filename)

        if os.path.exists(filepath):
            print(f"\n{filename} already exists, skipping.")
            continue

        print(f"\n{'='*70}")
        print(f"Generating IDX={IDX}: {filename}")
        print(f"{'='*70}")

        t_start = time.time()

        # --- Generate positions (matching MainNyx.m logic) ---
        # MainNyx uses time-based seed: IDX * clock_seconds
        # We use a deterministic seed based on IDX for reproducibility
        np.random.seed(IDX * 12345)

        # Build GP dict matching the MATLAB struct
        GP = {
            'RandomSet': RandomSet,
            'RandomFactor': RandFactor,
            'MinInterDistance': MinInterDistance,
            'NoCylinders': NoCylinders,
            'Width': Width,
            'Thickness': Thickness,
            'Radius': Radius,
            'Wavelength': lambda_wave,
            'IndexOfRefraction_Cylinder': IOR,
            'Period': period,
            'NoCylinderModes': NoCylinderModes,
            'NoPropagatingModes': NoPropagatingModes,
        }

        # Keep regenerating until we lose at most 7 cylinders
        ModifiedNoCylinders = 0
        attempts = 0
        while (NoCylinders - ModifiedNoCylinders) > 7:
            ModifiedNoCylinders, InitialPositions, RealPositions = \
                position_generator(GP)
            attempts += 1
            if attempts > 50:
                print(f"  WARNING: Could not place enough cylinders after "
                      f"{attempts} attempts. Got {ModifiedNoCylinders}/{NoCylinders}")
                break

        print(f"  Placed {ModifiedNoCylinders}/{NoCylinders} cylinders "
              f"(after {attempts} attempt(s))")

        # Python position_generator returns (N, 2) arrays
        # Python smatrix also expects (N, 2) for clocs
        clocs = RealPositions  # (N, 2)

        cepmus = np.column_stack([
            epsilon * np.ones(ModifiedNoCylinders),
            np.ones(ModifiedNoCylinders)
        ])
        crads = Radius * np.ones(ModifiedNoCylinders)
        cmmaxs = NoCylinderModes * np.ones(ModifiedNoCylinders, dtype=int)

        # --- Compute S-matrix ---
        print(f"  Computing S-matrix for {ModifiedNoCylinders} cylinders...")
        t0 = time.time()
        S, STP = smatrix_cascade(clocs, cmmaxs, cepmus, crads, period,
                                  lambda_wave, nmax, d, sp, 'On',
                                  cylinders_per_group=50, num_gpus=args.gpus)
        elapsed = time.time() - t0
        print(f"  S-matrix computed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"  S-matrix size: {S.shape}")

        # --- Save .mat file (matching MainNyx.m: save(filename) saves all workspace) ---
        save_dict = {
            'S': S,
            'd': d,
            'nmax': nmax,
            'NoEvaMode': NoEvaMode,
            'NoPropagatingModes': NoPropagatingModes,
            'clocs': clocs,
            'cmmaxs': cmmaxs,
            'cepmus': cepmus,
            'crads': crads,
            'NoCylinders': NoCylinders,
            'ModifiedNoCylinders': ModifiedNoCylinders,
            'InitialPositions': InitialPositions,
            'RealPositions': RealPositions,
            'Width': Width,
            'Thickness': Thickness,
            'Radius': Radius,
            'RandomSet': RandomSet,
            'RandFactor': RandFactor,
            'MinInterDistance': MinInterDistance,
            'Bufferlength': Bufferlength,
            'period': period,
            'lambda': lambda_wave,
            'IOR': IOR,
            'CascadeTol': CascadeTol,
            'NoCylinderModes': NoCylinderModes,
        }

        sio.savemat(filepath, save_dict, do_compression=True)
        total_time = time.time() - t_start
        print(f"  Saved {filepath} ({os.path.getsize(filepath) / 1e6:.1f} MB)")
        print(f"  Total time for IDX={IDX}: {total_time:.1f}s ({total_time/60:.1f} min)")

    print(f"\nAll done! Generated {len(idx_list)} file(s).")


if __name__ == '__main__':
    main()
