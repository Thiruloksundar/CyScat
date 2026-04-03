"""
MainReplica - Reproduce published far-field scattering results
Translated from MainReplica.m (Curtis Jin, 2011)

Computes scattering coefficients for a single large cylinder and plots
the far-field scattering pattern. Uses aperiodic mode (period=-1).

Usage:
    python main_replica.py
    python main_replica.py --radius 3 --wavelength 3 --epsilon 1.5
"""
import sys
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
sys.path.insert(0, './Scattering_Code')

from Scattering_Code.sall import sall
from Scattering_Code.ky import ky


def far_field_single_cylinder(cmmaxs, cepmus, crads, wavelength, nphi=100):
    """
    Compute far-field scattering pattern for a single cylinder.

    For a single cylinder at the origin, the scattered field in the far zone is:
    E_s(phi) = sum_m s_m * exp(i*m*phi) * H_m(kr)
    The scattering cross section is proportional to |f(phi)|^2 where
    f(phi) = sum_m s_m * exp(i*m*phi)

    Parameters
    ----------
    cmmaxs : array — max mode numbers
    cepmus : array — [epsilon, mu] per cylinder
    crads : array — radii
    wavelength : float
    nphi : int — number of angles

    Returns
    -------
    phi : (nphi,) angles in radians
    sigma : (nphi,) scattering cross section (2*pi*|f|^2)
    """
    k = 2 * np.pi / wavelength

    # Get Mie scattering coefficients
    s = sall(cmmaxs, cepmus, crads, wavelength)
    cmmax = int(cmmaxs[0])
    modes = np.arange(-cmmax, cmmax + 1)

    # Far-field pattern
    phi = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    f = np.zeros(nphi, dtype=complex)

    for i, m in enumerate(modes):
        f += s[i] * np.exp(1j * m * phi)

    # Scattering cross section: 2*pi*|f(phi)|^2
    sigma = 2 * np.pi * np.abs(f)**2

    return phi, sigma


def main():
    parser = argparse.ArgumentParser(description='Reproduce far-field scattering')
    parser.add_argument('--radius', type=float, default=3.0)
    parser.add_argument('--wavelength', type=float, default=3.0)
    parser.add_argument('--epsilon', type=float, default=1.5)
    parser.add_argument('--cmmax', type=int, default=50)
    parser.add_argument('--nphi', type=int, default=360)
    args = parser.parse_args()

    clocs = np.array([[0.0, 0.0]])
    cepmus = np.array([[args.epsilon, 1.0]])
    crads = np.array([args.radius])
    cmmaxs = np.array([args.cmmax])

    print("=" * 60)
    print(f"Far-Field Scattering: Single Cylinder")
    print(f"  radius={args.radius}, wavelength={args.wavelength}")
    print(f"  epsilon={args.epsilon}, cmmax={args.cmmax}")
    print(f"  ka = {2 * np.pi / args.wavelength * args.radius:.3f}")
    print("=" * 60)

    phi, sigma = far_field_single_cylinder(cmmaxs, cepmus, crads,
                                           args.wavelength, args.nphi)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Cartesian plot
    axes[0].plot(phi * 180 / np.pi, sigma, 'b-', linewidth=1.5)
    axes[0].set_xlabel('Angle (degrees)')
    axes[0].set_ylabel('σ(φ) = 2π|f(φ)|²')
    axes[0].set_title('Far-Field Scattering Cross Section')
    axes[0].grid(True, alpha=0.3)

    # Polar plot
    ax_polar = fig.add_subplot(122, projection='polar')
    ax_polar.plot(phi, sigma / np.max(sigma), 'b-', linewidth=1.5)
    ax_polar.set_title('Normalized Far-Field Pattern', pad=20)
    axes[1].set_visible(False)

    plt.suptitle(f'Single Cylinder: a={args.radius}, λ={args.wavelength}, '
                 f'ε={args.epsilon}, ka={2 * np.pi / args.wavelength * args.radius:.2f}',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig('result_replica.png', dpi=150)
    print("Plot saved to result_replica.png")

    # Print key values
    print(f"\nScattering cross section:")
    print(f"  Forward (0°):    {sigma[0]:.6f}")
    print(f"  Backward (180°): {sigma[len(sigma) // 2]:.6f}")
    print(f"  Max:             {np.max(sigma):.6f} at {phi[np.argmax(sigma)] * 180 / np.pi:.1f}°")
    print(f"  Total (integral): {np.sum(sigma) * 2 * np.pi / len(sigma):.6f}")


if __name__ == '__main__':
    main()
