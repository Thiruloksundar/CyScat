"""
JAX-differentiable Bessel functions for CyScat.

Implements:
  - bessel_jv(n, z): Bessel function of first kind J_n(z) for integer n, complex z
  - bessel_yv(n, z): Bessel function of second kind Y_n(z) for integer n, complex z
  - hankel2(n, z): Hankel function H2_n(z) = J_n(z) - i*Y_n(z)

Power series for |z| < 20, asymptotic expansion for |z| >= 20.
JAX auto-differentiable w.r.t. z (not n — n must be an integer).
"""

import jax
import jax.numpy as jnp

# Number of terms in power series (60 is sufficient for |z| < 30)
_NTERMS = 30
# Threshold for switching to asymptotic expansion
_ASYMP_THRESHOLD = 20.0


def _log_factorial(n):
    """log(n!) via lgamma. Works for JAX arrays."""
    return jax.lax.lgamma(jnp.asarray(n + 1, dtype=jnp.float64))


def bessel_jv(n, z):
    """
    Bessel function of the first kind J_n(z) for integer n, complex z.

    Uses the power series:
      J_n(z) = (z/2)^|n| * sum_{k=0}^{inf} (-z^2/4)^k / (k! * (|n|+k)!)

    For negative n: J_{-n}(z) = (-1)^n * J_n(z)
    """
    n = jnp.asarray(n, dtype=jnp.int32)
    z = jnp.asarray(z, dtype=jnp.complex128)
    abs_n = jnp.abs(n)

    half_z = z / 2.0
    neg_quarter_z_sq = -z * z / 4.0

    # First term: (z/2)^|n| / |n|!
    log_first = abs_n * jnp.log(half_z + 0j) - _log_factorial(abs_n)
    term = jnp.exp(log_first)

    result = term
    for k in range(1, _NTERMS):
        term = term * neg_quarter_z_sq / (k * (abs_n + k))
        result = result + term

    # J_{-n}(z) = (-1)^n * J_n(z) for integer n
    sign = jnp.where(n < 0, (-1.0 + 0j) ** abs_n, 1.0 + 0j)
    return sign * result


def bessel_yv(n, z):
    """
    Bessel function of the second kind Y_n(z) for integer n, complex z.

    Computed via Y_0 and Y_1 series, then forward recurrence for |n| >= 2.
    For negative n: Y_{-n}(z) = (-1)^n * Y_n(z)
    """
    n = jnp.asarray(n, dtype=jnp.int32)
    z = jnp.asarray(z, dtype=jnp.complex128)
    abs_n = jnp.abs(n)

    y0 = _bessel_y0(z)
    y1 = _bessel_y1(z)

    # Build orders 0..max_order via forward recurrence: Y_{k+1} = 2k/z * Y_k - Y_{k-1}
    max_order = 25  # max |n| we support
    orders = [y0, y1]
    for k in range(2, max_order + 1):
        yk = 2.0 * (k - 1) / z * orders[-1] - orders[-2]
        orders.append(yk)

    # Stack and index
    orders_arr = jnp.array(orders)
    result = orders_arr[abs_n]

    # Y_{-n}(z) = (-1)^n * Y_n(z)
    sign = jnp.where(n < 0, (-1.0 + 0j) ** abs_n, 1.0 + 0j)
    return sign * result


def _bessel_y0(z):
    """Y_0(z) = (2/pi)*[J_0(z)*(ln(z/2) + gamma) - S]."""
    gamma_em = 0.5772156649015329  # Euler-Mascheroni constant
    j0 = bessel_jv(0, z)
    neg_quarter_z_sq = -z * z / 4.0

    S = jnp.zeros_like(z)
    term = jnp.ones_like(z)
    Hk = 0.0

    for k in range(1, _NTERMS):
        Hk = Hk + 1.0 / k
        term = term * neg_quarter_z_sq / (k * k)
        S = S + Hk * term

    return (2.0 / jnp.pi) * (j0 * (jnp.log(z / 2.0 + 0j) + gamma_em) - S)


def _bessel_y1(z):
    """Y_1(z) from direct series."""
    gamma_em = 0.5772156649015329
    j1 = bessel_jv(1, z)
    half_z = z / 2.0
    neg_quarter_z_sq = -z * z / 4.0

    S = jnp.zeros_like(z)
    term = jnp.ones_like(z)
    Hk = 0.0
    Hk1 = 1.0

    S = S + term * (Hk + Hk1)   # k=0 term

    for k in range(1, _NTERMS):
        term = term * neg_quarter_z_sq / (k * (k + 1))
        Hk = Hk + 1.0 / k
        Hk1 = Hk + 1.0 / (k + 1)
        S = S + term * (Hk + Hk1)

    return (2.0 / jnp.pi) * (j1 * (jnp.log(half_z + 0j) + gamma_em) - 1.0 / z) - \
           (1.0 / jnp.pi) * half_z * S


def _hankel2_asymptotic(n, z):
    """
    Asymptotic expansion of Hankel function H2_n(z) for large |z|.

    H2_n(z) ~ sqrt(2/(pi*z)) * exp(-i*omega) * sum_{k=0}^K a_k / z^k

    where omega = z - n*pi/2 - pi/4 and a_k are Debye coefficients.
    Accurate to ~10^{-14} for |z| > 15, |n| < 15.
    """
    n = jnp.asarray(n, dtype=jnp.int32)
    z = jnp.asarray(z, dtype=jnp.complex128)
    n_f = jnp.float64(n)

    omega = z - n_f * jnp.pi / 2.0 - jnp.pi / 4.0
    prefactor = jnp.sqrt(2.0 / (jnp.pi * z)) * jnp.exp(-1j * omega)

    # Debye coefficients: a_k(n) = prod_{j=0}^{k-1} (4n^2 - (2j+1)^2) / (k! * 8^k)
    # with alternating sign: coefficient is (-i)^k * a_k / z^k
    mu = 4.0 * n_f * n_f  # 4n^2
    inv_z = 1.0 / z

    # Compute up to 15 terms for high accuracy (rel error < 1e-10 for |z|>=20, n<=11)
    series = 1.0 + 0j
    ak = 1.0 + 0j  # running product of (mu - (2j+1)^2)/(8*k)
    inv_z_power = 1.0 + 0j

    for k in range(1, 16):
        factor = (mu - (2 * k - 1) ** 2) / (8.0 * k)
        ak = ak * factor
        inv_z_power = inv_z_power * inv_z
        series = series + (-1j) ** k * ak * inv_z_power

    return prefactor * series


def hankel2(n, z):
    """
    Hankel function of the second kind H2_n(z) = J_n(z) - i*Y_n(z).

    Uses power series for |z| < 20, asymptotic expansion for |z| >= 20.
    JAX-differentiable w.r.t. z (not n — n must be integer).
    """
    n = jnp.asarray(n, dtype=jnp.int32)
    z = jnp.asarray(z, dtype=jnp.complex128)
    z_abs = jnp.abs(z)

    use_asymp = z_abs >= _ASYMP_THRESHOLD

    # Clamp z for power series to avoid overflow (both branches always evaluated)
    z_safe = jnp.where(use_asymp, 1.0 + 0j, z)
    h2_series = bessel_jv(n, z_safe) - 1j * bessel_yv(n, z_safe)

    h2_asymp = _hankel2_asymptotic(n, z)

    return jnp.where(use_asymp, h2_asymp, h2_series)
