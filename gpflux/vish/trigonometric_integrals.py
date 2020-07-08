import numpy as np
from scipy.special import comb, factorial2


__all__ = [
    "integrate_0_pi__J1_cos_k_sin_d",
]


def _fac2(n):
    return factorial2(n, exact=True)


def _is_odd(num):
    return (num % 2) != 0


def _is_even(num):
    return (num % 2) == 0


def integrate_0_upper__sin_n_cos_m(n: int, m: int, upper: str) -> float:
    """
    ∫ sin^n(x) cos^m(x) dx for x=0 to x=upper,
    for n >= 0 and m>=0.

    :param n: int, larger than 0
        power of the sine
    :param m: int, larger than 0
        power of the cosine
    :param upper: str, [`pi`, `half_pi`]
        upper limit of the integral
    """
    np.testing.assert_(n >= 0 and m >= 0)

    def remaining_integral_half_pi():
        if _is_odd(n) or _is_odd(m):
            return 1
        else:
            return np.pi / 2

    def remaining_integral_pi():
        if _is_odd(m):
            return 0
        elif _is_even(m) and _is_odd(n):
            return 2
        elif _is_even(m) and _is_even(n):
            return np.pi
        else:
            raise ValueError("Should never happen!")

    # (n-1)!! * (m-1)!! / (m+n)!!
    const = (_fac2(n - 1) * _fac2(m - 1)) / _fac2(n + m)

    if upper == "half_pi":
        return const * remaining_integral_half_pi()
    elif upper == "pi":
        return const * remaining_integral_pi()
    else:
        raise NotImplementedError("Upper value integral must be `half_pi` or `pi`")


def factorial2_consecutive_ints(p):
    """ (p-1)!! / (p!!) """
    if p <= 301:
        return factorial2(p - 1) / factorial2(p)
    else:
        return 1 / np.sqrt(np.pi * p / 2)


def int_0_pi_cos_p(p):
    """ ∫ cos^p x dx x=0..pi """
    if p % 2 == 0:
        return factorial2_consecutive_ints(p) * np.pi
    elif p % 2 == 1:
        return 0
    else:
        raise NotImplementedError


def int_0_pi_sin_p(p):
    """ ∫ sin^p x dx x=0..pi """
    if p == 1:
        return 2
    elif p % 2 == 0:
        return factorial2_consecutive_ints(p) * np.pi
    elif p % 2 == 1:
        return factorial2_consecutive_ints(p) * 2
    else:
        raise NotImplementedError


def integrate_0_pi__pi_minus_t_sin_n_cos_m(n: int, m: int) -> float:
    """
    Computes ∫ (π - t) * sin(t)^n * cos(t)^m dt, for t=0..pi

    We use integration by parts:
        u(t) = (π - t)  -> du = -dt
        dv = sin(t)^n * cos(t)^m dt  -> v(t) = ∫ sin(t)^n * cos(t)^m dt + C
            we pick the constant so that v(0) = 0

    <=> result = u(π) v(π) - u(0) v(0) + ∫ v(t) dt, t=0..π
    First two terms are zero as u(π) = 0 and v(0) = 0
    <=> result = ∫ v(t) dt, t=0..π
    <=> result = ∫ { ∫ sin(t)^n * cos(t)^m dt + C(n, m) } dt, t=0..π
    The inner indefinite integral can be solved quite easily when
    n is odd or m is odd, it is harder when both m and n are even.

    :param n: int, larger than 0
        power of the sine
    :param m: int, larger than 0
        power of the cosine

    """
    if _is_odd(n):
        k = (n - 1) // 2
        a, c = 0, 0
        for ll in range(k + 1):
            kCl = comb(k, ll)  # choose l out of k
            a += kCl * (-1) ** ll / (2 * ll + m + 1) * int_0_pi_cos_p(2 * ll + m + 1)
            c += kCl * (-1) ** ll / (2 * ll + m + 1)
        return c * np.pi - a

    elif _is_odd(m):
        k = (m - 1) // 2
        a = 0
        for ll in range(k + 1):
            kCl = comb(k, ll)  # choose l out of k
            a += kCl * (-1) ** ll / (2 * ll + n + 1) * int_0_pi_sin_p(2 * ll + n + 1)
        return a

    elif _is_even(n) and _is_even(m):

        def int_0_pi_integral_cos_p_2t(p):
            """
            computes ∫ v(t) dt, t=0..π
            where v(t) = ∫ cos(2t)^p dt s.t. v(0) = 0,

            => result = ∫ {∫ cos(2 τ)^p dτ, τ=0..t} dt, t=0..π
            """
            if _is_even(p):
                return factorial2_consecutive_ints(p) * np.pi ** 2 / 2
            else:
                return 0

        m, n, a = m // 2, n // 2, 0
        for ll in range(n + 1):
            for k in range(m + 1):
                consts = comb(n, ll) * comb(m, k) * (-1) ** ll
                a += consts * int_0_pi_integral_cos_p_2t(ll + k)

        return a / (2 ** (n + m))
    else:
        raise NotImplementedError


def integrate_0_pi__J1_cos_k_sin_d(k: int, d: int) -> float:
    """
    Computes ∫ J1(t) * cos^k(t) * sin^d(t) dt, t=0..π,
        with J1(t) = sin(t) + (π-t) * cos(t) for t in [0, π].

    :param k: int, >= 0.
        Power of the cosine
    :param d: int >=0,
        Power of the sine
    """

    # ∫ sin(t) * cos^k(t) * sin^d(t), dt t=0..π
    term1 = integrate_0_upper__sin_n_cos_m(d + 1, k, "pi")
    # ∫ (π - t) cos(t) * cos^k(t) * sin^d(t), dt t=0..π
    term2 = integrate_0_pi__pi_minus_t_sin_n_cos_m(d, k + 1)
    return term1 + term2
