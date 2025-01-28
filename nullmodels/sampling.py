import warnings
import numpy as np

from scipy.stats import binom
from scipy.optimize import brentq, elementwise


def hoef_abs_err(n: float, alpha=0.05) -> float:
    """
    Hoeffding inequality based estimate for absolute error lower bound
    with given sample size and significance.
    """
    return np.sqrt(-np.log(alpha / 2) / (2 * n))


def binom_abs_err(n: float, p: float, alpha=0.05) -> float:
    """
    Binomial distribution based estimate for absolute error
    with given sample size, significance and assumed true p.
    """
    q1 = binom.ppf(n=np.int64(n), p=p, q=alpha / 2)
    q2 = binom.ppf(n=np.int64(n), p=p, q=1 - alpha / 2)
    return np.maximum(p - q1 / n, q2 / n - p)  # type: ignore


def binom_relative_err(n: float, p: float, alpha=0.05) -> float:
    """
    Binomial distribution based estimate for relative error
    with given sample size, significance and assumed true p.
    """
    q1 = binom.ppf(n=np.int64(n), p=p, q=alpha / 2)
    q2 = binom.ppf(n=np.int64(n), p=p, q=1 - alpha / 2)
    return np.maximum(1 - q1 / (n * p), q2 / (n * p) - 1)  # type: ignore


def hoef_abs_n(error=0.05, alpha=0.05) -> float:
    """
    Hoeffding inequality based estimate for sample size lower bound
    with given maximal absolute error and significance.
    """
    return -np.log(alpha / 2) / (2 * error**2)


def binom_abs_n(p: float, error=0.05, alpha=0.05) -> float:
    """
    Binomial distribution based estimate for sample size lower bound
    with given maximal absolute error, significance and assumed
    true p.
    """

    def __opt(n):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            binom_left = p - binom.ppf(n=np.int64(n), p=p, q=alpha / 2) / n
            binom_right = binom.ppf(n=np.int64(n), p=p, q=1 - alpha / 2) / n - p

        return error - np.maximum(binom_left, binom_right)

    a, b = elementwise.bracket_root(f=__opt, xl0=1).bracket
    n = brentq(f=__opt, a=a, b=b)

    return n  # type: ignore


def binom_relative_n(p: float, error=0.05, alpha=0.05) -> float:
    """
    Binomial distribution based estimate for sample size lower bound
    with given maximal relative error, significance and assumed
    true p.
    """

    def __opt(n):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            binom_left = 1 - binom.ppf(n=np.int64(n), p=p, q=alpha / 2) / (n * p)
            binom_right = binom.ppf(n=np.int64(n), p=p, q=1 - alpha / 2) / (n * p) - 1

        return error - np.maximum(binom_left, binom_right)

    a, b = elementwise.bracket_root(f=__opt, xl0=1).bracket
    n = brentq(f=__opt, a=a, b=b)

    return n  # type: ignore


__all__ = [
    "hoef_abs_n",
    "binom_abs_n",
    "hoef_abs_err",
    "binom_abs_err",
    "binom_relative_n",
    "binom_relative_err",
]
