import numpy as np

from scipy.stats import binom
from scipy.optimize import brentq


def hoeffding_absolute(error=0.05, alpha=0.05) -> float:
    """
    Hoeffding inequality based estimate for sample size lower bound
    with given maximal absolute error and significance.
    """
    return -np.log(alpha / 2) / (2 * error**2)


def binom_absolute(p: float, error=0.05, alpha=0.05) -> float:
    n_1 = brentq(
        f=lambda n: p - error - binom.ppf(n=np.int32(n), p=p, q=alpha / 2) / n,
        a=1,
        b=hoeffding_absolute(error, alpha),
    )

    n_2 = brentq(
        f=lambda n: binom.ppf(n=np.int32(n), p=p, q=1 - alpha / 2) / n - p - error,
        a=1,
        b=hoeffding_absolute(error, alpha),
    )

    return np.maximum(n_1, n_2)


def binom_relative(p: float, error=0.05, alpha=0.05) -> float:
    n_1 = brentq(
        f=lambda n: 1 - error - binom.ppf(n=np.int32(n), p=p, q=alpha / 2) / (n * p),
        a=1,
        b=hoeffding_absolute(error / 2, alpha),
    )

    n_2 = brentq(
        f=lambda n: binom.ppf(n=np.int32(n), p=p, q=1 - alpha / 2) / (n*p) - error - 1,
        a=1,
        b=hoeffding_absolute(error / 2, alpha),
    )

    return np.maximum(n_1, n_2)
