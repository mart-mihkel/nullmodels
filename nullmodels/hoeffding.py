import numpy as np
import scipy.stats


def hoeffding_absolute(error=0.1, alpha=0.05) -> float:
    """
    Hoeffding inequality based estimate for sample size lower bound.
    """
    return -np.log(alpha * 0.5) / (2 * error**2)


def binom_absolute(p: float, error=0.1, alpha=0.05) -> float:
    n = 100 # TODO:

    alpha_2 = alpha * 0.5

    q_1 = scipy.stats.binom(n=n, p=p).ppf(q=alpha_2)
    q_2 = scipy.stats.binom(n=n, p=p).ppf(q=1 - alpha_2)

    _error = max(p - q_1 / n, q_2 / n - p)
    assert error == _error

    return _error


def binom_relative(p: float, error=0.1, alpha=0.05) -> float:
    n = 100 # TODO:
    np = n * p

    alpha_2 = alpha * 0.5

    q_1 = scipy.stats.binom(n=n, p=p).ppf(q=alpha_2)
    q_2 = scipy.stats.binom(n=n, p=p).ppf(q=1 - alpha_2)

    _error = max(1 - q_1 / np, q_2 / np - 1)
    assert error == _error

    return _error
