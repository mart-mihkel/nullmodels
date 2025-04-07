import numpy as np


def relative_accuracy(y_true: np.ndarray, y0: np.ndarray, y1: np.ndarray) -> float:
    """
    Relative accuracy

    Parameters
    ----------
    y_true: ndarray of shape (n,)
        True labels
    y0: ndarray of shape (n,)
        First model predictions
    y1: ndarray of shape (n,)
        Second model predictions

    Returns
    -------
    dif_acc : float
        Difference between accuracy of first and second model
    """
    idx = y0 != y1

    dif_y = y_true[idx]
    dif_y0 = y0[idx]
    dif_y1 = y1[idx]

    return np.sum(dif_y == dif_y0) - np.sum(dif_y - dif_y1)


def hoef_abs_k_drecall(error=0.05, alpha=0.05) -> float:
    r"""
    Hoeffding inequality based estimate for sample size lower bound to estimate
    labelled component in difference of two binary classifiers' recalls

    .. math::

        \hat{\eta}=\frac{1}{k}\sum_{i=1}^{k}(\[a_i=1\]-\[b_i=1\])\[y_i=1\]

    with given maximal absolute error and significance.

    Given the labelled component, the difference in recalls can be found
    :math:`\hat{r_1}-\hat{r_2}=\hat{\beta}\hat{\eta}/\hat{\nu}`, where

    .. math::

        \hat{\beta} &= \sum_{i=1}^{n} \[a_i \neq b_i\]
        \hat{\nu} &= \sum_{i=1}^{n} \[y_i=1\]

    """
    return -np.log(alpha / 2) * 2 / error**2


def hoef_abs_err_drecall(k: int | float, alpha=0.05) -> float:
    r"""
    Hoeffding inequality based estimate for absolute error lower bound to
    estimate labelled component in difference of two binary classifiers'
    recalls

    .. math::

        \hat{\eta} = \frac{1}{k} \sum_{i=1}^{k} (\[a_i = 1\] - \[b_i = 1\])\[y_i = 1\]

    with given labelled sample size and significance.

    Given the labelled component, the difference in recalls can be found
    :math:`\hat{r_1} - \hat{r_2} = \hat{\beta} \hat{\eta} / \hat{\nu}`, where

    .. math::

        \hat{\beta} &= \sum_{i=1}^{n} \[a_i \neq b_i\]
        \hat{\nu} &= \sum_{i=1}^{n} \[y_i = 1\]

    """
    return np.sqrt(-np.log(alpha / 2) * 2 / k)


__all__ = [
    "relative_accuracy",
    "hoef_abs_k_drecall",
    "hoef_abs_err_drecall",
]
