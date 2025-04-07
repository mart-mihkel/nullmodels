import numpy as np
import pandas as pd

from scipy.stats import hypergeom
from plotnine import aes, geom_step


def geom_pr_simulate(
    n_sim: int, n_samp: int, pos_rate=0.5, h0_correct=0.0, q=None, method="tail"
) -> geom_step:
    """
    Simulate and plot precision recall curves

    Parameters
    ----------
    n_sim : integer
        Number of simulations.
    n_samp : integer
        Number of samples in each simulation.
    pos_rate : float, default 0.5
        Proporion of positive classes in population. Must be between 0 and 1.
    h0_correct : float, default 0.0
        Proporion of samples nullmodel can correctly handle.
    q : float, default None
        Quantile of curves to plot.
    method : {'tail', 'body'}, default 'tail'
        Score randomization method

    Returns
    -------
    geom : geom_step
        ggplot geometry with precision recall curves
    """
    y_true, y_scores = __simulate(n_sim, n_samp, pos_rate, h0_correct, method)
    simulated_curves = [pr_curve(y_true, y_s)[:2] for y_s in y_scores]

    if q is not None:
        p, r = pr_quantile_interp(simulated_curves, q)
        df = pd.DataFrame({"precision": p, "recall": r})
        return geom_step(mapping=aes("recall", "precision"), data=df)
    else:
        p, r = np.hstack(simulated_curves)
        g = np.repeat(np.arange(n_sim), n_samp)
        df = pd.DataFrame({"precision": p, "recall": r, "group": g})
        return geom_step(
            mapping=aes("recall", "precision", group="group"),
            data=df,
            alpha=max(0.05, 1 / n_sim),
        )


def geom_pr_hypergeom(
    n_samp: int, pos_rate=0.5, h0_correct=0.0, q=0.9, method="tail"
) -> geom_step:
    """
    Compute and plot precision recall curve

    Parameters
    ----------
    n_samp : integer
        Number of samples in each simulation.
    pos_rate : float, default 0.5
        Proporion of positive classes in population. Must be between 0 and 1.
    h0_correct : float, default 0.5
        Proporion of samples nullmodel can correctly handle.
    q : float, default 0.9
        Quantile of curves to plot.
    method : {'tail', 'body'}, default 'tail'
        Score randomization method

    Returns
    -------
    geom : geom_step
        ggplot geometry with precision recall curve
    """
    p, r, _ = pr_quantile_hypergeom(
        n_samp, h0_correct=h0_correct, q=q, pos_rate=pos_rate, method=method
    )

    df = pd.DataFrame({"precision": p, "recall": r})
    return geom_step(aes("recall", "precision"), data=df)


def pr_curve(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute precision recall points.

    Parameters
    ----------
    y_true : ndarray
        Array with true labels. Values must be 1 or 0.
    y_score : ndarray
        Probability score estimates of positive class.

    Returns
    -------
    precision : ndarray
        Precision values.
    recall : ndarray
        Recall values.
    threshold : ndarray
        Thresholds for precision-recall points.
    """
    thresold_pred = np.less_equal.outer(y_score, y_score)

    n_pos = y_true.sum()
    p_pos = thresold_pred.sum(axis=1)
    t_pos = np.array([y_true[pred].sum() for pred in thresold_pred])

    return t_pos / p_pos, t_pos / n_pos, y_score


def pr_quantile_interp(
    curves: np.ndarray | list, q=0.9
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate q-th quantile of precision recall curves.

    Parameters
    ----------
    curves : ndarray
        Array of (precision, recall) tuples.
    q : float, default 0.9
        Quantile.

    Returns
    -------
    precision : ndarray
        q-th quantile of curve precisions
    recall : ndarray
        Interpolation knots.
    """

    def __decreasing(p, r):
        idx = np.flip(np.diff(r[::-1], prepend=2) > 0)
        return p[idx], r[idx]

    knots = np.linspace(0, 1)
    dec = [__decreasing(*c) for c in curves]
    interps = [np.interp(knots, xp=r[::-1], fp=p)[::-1] for p, r in dec]

    return np.quantile(interps, q=q, axis=0), knots


def pr_quantile_hypergeom(
    n_samp: int,
    q=0.9,
    pos_rate=0.5,
    h0_correct=0.0,
    method="tail",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute q-th quantile of precision recall curve with hypergeometric
    distribution

    Parameters
    ----------
    n_samp: integer
        Number of samples
    q : float, default 0.9
        Quantile.
    pos_rate : float, default 0.5
        Proporion of positive classes in population. Must be between 0 and 1.
    method : {'tail', 'body'}, default 'tail'
        Score randomization method

    Returns
    -------
    precision : ndarray of shape (n_samp,)
        Precision values.
    recall : ndarray of shape (n_samp,)
        Recall values.
    threshold : ndarray of shape (n_samp,)
        Thresholds for precision-recall points.
    """

    if method == "tail":
        return __pr_hypergeom_tail(
            n_samp=n_samp, q=q, pos_rate=pos_rate, h0_correct=h0_correct
        )
    elif method == "body":
        return __pr_hypergeom_body(
            n_samp=n_samp, q=q, pos_rate=pos_rate, h0_correct=h0_correct
        )
    else:
        raise ValueError(f"Invailid method, got '{method}', should be 'tail' or 'body'")


def __pr_hypergeom_tail(
    n_samp: int,
    q=0.9,
    pos_rate=0.5,
    h0_correct=0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute q-th quantile of precision recall curve with hypergeometric
    distribution with tail method

    Parameters
    ----------
    n_samp: integer
        Number of samples
    q : float, default 0.9
        Quantile.
    pos_rate : float, default 0.5
        Proporion of positive classes in population. Must be between 0 and 1.

    Returns
    -------
    precision : ndarray of shape (n_samp,)
        Precision values.
    recall : ndarray of shape (n_samp,)
        Recall values.
    threshold : ndarray of shape (n_samp,)
        Thresholds for precision-recall points.
    """
    th = np.arange(n_samp) / n_samp
    pos = int(n_samp * pos_rate)
    pred_pos = (n_samp * (1 - th)).astype(np.int32)

    lb = h0_correct / 2 * n_samp
    ub = (1 - h0_correct / 2) * n_samp
    n_hyp = (1 - h0_correct) * n_samp
    pos_hyp = (1 - h0_correct) * pos

    pred_pos_hyp = pred_pos[(lb < pred_pos) & (pred_pos < ub)] - lb
    true_pos_hyp = hypergeom.ppf(M=n_hyp, n=pred_pos_hyp, N=pos_hyp, q=q)

    # TODO: upper and lower bound true positives might be incorrect with
    #       certain edge case values of ``pos_rate``
    true_pos_u = np.full_like(pred_pos[pred_pos >= ub], pos)
    true_pos_l = pred_pos[lb >= pred_pos]
    true_pos = np.concatenate((true_pos_u, true_pos_hyp + lb, true_pos_l))

    return true_pos / pred_pos, true_pos / pos, th


def __pr_hypergeom_body(
    n_samp: int,
    q=0.9,
    pos_rate=0.5,
    h0_correct=0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute q-th quantile of precision recall curve with hypergeometric
    distribution with body method

    Parameters
    ----------
    n_samp: integer
        Number of samples
    q : float, default 0.9
        Quantile.
    pos_rate : float, default 0.5
        Proporion of positive classes in population. Must be between 0 and 1.

    Returns
    -------
    precision : ndarray of shape (n_samp,)
        Precision values.
    recall : ndarray of shape (n_samp,)
        Recall values.
    threshold : ndarray of shape (n_samp,)
        Thresholds for precision-recall points.

    Raises
    ------
    NotImplementedError
        Method 'body' is not implemented
    """
    raise NotImplementedError("Method 'body' is not implemented!")

    th = np.arange(n_samp) / n_samp
    pos = int(n_samp * pos_rate)
    pred_pos = (n_samp * (1 - th)).astype(np.int32)

    n_hyp = (1 - h0_correct) * n_samp
    pos_hyp = (1 - h0_correct) * pos
    idx_hyp = np.random.choice(n_samp, int(n_hyp), replace=False)

    pred_pos_hyp = pred_pos[idx_hyp] - ...
    true_pos_hyp = hypergeom.ppf(M=n_hyp, n=pred_pos_hyp, N=pos_hyp, q=q)

    true_pos = np.minimum(pred_pos, pos)
    true_pos[idx_hyp] = true_pos_hyp

    return true_pos / pred_pos, true_pos / pos, th


def __simulate(
    n_sim: int,
    n_samp: int,
    pos_rate=0.5,
    h0_correct=0.0,
    method="tail",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate nullmodels.

    Parameters
    ----------
    n_sim : integer
        Number of simulations.
    n_samp : integer
        Number of samples in each simulation.
    pos_rate : float, default 0.5
        Proporion of positive classes in population. Must be between 0 and 1.
    h0_correct : float, default 0.0
        Proporion of samples nullmodel can correctly handle.
    method : {'tail', 'body'}, default 'tail'
        Score randomization method

    Returns
    -------
    y_true : ndarray of shape (n_samp,)
        True labels
    y_scores : ndarray of shape (n_sim, n_samp).
        Simulated scores

    Raises
    ------
    ValueError
        When incorrect ``method`` is provided
    """

    def __randomize_score_tail(init_score):
        h0_correct_2 = h0_correct / 2
        lb = int(h0_correct_2 * n_samp)
        ub = int((1 - h0_correct_2) * n_samp)
        init_score[lb : ub + 1] = np.random.permutation(init_score[lb : ub + 1])
        return init_score

    def __randomize_score_body(init_score):
        subs_size = int(n_samp * (1 - h0_correct))
        subs_idx = np.random.choice(n_samp, subs_size, replace=False)
        init_score[subs_idx] = np.random.permutation(init_score[subs_idx])
        return init_score

    if method == "tail":
        method_f = __randomize_score_tail
    elif method == "body":
        method_f = __randomize_score_body
    else:
        raise ValueError(f"Invailid method: got '{method}', should be 'tail' or 'body'")

    n_pos = int(pos_rate * n_samp)
    n_neg = n_samp - n_pos

    init_scores = np.tile(np.linspace(0, 1, num=n_samp), n_sim).reshape((n_sim, n_samp))
    y_scores = np.apply_along_axis(method_f, axis=1, arr=init_scores)
    y_true = np.repeat((0, 1), (n_neg, n_pos))

    return y_true, y_scores


__all__ = [
    "pr_quantile_hypergeom",
    "pr_quantile_interp",
    "geom_pr_hypergeom",
    "geom_pr_simulate",
    "pr_curve",
]
