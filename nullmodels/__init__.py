import scipy
import plotnine
import numpy as np
import pandas as pd

from typing import Literal


def geom_pr_simulate(
    n_sim: int,
    n_samp: int,
    pos_rate: float = 0.5,
    h0_correct: float = 0.0,
    q: float | None = None,
    method: Literal["tail", "body"] = "tail",
    plot_smoothing: Literal["precision-envelope", "moving-average"] | None = None,
    **kwargs,
) -> plotnine.geom_step:
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
    q : float | None, default None
        Quantile of curves to plot.
    method : Literal['tail', 'body'], default 'tail'
        Score randomization method
    plot_smoothing : Literal["precision-envelope", "moving-average"] | None, default None
        Optionally apply a smoothing to plotted curves.
    **kwargs :
        Keyword arguments passed to ggplot geometry

    Returns
    -------
    geom : geom_step
        ggplot geometry with precision recall curves
    """
    y_true, y_scores = __simulate(n_sim, n_samp, pos_rate, h0_correct, method)
    simulated_curves = [pr_curve(y_true, y_s)[:2] for y_s in y_scores]

    smoothing_f = None
    if plot_smoothing == "moving-average":
        smoothing_f = moving_average
    elif plot_smoothing == "precision-envelope":
        smoothing_f = precision_envelope

    if q is not None:
        p, r = pr_quantile(np.array(simulated_curves), q)
        if smoothing_f:
            p, r = smoothing_f(p, r)

        return plotnine.geom_step(
            mapping=plotnine.aes("recall", "precision"),
            data=pd.DataFrame({"precision": p, "recall": r}),
            **kwargs,
        )
    else:
        if smoothing_f:
            simulated_curves = [smoothing_f(*c) for c in simulated_curves]

        p, r = np.hstack(simulated_curves)
        grp = np.repeat(np.arange(n_sim), n_samp)
        return plotnine.geom_step(
            mapping=plotnine.aes("recall", "precision", group="group"),
            data=pd.DataFrame({"precision": p, "recall": r, "group": grp}),
            **kwargs,
        )


def geom_pr_hypergeom(
    n_samp: int,
    pos_rate: float = 0.5,
    h0_correct: float = 0.0,
    q: float = 0.9,
    method: Literal["tail", "body"] = "tail",
    plot_smoothing: Literal["precision-envelope", "moving-average"] | None = None,
    **kwargs,
) -> plotnine.geom_step:
    """
    Compute precision recall curve and return a corresponding ggplot geometry

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
    method : Literal['tail', 'body'], default 'tail'
        Score randomization method
    plot_smoothing : Literal["precision-envelope", "moving-average"] | None, default None
        Optionally apply a smoothing to plotted curves.
    **kwargs :
        Keyword arguments passed to ggplot geometry

    Returns
    -------
    geom : geom_step
        ggplot geometry with precision recall curve
    """
    p, r, _ = pr_quantile_hypergeom(
        n_samp, h0_correct=h0_correct, q=q, pos_rate=pos_rate, method=method
    )

    if plot_smoothing == "moving-average":
        p, r = moving_average(p, r)
    elif plot_smoothing == "precision-envelope":
        p, r = precision_envelope(p, r)

    df = pd.DataFrame({"precision": p, "recall": r})
    return plotnine.geom_step(
        mapping=plotnine.aes("recall", "precision"), data=df, **kwargs
    )


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
    dec_idx = np.argsort(y_score)[::-1]
    y_score = y_score[dec_idx]
    y_true = y_true[dec_idx]

    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)

    precision = tps / (tps + fps)
    recall = tps / y_true.sum()

    return precision, recall, y_score


def precision_envelope(p: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Conpute precision envelope of precision-recall curve

    Parameters
    ----------
    precision : ndarray
        Precision values.
    recall : ndarray
        Recall values.

    Returns
    -------
    precision : ndarray
        Enveloping precision values.
    recall : ndarray
        Recall values.
    """
    inc_idx = np.argsort(r)
    p, r = p[inc_idx], r[inc_idx]
    p_envelope = np.maximum.accumulate(p[::-1])[::-1]
    return p_envelope, r


def moving_average(
    p: np.ndarray, r: np.ndarray, window: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute moving average on precision values

    Parameters
    ----------
    precision : ndarray
        Precision values.
    recall : ndarray
        Recall values.
    window : int, default 4
        Size of the sliding window

    Returns
    -------
    precision : ndarray
        Averaged precision values.
    recall : ndarray
        Recall values.
    """
    p_ma = np.convolve(p, np.ones(window) / window, mode="same")
    return p_ma, r


def pr_quantile(curves: np.ndarray, q: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute quantile of simulated precision-recall curves

    Parameters
    ----------
    curves : ndarray
        Array of (precisions, recalls) tuples.
    q : float, default 0.9
        Quantile.

    Returns
    -------
    precision : ndarray
        q-th quantile of curve precisions
    recall : ndarray
        Interpolation knots.
    """
    r_grid = np.linspace(0, 1)
    p_grids = [None] * curves.shape[0]
    for i, (p, r) in enumerate(curves):
        # only pick points where recall is strictly increasing
        r_uniq, uniq_idx = np.unique(r, return_index=True)
        p_uniq = p[uniq_idx]

        p_grids[i] = np.interp(r_grid, xp=r_uniq, fp=p_uniq)

    p_quantile = np.quantile(p_grids, q=q, axis=0)

    return p_quantile, r_grid


def pr_quantile_hypergeom(
    n_samp: int,
    q: float = 0.9,
    pos_rate: float = 0.5,
    h0_correct: float = 0.0,
    method: Literal["tail", "body"] = "tail",
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
    method : Literal['tail', 'body'], default 'tail'
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
    if method not in ["tail", "body"]:
        raise ValueError(f"Invailid method, got '{method}', should be 'tail' or 'body'")

    if method == "tail":
        return __pr_hypergeom_tail(
            n_samp=n_samp, q=q, pos_rate=pos_rate, h0_correct=h0_correct
        )
    elif method == "body":
        return __pr_hypergeom_body(
            n_samp=n_samp, q=q, pos_rate=pos_rate, h0_correct=h0_correct
        )


def __pr_hypergeom_tail(
    n_samp: int,
    q: float = 0.9,
    pos_rate: float = 0.5,
    h0_correct: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute q-th quantile of precision recall curve with hypergeometric
    distribution using tail method.
    """
    th = np.arange(n_samp) / n_samp
    pos = int(n_samp * pos_rate)
    pred_pos = (n_samp * (1 - th)).astype(np.int32)

    lb = h0_correct / 2 * n_samp
    ub = (1 - h0_correct / 2) * n_samp
    n_hyp = (1 - h0_correct) * n_samp
    pos_hyp = (1 - h0_correct) * pos

    pred_pos_hyp = pred_pos[(lb < pred_pos) & (pred_pos < ub)] - lb
    true_pos_hyp = scipy.stats.hypergeom.ppf(M=n_hyp, n=pred_pos_hyp, N=pos_hyp, q=q)

    # TODO: upper and lower bound true positives might be incorrect with
    #       certain edge case values of ``pos_rate``
    true_pos_u = np.full_like(pred_pos[pred_pos >= ub], pos)
    true_pos_l = pred_pos[lb >= pred_pos]
    true_pos = np.concatenate((true_pos_u, true_pos_hyp + lb, true_pos_l))

    return true_pos / pred_pos, true_pos / pos, th


def __pr_hypergeom_body(
    n_samp: int,
    q: float = 0.9,
    pos_rate: float = 0.5,
    h0_correct: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute q-th quantile of precision recall curve with hypergeometric
    distribution using body method.

    Raises
    ------
    NotImplementedError
        Method 'body' is not implemented
    """
    raise NotImplementedError("Method 'body' is not implemented!")


def __simulate(
    n_sim: int,
    n_samp: int,
    pos_rate: float = 0.5,
    h0_correct: float = 0.0,
    method: Literal["tail", "body"] = "tail",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate nullmodels.
    """

    if method not in ["tail", "body"]:
        raise ValueError(f"Invailid method, got '{method}', should be 'tail' or 'body'")

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

    n_pos = int(pos_rate * n_samp)
    n_neg = n_samp - n_pos

    init_scores = np.tile(np.linspace(0, 1, num=n_samp), n_sim).reshape((n_sim, n_samp))
    y_scores = np.apply_along_axis(method_f, axis=1, arr=init_scores)
    y_true = np.repeat((0, 1), (n_neg, n_pos))

    return y_true, y_scores


__all__ = [
    "pr_quantile_hypergeom",
    "pr_quantile",
    "pr_curve",
    "precision_envelope",
    "moving_average",
    "geom_pr_hypergeom",
    "geom_pr_simulate",
]
