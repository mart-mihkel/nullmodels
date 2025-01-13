import numpy as np
import scipy.stats
from plotnine import aes, geom_step, ggplot, labs


def run_simulations(
    n_sim: int,
    n_samp: int,
    pos_rate=0.5,
    h0_acc=0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate nullmodels.

    Parameters
    ----------
    n_sim : integer
        Number of simulations.
    n_samp : integer
        Number of samples in each simulation.
    pos_rate : float
        Proporion of positive classes in population.
        Must be between 0 and 1, default is 0.5.

    Returns
    -------
    (y_true, y_scores) : tuple
        Tuple of true labels with shape (n_samp,) and simulated nullmodel
        scores with shape (n_sim, n_samp).
    """

    def __scores_0(init_score):
        return np.random.permutation(init_score)

    def __scores_1(init_score):
        h0_acc_2 = h0_acc / 2
        lb, ub = int(h0_acc_2 * n_samp), int((1 - h0_acc_2) * n_samp)
        init_score[lb : ub + 1] = np.random.permutation(init_score[lb : ub + 1])
        return init_score

    n_pos = int(pos_rate * n_samp)
    n_neg = n_samp - n_pos

    y_true = np.repeat((0, 1), (n_neg, n_pos))

    init_scores = np.tile(np.linspace(0, 1, num=n_samp), n_sim).reshape((n_sim, n_samp))

    _y_scores = np.array([__scores_0(s) for s in init_scores])
    y_scores = np.array([__scores_1(s) for s in init_scores])

    return y_true, y_scores


def plot_simulations(
    n_sim: int,
    n_samp: int,
    pos_rate=0.5,
    q=None,
    h0_acc=0.0,
) -> ggplot:
    """
    Simulate and plot nullmodels.

    Parameters
    ----------
    n_sim : integer
        Number of simulations.
    n_samp : integer
        Number of samples in each simulation.
    pos_rate : float
        Proporion of positive classes in population.
        Must be between 0 and 1, default is 0.5.
    q : float
        Quantile of curves to plot, default is None.

    Returns
    -------
    fig : ggplot
        ggplot figure with simulations
    """
    y_true, y_scores = run_simulations(n_sim, n_samp, pos_rate, h0_acc=h0_acc)
    simulated_curves = [pr_curve(y_true, y_s) for y_s in y_scores]

    g = ggplot() + labs(x="recall", y="precision")

    if q is not None:
        prec_q, rec_q = pr_quantile_interp(simulated_curves, q)
        g += geom_step(aes(rec_q, prec_q))
    else:
        shapes = [c[0].shape[0] for c in simulated_curves]
        group = np.repeat(np.arange(n_sim), shapes)
        precs, recs, _ = np.hstack(simulated_curves)
        g += geom_step(aes(recs, precs, group=group), alpha=1 / n_sim)

    return g


def pr_curve(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute precision recall points.

    Parameters
    ----------
    y_true : array-like
        Array with true labels.
        Values must be 1 or 0.
    y_score : array-like
        Probability estimates of positive class.

    Returns
    -------
    (precision, recall, thresholds) : tuple
        Precision recall points at given classification thresholds.
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
    curves : array
        Array of (precision, recall) tuples.
    q : float
        Quantile, default is 0.9.

    Returns
    -------
    (precision, recall) : tuple
        Precision recall curve, with precision = q-th quantile of precisions
        and recall = interpolation knots.
    """

    def __decreasing(p, r):
        idx = np.flip(np.diff(r[::-1], prepend=2) > 0)
        return p[idx], r[idx]

    knots = np.linspace(0, 1)
    dec = [__decreasing(*c) for c in curves]
    interps = [np.interp(knots, xp=r[::-1], fp=p)[::-1] for p, r in dec]

    return np.quantile(interps, q=q, axis=0), knots


def pr_quantile_binom(
    n_samp: int, q=0.9, pos_rate=0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Approximate q-th quantile of precision recall curve with binomial
    distribution

    Parameters
    ----------
    n_samp: integer
        Number of samples
    q : float
        Quantile, default is 0.9.
    pos_rate : float
        Proporion of positive classes in population.
        Must be between 0 and 1, default is 0.5.

    Returns
    -------
    (precision, recall, threshold) : tuple
        Precision-recall curve points at given thresholds.
    """
    th = np.arange(n_samp) / n_samp
    pos = int(n_samp * pos_rate)
    pred_pos = (n_samp * (1 - th)).astype(np.int32)
    true_pos = scipy.stats.binom(n=pred_pos, p=pos_rate).ppf(q)

    return true_pos / pred_pos, true_pos / pos, th


def pr_quantile_hypergeom(
    n_samp: int, q=0.9, pos_rate=0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute q-th quantile of precision recall curve with hypergeometric
    distribution

    Parameters
    ----------
    n_samp: integer
        Number of samples
    q : float
        Quantile, default is 0.9.
    pos_rate : float
        Proporion of positive classes in population.
        Must be between 0 and 1, default is 0.5.

    Returns
    -------
    (precision, recall, threshold) : tuple
        Precision-recall curve points at given thresholds.
    """
    th = np.arange(n_samp) / n_samp
    pos = int(n_samp * pos_rate)
    pred_pos = (n_samp * (1 - th)).astype(np.int32)
    true_pos = scipy.stats.hypergeom(M=n_samp, n=pred_pos, N=pos).ppf(q)

    return true_pos / pred_pos, true_pos / pos, th


def pr_quantile_hypergeom_1(
    n_samp: int, h0_acc: float, q=0.9, pos_rate=0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h0_acc_2 = h0_acc / 2

    th = np.arange(n_samp) / n_samp

    pos = int(n_samp * pos_rate)
    pred_pos = (n_samp * (1 - th)).astype(np.int32)

    pred_pos_hyp = (
        pred_pos[(h0_acc_2 * n_samp < pred_pos) & (pred_pos < (1 - h0_acc_2) * n_samp)]
        - h0_acc_2 * n_samp
    )

    true_pos_hyp = scipy.stats.hypergeom(
        M=(1 - h0_acc) * n_samp, n=pred_pos_hyp, N=(1 - h0_acc) * pos
    ).ppf(q)

    true_pos = np.concatenate(
        (
            np.full_like(pred_pos[pred_pos >= (1 - h0_acc_2) * n_samp], pos),
            h0_acc_2 * n_samp + true_pos_hyp,
            pred_pos[h0_acc_2 * n_samp >= pred_pos],
        )
    )

    return true_pos / pred_pos, true_pos / pos, th
