import numpy as np

from scipy.stats import binom, hypergeom
from plotnine import ggplot, ggtitle, geom_step, aes, labs


def run_simulations(
    n_sim: int,
    n_samp: int,
    ppos=0.5,
    h0_acc=0.0,
    h0_th=0.5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate nullmodels

    Parameters
    ----------
    n_sim : integer
        Number of simulations.
    n_samp : integer
        Number of samples in each simulation.
    ppos : float
        Percentage of positive labels in a sample.
        Must be between 0 and 1.
        Default is 0.5.
    h0_acc : float
        Percentage of samples the nullmodel is rigged to label correctly with
        a treshold of `h0_th`.
        Must be between 0 and 1.
        Defualt is 0.0.
    h0_th : float
        Classification treshold when using `h0_acc` to rig the model. Consider
        a data point positive when `y_score >= h0_th`.
        Must be between 0 and 1.
        Defualt is 0.5.

    Returns
    -------
    (y_true, y_scores) : tuple
        Tuple of true labels with shape (n_samp,) and simulated nullmodel
        scores with shape (n_sim, n_samp).
    """
    n_pos = int(ppos * n_samp)
    n_neg = n_samp - n_pos

    y_true = np.repeat((0, 1), (n_neg, n_pos))
    init_scores = np.tile(np.linspace(0, 1, num=n_samp), n_sim) \
                    .reshape((n_sim, n_samp))

    y_scores = np.array(
        [__randomize_score(s, y_true, h0_acc, h0_th) for s in init_scores]
    )

    return y_true, y_scores


def plot_simulations(
    n_sim: int,
    n_samp: int,
    ppos=0.5,
    q=0.9,
    h0_acc=0.0,
    h0_th=0.5,
    plot_all=False
) -> ggplot:
    """
    Simulate nullmodels and plot q-th quantile of results.

    Parameters
    ----------
    n_sim : integer
        Number of simulations.
    n_samp : integer
        Number of samples in each simulation.
    ppos : float
        Percentage of positive labels in a samples.
        Must be between 0 and 1.
        Default is 0.5.
    q : float
        Quantile of simulations to plot.
        Default is 0.9.
    h0_acc : float
        Percentage of samples the nullmodel is rigged to label correctly with
        a treshold of `h0_th`.
        Must be between 0 and 1.
        Defualt is 0.0.
    h0_th : float
        Classification treshold when using `h0_acc` to rig the model.
        Must be between 0 and 1.
        Defualt is 0.5.
    plot_all : boolean
        Wether to plot all simulations.
        Defualt is False.

    Returns
    -------
    fig : ggplot
        ggplot figure with simulations
    """
    y_true, y_scores = run_simulations(n_sim, n_samp, ppos, h0_acc, h0_th)
    simulated_curves = [pr_curve(y_true, y_s) for y_s in y_scores]
    prec_q, rec_q = pr_quantile_interp(simulated_curves, q)

    g = ggplot() 
    g += ggtitle(f'n_sim={n_sim} n_samp={n_samp} ppos={ppos} acc={h0_acc} q={q}')
    g += labs(x='recall', y='precision')
    g += geom_step(aes(rec_q, prec_q))

    if not plot_all:
        return g

    shapes = [c[0].shape[0] for c in simulated_curves]
    group = np.repeat(np.arange(n_sim), shapes)
    precs, recs, _ = np.hstack(simulated_curves)

    return g + geom_step(aes(recs, precs, group=group), alpha=1/n_sim)


def pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute precision recall curve.

    Parameters
    ----------
    y_true : array
        Array with true labels.
        Values must be 1 or 0.
    y_score : array
        Probability estimates of positive class.

    Returns
    -------
    (precision, recall, thresholds) : tuple
        Precision recall points at given classification thresholds.
    """
    thresholds = np.sort(y_score)
    pred = np.less_equal.outer(thresholds, y_score)

    p_pos = pred.sum(axis=1)
    n_pos = y_true.sum()
    t_pos = np.array([y_true[p].sum() for p in pred])

    prec = t_pos / p_pos
    rec = t_pos / n_pos

    return prec, rec, thresholds


def pr_quantile_interp(
    curves: np.ndarray | list,
    q=0.9,
    n_knots=50
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate q-th quantile of precision recall curves.

    Parameters
    ----------
    curves : array
        Array of (precision, recall) tuples.
    q : float
        Quantile to compute.
        Default is 0.9.
    n_knots : integer
        Number of interpolation knots.
        Defualt is 50.

    Returns
    -------
    (precision, recall) : tuple
        Precision recall curve, with precision = q-th quantile of precisions
        and recall = interpolation knots.
    """
    knots = np.linspace(0, 1, num=n_knots)
    dec = [__decreasing(c[0], c[1]) for c in curves]
    interps = [np.interp(knots, xp=r[::-1], fp=p)[::-1] for p, r in dec]
    return np.quantile(interps, q=q, axis=0), knots



def pr_quantile_binom(
    n_samp: int,
    q=.9,
    ppos=.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Approximate q-th quantile of precision recall curve with binomial
    distribution

    Parameters
    ----------
    n_samp: integer
        Number of samples
    q : float
        Quantile of compute.
        Default is 0.9.
    ppos : float
        Percentage of positive labels in the sample.
        Must be between 0 and 1.
        Default is 0.5.

    Returns
    -------
    (precision, recall, threshold) : tuple
        Precision-recall curve points at given thresholds.
    """
    th = np.arange(n_samp) / n_samp
    pos = int(n_samp * ppos)
    pred_pos = (n_samp * (1 - th)).astype(np.int32)
    true_pos = binom(n=pred_pos, p=ppos).ppf(q)

    return true_pos / pred_pos, true_pos / pos, th


def pr_quantile_hypergeom(
    n_samp: int,
    q=.9,
    ppos=.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute q-th quantile of precision recall curve with hypergeometric 
    distribution
    
    Parameters
    ----------
    n_samp: integer
        Number of samples
    q : float
        Quantile of compute.
        Default is 0.9.
    ppos : float
        Percentage of positive labels in the sample.
        Must be between 0 and 1.
        Default is 0.5.

    Returns
    -------
    (precision, recall, threshold) : tuple
        Precision-recall curve points at given thresholds.
 
    """
    th = np.arange(n_samp) / n_samp
    pos = int(n_samp * ppos)
    pred_pos = (n_samp * (1 - th)).astype(np.int32)
    true_pos = hypergeom(M=n_samp, n=pred_pos, N=pos).ppf(q)

    return true_pos / pred_pos, true_pos / pos, th


def __randomize_score(
    scores: np.ndarray | list,
    y: np.ndarray, 
    h0_acc=0.0, 
    h0_th=0.5
) -> np.ndarray:
    """
    Randomly permute scores. Rig the scores for the model to have accuracy of
    at least `h0_acc` with treshold of `h0_th` if provided.
    """
    scores = np.random.permutation(scores)

    if h0_acc == 0:
        return scores

    n_rig = int(y.shape[0] * h0_acc)
    n_neg = n_rig // 2
    n_pos = n_rig - n_neg

    y_neg_i = np.random.choice(np.flatnonzero(y == 0), n_neg)
    y_pos_i = np.random.choice(np.flatnonzero(y == 1), n_pos)

    score_neg_i = np.flatnonzero(scores < h0_th)
    score_pos_i = np.flatnonzero(scores >= h0_th)

    neg = scores[score_neg_i]
    pos = scores[score_pos_i]

    if neg.shape[0] < n_neg:
        neg = np.append(neg, np.repeat(1 - h0_th, n_neg - neg.shape[0]))
    else:
        neg = np.random.choice(neg, n_neg)

    if pos.shape[0] < n_pos:
        pos = np.append(pos, np.repeat(h0_th, n_pos - pos.shape[0]))
    else:
        pos = np.random.choice(pos, n_pos)

    scores[y_neg_i], scores[y_pos_i] = neg, pos

    return scores


def __decreasing(p, r):
    """
    Pick points on precision recall curve where recall is decreasin.
    """
    idx = np.flip(np.diff(r[::-1], prepend=1.1) > 0)
    return p[idx], r[idx]


__all__ = [
    "pr_quantile_hypergeom",
    "pr_quantile_interp",
    "pr_quantile_binom",
    "pr_curve",
    "run_simulations",
    "plot_simulations",
]
