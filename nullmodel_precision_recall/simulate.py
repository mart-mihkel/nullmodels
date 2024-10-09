import numpy as np

from plotnine import ggplot, aes, ggtitle, labs, geom_step


def __randomize_score(
        score: np.ndarray | list,
        y=None, 
        acc=0.0, 
        threshold=0.5
) -> np.ndarray:
    """
    Randomly permute scores. Rig the scores for the model to have accuracy of
    at least `acc` with treshold of `threshold` if provided.
    """
    score = np.random.permutation(score)

    if y is None or acc == 0:
        return score

    n_rig = int(y.shape[0] * acc)
    n_neg = n_rig // 2
    n_pos = n_rig - n_neg

    y_neg_i = np.random.choice(np.flatnonzero(y == 0), n_neg)
    y_pos_i = np.random.choice(np.flatnonzero(y == 1), n_pos)

    score_neg_i = np.flatnonzero(score < threshold)
    score_pos_i = np.flatnonzero(score >= threshold)

    neg = score[score_neg_i]
    pos = score[score_pos_i]

    if neg.shape[0] < n_neg:
        neg = np.append(neg, np.repeat(1 - threshold, n_neg - neg.shape[0]))
    else:
        neg = np.random.choice(neg, n_neg)

    if pos.shape[0] < n_pos:
        pos = np.append(pos, np.repeat(threshold, n_pos - pos.shape[0]))
    else:
        pos = np.random.choice(pos, n_pos)

    score[y_neg_i], score[y_pos_i] = neg, pos

    return score


def __decreasing(p, r):
    """
    Pick points on precision recall curve where recall is decreasin.
    """
    idx = np.flip(np.diff(r[::-1], prepend=1.1) > 0)
    return p[idx], r[idx]


def simulate_nullmodels(
    n_sim: int,
    n_samp: int,
    ppos=0.5,
    acc=0.0,
    threshold=0.5
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
    acc : float
        Percentage of samples the nullmodel is rigged to label correctly with
        a treshold of `threshold`.
        Must be between 0 and 1.
        Defualt is 0.0.
    threshold : float
        Classification treshold when using `acc` to rig the model. Consider
        a data point positive when `y_score >= threshold`.
        Must be between 0 and 1.
        Defualt is 0.5.

    Returns
    -------
    (y_true, y_scores) : tuple
        Tuple of true labels with shape (n_samp,) and simulated nullmodel
        scores with shape (n_sim, n_samp).

    Examples
    --------
    >>> n_simulations = 10
    >>> n_samples = 100
    >>> y_true, y_scores = simulate_nullmodels(n_simulations, n_samples)
    """
    n_pos = int(ppos * n_samp)
    n_neg = n_samp - n_pos

    y_true = np.repeat((0, 1), (n_neg, n_pos))
    init_scores = np.tile(np.linspace(0, 1, num=n_samp), n_sim) \
                    .reshape((n_sim, n_samp))

    y_scores = np.array(
        [__randomize_score(s, y_true, acc, threshold) for s in init_scores]
    )

    return y_true, y_scores


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

    Examples
    --------
    >>> y_true, y_scores = simulate_nullmodels(1, 100)
    >>> y_score = y_scores[0]
    >>> prec, rec, thresholds = pr_curve(y_true, y_score)
    """
    thresholds = np.sort(y_score)
    pred = np.less_equal.outer(thresholds, y_score)

    p_pos = pred.sum(axis=1)
    n_pos = y_true.sum()
    t_pos = np.array([y_true[p].sum() for p in pred])

    prec = t_pos / p_pos
    rec = t_pos / n_pos

    return prec, rec, thresholds


def pr_curve_quantile(
    curves: np.ndarray | list,
    q=0.9,
    n_knots=50
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute q-th quantile of precision recall curves.

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

    Examples
    --------
    >>> quantile = 0.9
    >>> y_true, y_scores = simulate_nullmodels(10, 100)
    >>> pr_curves = [pr_curve(y_true, s) for s in y_scores]
    >>> prec_q, rec_q = pr_curve_quantile(curves, q=quantile)
    """
    knots = np.linspace(0, 1, num=n_knots)
    dec = [__decreasing(c[0], c[1]) for c in curves]
    interps = [np.interp(knots, xp=r[::-1], fp=p)[::-1] for p, r in dec]
    return np.quantile(interps, q=q, axis=0), knots


def plot_simulations(
    n_sim: int,
    n_samp: int,
    ppos=0.5,
    acc=0.0,
    q=0.9,
    threshold=0.5,
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
        Percentage of positive labels in a sample.
        Must be between 0 and 1.
        Default is 0.5.
    acc : float
        Percentage of samples the nullmodel is rigged to label correctly with
        a treshold of `threshold`.
        Must be between 0 and 1.
        Defualt is 0.0.
    q : float
        Quantile of simulations to plot.
        Default is 0.9.
    threshold : float
        Classification treshold when using `acc` to rig the model.
        Must be between 0 and 1.
        Defualt is 0.5.
    plot_all : boolean
        Wether to plot all simulations.
        Defualt is False.

    Returns
    -------
    fig : ggplot
        ggplot figure with simulations

    Examples
    -------
    >>> n_simulations = 100
    >>> n_samples = 1000
    >>> plot_simulations(n_simulations, n_samples)
    """
    y_true, y_scores = simulate_nullmodels(n_sim, n_samp, ppos, acc, threshold)
    simulated_curves = [pr_curve(y_true, y_s) for y_s in y_scores]
    prec_q, rec_q = pr_curve_quantile(simulated_curves, q)

    g = ggplot() 
    g += ggtitle(f'n_sim={n_sim} n_samp={n_samp} ppos={ppos} acc={acc} q={q}')
    g += labs(x='recall', y='precision')
    g += geom_step(aes(rec_q, prec_q))

    if not plot_all:
        return g

    shapes = [c[0].shape[0] for c in simulated_curves]
    group = np.repeat(np.arange(n_sim), shapes)
    precs, recs = np.hstack(simulated_curves)

    return g + geom_step(aes(recs, precs, group=group), alpha=1/n_sim)

