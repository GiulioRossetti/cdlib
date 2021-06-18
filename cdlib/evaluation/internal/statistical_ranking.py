import scipy as sp
import numpy as np


def friedman_test(*args: dict) -> [float, float, list, list]:
    """
    Performs a Friedman ranking test.
    Tests the hypothesis that in a set of k dependent samples groups (where k >= 2) at least two of the groups represent populations with different median values.

    Parameters
    ----------
    sample1, sample2, ... : array_like
        The sample measurements for each group.

    Returns
    -------
    F-value : float
        The computed F-value of the test.
    p-value : float
        The associated p-value from the F-distribution.
    rankings : array_like
        The ranking for each group.
    pivots : array_like
        The pivotal quantities for each group.

    References
    ----------
    M. Friedman, The use of ranks to avoid the assumption of normality implicit in the analysis of variance, Journal of the American Statistical Association 32 (1937) 674–701.
    D.J. Sheskin, Handbook of parametric and nonparametric statistical procedures. crc Press, 2003, Test 25: The Friedman Two-Way Analysis of Variance by Ranks
    """
    k = len(args)
    if k < 2:
        raise ValueError("Less than 2 levels")
    n = len(args[0])
    if len(set([len(v) for v in args])) != 1:
        raise ValueError("Unequal number of samples")

    rankings = []
    for i in range(n):
        row = [col[i] for col in args]
        row_sort = sorted(row)
        rankings.append(
            [row_sort.index(v) + 1 + (row_sort.count(v) - 1) / 2.0 for v in row]
        )

    rankings_avg = [np.mean([case[j] for case in rankings]) for j in range(k)]
    rankings_cmp = [
        r / np.lib.scimath.sqrt(k * (k + 1) / (6.0 * n)) for r in rankings_avg
    ]

    chi2 = ((12 * n) / float((k * (k + 1)))) * (
        (sum(r ** 2 for r in rankings_avg)) - ((k * (k + 1) ** 2) / float(4))
    )
    iman_davenport = ((n - 1) * chi2) / float((n * (k - 1) - chi2))

    p_value = 1 - sp.stats.f.cdf(iman_davenport, k - 1, (k - 1) * (n - 1))

    return iman_davenport, p_value, rankings_avg, rankings_cmp


def bonferroni_dunn_test(ranks: dict, control: str = None) -> [list, list, list, list]:
    """
    Performs a Bonferroni-Dunn post-hoc test using the pivot quantities obtained by a ranking test.
    Tests the hypothesis that the ranking of the control method is different to each of the other methods.

    Parameters
    ----------
    ranks : dictionary_like
        A dictionary with format 'groupname':'pivotal quantity'
    control : string optional
        The name of the control method (one vs all), default None (all vs all)

    Returns
    ----------
    comparisons : array-like
        Strings identifier of each comparison with format 'group_i vs group_j'
    Z-values : array-like
        The computed Z-value statistic for each comparison.
    p-values : array-like
        The associated p-value from the Z-distribution which depends on the index of the comparison
    Adjusted p-values : array-like
        The associated adjusted p-values which can be compared with a significance level

    References
    ----------
    O.J. Dunn, Multiple comparisons among means, Journal of the American Statistical Association 56 (1961) 52–64.
    """
    k = len(ranks)
    values = list(ranks.values())
    keys = list(ranks.keys())

    if control is None:
        control_i = values.index(min(values))
    else:
        control_i = keys.index(control)

    comparisons = [
        keys[control_i] + " vs " + keys[i] for i in range(k) if i != control_i
    ]
    z_values = [abs(values[control_i] - values[i]) for i in range(k) if i != control_i]
    p_values = [2 * (1 - sp.stats.norm.cdf(abs(z))) for z in z_values]
    # Sort values by p_value so that p_0 < p_1
    p_values, z_values, comparisons = map(
        list, zip(*sorted(zip(p_values, z_values, comparisons), key=lambda t: t[0]))
    )
    adj_p_values = [min((k - 1) * p_value, 1) for p_value in p_values]

    return comparisons, z_values, p_values, adj_p_values
