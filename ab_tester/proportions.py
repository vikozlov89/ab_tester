"""
This module contains functions needed to conduct statistical tests for proportions.
Formulas are taken from:
http://powerandsamplesize.com/Calculators
"""

import numpy as np
from scipy import stats
from statsmodels.stats.power import NormalIndPower


def one_sample_prop_test(p_prior: float, p_sample: float, n_sample: int, alternative: str = 'two-sided') -> float:
    """
    Tests the hypothesis that a sample proportion (p_sample) does not equal the prior proportion value (p_prior).
    H0 -> p_sample == p_prior
    Allows to perform both two-tail and one-tail tests:
    H1 -> p_sample != p_prior
    H1 -> p_sample < p_prior
    H1 -> p_sample > p_prior
    :param p_prior: float
        the prior proportion value
    :param p_sample: float
        the sample proportion value
    :param n_sample: int
        number of elements in the sample
    :param alternative:
        str defining the type of the test (one- or two-sided). May be 'two-sided' (default)
        , 'lower' (H1: p_sample < p_prior)
        , 'upper' (H1: p_sample > p_prior)
    :return: float
        p-value
    Use Cases
    _________
    >>> p_sample = 0.5 #sample mean
    >>> n_sample = 100 #sample size
    >>> p_prior = 0.4 #prior mean
    >>>
    >>> test_result_against_prior = one_sample_prop_test(p_prior,p_sample,n_sample)
    >>> np.round(test_result_against_prior,4)
    0.0412
    """
    if alternative not in ('two-sided', 'lower', 'upper'):
        raise Exception('alternative must be one of (two-sided, lower, upper)')
    SE = np.sqrt(p_prior * (1 - p_prior) / n_sample)
    effect = np.abs(p_prior - p_sample)
    Z = effect / SE
    norm_dist = stats.norm(0, 1)
    return (1 - norm_dist.cdf(Z)) * 2 if alternative == 'two-sided' \
        else (1 - norm_dist.cdf(Z)) if (alternative == 'lower' and p_sample < p_prior) \
                                       or (alternative == 'upper' and p_sample > p_prior) \
        else norm_dist.cdf(Z)


def two_sample_prop_test(p1: float, p2: float, n1: int, n2: int, alternative: str = 'two-sided') -> float:
    """
    Tests the hypotesis that a sample proportion (p1) does not equal the other sample proportion
     value (p2).
    :param p1: float
        proportion in the sample 1
    :param p2: float
        proportion in the sample 2
    :param n1: int
        size of the sample 1
    :param n2: int
        size of the sample 2
    :param alternative:
        str defining the type of the test (one- or two-sided). May be 'two-sided' (default), 'lower' (H1: p1 < p2),
        'upper' (H1: p1 > p2)
    :return: float
        p-value
    Use Cases
    _________
    >>> p1 = 0.5
    >>> p2 = 0.3
    >>> n1 = 100
    >>> n2 = 100
    >>> np.round(two_sample_prop_test(p1,p2,n1,n2,alternative='two-sided'),4)
    0.0039
    >>> np.round(two_sample_prop_test(p1,p2,n1,n2,alternative='upper'),4)
    0.0019
    """
    if alternative not in ('two-sided', 'lower', 'upper'):
        raise Exception('alternative must be one of (two-sided, lower, upper)')
    P = (p1 * n1 + p2 * n2) / (n1 + n2)
    SE = np.sqrt(P * (1 - P) * (1 / n1 + 1 / n2))
    effect = np.abs(p1 - p2)
    Z = effect / SE
    norm_dist = stats.norm(0, 1)
    return (1 - norm_dist.cdf(Z)) * 2 if alternative == "two-sided" \
        else (1 - norm_dist.cdf(Z)) if (alternative == 'lower' and p1 < p2) or (alternative == 'upper' and p1 > p2) \
        else norm_dist.cdf(Z)


def calculate_one_sample_normalized_effect(p0: float, p1: float) -> float:
    """
    Calculates a normalized effect (in terms of Z-score) for 1 sample.
    :param p0: float
        initial proportion value
    :param p1: float
        new proportion value
    :return: float
        normalized effect size
    """
    q1 = 1 - p1
    sd = np.sqrt(p1 * q1)
    effect = abs(p1 - p0)
    return effect / sd


def calculate_two_sample_normalized_effect(p1: float, p2: float, n1: int, n2: int) -> float:
    """
    Calculates a normalized effect (in terms of Z-score) for 2 samples.
    :param p1: float
        proportion 1 value
    :param p2: float
        proportion 2 value
    :param n1: int
        sample 1 size
    :param n2: int
        sample 2 size
    :return: float
        normalized effect
    """
    q1 = 1 - p1
    q2 = 1 - p2
    sd1 = np.sqrt(p1 * q1)
    sd2 = np.sqrt(p2 * q2)
    sd = (sd1 * n1 + sd2 * n2) / (n1 + n2)  # a total sd for two samples together
    effect = abs(p1 - p2)
    return effect / sd


def calc_power_one_sample_two_sided(p_init: float, p_sample: float, n_sample: int, alpha: float = 0.05) -> float:
    """
    Calculates power for 1-sample 2-sided test.
    The approach according to : http://powerandsamplesize.com/Calculators/Test-1-Proportion/1-Sample-Equality
    :param p_init: float
        initial proportion value
    :param p_sample: float
        sample proportion value
    :param n_sample: int
        sample size
    :param alpha: float
        alpha value (Type I error rate)
    :return: float
        test power (1 - Type II error rate)
    Use Cases
    _________
    >>> # Relatively large effect
    >>> p_init = 0.3
    >>> p_sample = 0.5
    >>> n_sample = 50
    >>> alpha = 0.05
    >>> np.round(calc_power_one_sample_two_sided(p_init,p_sample,n_sample,alpha),2)
    0.81
    >>> # Small effect
    >>> p_init = 0.55
    >>> p_sample = 0.6
    >>> n_sample = 50
    >>> alpha = 0.05
    >>> np.round(calc_power_one_sample_two_sided(p_init,p_sample,n_sample,alpha),2)
    0.11
    """
    norm_dist = stats.norm(0, 1)
    var_sample = p_sample * (1 - p_sample)
    z = abs(p_sample - p_init) / np.sqrt(var_sample / n_sample)

    z_corr = norm_dist.ppf(1 - alpha / 2)
    power = norm_dist.cdf(z - z_corr) + norm_dist.cdf(-z - z_corr)
    return power


def calc_power_one_sample_one_sided(p_init: float, p_sample: float, n_sample: int, alpha: float = 0.05) -> float:
    """
    Calculates power for 1-sample 1-sided test.
    The approach according to : http://powerandsamplesize.com/Calculators/Test-1-Proportion/1-Sample-1-Sided
    :param p_init: float
        initial proportion value
    :param p_sample: float
        sample proportion value
    :param n_sample: int
        sample size
    :param alpha: float
        alpha value (Type I error rate)
    :return: float
        test power (1 - Type II error rate)
    Use Cases
    _________
    >>> # Relatively large effect
    >>> p_init = 0.3
    >>> p_sample = 0.5
    >>> n_sample = 50
    >>> alpha = 0.05
    >>> np.round(calc_power_one_sample_one_sided(p_init,p_sample,n_sample,alpha),2)
    0.91
    >>> # Small effect
    >>> p_init = 0.55
    >>> p_sample = 0.6
    >>> n_sample = 50
    >>> alpha = 0.05
    >>> np.round(calc_power_one_sample_one_sided(p_init,p_sample,n_sample,alpha),2)
    0.17
    """
    norm_dist = stats.norm(0, 1)
    var_sample = p_sample * (1 - p_sample)
    var_init = p_init * (1 - p_init)
    var_ratio = np.sqrt(var_init / var_sample)
    effect = abs(p_sample - p_init)

    z_corr = norm_dist.ppf(1 - alpha)

    z = var_ratio * (effect * np.sqrt(n_sample) / np.sqrt(var_init) - z_corr)

    power = norm_dist.cdf(z)
    return power


def calc_power_one_sample(p_init: float
                          , p_sample: float
                          , n_sample: int
                          , alpha: float = 0.05
                          , one_tail: bool = False) -> float:
    """
    Calculates a test power for 1 sample. Uses either calc_power_one_sample_one_sided or calc_power_one_sample_two_sided
    function according to the one_tail parameter value.
    :param p_init: float
        initial proportion value
    :param p_sample: float
        sample proportion value
    :param n_sample: int
        sample size
    :param alpha: float
        alpha value (Type I error rate)
    :param one_tail: bool
        defines what test to use. If True, 1-sided test is used else - 2-sided.
    :return: float
        test power (1 - Type II error rate)
    """
    if one_tail:
        return calc_power_one_sample_one_sided(p_init, p_sample, n_sample, alpha)
    return calc_power_one_sample_two_sided(p_init, p_sample, n_sample, alpha)


def calc_power_two_sample(p1: float, p2: float, n1: int, n2: int, alpha: float,
                          alternative: str = 'two-sided') -> float:
    """
    :param p1: float
        proportion in the sample 1
    :param p2: float
        proportion in the sample 2
    :param n1: int
        size of the sample 1
    :param n2: int
        size of the sample 2
    :param alpha: float
        Type I error rate
    :param alternative: str
        can be 'two-sided', 'less' and 'greater'
    :return: float
        test power
    Use Cases
    _________
    >>> p1 = 0.35
    >>> p2 = 0.5
    >>> n1 = 100
    >>> n2 = 120
    >>> alpha = 0.05
    >>> res = calc_power_two_sample(p1,p2,n1,n2,alpha)
    >>> np.round(res,2)
    0.62
    """
    ratio = n2 / n1
    effect_size = calculate_two_sample_normalized_effect(p1, p2, n1, n2)
    nip = NormalIndPower()
    power = nip.solve_power(effect_size=effect_size,
                            ratio=ratio,
                            nobs1=n1,
                            alpha=alpha,
                            alternative=alternative)
    return power


def calc_sample_size_two_sample(p1: float, p2, ratio: float, power: float = 0.8, alpha: float = 0.05,
                                alternative: str = 'two-sided'):
    """
    Calcuates sample size for two proportions test.
    :param p1:
    :param p2:
    :param ratio:
    :param power:
    :param alpha:
    :param alternative:
    :return:
    Use Cases
    _________
    Use Cases
    _________
    >>> p1 = 0.35
    >>> p2 = 0.5
    >>> ratio = 1.2
    >>> alpha = 0.05
    >>> power = 0.8
    >>> res = calc_sample_size_two_sample(p1,p2,ratio,power,alpha)
    >>> res
    (153, 184)
    """
    effect_size = calculate_two_sample_normalized_effect(p1, p2, 1, np.int32(np.round(1 * ratio, 0)))
    nip = NormalIndPower()
    ssize = nip.solve_power(effect_size=effect_size,
                            ratio=ratio,
                            power=power,
                            alpha=alpha,
                            alternative=alternative)
    return np.int32(np.round(ssize, 0)), np.int32(np.round(ratio * ssize, 0))

if __name__ == '__main__':

    print(two_sample_prop_test(p1=0.896127,p2=0.930966,n1=568, n2=507,alternative='two-sided'))