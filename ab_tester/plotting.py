import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from ab_tester.proportions import calc_power_two_sample \
    , calc_sample_size_two_sample \
    , calc_power_one_sample


def plot_simulation(sim_results: np.array, alpha: float = 0.05, series_name: str = "") -> None:
    """
    Plots simulation result as a distplot with confidence intervals.

    :param sim_results: np.array with simulation results
    :param alpha: float value, alpha for confidence interval (from 0 to 1)
    :param series_name: str with the name of the simulated variable
    :return: None
    """
    limits = np.percentile(sim_results, q=[alpha * 100 / 2, (1 - alpha / 2) * 100])

    f = plt.plot(figsize=(15, 7))
    sns.distplot(sim_results)
    xmin, xmax, ymin, ymax = plt.axis()
    plt.vlines(x=limits
               , ymin=ymin
               , ymax=ymax
               , color='red'
               , linestyles='--'
               , label=f'CI, alpha={np.round(alpha, 2)}')
    plt.vlines(x=np.median(sim_results)
               , ymin=ymin
               , ymax=ymax
               , color='darkgrey'
               , linestyles='--'
               , label='Mean')
    plt.title("Bootstrap result" + (f" for {series_name}" if series_name else ""), fontsize=15)
    plt.ylabel("Density", fontsize=13)
    plt.xlabel("Values" if not series_name else series_name, fontsize=13)
    plt.legend(loc=1, fontsize=13)
    plt.show()


def plot_simulation_comparison(sim_results_series_1: np.array
                               , sim_results_series_2: np.array
                               , series_1_name: str = "Series 1"
                               , series_2_name: str = "Series 2"
                               , alpha: float = 0.05) -> None:
    """
    Plots bootstrap comparison of two series.

    :param sim_results_series_1: simulation results for series 1
    :param sim_results_series_2:  simulation results for series 2
    :param series_1_name: (optional) name for the series 1
    :param series_2_name: (optional) name for the series 2
    :param alpha: alpha value for CI calculations
    :return: None
    """
    diff = sim_results_series_1 - sim_results_series_2
    ci_series_1 = np.percentile(sim_results_series_1, q=(alpha * 100 / 2, (1 - alpha / 2) * 100))
    ci_series_2 = np.percentile(sim_results_series_2, q=(alpha * 100 / 2, (1 - alpha / 2) * 100))
    ci_diff = np.percentile(diff, q=(alpha * 100 / 2, (1 - alpha / 2) * 100))
    median_series_1 = np.median(sim_results_series_1)
    median_series_2 = np.median(sim_results_series_2)
    median_diff = np.median(diff)

    fig, ax = plt.subplots(ncols=2, figsize=(16, 7))
    sns.distplot(sim_results_series_1,
                 label=series_1_name,
                 ax=ax[0])
    sns.distplot(sim_results_series_2,
                 label=series_2_name,
                 ax=ax[0])

    ax_0_ymin, ax_0_ymax = ax[0].get_ylim()

    ax[0].vlines(x=ci_series_1,
                 ymin=ax_0_ymin,
                 ymax=ax_0_ymax,
                 linestyles='--',
                 color='red',
                 label=f"CI {series_1_name}, alpha = {round(alpha, 2)}")
    ax[0].vlines(x=ci_series_2,
                 ymin=ax_0_ymin,
                 ymax=ax_0_ymax,
                 linestyles='--',
                 color='brown',
                 label=f"CI {series_2_name}, alpha = {round(alpha, 2)}")
    ax[0].vlines(x=median_series_1,
                 ymin=ax_0_ymin,
                 ymax=ax_0_ymax,
                 linestyles='--',
                 color='darkgrey',
                 label=f"Medians")
    ax[0].vlines(x=median_series_2,
                 ymin=ax_0_ymin,
                 ymax=ax_0_ymax,
                 linestyles='--',
                 color='darkgrey')
    sns.distplot(diff,
                 label="Difference",
                 ax=ax[1])
    ax_1_ymin, ax_1_ymax = ax[1].get_ylim()
    ax[1].vlines(x=ci_diff,
                 ymin=ax_1_ymin,
                 ymax=ax_1_ymax,
                 linestyles='--',
                 color='red',
                 label=f"CI, alpha = {round(alpha, 2)}")
    ax[1].vlines(x=median_diff,
                 ymin=ax_1_ymin,
                 ymax=ax_1_ymax,
                 linestyles='--',
                 color='darkgrey')
    ax[0].legend(loc=1)
    ax[1].legend(loc=1)

    ax[0].set_title("Bootstrap Distributions", fontsize=12)
    ax[0].set_xlabel("Values", fontsize=12)
    ax[0].set_ylabel("Density", fontsize=12)

    ax[1].set_title("Difference Distributions", fontsize=12)
    ax[1].set_xlabel("Values", fontsize=12)
    ax[1].set_ylabel("Density", fontsize=12)
    plt.show()


def plot_two_proportions_power(p1: float
                               , p2: float
                               , baseline_power: float = 0.8
                               , alternative: str = 'two-sided'
                               , alpha: float = 0.05
                               , n_min: int = 500
                               , n_max: int = 10_000
                               , step: int = 500) -> None:
    """
    Plots a propotrions test power vs sample size (assuming sample sizes are equal).

    :param p1: float
        proportion in the sample 1
    :param p2: float
        proportion in the sample 2
    :param baseline_power: float
        power value you expect to get
    :param alternative: str
        can be 'two-sided', 'less' and 'greater'
    :param alpha: float
        test alpha parameter
    :param n_min: int
        minimum sample size
    :param n_max: int
        maximum sample size
    :param step: int
        step size
    :return: None
    """

    sample_size_array = np.arange(n_min, n_max + 1, step)
    powers = [calc_power_two_sample(p1=p1, p2=p2, n1=n, n2=n, alpha=alpha, alternative=alternative)
              for n in sample_size_array]
    ssize_for_baseline_power = calc_sample_size_two_sample(p1=p1
                                                           , p2=p2
                                                           , power=baseline_power
                                                           , alternative=alternative
                                                           , alpha=alpha
                                                           , ratio=1)[0]
    powers = np.array(powers)
    plt.figure(figsize=(15, 7))
    plt.plot(sample_size_array, powers, linewidth=2)
    xmin, xmax, ymin, ymax = plt.axis()
    plt.hlines(y=baseline_power, xmin=xmin, xmax=xmax, color='brown', linestyles='--', linewidth=2)
    plt.title('Power vs Sample Size Plot', fontsize=13)
    plt.xlabel('Sample Size', fontsize=13)
    plt.ylabel('Power', fontsize=13)
    plt.vlines(x=ssize_for_baseline_power
               , ymin=ymin
               , ymax=baseline_power
               , linestyles='--'
               , color='brown'
               , linewidth=2)
    plt.text(x=ssize_for_baseline_power + step / 10,
             y=baseline_power - 0.03,
             s=f"Sample Size = {ssize_for_baseline_power}",
             fontsize=12)
    plt.show()


def plot_one_proportion_power(p_init: float
                              , p_sample: float
                              , baseline_power: float = 0.8
                              , one_tail: bool = False
                              , alpha: float = 0.05
                              , n_min: int = 500
                              , n_max: int = 10_000
                              , step: int = 500) -> None:
    """
    Plots a propotrions test power vs sample size.

    :param p_init: float
        P to compare a sample P with
    :param p_sample: float
        sample P
    :param baseline_power: float
        power value you expect to get
    :param one_tail: bool
        flag marking if one-tail test should be performed
    :param alpha: float
        test alpha parameter
    :param n_min: int
        minimum sample size
    :param n_max: int
        maximum sample size
    :param step: int
        step size
    :return: None
    """
    sample_size_array = np.arange(n_min, n_max + 1, step)
    powers = [calc_power_one_sample(p_init=p_init
                                    , p_sample=p_sample
                                    , n_sample=n
                                    , alpha=alpha
                                    , one_tail=one_tail)
              for n in sample_size_array]
    powers = np.array(powers)
    if (powers >= baseline_power).sum():
        ssize_for_baseline_power = sample_size_array[powers >= baseline_power].min()
    else:
        ssize_for_baseline_power = 0

    plt.figure(figsize=(15, 7))
    plt.plot(sample_size_array, powers, linewidth=2)
    xmin, xmax, ymin, ymax = plt.axis()
    plt.hlines(y=baseline_power, xmin=xmin, xmax=xmax, color='brown', linestyles='--', linewidth=2)
    plt.title('Power vs Sample Size Plot', fontsize=13)
    plt.xlabel('Sample Size', fontsize=13)
    plt.ylabel('Power', fontsize=13)
    if ssize_for_baseline_power:
        plt.vlines(x=ssize_for_baseline_power
                   , ymin=ymin
                   , ymax=baseline_power
                   , linestyles='--'
                   , color='brown'
                   , linewidth=2)
        plt.text(x=ssize_for_baseline_power + step / 10,
                 y=baseline_power - 0.03,
                 s=f"Sample Size = {ssize_for_baseline_power}",
                 fontsize=12)
    plt.show()


if __name__ == '__main__':
    plot_one_proportion_power(0.41, 0.4, n_min=100, n_max=50_000, step=100, one_tail=True)
