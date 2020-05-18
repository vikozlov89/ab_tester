import numpy as np
import pandas as pd
from datetime import datetime, date
import tqdm
from numba import jit

"""
This module includes bootstrapping functions
"""


class BasicSimulator(object):
    """
    Basic simulator (abstract) class. Not intended to be used directly.
    """
    def __init__(self, boot_func, adjust_bias=True, adjust_variance=True):
        self.boot_func = boot_func
        self.adjust_bias = adjust_bias
        self.adjust_variance = adjust_variance

    def boot(self, sample, *args, **kwargs):
        pass

    def randomize_sample(self, sample):
        return np.random.choice(sample, sample.shape[0], replace=False)

    def calculate_bias(self, sample, simulation_result):
        return np.abs(simulation_result.mean() - sample.mean())


class JacknifeSimulator(BasicSimulator):
    """
    Jacknife realization.
    """
    def get_next_window(self, sample, i_start, i_end):
        return sample[i_start:i_end], i_start + 1, i_end + 1

    def calculate_number_of_steps(self, sample_size, batch_size):
        return sample_size - batch_size + 1

    def boot(self, sample, batch_size=100):
        res = []
        tmp = self.randomize_sample(sample)
        i_start = 0
        i_end = i_start + batch_size
        n_steps = self.calculate_number_of_steps(tmp.shape[0], batch_size)
        for i in range(n_steps):
            batch, i_start, i_end = self.get_next_window(tmp, i_start, i_end)
            res.append(self.boot_func(batch))
        res = np.array(res)
        return res, self.calculate_bias(sample, res)


class BatchesSimulator(BasicSimulator):
    """
    Calculates desired ststistics for batches of the original sample random elements.
    """
    def get_batches(self, sample, batch_size=100):
        num_batches = sample.shape[0] // batch_size
        return sample[:num_batches * batch_size].reshape((num_batches, batch_size))

    def boot(self, sample, batch_size=100):
        tmp = self.randomize_sample(sample)
        butches = self.get_batches(tmp, batch_size)
        res = self.boot_func(butches, axis=1)
        return res, self.calculate_bias(sample, res)


class BootstrapSimulator(BasicSimulator):
    """
    Classic bootstrap realization.
    """
    def boot(self, sample, n_boots=10_000, sample_size = None):
        res = np.zeros(n_boots)
        idmax = sample.shape[0]
        sample_size = idmax if not sample_size else sample_size
        for i in range(n_boots):
            res[i] = self.boot_func(sample[np.random.randint(0, idmax, sample_size)])
        return res, self.calculate_bias(sample, res)


class StratifiedBootstrap(BasicSimulator):
    """
    Performs bootstrap on a synthetic samples collected from the original sample in accordance to the defined structure.
    Should be used carefully in a case of having two samples with different structures by some important feature when
    there is no other way to balance samples.

    """
    def construct_query(self, index_names, index_values):

        query = ""
        i = 0
        for n, v in zip(index_names, index_values):
            if isinstance(v, str):
                query += (" and " if i else "") + f"{n} == '{v}' "
            elif isinstance(v, datetime):
                query += (" and " if i else "") + f"{n} == '{v}' "
            elif isinstance(v, date):
                query += (" and " if i else "") + f"{n} == '{v}' "
            else:
                query += (" and " if i else "") + f"{n} == {v} "
            i += 1

        return query

    def pack_to_list(self, to_pack):
        try:
            res = list(to_pack) if not isinstance(to_pack,str) else [to_pack]
        except:
            res = [to_pack]
        return res

    def make_stratified_random_sample(self, sample, structure_to_copy):
        cnames = structure_to_copy.index.names
        res = pd.DataFrame([])
        for vals in structure_to_copy.index:
            clevels = self.pack_to_list(vals)
            cnames = self.pack_to_list(cnames)
            n = structure_to_copy[tuple(clevels) if len(clevels) > 1 else clevels[0]]
            query = self.construct_query(cnames, clevels)
            tmp = sample.query(query)
            if not tmp.shape[0]:
                continue
            resampled_index = np.random.choice(tmp.index, size=n, replace=True)
            res = pd.concat((res, tmp.loc[resampled_index, :].copy()))
        return res.copy()

    def boot(self, sample, column_to_boot, structure_to_copy, n_boots=10_000, resample_freq=1, show_progress_bar = False):
        if resample_freq <= 0:
            raise Exception("resample_freq must be a positive number")
        res = []
        tmp = None
        from_last_resample = 1
        resamples_made = 0
        for i in range(n_boots) if not show_progress_bar else tqdm.trange(n_boots):
            if not i:
                tmp = self.make_stratified_random_sample(sample, structure_to_copy)
                resamples_made += 1
            if from_last_resample > resample_freq * resamples_made:
                resamples_made += 1
                tmp = self.make_stratified_random_sample(sample, structure_to_copy)
            if resample_freq > 1:
                ch = np.random.choice(tmp[column_to_boot].values, size=tmp.shape[0], replace=True)
            else:
                ch = tmp[column_to_boot].values
            res.append(self.boot_func(ch))
            from_last_resample += 1
        res = np.array(res)
        return res, self.calculate_bias(sample, res)


class ParametricSimulator(BasicSimulator):
    """
    Performs a simulation based on a given distribution.
    """
    def boot(self, sample, distribution_generator, distr_params: dict, n_boots: int = 10_000):
        res = np.zeros(n_boots)
        size = sample.shape[0]
        for i in range(n_boots):
            res[i] = self.boot_func(distribution_generator(size=size, **distr_params))
        return res, self.calculate_bias(sample, res)

@jit(nopython=True, parallel=True)
def bootstrap_mean(data, boots=10_000):
    size = data.shape[0]
    result = np.zeros(boots)
    for i in range(boots):
        result[i] = np.random.choice(data, size=size).mean()
    bias = result.mean() - data.mean()
    return result, bias


def bootstrap_samples_difference(sample1, sample2, boot_func, n_boots=10_000):
    """
    Compares two samples using bootstrap. Calculates the statistics distribution for every sample and both biases.
    Returns a dict with simulation results, difference between simulated statistics samples and biases.

    :param sample1: np.array
        sample 1 for bootstrap
    :param sample2: np.array
        sample 2 for bootstrap
    :param boot_func:
        function to be simulated
    :param n_boots:
        number of bootstrap iterations
    :return:
        dict
    """
    boot_sample_size = min(sample1.shape[0], sample2.shape[0])
    simulator = BootstrapSimulator(boot_func)
    sim_sample1, bias1 = simulator.boot(sample1, n_boots=n_boots, sample_size=boot_sample_size)
    sim_sample2, bias2 = simulator.boot(sample2, n_boots=n_boots, sample_size=boot_sample_size)
    diff = sim_sample1 - sim_sample2
    return dict(sim_sample1=sim_sample1, sim_sample2=sim_sample2, diff=diff, bias1=bias1, bias2=bias2)

def get_resampling_structure(sample1, sample2, group_by, boot_variable):
    """
    Calculates a structure for comparing samples with a stratified bootstrap. The final structure includes only groups
    present in both samples and averages the number of rows for each of them between samples (n_sample1 + n_sample2) / 2

    :param sample1: pd.DataFrame
        sample 1 for bootstrap
    :param sample2: pd.DataFrame
        sample 2 for bootstrap
    :param group_by: str
        field(s) to be grouped by
    :param boot_variable:
        feature to be simulated
    :return:
        pd.Series with a new structure
    """
    s1_struct = sample1.groupby(group_by)[[boot_variable]].count()
    s2_struct = sample2.groupby(group_by)[[boot_variable]].count()
    tmp = pd.merge(left=s1_struct,right=s2_struct,left_index=True,right_index=True,how='inner')
    return tmp.sum(axis=1).apply(lambda x: x //2)


def bootstrap_stratified_samples_difference(sample1, sample2, group_by, boot_variable
                                            , boot_func, n_boots=10_000, resample_freq=1,show_progress_bar=False):
    """
    Compares two samples with stratified bootstrap.

    :param sample1: pd.DataFrame
        sample 1 for bootstrap
    :param sample2: pd.DataFrame
        sample 2 for bootstrap
    :param group_by: str, list
        field(s) to be grouped by
    :param boot_variable: str
        feature to be simulated
    :param boot_func:
        function to be simulated
    :param n_boots:
        number of bootstrap iterations
    :param resample_freq:
        number of bootstrap iterations between resampling samples in order to create a predefined structures
    :return: dict

    """
    simulator = StratifiedBootstrap(boot_func)
    struct = get_resampling_structure(sample1, sample2, group_by, boot_variable)
    sim_sample1, bias1 = simulator.boot(sample1
                                        , column_to_boot = boot_variable
                                        , structure_to_copy = struct
                                        , n_boots=n_boots
                                        , resample_freq=resample_freq
                                        , show_progress_bar=show_progress_bar)
    sim_sample2, bias2 = simulator.boot(sample2
                                        , column_to_boot=boot_variable
                                        , structure_to_copy=struct
                                        , n_boots=n_boots
                                        , resample_freq=resample_freq
                                        , show_progress_bar=show_progress_bar)
    diff = sim_sample1 - sim_sample2
    return dict(sim_sample1=sim_sample1, sim_sample2=sim_sample2, diff=diff)


if __name__ == '__main__':

    sample = np.random.normal(size=100_000)
    start = datetime.now()

    means, bias = bootstrap_mean(sample)

    end = datetime.now()
    delta = end - start
    delta = delta.total_seconds()
    print(f"{delta} seconds passed")
    print(bias, means.mean(), sample.mean())


    # from plotting import plot_simulation_comparison

    # ones = pd.DataFrame(np.random.normal(0, 1, size=5_000).reshape((-1, 1)), columns=['y'])
    # twos = pd.DataFrame(np.random.normal(0.5, 1, size=5_000).reshape((-1, 1)), columns=['y'])
    # threes = pd.DataFrame(np.random.normal(1, 1, size=5_000).reshape((-1, 1)), columns=['y'])
    # ones['gr_num'] = 1
    # twos['gr_num'] = 2
    # threes['gr_num'] = 3
    #
    # control = pd.concat((
    #     ones.sample(n=1000, replace=False),
    #     twos.sample(n=1500, replace=False),
    #     threes.sample(n=1000, replace=False)
    # ))
    #
    # test = pd.concat((
    #     ones.sample(n=1500, replace=False),
    #     twos.sample(n=1000, replace=False),
    #     threes.sample(n=1000, replace=False)
    # ))
    #
    # test['y'] += 0.1
    #
    # res = bootstrap_stratified_samples_difference(control, test, 'gr_num', 'y'
    #                                                         , np.mean, n_boots=100, resample_freq=1)
    #
    # plot_simulation_comparison(res.get('sim_sample1'), res.get('sim_sample2'))


