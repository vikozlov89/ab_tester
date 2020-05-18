import unittest
import numpy as np
from ab_tester import bootstrap
import pandas as pd


class BootstrapModuleTestCase(unittest.TestCase):

    def test_randomized_sample_size(self):
        sample = np.arange(10)
        simulator = bootstrap.BasicSimulator(np.mean)
        rand_sample = simulator.randomize_sample(sample)
        self.assertEqual(sample.shape
                         , rand_sample.shape
                         , "Randomized sample size different from the initial sample size")

    def test_randomized_sample_order(self):
        sample = np.arange(10)
        simulator = bootstrap.BasicSimulator(np.mean)
        rand_sample = simulator.randomize_sample(sample)
        check_number = np.abs(rand_sample - sample).sum()
        self.assertGreater(check_number
                           , 0
                           , "Randomized sample elements are in the exactly the same order as in the initial sample"
                             ", no randomization occured")

    def test_randomized_sample_has_the_same_elements(self):
        sample = np.arange(10)
        simulator = bootstrap.BasicSimulator(np.mean)
        rand_sample = simulator.randomize_sample(sample)
        sorted_rand_sample = np.sort(rand_sample)
        check_number = np.abs(sorted_rand_sample - sample).sum()
        self.assertEqual(check_number
                         , 0
                         , "Randomized sample has duplicates or some elements different from the original sample")

    def test_jacknife_next_window_correct_batch(self):
        sample = np.arange(10)
        simulator = bootstrap.JacknifeSimulator(np.mean)
        batch_size = 3
        i_start = 0
        i_end = i_start + batch_size
        batch, i_start, i_end = simulator.get_next_window(sample, i_start, i_end)
        correct_batch = sample[0:batch_size]
        check_number = abs(correct_batch - batch).sum()
        self.assertEqual(check_number, 0, "Jacknife batches are incorrect")

    def test_jacknife_number_of_steps_calculation(self):
        simulator = bootstrap.JacknifeSimulator(np.mean)
        sample_size = 10
        n_steps_window_3 = simulator.calculate_number_of_steps(sample_size=sample_size, batch_size=3)
        n_steps_window_4 = simulator.calculate_number_of_steps(sample_size=sample_size, batch_size=4)

        self.assertTupleEqual((n_steps_window_3, n_steps_window_4)
                              , (8, 7)
                              , "Jacknife: incorrect number of steps calculation")

    def test_jacknife_get_next_window_i_end(self):
        sample = np.random.normal(size=10)
        simulator = bootstrap.JacknifeSimulator(np.mean)
        batch_size = 3
        n_steps = simulator.calculate_number_of_steps(sample.shape[0], batch_size)
        i_start = 0
        i_end = i_start + batch_size
        res = [i_end]
        for i in range(n_steps):
            batch, i_start, i_end = simulator.get_next_window(sample, i_start, i_end)
            res.append(i_end)
        # add 11 just because at the last step i_end will be sample_shape + 1
        check_tuple = (3, 4, 5, 6, 7, 8, 9, 10, 11)
        self.assertTupleEqual(tuple(res), check_tuple, "i_end defined incorrectly")

    def test_jacknife_get_next_window_i_start(self):
        sample = np.random.normal(size=10)
        simulator = bootstrap.JacknifeSimulator(np.mean)
        batch_size = 3
        n_steps = simulator.calculate_number_of_steps(sample.shape[0], batch_size)
        i_start = 0
        i_end = i_start + batch_size
        res = [i_start]
        for i in range(n_steps):
            batch, i_start, i_end = simulator.get_next_window(sample, i_start, i_end)
            res.append(i_start)
        # add 8 just because at the last step i_end will be n_steps + 1
        check_tuple = (0, 1, 2, 3, 4, 5, 6, 7, 8)
        self.assertTupleEqual(tuple(res), check_tuple, "i_start defined incorrectly")

    def test_jacknife_correct_number_of_results(self):
        sample = np.random.normal(size=10)
        simulator = bootstrap.JacknifeSimulator(np.mean)
        batch_size = 3
        results, bias = simulator.boot(sample, batch_size)
        self.assertEqual(results.shape[0], 8, "Incorrect number of results. Should be sample_size - batch_size + 1:"
                                              "10 - 3 + 1 = 8")

    def test_batches_simulator_correct_batches(self):
        sample = np.arange(10)
        simulator = bootstrap.BatchesSimulator(np.mean)
        batch_size = 3
        correct_batches = sample[:9].reshape((3, 3))
        batches = simulator.get_batches(sample, batch_size)
        check_number = np.abs(correct_batches - batches).sum()
        self.assertEqual(check_number, 0, "Incorrect batches created. Check numbers in batches.")

    def test_batches_simulator_correct_number_of_results(self):
        sample = np.arange(10)
        simulator = bootstrap.BatchesSimulator(np.mean)
        batch_size = 4
        correct_n = 2
        batches, bias = simulator.boot(sample, batch_size)
        self.assertEqual(batches.shape[0], correct_n)

    def test_stratified_bootstrap_resampling_correctess(self):

        ones = pd.DataFrame(np.random.normal(0, 1, size=5_000).reshape((-1, 1)), columns=['y'])
        twos = pd.DataFrame(np.random.normal(0.5, 1, size=5_000).reshape((-1, 1)), columns=['y'])
        threes = pd.DataFrame(np.random.normal(1, 1, size=5_000).reshape((-1, 1)), columns=['y'])
        ones['gr_num'] = 1
        twos['gr_num'] = 2
        threes['gr_num'] = 3

        control = pd.concat((
            ones.sample(n=1000, replace=False),
            twos.sample(n=1500, replace=False),
            threes.sample(n=1000, replace=False)
        ))

        test = pd.concat((
            ones.sample(n=1500, replace=False),
            twos.sample(n=1000, replace=False),
            threes.sample(n=1000, replace=False)
        ))

        str_boot = bootstrap.StratifiedBootstrap(np.mean)
        struct = control.groupby('gr_num').y.count()
        resampled = str_boot.make_stratified_random_sample(test, struct)
        new_struct = resampled.groupby('gr_num').y.count()

        res = np.abs(struct - new_struct).sum()
        self.assertEqual(res, 0)

    def test_bootstrap_stratified_samples_difference_works(self):

        ones = pd.DataFrame(np.random.normal(0, 1, size=5_000).reshape((-1, 1)), columns=['y'])
        twos = pd.DataFrame(np.random.normal(0.5, 1, size=5_000).reshape((-1, 1)), columns=['y'])
        threes = pd.DataFrame(np.random.normal(1, 1, size=5_000).reshape((-1, 1)), columns=['y'])
        ones['gr_num'] = 1
        twos['gr_num'] = 2
        threes['gr_num'] = 3

        control = pd.concat((
            ones.sample(n=1000, replace=False),
            twos.sample(n=1500, replace=False),
            threes.sample(n=1000, replace=False)
        ))

        test = pd.concat((
            ones.sample(n=1500, replace=False),
            twos.sample(n=1000, replace=False),
            threes.sample(n=1000, replace=False)
        ))

        res = bootstrap.bootstrap_stratified_samples_difference(control, test, 'gr_num', 'y'
                                                                , np.mean, n_boots=100, resample_freq=1)
        res = (res.get('sim_sample1').shape[0], res.get('sim_sample2').shape[0], res.get('diff').shape[0])
        self.assertTupleEqual(res, (100, 100, 100))

    def test_get_resampling_structure_works(self):

        ones = pd.DataFrame(np.random.normal(0, 1, size=5_000).reshape((-1, 1)), columns=['y'])
        twos = pd.DataFrame(np.random.normal(0.5, 1, size=5_000).reshape((-1, 1)), columns=['y'])
        threes = pd.DataFrame(np.random.normal(1, 1, size=5_000).reshape((-1, 1)), columns=['y'])
        ones['gr_num'] = 1
        twos['gr_num'] = 2
        threes['gr_num'] = 3

        control = pd.concat((
            ones.sample(n=10, replace=False),
            twos.sample(n=15, replace=False),
        ))

        test = pd.concat((
            ones.sample(n=15, replace=False),
            twos.sample(n=10, replace=False),
            threes.sample(n=10, replace=False)
        ))
        struct = bootstrap.get_resampling_structure(control, test, 'gr_num', 'y')
        right_struct = pd.Series([12, 12], index=[1, 2])
        res = pd.DataFrame({'s1':struct, 's2':right_struct}).fillna(0)
        self.assertEqual(res.apply(lambda x: x['s1']-x['s2'], axis=1).abs().sum(), 0)



if __name__ == '__main__':
    unittest.main()
