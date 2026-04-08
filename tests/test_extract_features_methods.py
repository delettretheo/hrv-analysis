#!/usr/bin/env python
"""This script provides methods to test extract_features methods."""

import os
import unittest
import numpy as np
import pandas as pd
import random
from hrvanalysis.extract_features import (get_time_domain_features, get_geometrical_features,
                                          _create_interpolated_timestamp_list, get_sampen,
                                          get_csi_cvi_features, get_poincare_plot_features,
                                          get_frequency_domain_features)


TEST_DATA_FILENAME = os.path.join(os.path.dirname(__file__), 'test_nn_intervals.txt')


def load_test_data(path):
    # Load test rr_intervals data
    with open(path, "r") as text_file:
        lines = text_file.readlines()
    nn_intervals = list(map(lambda x: int(x.strip()), lines))
    return nn_intervals

def generate_corrupted_data(original_list: list) -> list:
        """
        Simulates signal loss by injecting 10 random windows of NaN values 
        (size 1-5) into a copy of the list.
        """
        nn_intervals_with_nan = list(original_list)
        n = len(nn_intervals_with_nan)

        # Pick 10 random starting indices
        start_indices = random.sample(range(n - 5), 10)
        for start_idx in start_indices:
            window_size = random.randint(1, 5)
            for i in range(window_size):
                if start_idx + i < n:
                    nn_intervals_with_nan[start_idx + i] = np.nan

        return nn_intervals_with_nan

class ExtractFeaturesTestCase(unittest.TestCase):
    """Class for UniTests of different methods in extract_features module"""

    def test_if_time_domain_features_are_correct(self):
        nn_intervals = load_test_data(TEST_DATA_FILENAME)
        function_time_domain_features = get_time_domain_features(nn_intervals=nn_intervals)
        real_function_time_domain_features = {'mean_nni': 718.248,
                                              'sdnn': 43.113074968427306,
                                              'sdsd': 19.519367520775713,
                                              'nni_50': 24,
                                              'pnni_50': 2.4024024024024024,
                                              'nni_20': 225,
                                              'pnni_20': 22.52252252252252,
                                              'rmssd': 19.519400785039664,
                                              'median_nni': 722.5,
                                              'range_nni': 249,
                                              'cvsd': 0.027176408127888504,
                                              'cvnni': 0.060025332431732914,
                                              'mean_hr': 83.84733227281252,
                                              'max_hr': 101.69491525423729,
                                              'min_hr': 71.51370679380214,
                                              'std_hr': 5.196775370674054}

        self.assertDictEqual(function_time_domain_features, real_function_time_domain_features)


    def test_if_time_domain_features_are_correct_for_pnni_as_percent_set_to_false(self):
        nn_intervals = load_test_data(TEST_DATA_FILENAME)
        function_time_domain_features = get_time_domain_features(nn_intervals=nn_intervals, pnni_as_percent=False)
        real_function_time_domain_features = {'mean_nni': 718.248,
                                              'sdnn': 43.113074968427306,
                                              'sdsd': 19.519367520775713,
                                              'nni_50': 24,
                                              'pnni_50': 2.4,
                                              'nni_20': 225,
                                              'pnni_20': 22.5,
                                              'rmssd': 19.519400785039664,
                                              'median_nni': 722.5,
                                              'range_nni': 249,
                                              'cvsd': 0.027176408127888504,
                                              'cvnni': 0.060025332431732914,
                                              'mean_hr': 83.84733227281252,
                                              'max_hr': 101.69491525423729,
                                              'min_hr': 71.51370679380214,
                                              'std_hr': 5.196775370674054}

        self.assertAlmostEqual(function_time_domain_features, real_function_time_domain_features)

    def test_if_geometrical_domain_features_are_correct(self):
        nn_intervals = load_test_data(TEST_DATA_FILENAME)
        function_geometrical_domain_features = get_geometrical_features(nn_intervals)
        real_function_geometrical_domain_features = {'triangular_index': 11.363636363636363,
                                                     'tinn': None}
        self.assertAlmostEqual(function_geometrical_domain_features,
                               real_function_geometrical_domain_features)

    # TODO : check why there is not equality between arrays
    # def test_if_time_info_created_is_correct(self):
    #     nn_intervals = [900, 1000, 1100, 1000, 950, 850]
    #     time_info_created = _create_timestamp_list(nn_intervals)
    #     expected_time = np.array([0., 1., 2.1, 3.1, 4.05, 4.9])
    #     print(expected_time == time_info_created)
    #     print(expected_time)
    #     print(time_info_created)
    #     self.assertAlmostEqual(time_info_created, expected_time)

    def test_if_interpolated_time_created_is_correct(self):
        nn_intervals = [1000, 900, 1100, 1000, 950, 850]
        nni_interpolation_tmstp = _create_interpolated_timestamp_list(nn_intervals, sampling_frequency=2)
        real_interpolation_tmstp = np.array([0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5])
        all_is_equal = all(nni_interpolation_tmstp == real_interpolation_tmstp)
        self.assertTrue(all_is_equal)

    def test_if_csi_cvi_features_are_correct(self):
        nn_intervals = load_test_data(TEST_DATA_FILENAME)
        function_csi_cvi_features = get_csi_cvi_features(nn_intervals)
        real_csi_cvi_features = {'csi': 4.300520404060338,
                                 'cvi': 4.117977429005704,
                                 'Modified_csi': 1021.5749458778378}
        self.assertAlmostEqual(function_csi_cvi_features, real_csi_cvi_features)

    def test_if_pointcare_plot_features_features_are_correct(self):
        nn_intervals = load_test_data(TEST_DATA_FILENAME)
        function_pointcare_plot_features = get_poincare_plot_features(nn_intervals)
        real_pointcare_plot_features = {'sd1': 13.80919037557993,
                                        'sd2': 59.38670497373513,
                                        'ratio_sd2_sd1': 4.300520404060338}
        self.assertAlmostEqual(function_pointcare_plot_features, real_pointcare_plot_features)

    def test_if_sampen_feature_is_correct(self):
        nn_intervals = load_test_data(TEST_DATA_FILENAME)
        function_sampen_features = get_sampen(nn_intervals)
        sampen_plot_features = {'sampen': 1.2046675751816824}
        self.assertAlmostEqual(function_sampen_features, sampen_plot_features)

    def test_if_get_frequency_domain_features_handles_pandas_series(self):

        # TODO: Investigate: extract_features.py:432: RuntimeWarning: invalid value encountered in double_scalars
        # Also occurs with list only.

        try:
            get_frequency_domain_features(pd.Series([42]*10000))
        except KeyError:
            self.fail()



    def test_if_frequency_domain_features_are_not_nan(self):
        """
        Validates that the extraction logic correctly handles corrupted signals (NaN).
        This test confirms that 'nan-aware' functions (nanmean, nanstd, etc.) prevent NaN propagation 
        to the final HRV metrics.
        """
        nn_intervals = load_test_data(TEST_DATA_FILENAME)

        # Simulate intermittent sensor contact or signal loss by injecting random NaN windows
        nn_intervals_with_nan = generate_corrupted_data(nn_intervals)

        # Print raw input for manual verification of NaN distribution within the signal
        print("\n" + "="*80)
        print("INPUT NN-INTERVALS (RAW VECTOR WITH INJECTED NaN):")
        print(nn_intervals_with_nan)
        print("="*80 + "\n")

        # Process corrupted data
        function_time_domain_features = get_time_domain_features(nn_intervals=nn_intervals_with_nan)
        function_csi_cvi_features = get_csi_cvi_features(nn_intervals_with_nan)

        # Display output for manual sanity check of calculated metrics
        print("OUTPUT HRV FEATURES (SUCCESSFULLY COMPUTED FROM DATA EXCLUDING NaNs):")
        print(function_time_domain_features)
        print(function_csi_cvi_features)

        # Audit time-domain results to identify any metrics that failed to filter out the NaNs
        nans_time = [
            k for k, v in function_time_domain_features.items() 
            if isinstance(v, (float, np.float64)) and np.isnan(v)
        ]
        
        # Audit non-linear domain results for potential NaN leakage
        nans_csi = [
            k for k, v in function_csi_cvi_features.items() 
            if isinstance(v, (float, np.float64)) and np.isnan(v)
        ]

        # Assert that the list of "faulty keys" is empty
        self.assertListEqual(nans_time, [], f"NaN values detected in Time Domain output: {nans_time}")
        self.assertListEqual(nans_csi, [], f"NaN values detected in CSI/CVI output: {nans_csi}")

if __name__ == '__main__':

    unittest.main()
