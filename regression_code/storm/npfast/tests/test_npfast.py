
from __future__ import print_function

import time
import collections

import numpy as np
import scipy.stats

from regression_code.storm import npfast
from regression_code.storm.tests import test_datasets
from regression_code.storm.tests import test_utils


sum_functions = collections.OrderedDict([
    ['np.sum', lambda X: X.sum(0)],
    ['thread_sum', lambda X: npfast.thread_sum(X, 0)],
    ['dot_sum', lambda X: npfast.dot_sum(X, 0)],
])


mean_functions = collections.OrderedDict([
    ['np.mean', lambda X: X.mean(0)],
    ['dot_mean', lambda X: npfast.dot_mean(X, 0)],
])


std_functions = collections.OrderedDict([
    ['np.std', lambda X: X.std(0)],
    ['arithmetic_std', lambda X: (((X - X.mean(0)) ** 2).sum(0) / X.shape[0]) ** .5],
    ['dot_std', lambda X: npfast.dot_std(X, 0)],
    ['dot_std_inplace', lambda X: npfast.dot_std_inplace(X, 0)],
    ['numexpr_std', lambda X: npfast.numexpr_std(X, 0)],
    ['dot_std_demeaned', lambda X: npfast.dot_std(X, 0, demean=False)],
    ['dot_std_inplace_demeaned', lambda X: npfast.dot_std_inplace(X, 0, demean=False)],
    ['numexpr_std_demeaned', lambda X: npfast.numexpr_std(X, 0, demean=False)],
])


zscore_functions = collections.OrderedDict([
    ['scipy.zscore', lambda X: scipy.stats.zscore(X)],
    ['np_arithmetic', lambda X: (X - X.mean(0)) / X.std(0)],
    ['dot', lambda X: npfast.dot_zscore(X, 0)],
    ['dot_inplace', lambda X: npfast.dot_zscore_inplace(X, 0)],
    ['arithmetic_demeaned', lambda X: X / X.std(0)],
    ['dot_demeaned', lambda X: npfast.dot_zscore(X, 0, demean=False)],
    ['dot_inplace_demeaned', lambda X: npfast.dot_zscore_inplace(X, 0, demean=False)],
    ['numexpr', lambda X: npfast.numexpr_zscore(X, 0)],
    ['numexpr_demeaned', lambda X: npfast.numexpr_zscore(X, 0, demean=False)],
    ['numexpr_inplace', lambda X: npfast.numexpr_zscore_inplace(X, 0)],
    ['numexpr_inplace_demeaned', lambda X: npfast.numexpr_zscore_inplace(X, 0, demean=False)],
])


# only correlation of data with itself, but # of operations is the same
correlation_functions = collections.OrderedDict([
    ['scipy_zscore', lambda X, Y: (scipy.stats.zscore(X) * scipy.stats.zscore(Y)).mean(0)],
    ['arithmetic', lambda X, Y: (((X - X.mean(0)) / X.std(0)) * ((Y - Y.mean(0)) / Y.std(0))).mean(0)],
    ['dot', lambda X, Y: npfast.dot_correlate(X, Y, 0)],
    ['dot_inplace', lambda X, Y: npfast.dot_correlate_inplace(X, Y, 0, pollute='left')],
    ['numexpr', lambda X, Y: npfast.numexpr_correlate(X, Y, 0)],
    ['arithmetic_1_zscored', lambda X, Y: (X * ((Y - Y.mean(0)) / Y.std(0))).mean(0)],
    ['inplace_1_zscored', lambda X, Y: npfast.dot_correlate_inplace(X, Y, 0, zscore_left=False, pollute='left')],
    ['arithmetic_2_zscored', lambda X, Y: (X * Y).mean(0)],
    ['inplace_2_zscored', lambda X, Y: npfast.dot_correlate_inplace(X, Y, 0, zscore_left=False, zscore_right=False, pollute='left')],
    ['numexpr_1_zscored', lambda X, Y: npfast.numexpr_correlate(X, Y, 0, zscore_left=False)],
    ['numexpr_2_zscored', lambda X, Y: npfast.numexpr_correlate(X, Y, 0, zscore_left=False, zscore_right=False)],
])


isnan_functions = collections.OrderedDict([
    ['np.isnan', lambda X: np.isnan(X)],
    ['numexpr_isnan', lambda X: npfast.numexpr_isnan(X)],
])


nan_to_num_functions = collections.OrderedDict([
    ['np.nan_to_num', lambda X: np.nan_to_num(X)],
    ['numexpr', lambda X: npfast.numexpr_nan_to_num(X)],
    ['numexpr_inplace', lambda X: npfast.numexpr_nan_to_num_inplace(X)],
])


copy_functions = collections.OrderedDict([
    ['np.copy', lambda X: np.copy(X)],
    ['numexpr_copy', lambda X: npfast.numexpr_copy(X)],
])


all_functions = collections.OrderedDict([
    ['sum functions', sum_functions],
    ['mean functions', mean_functions],
    ['std functions', std_functions],
    ['zscore functions', zscore_functions],
    ['correlation functions', correlation_functions],
    ['isnan functions', isnan_functions],
    ['nan_to_num functions', nan_to_num_functions],
    ['copy functions', copy_functions],
])


two_arg_functions = [
    'correlation functions',
]


def test_precision(X=None, X2=None, add_nans=10):
    """tests precision fo each set of functions

    - the first result in each result type is taken as the ground truth

    Parameters
    ----------
    - X:
    - X2:
    - add_nans:
    """

    if X is None:
        X, Y = test_datasets.create_random_XY('medium')

    if X2 is None:
        X2 = np.random.rand(*X.shape).astype(X.dtype)

    # add random nans to data
    for i in range(add_nans):
        X[np.random.randint(X.shape[0]), np.random.randint(X.shape[1])] = np.nan
        X2[np.random.randint(X.shape[0]), np.random.randint(X.shape[1])] = np.nan

    # test accuracy for each function set
    for set_name, function_set in all_functions.items():
        print('Testing precision of', set_name)
        X_copy = np.copy(X)

        # compute baseline result
        if set_name in two_arg_functions:
            X2_copy = np.copy(X2)
            baseline_result = function_set.values()[0](X_copy, X2_copy)
        else:
            baseline_result = function_set.values()[0](X_copy)

        # compute result for each function in set
        for function_name, function in function_set.items():
            X_copy = np.copy(X)
            if '_demeaned' in function_name:
                X_copy = X_copy - X_copy.mean(0)
            elif '_zscored' in function_name:
                X_copy = scipy.stats.zscore(X_copy)

            if set_name in two_arg_functions:
                X2_copy = np.copy(X2)
                if '_2_zscored' in function_name:
                    X2_copy = scipy.stats.zscore(X2_copy)

                function_result = function(X_copy, X2_copy)
            else:
                function_result = function(X_copy)

            # test equality of results
            test_utils.is_equal(
                baseline_result,
                function_result,
                '- ' + function_name,
            )
        print()


def compute_time_results(matrix_sets, functions, outer_repeats=10,
                         inner_repeats=10, two_arg=False):
    """compute mean time of each function applied to each matrix

    Parameters
    ----------
    - matrix_sets: dict of tuple of arrays to use as test data
    - functions: dict of {str_name: function}
    - outer_repeats: int of outer repeat count
    - inner_repeats: int of inner repeat count
    """
    results = {}
    for name, matrices in matrix_sets.items():
        results[name] = np.zeros((len(functions), len(matrices), outer_repeats))
        for repeat in range(outer_repeats):
            for f, (f_name, function) in enumerate(functions.items()):
                for m, matrix in enumerate(matrices):
                    start = time.time()
                    for i in range(inner_repeats):
                        if two_arg:
                            function(matrix, matrix)
                        else:
                            function(matrix)
                    end = time.time()
                    results[name][f, m, repeat] = (end - start) / inner_repeats
        results[name] = results[name].mean(2)
    return results


def tabulate_time_results(results, matrices, functions):
    """print results in a table

    Parameters
    ----------
    - results: dict from output of compute_results()
    - matrices: dict of matrices used for testing
    - functions: tuple of (name, function)
    """
    print()
    for name in reversed(sorted(results.keys())):
        max_name = max(len(f_name) for f_name, function in functions.items())
        shapes = []
        for matrix in matrices[name]:
            shape = str(matrix.shape[0]) + 'x' + str(matrix.shape[1])
            shape = '{:>12}'.format(shape)[:12]
            shapes.append(shape)

        print('Matrix Set:', name)
        print('=' * len('Matrix Set: ' + name))
        print()
        print('Absolute Times')
        print('--------------')
        print('{:>{}}'.format('', max_name) + ' ', '  '.join(shapes))
        for f, (f_name, function) in enumerate(functions.items()):
            times = [
                '{:12.5}'.format(result)[:12]
                for result in results[name][f, :]
            ]
            print('{:>{}}'.format(f_name, max_name) + ' ', '  '.join(times))
        print()
        print()
        print('Relative Times')
        print('--------------')
        print('{:>{}}'.format('', max_name) + ' ', '  '.join(shapes))
        for f, (f_name, function) in enumerate(functions.items()):
            normalized = results[name][f, :] / results[name][0, :]
            times = ['{:>12.5}'.format(result)[:12] for result in normalized]
            print('{:>{}}'.format(f_name, max_name) + ' ', '  '.join(times))
        print()
        print()
        print()
