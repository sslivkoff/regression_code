
from __future__ import print_function

import collections
import time

import numpy as np

from regression_code.storm import utils

import sys
sys.path.append('/auto/k1/storm/python_path/storm')
from storm.utils.strs import rprint


tols = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]


def is_equal(results1, results2, name=None, tols=tols, indent=''):
    """print the tolerances to which arrays or dicts of arrays are equal

    Parameters
    ----------
    - results1: array, or dict of str -> array
    - results2: array, or dict of str -> array
    - name: name to use for results if none provided
    - tols: iterable of tolerances to check precision of
    - indent: str indent to add to print statements
    """
    if not isinstance(results1, dict):
        results1 = {name: results1}
    if not isinstance(results2, dict):
        results2 = {name: results2}

    result_types = sorted(set(results1.keys()) & set(results2.keys()))
    tolerances = {
        result_type: {'abs': {}, 'rel': {}}
        for result_type in result_types
    }

    for result_type in result_types:
        result1 = results1[result_type]
        result2 = results2[result_type]

        # allow for ints, floats, lists, etc
        if not isinstance(result1, np.ndarray):
            result1 = np.array(result1)
            result2 = np.array(result2)

        # use np.isclose to test equality
        results_equal = np.isclose(result1, result2)
        results_equal = np.all(results_equal)
        print(indent + result_type, ': all_close=' + str(results_equal))

        if not results_equal:
            N = result1.size

            # compute relative differences
            rel_difference = np.stack(
                [
                    result1 / result2 - 1,
                    result2 / result1 - 1,
                ],
            )
            rel_difference = np.abs(rel_difference)
            rel_difference = rel_difference.max(axis=0)
            for tol in tols:
                tol_str = '{:.0e}'.format(tol)

                n_same = N - (rel_difference > tol).sum()
                print(
                    indent + '    - rel within */', tol_str + ':',
                    n_same,
                    '/',
                    result1.size,
                )
                tolerances[result_type]['rel'][tol_str] = n_same
                if n_same == N:
                    break
            tolerances[result_type]['max_rel'] = tol_str

            # compute absolute differences
            abs_difference = np.abs(result1 - result2)
            for tol in tols:
                tol_str = '{:.0e}'.format(tol)

                n_same = (abs_difference < tol).sum()
                n_same += (np.isnan(result1) + np.isnan(result2)).sum()
                print(
                    indent + '    - abs within +-', tol_str + ':',
                    n_same,
                    '/',
                    result1.size,
                )
                tolerances[result_type]['abs'][tol_str] = n_same
                if n_same == N:
                    break
            tolerances[result_type]['max_abs'] = tol_str

        if 'max_abs' not in tolerances[result_type]:
            tolerances[result_type]['max_abs'] = '0'
        if 'max_rel' not in tolerances[result_type]:
            tolerances[result_type]['max_rel'] = '0'

    return tolerances


def benchmark_functions(functions, save_outputs=False, datasets=None,
                        common=None, outer_loops=3, inner_loops=3):
    """benchmark timings of functions

    - for each (dataset, function) pair will call function(**dataset, **common)
        - will call each pair (outer_loops * inner_loops) times
    - use is_equal() on outputs to perform precision tests

    Parameters
    ----------
    - functions: dict of {function_name: function} of functions to call
    - save_outputs: bool of whether to record outputs of each function call
    - record_times: bool of whether to record timings of each function call
    - datasets: dict of {dataset_name: dataset} of datasets to use
    - common: dict of common kwargs for each function call
    - outer_loops: int number of times to run all functions on a dataset
    - inner_loops: int number of times to run each function in a row

    Returns
    -------
    - results: dict of results
        - 'times': nested dict of arrays of timings in seconds
            - index ordering: times[data_name][f_name][outer, inner]
        - 'mean_times': nested dict of arrays of mean timings in seconds
            - index ordering: times[data_name][f_name]
        - 'outputs': nested dict of calls
            - index ordering: outputs[data_name][f_name][outer][inner]
    """
    if datasets is None:
        datasets = {'single dataset': {}}
    if common is None:
        common = collections.OrderedDict()

    # initialize times
    times = collections.OrderedDict()
    for data_name in datasets.keys():
        times[data_name] = collections.OrderedDict()
        for f_name in functions.keys():
            times[data_name][f_name] = np.ones((outer_loops, inner_loops))
            times[data_name][f_name] *= np.inf

    # initialize outputs
    all_outputs = collections.OrderedDict()
    for data_name in datasets.keys():
        all_outputs[data_name] = collections.OrderedDict()
        for f_name in functions.keys():
            all_outputs[data_name][f_name] = [[] for outer in range(outer_loops)]

    # iterate through functions
    for data_name, dataset in datasets.items():
        for outer in range(outer_loops):
            for f_name, ridge_function in functions.items():
                for inner in range(inner_loops):
                    kwargs = utils.merge(common, dataset)
                    start = time.time()
                    f_outputs = ridge_function(**kwargs)
                    end = time.time()
                    times[data_name][f_name][outer, inner] = end - start

                    if save_outputs:
                        all_outputs[data_name][f_name][outer].append(f_outputs)

    # format results
    results = collections.OrderedDict()
    if save_outputs:
        results['outputs'] = all_outputs
    results['times'] = times

    # compute time stats
    mean_times = collections.OrderedDict()
    std_times = collections.OrderedDict()
    for data_name in datasets.keys():
        mean_times[data_name] = collections.OrderedDict()
        std_times[data_name] = collections.OrderedDict()
        for f_name in functions.keys():
            mean_times[data_name][f_name] = times[data_name][f_name].mean()
            std_times[data_name][f_name] = times[data_name][f_name].std()
    results['mean_times'] = mean_times
    results['std_times'] = std_times

    return results


def print_times(results, stds=False):
    """print out timings of functions benchmarked in benchmark_functions()

    Parameters
    ----------
    - results: dict of results returned by benchmark_functions()
    - std: bool of whether to print out standard deviations
    """
    print('Mean Function Times')
    print('===================')
    for dataset_name in results['mean_times'].keys():

        print()
        print(dataset_name)
        print('-' * len(dataset_name))

        for f_name in results['mean_times'][dataset_name]:
            mean_time = results['mean_times'][dataset_name][f_name]
            std_time = results['std_times'][dataset_name][f_name]
            if stds:
                std_str = ' ({:.4})'.format(std_time)
            else:
                std_str = ''
            print('-', f_name + ':', '{:.4}'.format(mean_time) + std_str)


def print_output_precision(results, baseline_function=None, keys=None):
    """print out precision of functions benchmarked in benchmark_functions()

    Parameters
    ----------
    - results: dict of results returned by benchmark_functions()
    - baseline_function: str name of function to use as baseline
    - keys: iterable of output keys to check precision of
    """
    if baseline_function is None:
        baseline_function = results['outputs'].values()[0].keys()[0]

    if keys is None:
        keys = results['outputs'].values()[0].values()[0][0][0].keys()

    for dataset_name, dataset_outputs in results['outputs'].items():
        dataset_str = 'Dataset: ' + dataset_name
        print('=' * 80)
        print(dataset_str)
        print('=' * 80)

        baseline_outputs = dataset_outputs[baseline_function][0][0]

        for f_name, all_f_outputs in dataset_outputs.items():
            print()
            f_str = 'Function: ' + f_name
            if f_name == baseline_function:
                f_str += ' (baseline)'
            print(f_str)
            print('-' * len(f_str))

            n_outer = len(all_f_outputs)
            n_inner = len(all_f_outputs[0])
            n_total = n_outer * n_inner
            if n_total > 1:
                indent = '    '
            else:
                indent = ''

            for outer in range(n_outer):
                for inner in range(n_inner):
                    if n_total > 1:
                        n_run = outer * n_inner + inner + 1
                        print('run', n_run, '/', n_total)

                    for key in keys:
                        f_outputs = all_f_outputs[outer][inner]
                        is_equal(baseline_outputs, f_outputs, indent=indent)

            print()
        print()
