"""use other functions' apis to call storm functions


- an Exception will be raised if an untranslated feature is used


Functions Translated
--------------------
- huth.ridge.ridge_corr()
- huth.ridge.bootstrap_ridge()
- aone.models.solve_l2()
- aone.models.solve_l2_primal()
- aone.models.solve_l2_dual()
- aone.models.cvridge()
- aone.models.kernel_cvridge()
"""

import inspect

import numpy as np

from .... import huth
from .... import aone
from .. import cv
from .. import ridge


def convert_to_kwargs(f, args, kwargs, populate_defaults=True):
    """convert a (*args, **kwargs) pair to a simple **kwargs dict

    Parameters
    ----------
    - f: function being called
    - args: iterable of args
    - kwargs: dict of kwargs
    - populate_defaults: bool of whether to populate all kwargs with defaults
    """
    argspec = inspect.getargspec(f)

    # convert args to kwargs
    for name, value in zip(argspec.args, args):
        kwargs[name] = value

    # populate defaults
    if populate_defaults:
        n_nondefaults = len(argspec.args) - len(argspec.defaults)
        for name, value in zip(argspec.args[n_nondefaults:], argspec.defaults):
            kwargs.setdefault(name, value)

    return kwargs


#
# # huth.ridge
#

def ridge_corr(*args, **kwargs):
    """use api of huth.ridge.ridge_corr() to call storm.ridge.solve_ridge()"""
    kwargs = convert_to_kwargs(huth.ridge.ridge_corr, args, kwargs)

    # convert keys and values
    kwargs['Xtrain'] = kwargs.pop('Rstim')
    kwargs['Ytrain'] = kwargs.pop('Rresp')
    kwargs['Xtest'] = kwargs.pop('Pstim')
    kwargs['Ytest'] = kwargs.pop('Presp')
    kwargs['ridges'] = kwargs.pop('alphas')
    if not kwargs['use_corr']:
        kwargs['metric'] = 'R^2'

    # forbid untranslated features
    if (
        kwargs['normalpha']
        or not np.isclose(kwargs['singcutoff'], 1e-10)
    ):
        raise NotImplementedError('feature not translated')

    # remove keys
    for key in [
        'normalpha',
        'corrmin',
        'singcutoff',
        'use_corr',
        'logger',
    ]:
        kwargs.pop(key)

    results = ridge.solve_ridge(**kwargs)
    n_ridges = len(kwargs['ridges'])
    return [results['performance'][i, :] for i in range(n_ridges)]


def bootstrap_ridge(*args, **kwargs):
    """use api of huth.ridge.bootstrap_ridge() to call storm.ridge.solve_ridge()"""
    kwargs = convert_to_kwargs(huth.ridge.bootstrap_ridge, args, kwargs)

    # convert keys and values
    kwargs['Xtrain'] = kwargs.pop('Rstim')
    kwargs['Ytrain'] = kwargs.pop('Rresp')
    kwargs['Xtest'] = kwargs.pop('Pstim')
    kwargs['Ytest'] = kwargs.pop('Presp')
    kwargs['ridges'] = kwargs.pop('alphas')
    kwargs['n_folds'] = kwargs.pop('nboots')
    kwargs['len_blocks'] = kwargs.pop('chunklen')
    n_training_samples = float(kwargs['Xtrain'].shape[0])
    fraction = kwargs.pop('nchunks') * kwargs['len_blocks'] / n_training_samples
    kwargs['train_fraction'] = fraction
    if kwargs['single_alpha']:
        kwargs['locality'] = 'global'
    else:
        kwargs['locality'] = 'local'
    if not kwargs['use_corr']:
        kwargs['metric'] = 'R^2'

    # forbid untranslated features
    if (
        kwargs['joined'] is not None
        or not np.isclose(kwargs['singcutoff'], 1e-10)
    ):
        raise NotImplementedError('feature not translated')

    # remove keys
    for key in [
        'normalpha',
        'corrmin',
        'joined',
        'singcutoff',
        'single_alpha',
        'use_corr',
        'logger',
    ]:
        kwargs.pop(key)

    # call function
    kwargs['raw_cv_surface'] = True
    results = cv.cv_ridge(**kwargs)

    # convert output
    local_optimal_ridge = results['local_optimal_ridge']
    for key in list(results.keys()):
        if key.startswith('global_') or key.startswith('local_'):
            results[key.lstrip('global_').lstrip('local_')] = results[key]
            del results[key]
    return (
        results['weights'],
        results['performance'],
        local_optimal_ridge,
        np.moveaxis(results['raw_cv_surface'], 0, 2),
        np.array([]),
    )


#
# # aone.models
#


def solve_l2(*args, **kwargs):
    """use api of aone.models.solve_l2() to call storm.ridge.solve_ridge()"""
    kwargs = convert_to_kwargs(aone.models.solve_l2, args, kwargs)

    kwargs['ridges'] = [kwargs.pop('ridge')]
    for key in ['weights', 'predictions', 'performance']:
        kwargs.setdefault(key, False)

    kwargs.pop('kernel_name')
    kwargs.pop('kernel_param')
    kwargs.pop('kernel_weights')
    if 'EPS' in kwargs:
        kwargs.pop('EPS')

    return ridge.solve_ridge(**kwargs)


def solve_l2_primal(*args, **kwargs):
    """use api of aone.models.solve_l2_primal() to call storm.ridge.solve_ridge()"""
    kwargs = convert_to_kwargs(aone.models.solve_l2_primal, args, kwargs)
    kwargs['Xtest'] = kwargs.pop('Xval')
    kwargs['Ytest'] = kwargs.pop('Yval')
    kwargs.pop('EPS')
    return ridge.solve_ridge(**kwargs)


def solve_l2_dual(*args, **kwargs):
    """use api of aone.models.solve_l2_dual() to call storm.ridge.solve_ridge()"""
    kwargs = convert_to_kwargs(aone.models.solve_l2_dual, args, kwargs)
    kwargs['Ytest'] = kwargs.pop('Yval')
    kwargs['Ktest'] = kwargs.pop('Kval')
    kwargs.pop('EPS')

    if kwargs['weights']:
        raise NotImplementedError('feature not translated')

    return ridge.solve_ridge(**kwargs)


def cvridge(*args, **kwargs):
    """use api of aone.models.cvridge() to call storm.ridge.cv_ridge()"""
    kwargs = convert_to_kwargs(aone.models.cvridge, args, kwargs)

    # convert keys and values
    kwargs['n_folds'] = kwargs.pop('nfolds')
    kwargs['len_blocks'] = kwargs.pop('blocklen')
    kwargs['train_fraction'] = kwargs.pop('trainpct')
    if kwargs['verbose']:
        kwargs['verbose'] = 2
    if kwargs['withinset_test']:
        if kwargs.get('Xtest') is None:
            kwargs['Xtest'] = kwargs['Xtrain']
        if kwargs.get('Ktest') is None:
            kwargs['Ktest'] = kwargs['Ktrain']
        if kwargs.get('Ytest') is None:
            kwargs['Ytest'] = kwargs['Ytrain']

    # forbid untranslated features
    if (
        kwargs['Li'] is not None
        or kwargs['folds'] != 'cv'
        or kwargs['Li'] is not None
        or kwargs['kernel_name'] != 'linear'
        or kwargs['kernel_params'] is not None
        or kwargs['kernel_weights']
        or not np.isclose(kwargs['EPS'], 1e-10)
    ):
        raise NotImplementedError('feature not translated')

    # remove keys
    for key in [
        'Li',
        'folds',
        'kernel_name',
        'kernel_params',
        'kernel_weights',
        'EPS',
        'withinset_test',
    ]:
        kwargs.pop(key)

    # call function
    kwargs['locality'] = 'global'
    kwargs['raw_cv_surface'] = True
    results = cv.cv_ridge(**kwargs)


    final_results = {'cvresults': results['raw_cv_surface'][:, np.newaxis, ...]}
    if 'global_weights' in results:
        final_results['weights'] = results['global_weights']
    if 'global_predictions' in results:
        final_results['predictions'] = results['global_predictions']
    if 'global_performance' in results:
        final_results['performance'] = results['global_performance'][np.newaxis, :]

    return final_results


def kernel_cvridge(*args, **kwargs):
    """use api of aone.models.kernel_cvridge() to call storm.ridge.cv_ridge()"""
    kwargs = convert_to_kwargs(aone.models.kernel_cvridge, args, kwargs)

    # convert keys and values
    kwargs['n_folds'] = kwargs.pop('nfolds')
    kwargs['len_blocks'] = kwargs.pop('blocklen')
    kwargs['train_fraction'] = kwargs.pop('trainpct')
    if kwargs['verbose']:
        kwargs['verbose'] = 2

    # forbid untranslated features
    if (
        kwargs['folds'] != 'cv'
        or not np.isclose(kwargs['EPS'], 1e-10)
    ):
        raise NotImplementedError('feature not translated')

    # remove keys
    for key in [
        'folds',
        'EPS',
    ]:
        kwargs.pop(key)

    # call function
    kwargs['locality'] = 'global'
    kwargs['raw_cv_surface'] = True
    results = cv.cv_ridge(**kwargs)

    final_results = {'cvresults': results['raw_cv_surface']}
    if 'global_predictions' in results:
        final_results['predictions'] = results['global_predictions']
    if 'global_performance' in results:
        final_results['performance'] = results['global_performance'][np.newaxis, :]

    return final_results
