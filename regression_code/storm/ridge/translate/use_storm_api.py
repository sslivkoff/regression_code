"""use storm api to call other functions

- these translations are less developed than use_other_api translations
"""

import functools

import numpy as np

from .... import huth
from .... import aone


def transform_kwargs(function, kwargs_transformers=None,
                     output_transformer=None):
    """transforms a wrapped function's kwargs before calling it

    Parameters
    ----------
    - function: function to be decorated
    - kwargs_transformers: iterable of functions that modify a dict in-place
    """
    if kwargs_transformers is None:
        kwargs_transformers = []
    if output_transformer is None:
        output_transformer = lambda outputs: outputs

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        for transformer in kwargs_transformers:
            transformer(kwargs)
        outputs = function(*args, **kwargs)
        return output_transformer(outputs)
    return wrapper


def kernelize_kwargs(kwargs):
    """transform X data into kernels"""
    if 'Xtrain' in kwargs:
        kwargs['Ktrain'] = kwargs['Xtrain'].dot(kwargs['Xtrain'].T)

        if 'Xtest' in kwargs:
            kwargs['Ktest'] = kwargs['Xtest'].dot(kwargs['Xtrain'].T)
            kwargs.pop('Xtest')

        kwargs.pop('Xtrain')


def make_huth_kwargs(kwargs):
    """translate kwargs into huth.ridge api"""
    kwargs['Rstim'] = kwargs.pop('Xtrain')
    kwargs['Pstim'] = kwargs.pop('Xtest')
    kwargs['Rresp'] = kwargs.pop('Ytrain')
    kwargs['Presp'] = kwargs.pop('Ytest')
    kwargs['alphas'] = kwargs.pop('ridges')

    if kwargs.get('metric') == 'R^2':
        kwargs['use_corr'] = False
    if 'n_folds' in kwargs:
        kwargs['nboots'] = kwargs.pop('n_folds')

        test_fraction = 1 - kwargs.pop('train_fraction', .8)
        len_blocks = kwargs.pop('len_blocks', 5)

        n_samples = kwargs['Rstim'].shape[0]
        kwargs['nchunks'] = int(n_samples * test_fraction / len_blocks)
        kwargs['chunklen'] = len_blocks

    for key in [
        'weights',
        'predictions',
        'performance',
        'verbose',
        'locality',
        'metric',
        'Ytest_zscored',
    ]:
        if key in kwargs:
            kwargs.pop(key)


def make_aone_kwargs(kwargs, cv=False):
    """translate kwargs into aone.ridge api"""
    if not cv:
        if 'Xtest' in kwargs:
            kwargs['Xval'] = kwargs.pop('Xtest')
        if 'Ytest' in kwargs:
            kwargs['Yval'] = kwargs.pop('Ytest')
        if 'Ktest' in kwargs:
            kwargs['Kval'] = kwargs.pop('Ktest')
    if 'n_folds' in kwargs:
        kwargs['nfolds'] = kwargs.pop('n_folds')

    for key in [
        'locality',
        'Ytest_zscored',
    ]:
        if key in kwargs:
            kwargs.pop(key)


#
# # ridge functions
#

ridge_corr_wrapper = transform_kwargs(
    huth.ridge.ridge_corr,
    kwargs_transformers=[make_huth_kwargs],
    output_transformer=lambda outputs: {
        'performance': np.stack(outputs, axis=0),
    },
)
solve_l2_primal_wrapper = transform_kwargs(
    aone.models.solve_l2_primal,
    kwargs_transformers=[make_aone_kwargs],
)
solve_l2_dual_wrapper = transform_kwargs(
    aone.models.solve_l2_dual,
    kwargs_transformers=[kernelize_kwargs, make_aone_kwargs],
)


#
# # cv functions
#

bootstrap_ridge_wrapper = transform_kwargs(
    huth.ridge.bootstrap_ridge,
    kwargs_transformers=[make_huth_kwargs],
    output_transformer=lambda wt, corrs, valphas, allRcorrs, valinds: {
        'weights': wt,
        'performance': corrs,
        'local_optimal_ridge': valphas,
    },
)
cvridge_wrapper = transform_kwargs(
    aone.models.cvridge,
    kwargs_transformers=[functools.partial(make_aone_kwargs, cv=True)],
)
kernel_cvridge_wrapper = transform_kwargs(
    aone.models.kernel_cvridge,
    kwargs_transformers=[
        kernelize_kwargs,
        functools.partial(make_aone_kwargs, cv=True)
    ],
)
