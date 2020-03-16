"""main function for solving ridge regression problems"""

from __future__ import print_function

import copy

import numpy as np
import scipy.stats

from .. import utils
from . import solvers


def solve_ridge(Xtrain=None, Ytrain=None, Xtest=None, Ytest=None, Ktrain=None,
                Ktest=None, ridges=None, ridge_indices=None, weights=True,
                predictions=True, performance=True, metric='r',
                dtype=None, verbose=2, Ytest_zscored=False, solver=None,
                return_solver=False, factors=None, return_factors=False,
                factors_only=False, sizes=None, rescale=None, fold=None,
                out_path=None):
    """solve ridge regression problem

    Solvers
    -------
    - solvers:
        - 'svd': np.linalg.svd(Xtrain)
        - 'eig': np.linalg.eigh(Xtrain.T.dot(Xtrain))
        - 'eig_dual': np.linalg.eigh(Ktrain)
        - 'cho': (Xtrain.T.dot(Xtrain) + ridge ** 2 * I) Z = Xtrain.T
        - 'cho_dual':
            1. yes weights: (Ktrain + aI)T ZT = X
            2. no weights: (Ktrain + aI)T ZT = Ktest
        - 'qr': QR = scipy.linalg.lu_factor(XTX_ridged); Z = (QR)-1 Xtrain.T
        - 'qr_dual': QR = scipy.linalg.qr(Ktrain_ridged.T)
            1. yes weights: ZT = scipy.linalg.solve_triangular(R, Q.T Xtrain)
            2. no weights: ZT = scipy.linalg.solve_triangular(R, Q.T Ktest.T)
        - 'lu': PLU = scipy.linalg.lu_factor(XTX_ridged); Z = (PLU)-1 Xtrain.T
        - 'lu_dual': PLU = scipy.linalg.lu_factor(Ktrain_ridged.T)
            1. yes weights: ZT = scipy.linalg.lu_solve(PLU, Xtrain)
            2. no weights: ZT = scipy.linalg.lu_solve(PLU, Ktest.T)
        - 'inv': np.linalg.inv(Xtrain.T.dot(Xtrain) + ridge ** 2 * I)
        - 'inv_dual': np.linalg.inv(Ktrain + ridge ** 2 * I)
    - see solvers.py for more information


    Improving Performance
    ---------------------
    - use Ytest_zscored=True if Ytest is already zscored
    - limit the number of returned results
    - reuse factors across multiple calls
        - precompute kernels
        - save factors with factors_only=True or return_factors=True
    - specify data matrices as strs or loader functions for lazy loading
    - give initialized arrays to weights, predictions, and performance arguments
        - useful if copying results to new datastructure afterward


    Sizes
    -----
    - n: number of regressors
    - v: number of regressands
    - m: number of training samples
    - s: number of testing samples
    - r: number of ridge values


    Parameters
    ----------
    - Xtrain: (m, n) array of training regressor data
    - Ytrain: (m, v) array of training regressand data
    - Xtest: (s, n) array of testing regressor data
    - Ytest: (s, v) array of testing regressand data
    - Ktrain: (m, m) array of training kernel
    - Ktest: (s, m) array of testing kernel
    - ridges: iterable of ridge values to use, None for ols
    - ridge_indices: dict of ridge values to compute for each index
        - allows for using different ridge values for each regressand
        - {ridge_value: regressand_indices} where regressand_indices is list
        - if None, all ridge values used for all regressands
        - regressand sets should be mutually exclusive
    - weights: bool or storage array, tells whether to return weights
    - predictions: bool or storage array, tells whether to return predictions
    - performance: bool or storage array, tells whether to return performance
    - metric: str of performance metric to use
        - 'r': pearson's r
        - 'R^2': coefficient of determination
    - dtype: dtype of results, default is Xtrain.dtype
    - verbose: int verbosity level
        - >= 2: print summary message in beginning
        - >= 1: print performance quantiles for each ridge
    - Ytest_zscored: bool of whether Ytest is zscored
    - solver: str of which solver to use (see "Solvers" above)
    - return_solver: bool of whether to return solver name in results
    - factors: dict of factors used for decomposition
    - return_factors: bool of whether to return decomposition factors in results
    - factors_only: bool of whether to only compute factors and skip solution
    - sizes: dict of sizes problem, required when factors are being passed
    - rescale: bool of whether to rescale weights and performance by ridge value
    - fold: 2-tuple of iterables of indices to use for training/testing split
    - out_path: str of path at which to save results


    Returns
    -------
    - outputs: dict of {str_name: result}
        - 'weights': (r, n, v), returned when weights is not None
        - 'predictions': (r, s, v) array, returned when predictions is not None
        - 'performance': (r, v) array, returned when performance is not None
        - 'solver': str of solver name, returned when return_solver=True
        - 'factors': dict of factor arrays, returned when return_factors=True
    - if ridge_indices is specified, r = 1
    - factors and solver are the only outputs when factors_only=True
    """
    # format ridges as array
    if ridges is None and ridge_indices is not None:
        ridges = ridge_indices.keys()
    ridges = np.array(sorted(ridges), dtype=float)

    # gather outputs
    outputs = {
        'weights': weights,
        'predictions': predictions,
        'performance': performance,
    }

    # preprocess data if factors are not provided
    if factors is None:
        datas = {
            'Xtrain': Xtrain,
            'Ytrain': Ytrain,
            'Xtest': Xtest,
            'Ytest': Ytest,
            'Ktrain': Ktrain,
            'Ktest': Ktest,
        }
        datas = preprocess_data(datas, Ytest_zscored, fold, **outputs)
    else:
        datas = factors

    # initialize output data structure
    sizes = get_sizes(datas, ridges, sizes=sizes)
    initialize_outputs(outputs, datas, ridges, ridge_indices, dtype, sizes)

    # get solver
    solver, (factorize, solve) = solvers.get_solver(solver, ridges, **datas)

    # print preamble
    if verbose >= 2:
        print_preamble(ridges, ridge_indices, datas, outputs, solver, metric)
        print()

    # compute factors
    if factors is None:
        factors = factorize(outputs, ridges, **datas)
    if factors_only:
        return {'solver': solver, 'factors': factors, 'sizes': sizes}

    # compute results
    solve_kwargs = utils.merge(outputs, factors, verbose=verbose, metric=metric)
    if ridge_indices is None:
        solve(ridges=ridges, **solve_kwargs)
        if rescale:
            ridge_rescale(outputs, factors, ridges, m=sizes['n_regressands'])
    else:
        solve_indices(ridge_indices, sizes, solve, rescale, **solve_kwargs)

    # add metadata
    if return_solver:
        outputs['sovler'] = solver
    if return_factors:
        outputs['factors'] = factors

    if out_path is not None:
        raise NotImplementedError()

    return outputs


def preprocess_data(datas, Ytest_zscored, fold, weights, predictions,
                    performance):
    """preprocess data

    Parameters
    ----------
    - [see solve_ridge()]
    """
    # ensure proper data is provided
    if (
        (datas.get('Xtrain') is None and datas.get('Ktrain') is None)
        or datas.get('Ytrain') is None
    ):
        raise Exception('must provide training data')
    if (
        (datas.get('Xtest') is None and datas.get('Ktest') is None)
        or datas.get('Ytest') is None
    ):
        if (performance or predictions) and fold is None:
            raise Exception('test data required for requested results')

    # load data if provided as specs
    datas = utils.load_data_specs(datas)

    # zscore Ytest if need be
    if not Ytest_zscored and datas.get('Ytest') is not None:
        datas['Ytest'] = scipy.stats.zscore(datas.get('Ytest'))

    # fold data into training and testing
    if fold is not None:
        assert datas.get('Xtest') is None, 'do not specify Xtest and fold'
        assert datas.get('Ktest') is None, 'do not specify Ktest and fold'
        assert datas.get('Ytest') is None, 'do not specify Ytest and fold'
        datas = utils.fold_data(fold, datas)

    # remove None entries
    datas = {key: value for key, value in datas.items() if value is not None}

    return datas


def get_sizes(datas, ridges=None, sizes=None):
    """gather the sizes of the data

    Parameters
    ----------
    - data: dict of input data arrays
    - ridges: iterable of ridge values
    - sizes: dict of sizes, does not necessarily contain all entries
    """
    if sizes is None:
        sizes = {}

    if 'sizes' in datas:
        sizes.update(datas['sizes'])

    else:
        # gather counts of training samples and regressors
        if datas.get('Xtrain') is not None:
            m, n = datas['Xtrain'].shape
        else:
            n = None
            m = datas['Ktrain'].shape[0]

        # gather counts of testing samples and regressands
        if datas.get('Ytest') is not None:
            s, v = datas['Ytest'].shape
        else:
            s = None
            v = datas['Ytrain'].shape[1]

        # gather counts of ridge values
        if ridges is not None:
            r = len(ridges)
        else:
            r = 1

        computed_sizes = {
            'n_ridges': r,
            'n_train_samples': m,
            'n_regressors': n,
            'n_test_samples': s,
            'n_regressands': v,
        }
        for key, value in computed_sizes.items():
            if (value is not None) or (sizes.get(key) is None):
                sizes[key] = value

    return sizes


def initialize_outputs(outputs, inputs, ridges, ridge_indices, dtype, sizes):
    """initialize data structures for ridge regression results

    Parameters
    ----------
    - outputs: dict of {str output_name: bool should_return}
    - [see solve_ridge()]
    """
    # determine output shapes
    if ridge_indices is not None:
        r = 1
    else:
        r = len(ridges)
    if sizes is None:
        sizes = get_sizes(inputs)
    s = sizes['n_test_samples']
    n = sizes['n_regressors']
    v = sizes['n_regressands']
    shapes = {
        'weights': [r, n, v],
        'predictions': [r, s, v],
        'performance': [r, v],
    }

    # determine output dtype
    if dtype is None:
        if inputs.get('Xtrain') is not None:
            dtype = inputs['Xtrain'].dtype
        elif inputs.get('Ktrain') is not None:
            dtype = inputs['Ktrain'].dtype
        else:
            for data in inputs:
                if isinstance(data, np.ndarray):
                    dtype = data.dtype
                    break
            else:
                raise Exception('no suitable dtype found')

    # initialize each output if it does not already exist
    # initialized = {}
    for name, shape in shapes.items():
        if outputs.get(name):
            if isinstance(outputs[name], np.ndarray):
                # run checks that will raise Exceptions in np.dot()
                if outputs[name].shape != shape:
                    raise Exception('provided matrix is wrong shape')
                if not outputs[name].flags['C_CONTIGUOUS']:
                    raise Exception('np.dot\'s out requires C-contiguous array')
                if not dtype == outputs[name]:
                    raise Exception('output is of wrong dtype')

                # initialized[name] = outputs[name]
            else:
                # initialized[name] = np.zeros(shape, dtype=dtype)
                outputs[name] = np.zeros(shape, dtype=dtype)
        else:
            if name in outputs:
                outputs.pop(name)
    # outputs = initialized

    # convert input dtypes
    for key in list(inputs.keys()):
        if inputs[key] is not None and inputs[key].dtype != dtype:
            inputs[key] = inputs[key].astype(dtype)

    return outputs


def print_preamble(ridges, ridge_indices, inputs, outputs, solver=None,
                   metric=None, print_header=True):
    """print preamble to ridge solver

    Parameters
    ----------
    - ridges: array of ridge parameters used
    - ridge_indices: dict of {ridge_value: regressand_indices}
    - results: dict of results to return
    - inputs: dict of {str: array} of input data
    - outputs: dict of {str: array} of output data
    - solver: str of solver name
    - print_header: bool of whether to print header
    - metric: str metric of performance
    """
    # compute relative sizes of datasets
    sizes = get_sizes(inputs)

    if sizes.get('n_test_samples') is not None:
        fractions = utils.fraction_str(
            sizes['n_train_samples'],
            sizes['n_test_samples'],
        )
    else:
        fractions = None

    # print header
    if print_header:
        print('Fitting ridge model...')
        if fractions is not None:
            print('- (train, test): (' + str(fractions) + ')')

    # print main parameters
    input_order = ['Xtrain', 'Ktrain', 'Ytrain', 'Xtest', 'Ktrain', 'Ytest']
    input_order += sorted(k for k in inputs.keys() if k not in input_order)
    output_order = ['weights', 'predictions', 'performance']
    output_order += sorted(k for k in outputs.keys() if k not in output_order)
    print('- n_ridges:', len(ridges))
    if ridge_indices is not None:
        print('- regressands per ridge:')
        for ridge, indices in ridge_indices.items():
            print('-', str(ridge) + ':', len(indices))
    print('- n_regressors:', sizes['n_regressors'])
    print('- n_regressands:', sizes['n_regressands'])
    print('- n_train_samples:', sizes['n_train_samples'])
    print('- n_test_samples:', sizes['n_test_samples'])

    ordered_inputs = [k for k in input_order if inputs.get(k) is not None]
    ordered_outputs = [k for k in output_order if outputs.get(k) is not None]
    print('- inputs:', ', '.join(ordered_inputs))
    print('- outputs:', ', '.join(ordered_outputs))

    # print dtypes
    input_dtypes = set()
    for data in inputs.values():
        if isinstance(data, np.ndarray):
            input_dtypes.add(data.dtype)
    output_dtypes = set()
    for data in outputs.values():
        if isinstance(data, np.ndarray):
            output_dtypes.add(data.dtype)
    all_dtypes = input_dtypes | output_dtypes
    if len(all_dtypes) == 1:
        print('- dtype:', all_dtypes.pop())
    else:
        print('- dtypes:')

        # print input dtypes
        if len(input_dtypes) == 1:
            print('    - inputs:', input_dtypes.pop())
        else:
            for name in input_order:
                if name in inputs and isinstance(inputs[name], np.ndarray):
                    print('    - ' + name + ':', inputs[name].dtype)

        # print output dtypes
        if len(output_dtypes) == 1:
            print('    - outputs:', output_dtypes.pop())
        else:
            for name in output_order:
                if name in outputs and isinstance(outputs[name], np.ndarray):
                    print('    - ' + name + ':', outputs[name].dtype)

    if solver is not None:
        print('- solver:', solver)
    if 'performance' in outputs and metric is not None:
        print('- performance metric:', metric)


def ridge_rescale(outputs, factors, ridges, m, normalize_by='diagonal'):
    """rescale ridge outputs inplace

    - assumes that training regressors are zscored


    Normalization
    -------------
    - by diagonal: rescale by (1 + ridge / m)
    - by determinant: rescale by geometric_mean(1 + ridge / spectrum ^ 2)


    Parameters
    ----------
    - outputs: outputs to rescale
    - ridges: number or iterable of ridge values used
    - m: int number of training samples
    - normalize_by: method to use for rescaling
    """
    if normalize_by == 'diagonal':
        rescale_factor = 1 + (ridges ** 2 / m)
    elif normalize_by == 'determinant':
        if 'S' in factors:
            spectrum = factors['S'] ** 2
        elif 'L' in factors and factors['L'].ndim == 1:
            spectrum = factors['L']
        elif 'spectrum' in factors:
            spectrum = spectrum
        else:
            P, L, U = scipy.linalg.lu(factors['Xtrain'])
            spectrum = (np.diag(L) ** (1 / m) * (np.diag(U) ** (1 / m)))
            factors['spectrum'] = spectrum

        spectrum = spectrum[:, np.newaxis]
        rescale_factor = np.product((1 + ridges / spectrum) ** (1 / m), axis=0)
    else:
        raise NotImplementedError()

    if 'weights' in outputs:
        outputs['weights'] *= rescale_factor
    if 'predictions' in outputs:
        outputs['predictions'] *= rescale_factor

    return outputs


def solve_indices(ridge_indices, sizes, solve, rescale, metric, verbose,
                  weights=None, predictions=None, performance=None, **factors):
    """solve ridge problem on a subset of regressands for each ridge parameter

    - solve_indices() is merely a helper function to solve_ridge()
        - invoke this behavior by passing ridge_indices to solve_ridge()

    Parameters
    ----------
    - [see solve_ridge()]
    """
    if not rescale:
        raise Exception('must rescale to make indices comparable')

    outputs = {
        'weights': weights,
        'predictions': predictions,
        'performance': performance,
    }

    for ridge in sorted(ridge_indices.keys()):
        indices = ridge_indices[ridge]
        if len(indices) == 0:
            continue

        # slice ridge factors by index
        ridge_factors = copy.copy(factors)
        for name, factor in ridge_factors.items():
            if name.startswith('Y') or name.endswith('Y'):
                ridge_factors[name] = factor[:, indices]
            else:
                ridge_factors[name] = factor

        # initialize temporary outputs, cannot populate outputs directly
        # list indexing of last dimension switches to fortran order... ^ _ ^
        # could be initialized as fortran array instead?
        ridge_sizes = copy.copy(sizes)
        ridge_sizes['n_regressands'] = len(indices)
        ridge_outputs = initialize_outputs(
            ridges=ridge_indices.keys(),
            ridge_indices=ridge_indices,
            dtype=outputs.values()[0].dtype,
            sizes=ridge_sizes,
            inputs=ridge_factors,
            outputs={key: value is not None for key, value in outputs.items()},
        )

        # solve for ridge indices
        ridge_kwargs = utils.merge(ridge_factors, ridge_outputs)
        solve(ridges=[ridge], verbose=verbose, metric=metric, **ridge_kwargs)

        # rescale using ridge values
        ridge_rescale(ridge_outputs, factors, ridge, m=sizes['n_regressands'])

        # populate final outputs
        for name, output in outputs.items():
            if output is not None:
                if output.ndim == 2:
                    output[:, indices] = ridge_outputs[name]
                elif output.ndim == 3:
                    output[:, :, indices] = ridge_outputs[name]
                else:
                    raise Exception('output has improper dimensions')
