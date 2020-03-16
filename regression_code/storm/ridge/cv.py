"""functions for cross validated ridge regression"""

from __future__ import print_function

import functools

import numpy as np
import scipy

from glabtools.cluster import cmap

from .. import npfast
from .. import utils
from . import ridge
from . import solvers


def cv_ridge(Xtrain=None, Ytrain=None, Xtest=None, Ytest=None, Ktrain=None,
             Ktest=None, ridges=None, weights=True, predictions=True,
             performance=True, cv_surface=False, raw_cv_surface=False,
             locality='both', Ytest_zscored=False, cluster=False, folds=None,
             n_folds=5, train_fraction=0.8, len_blocks=5, seed=0, verbose=2,
             n_train_samples=None, return_ridge_indices=False,
             multithread=False, **solve_kwargs):
    """cross-validated ridge regression

    - additional kwargs will be passed to solve_ridge()


    CV Procedure
    ------------
    1. split training data into folds
    2. use boostrap CV on training data folds to learn optimal hyperparameters
        - treat fold as testing data, and rest of data as training data
        - optimal parameters are computed as a mean of optima across folds
    3. fit weights using all training data and optimal hyperparameters
    4. compute predictions and performance of fit model using testing data
    5. return weights, predictions, and/or performance


    Sizes
    -----
    - n: number of regressors
    - v: number of regressands
    - m: number of training samples
    - s: number of testing samples
    - r: number of ridge values
    - f: number of folds


    Parameters
    ----------
    - Xtrain: (m, n) array of training regressor data
    - Ytrain: (m, v) array of training regressand data
    - Xtest: (s, n) array of testing regressor data
    - Ytest: (s, v) array of testing regressand data
    - Ktrain: (m, m) array of training kernel
    - Ktest: (s, m) array of testing kernel
    - ridges: iterable of ridge values
    - weights: bool of whether to include weights in results
    - predictions: bool of whether to include predictions in results
    - performance: bool of whether to include performance in results
    - cv_surface: bool of whether to include mean cv performance in results
    - raw_cv_surface: bool of whether to include all cv performance in results
    - locality: str of 'global' or 'local' or 'both' for parameter selection
    - Ytest_zscored: bool of whether Ytest has already been zscored
    - cluster: bool of whether to run jobs on cluster
    - folds: iterable of (training, testing) index arrays
    - n_folds: int number of folds
    - train_fraction: float fraction of data to use for training
    - len_blocks: int length of blocks used in fold generation
    - seed: int of random seed to use for fold generation
    - verbose: int or bool of verbosity level
        - >= 1: show basic summary information
        - >= 2: show performance of each ridge parameter for each fold
    - n_train_samples: int m, required if arrays given as nonarray references
    - multithread: bool or int of whether to use threads, int indicates how many
    - solve_kwargs: dict of additional kwargs to pass to solve_ridge()


    Returns
    -------
    - results: dict containing various results
        - global_weights: (n, v) array of weights of global ridge optima
        - global_predictions: (s, v) array of predictions of global ridge optima
        - global_performance: (v) array of performance of global ridge optima
        - local_weights: (n, v) array of weights of local ridge optima
        - local_predictions: (s, v) array of predictions of local ridge optima
        - local_performance: (v) array of performance of local ridge optima
        - cv_surface: (r, v) array of cv performance
        - raw_cv_surface: (f, r, v) array of cv_surface prior to fold averaging
        - ridge_indices: dict of {ridge_value: indices} of optimal local values
    """
    # group parameters
    datas = {
        'Xtrain': Xtrain,
        'Ytrain': Ytrain,
        'Xtest': Xtest,
        'Ytest': Ytest,
        'Ktrain': Ktrain,
        'Ktest': Ktest,
    }
    outputs = {
        'weights': weights,
        'predictions': predictions,
        'performance': performance,
    }
    ridges = np.array(sorted(ridges), dtype=float)

    # assert proper data is specified
    assert (Xtrain is not None) or (Ktrain is not None)
    assert Ytrain is not None
    if predictions or performance:
        assert (Ktest is not None or (Xtest is not None and Xtrain is not None))
        assert Ytest is not None

    # construct folds
    np.random.seed(seed)
    if folds is None:
        if n_train_samples is None:
            n_train_samples = ridge.get_sizes(datas)['n_train_samples']
        folds = utils.generate_folds(
            n_samples=n_train_samples,
            n_folds=n_folds,
            train_fraction=train_fraction,
            len_blocks=len_blocks,
        )

    # print summary
    if verbose >= 1:
        print_cv_preamble(ridges, cluster, locality, folds, datas, outputs, **solve_kwargs)
        if verbose == 1:
            print()

    # compute fold results
    training_data = {'Xtrain': Xtrain, 'Ytrain': Ytrain, 'Ktrain': Ktrain}
    fold_kwargs = utils.merge(
        training_data,
        solve_kwargs,
        folds=folds,
        ridges=ridges,
        verbose=verbose,
    )

    # case: compute using cluster
    if cluster:
        job_kwargs = [{'f': f} for f in range(len(folds))]
        fold_results = cmap.cmap(
            compute_fold,
            job_kwargs,
            common=fold_kwargs,
            verbose=False,
        )
        datas = utils.load_data_specs(datas)  # start loading these datas

    # case: compute using local machine
    else:
        if multithread:
            if isinstance(multithread, int):
                n_threads = multithread
            else:
                n_threads = None
            f = functools.partial(compute_fold, **fold_kwargs)
            fold_results = utils.thread_map(f, range(len(folds)), n_threads)
        else:
            fold_results = []
            datas = utils.load_data_specs(datas)  # start loading these datas
            for f in range(len(folds)):
                fold_result = compute_fold(f=f, **fold_kwargs)
                fold_results.append(fold_result)

    fold_results = [fold_result['performance'] for fold_result in fold_results]

    # compute ridge surface
    results = {}
    if cv_surface or weights or predictions or performance:
        cv_surf = npfast.list_mean([result for result in fold_results])
    if cv_surface:
        results['cv_surface'] = cv_surf
    if raw_cv_surface:
        results['raw_cv_surface'] = np.stack(fold_results, axis=0)
    # del fold_result  # see if these help memory
    # del fold_results

    # compute optimal ridge values
    r_global_optima = npfast.nanmean(cv_surf, axis=1).argmax()
    results['global_optimal_ridge'] = ridges[r_global_optima]
    results['local_optimal_ridge'] = ridges[cv_surf.argmax(axis=0)]
    ridge_indices = {}
    for alpha in ridges:
        indices = np.nonzero(results['local_optimal_ridge'] == alpha)[0]
        ridge_indices[alpha] = indices
    if return_ridge_indices:
        results['ridge_indices'] = ridge_indices

    # display optimal ridge values
    if verbose >= 1:
        print()
        print('Globally-Optimal Ridge:', results['global_optimal_ridge'])
        print('Locally-Optimal Ridges:')
        index_digits = int(np.floor(np.log10(len(ridges) + 1)) + 1)
        ridge_digits = int(max(np.floor(np.log10(max(np.abs(ridges)))), 0) + 2)
        ridge_template = '{0:' + str(ridge_digits) + '.1f}'
        for alpha in ridges:
            indices = ridge_indices[alpha]
            ridge_str = ridge_template.format(alpha)[:ridge_digits]
            print(' ' * index_digits + '   ' + ridge_str + ' ', indices.size)
        if verbose == 1:
            print()

    # compute results based on optimal ridge parameters
    if weights or predictions or performance:
        common_kwargs = utils.merge(datas, outputs, solve_kwargs)
        common_kwargs['verbose'] = 0
        common_kwargs['Ytest_zscored'] = Ytest_zscored
        locality_kwargs = {}

        # gather global parameters
        if locality in ('global', 'both'):
            locality_kwargs['global'] = {'ridges': [ridges[r_global_optima]]}

        # gather local parameters
        if locality in ('local', 'both'):
            locality_kwargs['local'] = {
                'ridge_indices': ridge_indices,
                'rescale': True,
            }

        # compute results for each locality
        for loc, kwargs in locality_kwargs.items():
            if verbose:
                if verbose >= 2 and common_kwargs['performance']:
                    print()
                print('Fit with ' + loc + 'ly-optimal parameters')
            loc_results = ridge.solve_ridge(**utils.merge(common_kwargs, kwargs))

            # remove singleton dimension
            loc_results = {key: value[0, ...] for key, value in loc_results.items()}

            if verbose >= 1 and common_kwargs['performance']:
                if 'ridge_indices' in kwargs:
                    performances = {}
                    for alpha, indices in ridge_indices.items():
                        performances[alpha] = loc_results['performance'][indices]
                    for r, (alpha, loc_performance) in enumerate(sorted(performances.items())):
                        solvers.print_performance(loc_performance, r, ridges)
                    solvers.print_performance(loc_results['performance'], 'all', ridges)
                else:
                    solvers.print_performance(loc_results['performance'], 0, ridges)

            for key, value in loc_results.items():
                results[loc + '_' + key] = value

    return results


def print_cv_preamble(ridges, cluster, locality, folds, inputs, outputs,
                      **solve_kwargs):
    """print summary of cv_ridge()

    Parameters
    ----------
    - [see cv_ridge()]
    """
    # compute relative sizes of datasets
    sizes = ridge.get_sizes(inputs)
    n_training = sum(training.size for training, _ in folds) / len(folds)
    n_validation = sum(validation.size for _, validation in folds) / len(folds)
    if sizes.get('n_test_samples') is not None:
        n_test = sizes['n_test_samples']
        fractions = utils.fraction_str(n_training, n_validation, n_test)
        fraction_str = '- (train, validate, test):'
    else:
        fractions = utils.fraction_str(n_training, n_validation)
        fraction_str = '- (train, validate):'

    # print main parameters
    print('Fitting cross-validated ridge model...')
    print(fraction_str, fractions)
    print('- n_folds:', len(folds))
    print('- using cluster:', cluster)
    print('- globally optimal parameters:', locality in ('global', 'both'))
    print('- locally optimal parameters:', locality in ('local', 'both'))

    # print preamble of solver function
    outputs = {key: value for key, value in outputs.items() if value}
    ridge.print_preamble(ridges, None, inputs, outputs, print_header=False)

    # print other parameters passed to solve_ridge
    if len(solve_kwargs) > 0:
        print('- additional kwargs:')
        for key in sorted(solve_kwargs.keys()):
            print('    -', str(key) + ':', solve_kwargs[key])
    else:
        print('- additional kwargs: None')


def compute_fold(f, folds, Xtrain, Ytrain, ridges, verbose, Ktrain=None,
                 **solve_kwargs):
    """compute a single cross validation fold

    Parameters
    ----------
    - f: int fold index
    - folds: iterable of (training, validation) pairs of arrays of indices
    - Xtrain: (m, n) array of training regressor data
    - Ytrain: (m, v) array of training regressand data
    - Ytrain: array training regressand
    - ridges: iterable of ridge values
    - verbose: bool of verbosity level
    - Ktrain: (m, m) array of training kernel
    - solve_kwargs: dict of addtional kwargs passed to solve_ridge
    """
    if verbose:
        training, validation = folds[f]
        lengths = (len(training), len(validation))
        fractions = utils.fraction_str(*lengths)
        amounts = '(train, validate)=' + str(lengths) + '=' + fractions
        if verbose >= 2:
            print()
        print('Fold', str(f + 1) + '/' + str(len(folds)) + ',', amounts)

    # compute results
    solve_kwargs['rescale'] = False
    fold_results = ridge.solve_ridge(
        Xtrain=Xtrain,
        Ytrain=Ytrain,
        Ktrain=Ktrain,
        ridges=ridges,
        verbose=(1 if verbose >= 2 else 0),
        weights=False,
        predictions=False,
        performance=True,
        fold=folds[f],
        **solve_kwargs
    )

    return fold_results


def nested_cv_ridge(Xtrain=None, Ytrain=None, ridges=None, weights=True,
                    predictions=True, performance=True, locality='both',
                    cluster=False, outer_folds=None, inner_folds=None,
                    outer_fold_params=None, inner_fold_params=None,
                    Ytrain_zscored=False, verbose=2, **solve_kwargs):
    """nested cross validated ridge regression

    - used when a separate test dataset is not available


    Combining Values Across Folds
    -----------------------------
    - ridge parameters: concatenate optimal ridge values for each outer fold
    - weights: rescale weights from each other fold and take mean
    - predictions: predict held-out data with rescaled weights, concatenate each
    - performance: concatenate predictions as described above, compare to Ytrain


    Parameters
    ----------
    - Xtrain: (m, n) array of training regressor data
    - Ytrain: (m, v) array of training regressand data
    - ridges: iterable of ridge values
    - weights: bool of whether to include weights in results
    - predictions: bool of whether to include predictions in results
    - performance: bool of whether to include performance in results
    - locality: str of 'global' or 'local' or 'both' for parameter selection
    - cluster: bool of whether to use cluster
    - outer_folds: iterable of (training, testing) index arrays for outer folds
    - inner_folds: iterable of (training, testing) index arrays for inner folds
    - outer_fold_params: dict of outer fold kwargs for utils.generate_folds()
    - inner_fold_params: dict of inner fold kwargs for utils.generate_folds()
    - Ytrain_zscored: bool of whether Ytrain has been zscored
    - verbose: int or bool of verbosity level
    - solve_kwargs: dict of additional kwargs for solve_ridge()
    """
    datas = {'Xtrain': Xtrain, 'Ytrain': Ytrain}
    datas = {key: value for key, value in datas.items() if value is not None}
    outputs = {
        'weights': weights,
        'predictions': predictions,
        'performance': performance,
    }

    # generate outer fold indices
    if outer_folds is None:
        if all(isinstance(data, np.ndarray) for data in datas.values()):
            # case: data is given as arrays
            if outer_fold_params is None:
                outer_fold_params = {}
            outer_fold_params.setdefault('n_folds', 10)
            outer_folds = utils.generate_folds(
                n_samples=Ytrain.shape[0],
                overlapping=False,
                **outer_fold_params
            )
        else:
            # case: data is given as iterables of arrays
            shapes = [data.shape[0] for data in datas['Ytrain']]
            lower_bounds = [0] + np.cumsum(shapes)[:-1]
            upper_bounds = np.cumsum(shapes)
            outer_folds = []
            for lower, upper in zip(lower_bounds, upper_bounds):
                validation = np.arange(lower, upper)
                training = np.arange(0, upper_bounds[-1])[~validation]
                outer_folds.append((training, validation))
            datas = {
                name: np.concatenate(data, axis=0)
                for name, data in datas.items()
            }

    # print summary
    if verbose >= 1:
        print_cv_preamble(
            ridges,
            cluster,
            locality,
            outer_folds,
            datas,
            outputs,
            **solve_kwargs
        )
        if verbose == 1:
            print()

    # compute each outer folds
    if inner_fold_params is None:
        inner_fold_params = {}
    fold_kwargs = utils.merge(
        outputs,
        inner_fold_params,
        solve_kwargs,
        datas,
        outer_folds=outer_folds,
        locality=locality,
        ridges=ridges,
        verbose=verbose,
    )
    if cluster:
        folds = [{'f': f} for f in range(len(outer_folds))]
        fold_results = cmap.cmap(
            compute_outer_fold, folds,
            common=fold_kwargs,
            verbose=False,
        )
    else:
        fold_results = []
        for f in range(len(outer_folds)):
            fold_results.append(compute_outer_fold(f=f, **fold_kwargs))

    if verbose:
        if verbose >= 2:
            print()
            print()
        print('Outer folds complete, combining results')

    # combine results
    results = {}
    get_key = lambda key: [fold_result[key] for fold_result in fold_results]
    mean = lambda key: npfast.list_mean(get_key(key))
    concatenate = lambda key: np.concatenate(get_key(key), axis=0)

    # combine lambda values
    if locality in ('local', 'both'):
        results['local_optimal_ridge'] = concatenate('local_optimal_ridge')
    if locality in ('global', 'both'):
        results['global_optimal_ridge'] = np.array(get_key('global_optimal_ridge'))

    # weights is average of each fold's weights
    if weights:
        if locality in ('local', 'both'):
            results['local_weights'] = mean('local_weights')
        if locality in ('global', 'both'):
            results['global_weights'] = mean('global_weights')

    # predictions is concatenation of each fold's prediction
    if predictions or performance:
        if locality in ('local', 'both'):
            local_predictions = concatenate('local_predictions')
        if locality in ('global', 'both'):
            global_predictions = concatenate('global_predictions')

    # save predictions if need be
    if predictions:
        if locality in ('local', 'both'):
            results['local_predictions'] = local_predictions
        if locality in ('global', 'both'):
            results['global_predictions'] = global_predictions

    # performance is the concatenated predictions correlated with training data
    if performance:
        if not Ytrain_zscored:
            Ytrain = scipy.stats.zscore(Ytrain)

        correlate = lambda Yhat: npfast.correlate(
            Yhat,
            Ytrain,
            axis=0,
            zscore_right=False,
        )

        if 'global_predictions' in results:
            results['global_performance'] = correlate(global_predictions)
        if 'local_predictions' in results:
            results['local_performance'] = correlate(local_predictions)

    return results


def compute_outer_fold(f, outer_folds, ridges, weights, predictions,
                       performance, Xtrain=None, Ytrain=None, verbose=False,
                       Ktrain=None, **solve_kwargs):
    """compute an outer fold of nested cross validation

    Parameters
    ----------
    - f: int fold index
    - outer_folds: list of (training, validation) pairs of arrays of indices
    - ridges: iterable of ridge values
    - weights: bool of whether to include weights in results
    - predictions: bool of whether to include predictions in results
    - performance: bool of whether to include performance in results
    - Xtrain: (m, n) array of training regressor data
    - Ytrain: (m, v) array of training regressand data
    - verbose: bool or int of verbosity level
    - Ktrain: (m, m) array of training kernel
    - solve_kwargs: dict of additional kwargs for solve_ridge()
    """
    training, validation = outer_folds[f]

    if verbose:
        training, validation = outer_folds[f]
        lengths = (len(training), len(validation))
        fractions = utils.fraction_str(*lengths)
        amounts = '(train, validate)=' + str(lengths) + '=' + fractions
        if verbose >= 2:
            print()
            print()
        print('Outer Fold', str(f + 1) + '/' + str(len(outer_folds)) + ',', amounts)

    folded_datas = utils.fold_data(
        outer_folds[f],
        {'Xtrain': Xtrain, 'Ytrain': Ytrain, 'Ktrain': Ktrain},
    )

    if verbose >= 2:
        inner_verbose = 2
    else:
        inner_verbose = 0
    fold_result = cv_ridge(
        ridges=ridges,
        weights=weights,
        predictions=(predictions or performance),
        performance=False,
        verbose=inner_verbose,
        rescale=True,
        **utils.merge(folded_datas, solve_kwargs)
    )

    return fold_result
