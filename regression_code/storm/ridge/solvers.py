"""solvers for ridge regression problem


Solvers
-------
- solvers differ in speed and memory usage
    - solve_ridge() heuristically picks the most efficent solver if solver=None
    - solver heuristics depend on:
        - n: number of regressors
        - v: number of regressands
        - m: number of training samples
        - s: number of testing samples
        - r: number of ridge values


Implementation
--------------
- each solver type is organized into two functions
    - factorize(): decomposes data matrices into factors
    - solve(): uses the decomposed factors to compute results
- factor computation is organized in this way to help with:
    - reuse across multiple calls to solve_ridge()
    - minimization of data transfer in distributed contexts
    - explicit book keeping of data used by each sovler
- can compute factors easily using ridge.solve_ridge(..., factors_only=True)
- these functions are intended to be used by ridge.solve_ridge()
    - need to preintialize outputs and provide proper set of factors
- solvers assume:
    - outputs are already initialized
    - Ytest is zscored


Optimizations
-------------
- memory for results is preallocated and populated using np.dot's out argument
- use npfast.multi_dot for matrix chain multiplication
- use npfast.correlate for computing performance
- use different array factors when computing single-ridge solutions
- touch Y-sized matrices as little as possible
- solvers assume
    - (n_regressands >> n_regressors) and (n_regressands >> n_samples)
    - (n_training_samples > n_testing_samples)
    - (n_ridges == 1) or (n_ridges >> 1)
- see "Improving Performance" in docstring of solve_ridge()


Formulations
------------
- primal:
    - B = (XT X + aI)-1 XT Y
    - Yhat = Xtest B
           = Xtest (XT X + aI)-1 XT Y
- dual:
    - A = (X XT + aI)-1 Y
        = (K + aI)-1
    - B = XT A
        = XT (K + aI)-1 Y
    - Yhat = Xtest B
           = Xtest XT A
           = Ktest (K + aI)-1 Y
    - Ktrain = Xtrain Xtrain.T
    - Ktest = Xtest Xtrain.T
- svd:
    - X = U S V
    - B = V S (S2 + ridge ** 2)-1 UT Y
    - Yhat = Xtest V S (S2 + ridge ** 2)-1 UT Y
- eig:
    - XTX = V L VT
    - D = 1 / (L + ridge ** 2)
    - B = V D VT XT Y
    - Yhat = Xtest V D VT XT Y
- eig_dual:
    - XXT = V L VT
    - D = 1 / (L + ridge ** 2)
    - B = XT V D VT Y
    - Yhat = Xtest XT V D VT Y
- cho:
    - (XTX + aI) Z = XT
    - Z = (XTX + aI)-1 XT
    - B = Z Y
    - Yhat = Xtest Z Y
- cho_dual:
    - weights
        - (Ktrain + aI)T ZT = X
        - Z = XT (Ktrain + aI)-1
        - B = Z Y
        - Yhat = Xtest Z Y
    - no weights
        - (Ktrain + aI)T ZT = KtestT
        - Z = Ktest (Ktrain + aI)-1
        - Yhat = Z Y
- qr:
    - (XTX + aI) Z = XT
    - (XTX + aI) = QR
    - R Z = QT XT
    - Z = (XTX + aI)-1 XT
    - B = Z Y
    - Yhat = Xtest Z Y
- qr_dual:
    - weights
        - (Ktrain + aI)T ZT = X
        - (Ktrain + aI)T = QR
        - R ZT = QT X
        - Z = XT (Ktrain + aI)-1
        - B = Z Y
        - Yhat = Xtest Z Y
    - no weights
        - (Ktrain + aI)T ZT = KtestT
        - (Ktrain + aI)T = QR
        - R ZT = QT Ktest
        - Z = Ktest (Ktrain + aI)-1
        - Yhat = Z Y
- lu:
    - (XTX + aI) Z = XT
    - (XTX + aI) = PLU
    - L U Z = PT XT
    - Z = (XTX + aI)-1 XT
    - B = Z Y
    - Yhat = Xtest Z Y
- lu_solve:
    - weights
        - (Ktrain + aI)T ZT = X
        - (Ktrain + aI)T = PLU
        - L U ZT = PT X
        - Z = XT (Ktrain + aI)-1
        - B = Z Y
        - Yhat = Xtest Z Y
    - no weights
        - (Ktrain + aI)T ZT = KtestT
        - (Ktrain + aI)T = PLU
        - L U ZT = PT Ktest
        - Z = Ktest (Ktrain + aI)-1
        - Yhat = Z Y
- inv:
    - B = (XTX + ridge ** 2 * I)-1 XT Y
    - Yhat = Xtest (XTX + ridge ** 2 * I)-1 XT Y
- inv_dual:
    - B = XT (XXT + ridge ** 2 * I)-1 Y
    - Yhat = Xtest XT (XXT + ridge ** 2 * I)-1 Y


Functions Used
--------------
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
"""

from __future__ import print_function

import numpy as np
import scipy

from .. import npfast
from .. import utils


eps = 1e-10


solvers = [
    'svd',
    'eig',
    'eig_dual',
    'cho',
    'cho_dual',
    'qr',
    'qr_dual',
    'lu',
    'lu_dual',
    'inv',
    'inv_dual',
]


def compute_performance(Ytest, Yhat, metric='r'):
    """compute performance using a variety of performance metrics

    Parameters
    ----------
    - Ytest: array of observations
    - Yhat: array of model predictions
    - metric: str of performance metric to use
    """
    # case: person's r
    if metric == 'r':
        return npfast.correlate(Yhat, Ytest, axis=0, zscore_right=False)

    # case: pearson's r with higher precision
    elif metric == 'r_precise':
        Yhat = scipy.stats.zscore(Yhat)
        Yhat *= Ytest
        return Yhat.mean(0)

    # case: coefficient of determination
    elif metric == 'R^2':
        import warnings
        warnings.warn('need to debug R^2 computations')
        return npfast.R_squared(Yhat=Yhat, Y=Ytest, axis=0, unit_variance=True)


def print_performance(performance, index, ridges):
    """print performance with percentiles

    - does not use np.percentile because it doesn't handle nans

    Parameters
    ----------
    - performance: array of performance correlation
    - index: int ridge parameter index
    - ridges: iterable of ridge values
    """
    # compute formatting parameters
    index_digits = int(np.floor(np.log10(len(ridges) + 1)) + 1)
    ridge_digits = int(max(np.floor(np.log10(max(np.abs(ridges)))), 0) + 2)
    if isinstance(index, (int, float)):
        index_template = '{:0' + str(index_digits) + '}'
        index_str = index_template.format(index + 1) + ': '
        ridge_template = '{0:' + str(ridge_digits) + '.1f}'
        ridge_str = ridge_template.format(ridges[index])[:(ridge_digits)] + ' '
    elif isinstance(index, str):
        index_str = ' ' * index_digits + '  '
        ridge_str = index.rjust(ridge_digits)[:ridge_digits] + ' '
    else:
        raise Exception('index not understood')

    # compute quantiles
    quantile_fractions = [.05, .25, .50, .75, .95]
    indices = np.floor(performance.shape[0] * np.array(quantile_fractions))
    indices = indices.astype(int)
    if len(indices) > len(performance):
        performance = np.concatenate((performance, [0] * (len(indices) - len(performance))))
    quantiles = performance[np.argsort(performance)[indices]]
    quantile_strs = []
    for quantile in quantiles:
        quantile_strs.append('{0:+0.03f}'.format(quantile))
    quantile_str = '(5%, 25%, 50%, 75%, 95%)=(' + ', '.join(quantile_strs) + ')'

    # print performance
    print(index_str, ridge_str, quantile_str)


def get_solver(solver, ridges, Xtrain=None, Ytrain=None, Xtest=None, Ytest=None,
               Ktrain=None, Ktest=None):
    """return appropriate factorize() and solve() functions

    Parameters
    ----------
    - sovler: str name of sovler to get, if None will determine using heuristics
    - [see solve_ridge() for others]

    Returns
    -------
    - sovler: str name of solver being returned
    - (factorize, solve): functions for solving, work as solve(factorize(...))
    """
    # determine solver if not specified
    if solver is None:
        if Xtrain is not None:
            m, n = Xtrain.shape
        else:
            m = Ktrain.shape[0]
            n = None

        if len(ridges) == 1 and min(ridges) >= 1:
            if (Ktrain is not None) or (Ktest is not None) or (m >= n):
                solver = 'cho_dual'
            else:
                solver = 'cho'
        elif (Ktrain is not None) or (Ktest is not None) or (m > n):
            solver = 'eig_dual'
        else:
            solver = 'svd'

    functions = {
        'svd': (svd_factorize, svd_solve),
        'eig': (eig_factorize, eig_solve),
        'eig_dual': (eig_dual_factorize, eig_dual_solve),
        'cho': (cho_factorize, cho_solve),
        'cho_dual': (cho_dual_factorize, cho_dual_solve),
        'qr': (cho_factorize, qr_solve),
        'qr_dual': (cho_dual_factorize, qr_dual_solve),
        'lu': (cho_factorize, lu_solve),
        'lu_dual': (cho_dual_factorize, lu_dual_solve),
        'inv': (inv_factorize, inv_solve),
        'inv_dual': (inv_dual_factorize, inv_dual_solve),
    }

    if solver not in functions:
        raise Exception('solver not found: ' + str(solver))

    return solver, functions[solver]


def svd_factorize(outputs, ridges, Xtrain, Ytrain, Xtest, Ytest):
    """return factors needed to solve ridge problem via svd_solve()

    Parameters
    ----------
    - [see solve_ridge()]
    """
    U, S, VT = np.linalg.svd(Xtrain, full_matrices=False)
    V = VT.T
    keep = S > eps
    U = U[:, keep]
    S = S[keep]
    V = V[:, keep]

    factors = {'S': S}

    if len(ridges) > 1:
        factors['UTY'] = np.dot(U.T, Ytrain)
    else:
        factors['U'] = U
        factors['Ytrain'] = Ytrain

    if 'weights' in outputs:
        factors['V'] = V

    if ('predictions' in outputs) or ('performance' in outputs):
        if 'weights' in outputs:
            factors['Xtest'] = Xtest
        else:
            factors['XtestV'] = np.dot(Xtest, V)

    if 'performance' in outputs:
        factors['Ytest'] = Ytest

    return factors


def svd_solve(ridges, S, Ytrain=None, Xtest=None, XtestV=None, Ytest=None,
              V=None, U=None, UTY=None, weights=None, predictions=None,
              performance=None, verbose=False, metric='r'):
    """solve ridge problem using singular value decomposition

    Parameters
    ----------
    - [see solve_ridge() and svd_factorize()]
    """
    outputs = {
        'weights': weights,
        'predictions': predictions,
        'performance': performance,
    }

    for r, ridge in enumerate(ridges):
        # update output locations
        out = {k: v[r, ...] for k, v in outputs.items() if v is not None}
        dot = lambda *args: npfast.multi_dot(*args[:-1], out=out.get(args[-1]))

        # transform singular values
        D = S / (S ** 2 + ridge ** 2)

        # compute model weights
        if weights is not None:
            if UTY is not None:
                B = dot(V * D, UTY, 'weights')
            else:
                B = dot(V * D, U.T, Ytrain, 'weights')
        else:
            B = None

        # compute model predictions
        if (predictions is not None) or (performance is not None):
            if weights is not None:
                Yhat = dot(Xtest, B, 'predictions')
            else:
                if UTY is not None:
                    Yhat = dot(XtestV * D, UTY, 'predictions')
                else:
                    Yhat = dot(XtestV * D, U.T, Ytrain, 'predictions')

        # compute model performance
        if performance is not None:
            out['performance'][:] = compute_performance(Ytest, Yhat, metric)
            if verbose >= 1:
                print_performance(out['performance'], r, ridges)


def eig_factorize(outputs, ridges, Xtrain, Ytrain, Xtest, Ytest):
    """return factors needed to solve ridge problem via eig_solve()

    Parameters
    ----------
    - [see solve_ridge()]
    """
    factors = {}

    L, V = np.linalg.eigh(np.dot(Xtrain.T, Xtrain))
    keep = L > eps
    L = L[keep]
    V = V[:, keep]

    factors = {'L': L}

    if len(ridges) > 1:
        factors['VTXTY'] = np.linalg.multi_dot([V.T, Xtrain.T, Ytrain])
    else:
        factors['Ytrain'] = Ytrain
        factors['VTXT'] = V.T.dot(Xtrain.T)

    if 'weights' in outputs:
        factors['V'] = V

    if ('predictions' in outputs) or ('performance' in outputs):
        if 'weights' in outputs:
            factors['Xtest'] = Xtest
        else:
            factors['XtestV'] = Xtest.dot(V)

    if 'performance' in outputs:
        factors['Ytest'] = Ytest

    return factors


def eig_solve(ridges, L, Ytrain=None, Xtest=None, XtestV=None, Ytest=None,
              V=None, VTXT=None, VTXTY=None, weights=None, predictions=None,
              performance=None, verbose=False, metric='r'):
    """solve ridge problem using eigendecomposition

    Parameters
    ----------
    - [see solve_ridge() and eig_factorize()]
    """
    outputs = {
        'weights': weights,
        'predictions': predictions,
        'performance': performance,
    }

    for r, ridge in enumerate(ridges):
        # update output locations
        out = {k: v[r, ...] for k, v in outputs.items() if v is not None}
        dot = lambda *args: npfast.multi_dot(*args[:-1], out=out.get(args[-1]))

        # transform eigenvalues
        D = 1 / (L + ridge ** 2)

        # compute model weights
        if (weights is not None):
            if VTXTY is not None:
                B = dot(V * D, VTXTY, 'weights')
            else:
                B = dot((V * D), VTXT, Ytrain, 'weights'),
        else:
            B = None

        # compute model predictions
        if (predictions is not None) or (performance is not None):
            if B is not None:
                Yhat = dot(Xtest, B, 'predictions')
            elif VTXTY is not None:
                Yhat = dot(XtestV * D, VTXTY, 'predictions')
            else:
                Yhat = dot(XtestV * D, VTXT, Ytrain, 'predictions')

        # compute model performance
        if performance is not None:
            out['performance'][:] = compute_performance(Ytest, Yhat, metric)
            if verbose >= 1:
                print_performance(out['performance'], r, ridges)


def eig_dual_factorize(outputs, ridges, Ytrain, Ytest, Xtrain=None, Xtest=None,
                       Ktrain=None, Ktest=None):
    """return factors needed to solve ridge problem via kernel_solve()

    Parameters
    ----------
    - [see solve_ridge()]
    """

    # compute kernels if need be
    if Ktrain is None:
        Ktrain = utils.linear_kernel(Xtrain, Xtrain)
    if Ktest is None:
        Ktest = utils.linear_kernel(Xtest, Xtrain)

    L, V = np.linalg.eigh(Ktrain)
    keep = L > eps
    L = L[keep]
    V = V[:, keep]

    factors = {'L': L}

    if len(ridges) > 1:
        factors['VTY'] = np.dot(V.T, Ytrain)
    else:
        factors['V'] = V
        factors['Ytrain'] = Ytrain

    if 'weights' in outputs:
        factors['XTV'] = Xtrain.T.dot(V)

    if ('predictions' in outputs) or ('performance' in outputs):
        if 'weights' in outputs:
            factors['Xtest'] = Xtest
        else:
            if Ktest is None:
                Ktest = Xtest.dot(Xtrain.T)
            factors['KtestV'] = Ktest.dot(V)

    if 'performance' in outputs:
        factors['Ytest'] = Ytest

    # factors['Ktest'] = Ktest
    return factors


def eig_dual_solve(ridges, L, Ytrain=None, Xtest=None, Ktest=None, KtestV=None,
                   Ytest=None, V=None, XTV=None, VTY=None, weights=None,
                   predictions=None, performance=None, verbose=False,
                   metric='r'):
    """solve ridge problem using eigendecomposition of dual formulation

    Parameters
    ----------
    - [see solve_ridge() and eig_dual_factorize()]
    """
    outputs = {
        'weights': weights,
        'predictions': predictions,
        'performance': performance,
    }

    for r, ridge in enumerate(ridges):
        # update output locations
        out = {k: v[r, ...] for k, v in outputs.items() if v is not None}
        dot = lambda *args: npfast.multi_dot(*args[:-1], out=out.get(args[-1]))

        # transform eigenvalues
        D = 1 / (L + ridge ** 2)

        # compute kernel weights and model weights
        if weights is not None:
            if VTY is not None:
                B = dot(XTV * D, VTY, 'weights')
            else:
                B = dot(XTV * D, V.T, Ytrain, 'weights')
        else:
            B = None

        # compute model predictions
        if (performance is not None) or (predictions is not None):
            if B is not None:
                Yhat = dot(Xtest, B, 'predictions')
            elif VTY is not None:
                Yhat = dot(KtestV * D, VTY, 'predictions')
            else:
                Yhat = dot(KtestV * D, V.T, Ytrain, 'predictions')

        # compute model performance
        if performance is not None:
            out['performance'][:] = compute_performance(Ytest, Yhat, metric)
            if verbose >= 1:
                print_performance(out['performance'], r, ridges)


def cho_factorize(outputs, ridges, Xtrain, Ytrain, Xtest, Ytest):
    """return factors needed to solve ridge problem via cho_solve()

    Parameters
    ----------
    - [see solve_ridge()]
    """
    factors = {
        'Xtrain': Xtrain,
        'Ytrain': Ytrain,
        'XTX': Xtrain.T.dot(Xtrain),
    }

    if ('predictions' in outputs) or ('performance' in outputs):
        factors['Xtest'] = Xtest

    if 'performance' in outputs:
        factors['Ytest'] = Ytest

    return factors


def cho_solve(ridges, Xtrain, Ytrain, XTX, Xtest=None, Ytest=None, weights=None,
              predictions=None, performance=None, verbose=False, metric='r'):
    """solve ridge problem using cholesky decomposition

    Formulation
    -----------
    (XTX + aI) Z = XT
    -> Z = (XTX + aI)-1 XT
    -> B = Z Y
    -> Yhat = Xtest Z Y

    Parameters
    ----------
    - [see solve_ridge() and cho_factorize()]
    """
    outputs = {
        'weights': weights,
        'predictions': predictions,
        'performance': performance,
    }
    for value in outputs.values():
        if value is not None:
            dtype = value.dtype
            break
    else:
        dtype = Xtrain.dtype

    for r, ridge in enumerate(ridges):
        # update output locations
        out = {k: v[r, ...] for k, v in outputs.items() if v is not None}
        dot = lambda *args: npfast.multi_dot(*args[:-1], out=out.get(args[-1]))

        # compute Z factor
        XTX_ridged = np.eye(*XTX.shape)
        XTX_ridged *= ridge ** 2
        XTX_ridged += XTX
        Z = scipy.linalg.solve(
            XTX_ridged,
            Xtrain.T,
            overwrite_a=True,
            sym_pos=True,
        )
        if Z.dtype != dtype:
            Z = Z.astype(dtype)

        # compute weights
        if weights is not None:
            B = dot(Z, Ytrain, 'weights')
        else:
            B = None

        # compute predictions
        if (predictions is not None) or (performance is not None):
            if B is not None:
                Yhat = dot(Xtest, B, 'predictions')
            else:
                Yhat = dot(Xtest, Z, Ytrain, 'predictions')

        # compute model performance
        if performance is not None:
            out['performance'][:] = compute_performance(Ytest, Yhat, metric)
            if verbose >= 1:
                print_performance(out['performance'], r, ridges)


def cho_dual_factorize(outputs, ridges, Xtrain=None, Ytrain=None, Xtest=None,
                       Ytest=None, Ktrain=None, Ktest=None):
    """return factors needed to solve ridge problem via cho_dual_solve()

    Parameters
    ----------
    - [see solve_ridge()]
    """
    if Ktrain is None:
        Ktrain = Xtrain.dot(Xtrain.T)

    factors = {
        'Xtrain': Xtrain,
        'Ytrain': Ytrain,
        'Ktrain': Ktrain,
    }

    if ('predictions' in outputs) or ('performance' in outputs):
        if 'weights' in outputs:
            factors['Xtest'] = Xtest
        else:
            if Ktest is None:
                Ktest = Xtest.dot(Xtrain.T)
            factors['Ktest'] = Ktest

    if 'performance' in outputs:
        factors['Ytest'] = Ytest

    return factors


def cho_dual_solve(ridges, Xtrain=None, Ktrain=None, Ytrain=None, Xtest=None,
                   Ktest=None, Ytest=None, weights=None, predictions=None,
                   performance=None, verbose=False, metric='r'):
    """solve ridge problem using cholesky decomposition

    Formulation
    -----------
    - weights
        (Ktrain + aI)T ZT = X
        -> Z = XT (Ktrain + aI)-1
        -> B = Z Y
        -> Yhat = Xtest Z Y
    - no weights
        (Ktrain + aI)T ZT = KtestT
        -> Z = Ktest (Ktrain + aI)-1
        -> Yhat = Z Y

    Parameters
    ----------
    - [see solve_ridge() and cho_dual_factorize()]
    """
    outputs = {
        'weights': weights,
        'predictions': predictions,
        'performance': performance,
    }
    for value in outputs.values():
        if value is not None:
            dtype = value.dtype
            break

    for r, ridge in enumerate(ridges):
        # update output locations
        out = {k: v[r, ...] for k, v in outputs.items() if v is not None}
        dot = lambda *args: npfast.multi_dot(*args[:-1], out=out.get(args[-1]))

        # compute ridged kernel
        Ktrain_ridged = np.eye(*Ktrain.shape)
        Ktrain_ridged *= ridge ** 2
        Ktrain_ridged += Ktrain

        # formulate as Z = XT (Ktrain + aI)-1
        if weights is not None:
            ZT = scipy.linalg.solve(
                Ktrain_ridged.T,
                Xtrain,
                sym_pos=True,
                overwrite_a=True,
            )
            if ZT.dtype != dtype:
                ZT = ZT.astype(dtype)
            B = dot(ZT.T, Ytrain, 'weights')
            if (predictions is not None) or (performance is not None):
                Yhat = dot(Xtest, B, 'predictions')

        # formulate as Z = Ktest (Ktrain + aI)-1
        elif (predictions is not None) or (performance is not None):
            ZT = scipy.linalg.solve(
                Ktrain_ridged.T,
                Ktest.T,
                sym_pos=True,
                overwrite_a=True,
            )
            if ZT.dtype != dtype:
                ZT = ZT.astype(dtype)
            Yhat = dot(ZT.T, Ytrain, 'predictions')

        # compute model performance
        if performance is not None:
            out['performance'][:] = compute_performance(Ytest, Yhat, metric)
            if verbose >= 1:
                print_performance(out['performance'], r, ridges)


def qr_solve(ridges, Xtrain, Ytrain, XTX, Xtest=None, Ytest=None,
             weights=None, predictions=None, performance=None,
             verbose=False, metric='r'):
    """solve ridge problem using QR decomposition

    Formulation
    -----------
    (XTX + aI) Z = XT
    QR = (XTX + aI)
    R Z = QT XT
    -> Z = (XTX + aI)-1 XT
    -> B = Z Y
    -> Yhat = Xtest Z Y

    Parameters
    ----------
    - [see solve_ridge() and cho_factorize()]
    """
    outputs = {
        'weights': weights,
        'predictions': predictions,
        'performance': performance,
    }

    for r, ridge in enumerate(ridges):
        # update output locations
        out = {k: v[r, ...] for k, v in outputs.items() if v is not None}
        dot = lambda *args: npfast.multi_dot(*args[:-1], out=out.get(args[-1]))

        # compute Z factor
        XTX_ridged = np.eye(*XTX.shape)
        XTX_ridged *= ridge ** 2
        XTX_ridged += XTX
        Q, R = np.linalg.qr(XTX_ridged)
        Z = scipy.linalg.solve_triangular(
            R,
            np.dot(Q.T, Xtrain.T),
            overwrite_b=True,
        )

        # compute weights
        if weights is not None:
            B = dot(Z, Ytrain, 'weights')
        else:
            B = None

        # compute predictions
        if (predictions is not None) or (performance is not None):
            if B is not None:
                Yhat = dot(Xtest, B, 'predictions')
            else:
                Yhat = dot(Xtest, Z, Ytrain, 'predictions')

        # compute model performance
        if performance is not None:
            out['performance'][:] = compute_performance(Ytest, Yhat, metric)
            if verbose >= 1:
                print_performance(out['performance'], r, ridges)


def qr_dual_solve(ridges, Xtrain=None, Ktrain=None, Ytrain=None, Xtest=None,
                  Ktest=None, Ytest=None, weights=None, predictions=None,
                  performance=None, verbose=False, metric='r'):
    """solve ridge problem using QR decomposition of dual formulation

    Formulation
    -----------
    - weights
        (Ktrain + aI)T ZT = X
        -> Z = XT (Ktrain + aI)-1
        -> B = Z Y
        -> Yhat = Xtest Z Y
    - no weights
        (Ktrain + aI)T ZT = KtestT
        -> Z = Ktest (Ktrain + aI)-1
        -> Yhat = Z Y

    Parameters
    ----------
    - [see solve_ridge() and cho_dual_factorize()]
    """
    outputs = {
        'weights': weights,
        'predictions': predictions,
        'performance': performance,
    }

    for r, ridge in enumerate(ridges):
        # update output locations
        out = {k: v[r, ...] for k, v in outputs.items() if v is not None}
        dot = lambda *args: npfast.multi_dot(*args[:-1], out=out.get(args[-1]))

        # compute ridged kernel
        Ktrain_ridged = np.eye(*Ktrain.shape)
        Ktrain_ridged *= ridge ** 2
        Ktrain_ridged += Ktrain
        Q, R = np.linalg.qr(Ktrain_ridged.T)

        # formulate as Z = XT (Ktrain + aI)-1
        if weights is not None:
            ZT = scipy.linalg.solve_triangular(
                R,
                Q.T.dot(Xtrain),
                overwrite_b=True,
            )
            B = dot(ZT.T, Ytrain, 'weights')
            if (predictions is not None) or (performance is not None):
                Yhat = dot(Xtest, B, 'predictions')

        # formulate as Z = Ktest (Ktrain + aI)-1
        elif (predictions is not None) or (performance is not None):
            ZT = scipy.linalg.solve_triangular(
                R,
                Q.T.dot(Ktest.T),
                overwrite_b=True,
            )
            Yhat = dot(ZT.T, Ytrain, 'predictions')

        # compute model performance
        if performance is not None:
            out['performance'][:] = compute_performance(Ytest, Yhat, metric)
            if verbose >= 1:
                print_performance(out['performance'], r, ridges)


def qr_lls(X, Y, overwrite_Y=False, check_finite=True):
    """compute linear least squares solution to Y = XB using QR decomposition

    Parameters
    ----------
    - X: (n_samples, n_regressors) array
    - Y: (n_samples, n_regressands) array
    - overwrite_Y: bool of whether to pollute Y for performance gain
    - check_finite: bool of whether to check for nan's and inf's in X and Y
    """
    m, n = X.shape

    # case: skinny matrix
    if m >= n:
        (R,) = scipy.linalg.qr(X, mode='r')
        XTY = np.dot(X.T, Y)
        intermediate = scipy.linalg.solve_triangular(
            R[:n, :].T,
            XTY,
            overwrite_b=True,
            lower=True,
            check_finite=check_finite,
        )
        B = scipy.linalg.solve_triangular(
            R[:n, :],
            intermediate,
            check_finite=check_finite,
        )

    # case: fat matrix
    else:
        (R,) = scipy.linalg.qr(X.T, mode='r')
        intermediate = scipy.linalg.solve_triangular(
            R[:m, :].T,
            Y,
            overwrite_b=overwrite_Y,
            lower=True,
            check_finite=check_finite,
        )
        A = scipy.linalg.solve_triangular(
            R[:m, :],
            intermediate,
            check_finite=check_finite,
        )
        B = np.dot(X.T, A)

    return B


def lu_solve(ridges, Xtrain, Ytrain, XTX, Xtest=None, Ytest=None,
             weights=None, predictions=None, performance=None,
             verbose=False, metric='r'):
    """solve ridge problem using LU decomposition

    Formulation
    -----------
    (XTX + aI) Z = XT
    P L U = (XTX + aI)
    L U Z = PT XT
    -> Z = (XTX + aI)-1 XT
    -> B = Z Y
    -> Yhat = Xtest Z Y

    Parameters
    ----------
    - [see solve_ridge() and cho_factorize()]
    """
    outputs = {
        'weights': weights,
        'predictions': predictions,
        'performance': performance,
    }

    for r, ridge in enumerate(ridges):
        # update output locations
        out = {k: v[r, ...] for k, v in outputs.items() if v is not None}
        dot = lambda *args: npfast.multi_dot(*args[:-1], out=out.get(args[-1]))

        # compute Z factor
        XTX_ridged = np.eye(*XTX.shape)
        XTX_ridged *= ridge ** 2
        XTX_ridged += XTX

        LU, P = scipy.linalg.lu_factor(XTX_ridged)
        Z = scipy.linalg.lu_solve((LU, P), Xtrain.T)

        # compute weights
        if weights is not None:
            B = dot(Z, Ytrain, 'weights')
        else:
            B = None

        # compute predictions
        if (predictions is not None) or (performance is not None):
            if B is not None:
                Yhat = dot(Xtest, B, 'predictions')
            else:
                Yhat = dot(Xtest, Z, Ytrain, 'predictions')

        # compute model performance
        if performance is not None:
            out['performance'][:] = compute_performance(Ytest, Yhat, metric)
            if verbose >= 1:
                print_performance(out['performance'], r, ridges)


def lu_dual_solve(ridges, Xtrain=None, Ktrain=None, Ytrain=None, Xtest=None,
                  Ktest=None, Ytest=None, weights=None, predictions=None,
                  performance=None, verbose=False, metric='r'):
    """solve ridge problem using LU decomposition of dual formulation

    Formulation
    -----------
    - weights
        (Ktrain + aI)T ZT = X
        -> Z = XT (Ktrain + aI)-1
        -> B = Z Y
        -> Yhat = Xtest Z Y
    - no weights
        (Ktrain + aI)T ZT = KtestT
        -> Z = Ktest (Ktrain + aI)-1
        -> Yhat = Z Y

    Parameters
    ----------
    - [see solve_ridge() and cho_dual_factorize()]
    """
    outputs = {
        'weights': weights,
        'predictions': predictions,
        'performance': performance,
    }

    for r, ridge in enumerate(ridges):
        # update output locations
        out = {k: v[r, ...] for k, v in outputs.items() if v is not None}
        dot = lambda *args: npfast.multi_dot(*args[:-1], out=out.get(args[-1]))

        # compute ridged kernel
        Ktrain_ridged = np.eye(*Ktrain.shape)
        Ktrain_ridged *= ridge ** 2
        Ktrain_ridged += Ktrain
        LU, P = scipy.linalg.lu_factor(Ktrain_ridged.T)

        # formulate as Z = XT (Ktrain + aI)-1
        if weights is not None:
            ZT = scipy.linalg.lu_solve((LU, P), Xtrain)
            B = dot(ZT.T, Ytrain, 'weights')
            if (predictions is not None) or (performance is not None):
                Yhat = dot(Xtest, B, 'predictions')

        # formulate as Z = Ktest (Ktrain + aI)-1
        elif (predictions is not None) or (performance is not None):
            ZT = scipy.linalg.lu_solve((LU, P), Ktest.T)
            Yhat = dot(ZT.T, Ytrain, 'predictions')

        # compute model performance
        if performance is not None:
            out['performance'][:] = compute_performance(Ytest, Yhat, metric)
            if verbose >= 1:
                print_performance(out['performance'], r, ridges)


def inv_factorize(outputs, ridges, Xtrain, Ytrain, Xtest, Ytest, Ktrain=None,
                  Ktest=None):
    """return factors needed to solve ridge problem via inv_solve()

    Parameters
    ----------
    - [see solve_ridge()]
    """
    factors = {'XTX': Xtrain.T.dot(Xtrain)}

    if len(ridges) > 1:
        factors['XTY'] = Xtrain.T.dot(Ytrain)
    else:
        factors['Xtrain'] = Xtrain
        factors['Ytrain'] = Ytrain

    if ('predictions' in outputs) or ('performance' in outputs):
        factors['Xtest'] = Xtest

    if 'performance' in outputs:
        factors['Ytest'] = Ytest

    return factors


def inv_solve(ridges, XTX, Xtrain=None, Ytrain=None, Xtest=None, Ytest=None,
              XTY=None, weights=None, predictions=None, performance=None,
              verbose=False, metric='r'):
    """solve ridge problem using cho inversion

    Parameters
    ----------
    - [see solve_ridge() and inv_factorize()]
    """
    outputs = {
        'weights': weights,
        'predictions': predictions,
        'performance': performance,
    }

    for r, ridge in enumerate(ridges):
        # update output locations
        out = {k: v[r, ...] for k, v in outputs.items() if v is not None}
        dot = lambda *args: npfast.multi_dot(*args[:-1], out=out.get(args[-1]))

        # compute weights
        XTX_ridged = np.eye(*XTX.shape)
        XTX_ridged *= ridge ** 2
        XTX_ridged += XTX
        XTX_inv = np.linalg.inv(XTX_ridged)
        if weights is not None:
            if XTY is not None:
                B = dot(XTX_inv, XTY, 'weights')
            else:
                B = dot(XTX_inv, Xtrain.T, Ytrain, 'weights')
        else:
            B = None

        # compute predictions
        if (predictions is not None) or (performance is not None):
            if B is not None:
                Yhat = dot(Xtest, B, 'predictions')
            elif XTY is not None:
                Yhat = dot(Xtest, XTX_inv, XTY, 'predictions')
            else:
                Yhat = dot(Xtest, XTX_inv, Xtrain.T, Ytrain, 'predictions')

        # compute model performance
        if performance is not None:
            out['performance'][:] = compute_performance(Ytest, Yhat, metric)
            if verbose >= 1:
                print_performance(out['performance'], r, ridges)


def inv_dual_factorize(outputs, ridges, Xtrain=None, Ytrain=None, Xtest=None,
                       Ytest=None, Ktrain=None, Ktest=None):
    """return factors needed to solve ridge problem via inv_dual_solve()

    Parameters
    ----------
    - [see solve_ridge()]
    """
    if Ktrain is None:
        Ktrain = Xtrain.dot(Xtrain.T)

    factors = {
        'Ytrain': Ytrain,
        'Ktrain': Ktrain,
    }

    if 'weights' in outputs:
        factors['Xtrain'] = Xtrain

    if ('predictions' in outputs) or ('performance' in outputs):
        if 'weights' in outputs:
            factors['Xtest'] = Xtest
        else:
            if Ktest is None:
                Ktest = Xtest.dot(Xtrain.T)
            factors['Ktest'] = Ktest

    if 'performance' in outputs:
        factors['Ytest'] = Ytest

    return factors


def inv_dual_solve(ridges, Ytrain, Ktrain, Xtrain=None, Ktest=None, Xtest=None,
                   Ytest=None, weights=None, predictions=None, performance=None,
                   verbose=False, metric='r'):
    """solve ridge problem using cho inversion of dual formulation

    Parameters
    ----------
    - [see solve_ridge() and inv_factorize()]
    """
    outputs = {
        'weights': weights,
        'predictions': predictions,
        'performance': performance,
    }

    for r, ridge in enumerate(ridges):
        # update output locations
        out = {k: v[r, ...] for k, v in outputs.items() if v is not None}
        dot = lambda *args: npfast.multi_dot(*args[:-1], out=out.get(args[-1]))

        # compute inversion factor
        Ktrain_ridged = np.eye(*Ktrain.shape)
        Ktrain_ridged *= ridge ** 2
        Ktrain_ridged += Ktrain
        Ktrain_inv = np.linalg.inv(Ktrain_ridged)

        # compute weights
        if weights is not None:
            B = dot(Xtrain.T, Ktrain_inv, Ytrain, 'weights')
        else:
            B = None

        # compute predictions
        if (predictions is not None) or (performance is not None):
            if B is not None:
                Yhat = dot(Xtest, B, 'predictions')
            else:
                Yhat = dot(Ktest, Ktrain_inv, Ytrain, 'predictions')

        # compute model performance
        if performance is not None:
            out['performance'][:] = compute_performance(Ytest, Yhat, metric)
            if verbose >= 1:
                print_performance(out['performance'], r, ridges)
