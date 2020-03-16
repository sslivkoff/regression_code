"""arithmetic functions"""

import multiprocessing
import multiprocessing.pool
import warnings

import numpy as np

try:
    import numexpr
    numexpr.set_num_threads(multiprocessing.cpu_count())
    numexpr.set_vml_num_threads(multiprocessing.cpu_count())
except ImportError:
    warnings.warn('numexpr not detected, use `sudo pip install numexpr`')
    numexpr = None

from . import arrays


#
# # main functions
#

def sum(X, axis):
    """compute sum of array along an axis

    Parameters
    ----------
    - X: array to compute sum of
    - axis: int axis along which to compute sum
    """
    return dot_sum(X, axis)


def list_sum(Xs):
    """compute sum of iterable of arrays

    Parameters
    ----------
    - Xs: iterable of numpy arrays
    """
    if numexpr is not None:
        return numexpr_list_sum(Xs)
    else:
        return arith_list_sum(Xs)


def mean(X, axis):
    """compute mean of array along an axis

    Parameters
    ----------
    - X: array to compute mean of
    - axis: int axis along which to compute mean
    """
    return dot_mean(X, axis)


def nanmean(X, axis):
    """compute mean of array along axis, ignoring nan values

    Parameters
    ----------
    - X: array to compute mean of
    - axis: int axis along which to compute mean
    """
    return mean(arrays.nan_to_num(X), axis)


def list_mean(Xs):
    """compute mean of iterable of arrays

    Parameters
    ----------
    - Xs: iterable of numpy arrays
    """
    if numexpr is not None:
        return numexpr_list_mean(Xs)
    else:
        return arith_list_mean(Xs)


def std(X, axis, demean=True):
    """compute standard deviation of array along an axis

    Parameters
    ----------
    - X: array to compute standard deviation of
    - axis: int axis along which to compute standard deviation
    - demean: bool of whether to demean array
    """
    if numexpr is not None:
        return numexpr_std(X, axis, demean=demean)
    else:
        return dot_std(X, axis, demean=demean)


def zscore(X, axis, demean=True):
    """compute zscore of array

    Parameters
    ----------
    - X: array to zscore
    - axis: axis along which to compute zscore
    - demean: bool of whether to demean array
    """
    if numexpr is not None:
        return numexpr_zscore(X, axis, demean=demean)
    else:
        return dot_zscore(X, axis, demean=demean)


def zscore_inplace(X, axis, demean=True):
    """compute zscore of array inplace

    Parameters
    ----------
    - X: array to zscore
    - axis: axis along which to compute zscore
    - demean: bool of whether to demean array
    """
    if numexpr is not None:
        return numexpr_zscore_inplace(X, axis, demean=demean)
    else:
        return dot_zscore_inplace(X, axis, demean=demean)


def correlate(X1, X2, axis, zscore_left=True, zscore_right=True):
    """compute correlation between arrays along an axis

    Parameters
    ----------
    - X1: array to compute correlation of
    - X2: array to compute correlation of
    - axis: int axis along which to compute correlation
    - zscore_left: bool of whether to zscore array 1
    - zscore_right: bool of whether to zscore array 2
    """
    kwargs = {'zscore_left': zscore_left, 'zscore_right': zscore_right}
    if numexpr is not None:
        return numexpr_correlate(X1, X2, axis, **kwargs)
    else:
        return dot_correlate(X1, X2, axis, **kwargs)


def R_squared(Y, Yhat, axis, unit_variance=False, overwrite_Yhat=False):
    """compute coefficient of determination along an axis

    Parameters
    ----------
    - Y: array of observed values
    - Yhat: array of model predictions
    - axis: int axis along which to compute R^2
    - unit_variance: bool of whether Y has unit variance
    - overwrite_Yhat: bool of whether Yhat can be polluted for speed
    """
    kwargs = {'unit_variance': unit_variance, 'overwrite_Yhat': overwrite_Yhat}
    if numexpr is not None:
        return numexpr_R_squared(Y, Yhat, axis, **kwargs)
    else:
        return dot_R_squared(Y, Yhat, axis, **kwargs)


#
# # dot functions
#

def dot_sum(X, axis):
    """compute sum of array along an axis

    Parameters
    ----------
    - X: array to compute sum of
    - axis: int axis along which to compute sum
    """
    if X.ndim == 1 or (X.ndim == 2 and axis == 0):
        return np.dot(np.ones(X.shape[0], dtype=X.dtype), X)
    elif (X.ndim == 2 and axis == 1):
        return np.dot(X, np.ones(X.shape[1], dtype=X.dtype))
    else:
        raise NotImplementedError()


def dot_mean(X, axis):
    """compute mean of array along an axis

    Parameters
    ----------
    - X: array to compute mean of
    - axis: int axis along which to compute mean
    """
    if axis is None:
        return dot_sum(X, axis=axis) / X.dtype.type(X.size)
    else:
        return dot_sum(X, axis=axis) / X.dtype.type(X.shape[axis])


def dot_std(X, axis, demean=True):
    """compute standard deviation of array along an axis

    Parameters
    ----------
    - X: array to compute standard deviation of
    - axis: int axis along which to compute standard deviation
    - demean: bool of whether to demean array
    """
    X_std = dot_sum(X ** 2, axis=axis)
    if demean:
        X_std -= dot_sum(X, axis=axis) ** 2 / X.shape[axis]
    X_std **= .5
    X_std /= (X.shape[axis] ** .5)
    return X_std


def dot_zscore(X, axis, demean=True):
    """compute zscore of array

    Parameters
    ----------
    - X: array to zscore
    - axis: axis along which to compute zscore
    - demean: bool of whether to demean array
    """
    if demean:
        return (X - dot_mean(X, axis=axis)) / dot_std(X, axis=axis)
    else:
        return X / dot_std(X, axis=axis)


def dot_correlate(X1, X2, axis, zscore_left=True, zscore_right=True):
    """compute correlation between arrays along an axis

    Parameters
    ----------
    - X1: array to compute correlation of
    - X2: array to compute correlation of
    - axis: int axis along which to compute correlation
    - zscore_left: bool of whether to zscore array 1
    - zscore_right: bool of whether to zscore array 2
    """
    if zscore_left:
        X1 = dot_zscore(X1, axis=axis)
        if zscore_right:
            X1 *= dot_zscore(X2, axis=axis)
        else:
            X1 *= X2
        return dot_mean(X1, axis=axis)
    elif zscore_right:
        X2 = dot_zscore(X2, axis=axis)
        if zscore_left:
            X2 *= dot_zscore(X1, axis=axis)
        else:
            X2 *= X1
        return dot_mean(X2, axis=axis)
    else:
        return dot_sum(X1 * X2, axis=axis)


def dot_R_squared(Y, Yhat, axis, unit_variance=False, overwrite_Yhat=False):
    """compute coefficient of determination along an axis

    Parameters
    ----------
    - Y: array of observed values
    - Yhat: array of model predictions
    - axis: int axis along which to compute R^2
    - unit_variance: bool of whether Y has unit variance
    - overwrite_Yhat: bool of whether Yhat can be polluted for speed
    """
    if overwrite_Yhat:
        Yhat -= Y
        Yhat **= 2
        res_sq = Yhat
    else:
        res_sq = Yhat - Y
        res_sq **= 2

    res_sq_sum = dot_sum(res_sq, axis=axis)

    if unit_variance:
        res_sq_sum *= -1
        res_sq_sum += 1
        return res_sq_sum
    else:
        res_sq_sum /= Y.var(axis=axis)
        res_sq_sum *= -1
        res_sq_sum += 1
        return res_sq_sum


def dot_std_inplace(X, axis, demean=True):
    """compute standard deviation of array along an axis, and pollute array

    Parameters
    ----------
    - X: array to compute standard deviation of
    - axis: int axis along which to compute standard deviation
    - demean: bool of whether to demean array
    """
    if demean:
        X_sum_squared = dot_sum(X, axis=axis)
        X_sum_squared **= 2
        X_sum_squared /= X.shape[axis]
    X **= 2
    X_std = dot_sum(X, axis)
    if demean:
        X_std -= X_sum_squared
    X_std **= .5
    X_std /= (X.shape[axis] ** .5)
    return X_std


def dot_zscore_inplace(X, axis, demean=True):
    """compute zscore of array inplace

    Parameters
    ----------
    - X: array to zscore
    - axis: axis along which to compute zscore
    - demean: bool of whether to demean array
    """
    if demean:
        X -= dot_mean(X, axis=axis)
    X /= dot_std(X, axis=axis, demean=False)
    return X


def dot_correlate_inplace(X1, X2, axis, zscore_left=True, zscore_right=True, demean1=True,
                          demean2=True, pollute=None):
    """compute correlation between arrays along an axis, polluting an input

    Parameters
    ----------
    - X1: array to compute correlation of
    - X2: array to compute correlation of
    - axis: int axis along which to compute correlation
    - zscore_left: bool of whether to zscore array 1
    - zscore_right: bool of whether to zscore array 2
    """
    if pollute is None:
        raise Exception('must specify pollute=\'left\' or pollute=\'right\'')
    else:
        if zscore_left:
            dot_zscore_inplace(X1, axis=axis, demean=demean1)
        if zscore_right:
            dot_zscore_inplace(X2, axis=axis, demean=demean2)

        if pollute == 'left':
            X1 *= X2
            return dot_mean(X1, axis=axis)
        elif pollute == 'right':
            X2 *= X1
            return dot_mean(X2, axis=axis)


#
# # numexpr functions
#

def numexpr_list_sum(Xs):
    """compute sum of list of arrays

    Parameters
    ----------
    - Xs: iterable of numpy arrays
    """
    X_sum = np.zeros(Xs[0].shape)
    for X in Xs:
        numexpr.evaluate('X_sum + X', out=X_sum)
    return X_sum


def numexpr_list_mean(Xs):
    """compute mean of list of arrays

    Parameters
    ----------
    - Xs: iterable of numpy arrays
    """
    Xs_sum = numexpr_list_sum(Xs)
    N = len(Xs)
    return numexpr.evaluate('Xs_sum / N')


def numexpr_std(X, axis, demean=True):
    """compute standard deviation of array along an axis

    Parameters
    ----------
    - X: array to compute standard deviation of
    - axis: int axis along which to compute standard deviation
    - demean: bool of whether to demean array
    """
    if demean:
        X_mean = mean(X, axis)
        X_std = dot_sum(numexpr.evaluate('(X - X_mean) ** 2'), axis=axis)
    else:
        X_std = dot_sum(numexpr.evaluate('X ** 2'), axis=axis)
    N = X.dtype.type(X.shape[axis])
    return numexpr.evaluate('(X_std ** .5) / (N ** .5)', out=X_std)


def numexpr_zscore(X, axis, demean=True):
    """compute zscore of array

    Parameters
    ----------
    - X: array to zscore
    - axis: axis along which to compute zscore
    - demean: bool of whether to demean array
    """
    if demean:
        X_mean = mean(X, axis)
        X_std = std(X, axis)
        return numexpr.evaluate('(X - X_mean) / X_std')
    else:
        X_std = std(X, axis)
        return numexpr.evaluate('X / X_std')


def numexpr_zscore_inplace(X, axis, demean=True):
    """compute zscore of array inplace

    Parameters
    ----------
    - X: array to zscore
    - axis: axis along which to compute zscore
    - demean: bool of whether to demean array
    """
    if demean:
        X_mean = mean(X, axis)
        X_std = std(X, axis)
        return numexpr.evaluate('(X - X_mean) / X_std', out=X)
    else:
        X_std = std(X, axis)
        return numexpr.evaluate('X / X_std', out=X)


def numexpr_correlate(X1, X2, axis, zscore_left=True, zscore_right=True):
    """compute correlation between arrays along an axis

    Parameters
    ----------
    - X1: array to compute correlation of
    - X2: array to compute correlation of
    - axis: int axis along which to compute correlation
    - zscore_left: bool of whether to zscore array 1
    - zscore_right: bool of whether to zscore array 2
    """
    if zscore_left and zscore_right:
        X1_mean = mean(X1, axis)
        X1_std = std(X1, axis)
        X2_mean = mean(X2, axis)
        X2_std = std(X2, axis)
        expr = '(X1 - X1_mean) * ((X2 - X2_mean) / (X1_std * X2_std))'
        return mean(numexpr.evaluate(expr), axis)
    elif zscore_left and not zscore_right:
        X1_mean = mean(X1, axis)
        X1_std = std(X1, axis)
        expr = '((X1 - X1_mean) / X1_std) * X2'
        return mean(numexpr.evaluate(expr), axis)
    elif not zscore_left and zscore_right:
        X2_mean = mean(X2, axis)
        X2_std = std(X2, axis)
        expr = 'X1 * ((X2 - X2_mean) / X2_std)'
        return mean(numexpr.evaluate(expr), axis)
    elif not zscore_left and not zscore_right:
        return mean(numexpr.evaluate('X1 * X2'), axis)


def numexpr_R_squared(Y, Yhat, axis, unit_variance=False, overwrite_Yhat=False):
    """compute coefficient of determination along an axis

    Parameters
    ----------
    - Y: array of observed values
    - Yhat: array of model predictions
    - axis: int axis along which to compute R^2
    - unit_variance: bool of whether Y has unit variance
    - overwrite_Yhat: bool of whether Yhat can be polluted for speed
    """
    if overwrite_Yhat:
        res_sq = numexpr.evaluate('(Yhat - Y) ** 2', out=Yhat)
    else:
        res_sq = numexpr.evaluate('(Yhat - Y) ** 2')

    res_sq_sum = dot_sum(res_sq, axis=axis)
    if unit_variance:
        return numexpr.evaluate('1 - res_sq_sum')
    else:
        Y_variance = Y.var(axis=axis)
        return numexpr.evaluate('1 - (res_sq_sum / Y_variance)')


#
# # thread functions
#


def thread_sum(X, axis):
    """compute sum along an axis

    Parameters
    ----------
    - X: array to compute sum of
    - axis: int axis along which to compute sum
    """
    if X.ndim == 2:
        split_axis = int(not axis)
        n_threads = multiprocessing.cpu_count()
        pool = multiprocessing.pool.ThreadPool(processes=n_threads)
        X_split = np.array_split(X, n_threads, axis=split_axis)
        return np.concatenate(pool.map(lambda X: X.sum(axis), X_split), axis=0)
    else:
        raise NotImplementedError()


#
# # arithmetic functions
#

def arith_list_sum(Xs):
    """compute sum of list of arrays

    Parameters
    ----------
    - Xs: iterable of numpy arrays
    """
    X_sum = np.zeros(Xs[0].shape)
    for X in Xs:
        X_sum += X
    return X_sum


def arith_list_mean(Xs):
    """compute mean of list of arrays

    Parameters
    ----------
    - Xs: iterable of numpy arrays
    """
    X_mean = arith_list_sum(Xs)
    X_mean /= len(Xs)
    return X_mean


#
# # matrix multiplication
#

def multi_dot(*arrays, **kwargs):
    """compute chain of matrix dot products in most efficient order

    - like np.linalg.multi_dot, but handles 'out' argument
    - number of ways to evaluate chain is C(n-1), the (n-1)th Catalan number
        - C(n-1) for n in range(2, 10): 1, 2, 5, 14, 42, 132, 429, 1430, 4862
        - see https://en.wikipedia.org/wiki/Matrix_chain_multiplication
    - numpy's algorithm is O(n^3)
        - becomes very expensive for large lists of arrays
        - much more efficient algorithms exist on the order of O(n log n)

    - is there an easy workaround for 1d arrays?
        - maybe broadcast them? are they always representative of square matrices?

    Parameters
    ----------
    - arrays: iterable of arrays to dot multiply as chain
    - kwargs: [only here because python 2 can't have args after *args]
        - out: array in which to store output
    """
    for array in arrays:
        if array.ndim != 2:
            raise Exception('all arrays must be length 2')

    out = kwargs.get('out')

    if len(arrays) == 1:
        if out is not None:
            out[:] = arrays[0]
        return arrays[0]

    elif len(arrays) == 2:
        A, B = arrays
        return np.dot(A, B, out=out)

    elif len(arrays) == 3:
        A, B, C = arrays
        a0, a1b0 = A.shape
        b1c0, c1 = C.shape
        cost1 = a0 * b1c0 * (a1b0 + c1)
        cost2 = a1b0 * c1 * (a0 + b1c0)
        if cost1 < cost2:
            return np.dot(np.dot(A, B), C, out=out)
        else:
            return np.dot(A, np.dot(B, C), out=out)

    else:
        from numpy.linalg.linalg import _multi_dot_matrix_chain_order
        order = _multi_dot_matrix_chain_order(arrays)

        def _multi_dot(arrays, order, i, j, out=None):
            """based on np.linalg.linalg._multi_dot"""
            if i == j:
                return arrays[i]
            else:
                return np.dot(
                    _multi_dot(arrays, order, i, order[i, j]),
                    _multi_dot(arrays, order, order[i, j] + 1, j),
                    out=out,
                )

        return _multi_dot(arrays, order, 0, len(arrays) - 1, out=out)
