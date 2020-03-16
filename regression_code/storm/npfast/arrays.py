"""basic array functions"""

import multiprocessing
import warnings

import numpy as np

try:
    import numexpr
    numexpr.set_num_threads(multiprocessing.cpu_count())
    numexpr.set_vml_num_threads(multiprocessing.cpu_count())
except ImportError:
    warnings.warn('numexpr not detected, use `sudo pip install numexpr`')
    numexpr = None


def astype(array, dtype):
    """cast array to dtype

    Parameters
    ----------
    - array: array
    - dtype: dtype to cast to
    """
    if numexpr is None:
        return array.astype(dtype)
    result = np.zeros(array.shape, dtype=dtype)
    return numexpr.evaluate('array', out=result, casting='unsafe')


def concatenate(arrays, axis, dtype=None, out=None):
    """concatenate arrays along axis

    Parameters
    ----------
    - arrays: iterable of arrays
    - axis: int axis to concatenate
    - dtype: dtype of result
    - out: array in which to store result
    """
    # compute sizes
    ndim = arrays[0].ndim
    other_axes = [other for other in range(arrays[0].ndim) if other != axis]
    other_lengths = [arrays[0].shape[other_axis] for other_axis in other_axes]
    axis_lengths = [array.shape[axis] for array in arrays]
    axis_length = np.sum(axis_lengths)
    result_shape = other_lengths[:axis] + [axis_length] + other_lengths[axis:]

    # ensure sizes and dtypes are proper
    for a, array in enumerate(arrays):
        if len(array.shape) != ndim:
            raise Exception('array' + str(a) + 'has wrong dimensions')
        for ol, length in enumerate(other_lengths):
            if array.shape[other_axes[ol]] != length:
                raise Exception('bad axis ' + str(ol) + ' of array ' + str(a))
    if out is not None:
        if out.shape != result_shape:
            raise Exception('out does not have shape ' + str(result_shape))
        if dtype is not None and out.dtype != dtype:
            raise Exception('out does not have dtype ' + str(dtype))

    # initialize output
    if out is None:
        out = np.zeros(result_shape, dtype=dtype)

    # fall back to numpy if numexpr not available
    if numexpr is None:
        out[:] = np.concatenate(arrays, axis=axis)
        return out

    # populate output
    start = 0
    slices = [slice(None) for d in range(ndim)]
    for array in arrays:
        end = start + array.shape[axis]
        slices[axis] = slice(start, end)
        numexpr.evaluate('array', out=out[slices])
        start = end

    return out


def isnan(X):
    """evaluate whether elements of X are infinite

    Parameters
    ----------
    - X: array to evaluate nan values of
    """
    if numexpr is not None:
        return numexpr.evaluate('X != X')
    else:
        return np.isnan(X)


def nan_to_num(X):
    """convert infinite values in array to 0

    Parameters
    ----------
    - X: array whose infinite values to convert
    """
    if numexpr is not None:
        X = copy(X)
        X[isnan(X)] = 0
        return X
    else:
        return np.nan_to_num(X)


def nan_to_num_inplace(X):
    """convert infinite values in array to 0 inplace

    Parameters
    ----------
    - X: array whose infinite values to convert
    """
    if numexpr is not None:
        X[isnan(X)] = 0
        return X
    else:
        X[np.isnan(X)] = 0
        return X


def copy(X):
    """return a copy of X

    Parameters
    ----------
    - X: array to copy
    """
    if numexpr is not None:
        return numexpr.evaluate('X')
    else:
        return np.copy(X)
