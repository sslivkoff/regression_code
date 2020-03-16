"""shifting operations"""

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

from . import arrays


def create_shift_matrix(size, offset=1):
    """create a shift operator matrix


    - shifts all items in matrix by offset positions
        - (shift_matrix * X) shifts rows
        - (X * shift_matrix) shifts columns
    - using shift_rows() or shift_columns() directly is more efficient


    Parameters
    ----------
    - size: int number of samples
    - offset: int size of shift
    """
    return np.diag(np.ones(size - offset), -offset)


def shift_rows(X, offset):
    """shift rows of array

    Parameters
    ----------
    - X: array
    - offset: int size of shift, or iterable of offsets
        - if iterable, will return shifted copies concatenated
    """
    m, n = X.shape
    if offset > 0:
        return arrays.concatenate([np.zeros((offset, n)), X[:-offset, :]], axis=0)
    elif offset < 0:
        return arrays.concatenate([X[-offset:, :], np.zeros((-offset, n))], axis=0)
    else:
        if numexpr is not None:
            return numexpr.evaluate('X')
        else:
            return X.copy()


def shift_columns(X, offset):
    """shift columns of array

    Parameters
    ----------
    - X: array
    - offset: int size of shift, or iterable of offsets
        - if iterable, will return shifted copies concatenated
    """
    m, n = X.shape
    if offset > 0:
        return arrays.concatenate([np.zeros((m, offset)), X[:, :-offset]], axis=1)
    elif offset < 0:
        return arrays.concatenate([X[:, offset:], np.zeros((m, -offset))], axis=1)
    else:
        if numexpr is not None:
            return numexpr.evaluate('X')
        else:
            return X.copy()


def make_shifted_copies(X, row_shifts=None, column_shifts=None):
    """make shifted copies of X


    - this function could be more efficient by skipping the intermediate concatenations
    - should specify exactly one of row_shifts or column_shifts


    Parameters
    ----------
    - X: array
    - row_shifts: iterable of shift sizes
    - column_shifts: iterable of shift sizes
    - concatenate: bool of whether to concatenate copies along opposite axis
    """
    assert (
        (row_shifts is not None and column_shifts is None)
        or (row_shifts is None and column_shifts is not None)
    )

    m, n = X.shape

    # if row_shifts is not None:
    #     n_shifts = len(row_shifts)
    #     X_shifted = np.zeros((m, n * n_shifts))
    #     # for shift in shifts

    # create shifted copies
    if row_shifts is not None:
        X_shifted = [shift_rows(X, offset) for offset in row_shifts]
    else:
        X_shifted = [shift_columns(X, offset) for offset in column_shifts]

    concatenation_axis = int(row_shifts is not None)
    return arrays.concatenate(X_shifted, axis=concatenation_axis)


def compute_XTX_shifted(X, shifts, upper_only=False):
    """computes XTX with extra shifted copies along the first dimension of X


    Basic Idea
    ----------
    - main equation:
        X.T (shift.T ^ d1) (shift ^ d2) X
        ==
        X.T (shift ^ |d1 - d2|) X - trimmed_subspace
    - cache (X.T (shift ^ (d2 - d1)) X) matrices


    Parameters
    ----------
    - X: array
    - shifts: iterable of int shifts
    """
    if not all(shift >= 0 for shift in shifts):
        raise NotImplementedError('negative shifts not implemented')

    m, n = X.shape
    n_shifts = len(shifts)

    XTX_shifted = np.zeros((n * n_shifts, n * n_shifts))

    # compute blocks in upper triangle
    XT_shifted_Xs = {}
    for row, row_shift in enumerate(shifts):
        for col, col_shift in list(enumerate(shifts))[(row + 1):]:
            # locate destination
            row_slice = slice(row * n, (row + 1) * n)
            column_slice = slice(col * n, (col + 1) * n)
            destination = XTX_shifted[row_slice, column_slice]

            # compute shifted base matrix
            diff = col - row
            if diff not in XT_shifted_Xs:
                # XT_shifted_Xs[diff] = X.T.dot(shift_rows(X, col))
                XT_shifted_Xs[diff] = X[col:, :].T.dot(X[:-col, :])
            XT_shifted_X = XT_shifted_Xs[diff]

            # subtract trimmed subspace
            left = slice(-(row + 1), None)
            right = slice(-(col + 1), -diff)
            if numexpr is not None:
                destination[:] = X[left, :].T.dot(X[right, :])
                numexpr.evaluate('XT_shifted_X - destination', out=destination)
            else:
                destination[:] = XT_shifted_X - X[left, :].T.dot(X[right, :])

    # lower triangular components
    if not upper_only:
        if numexpr is not None:
            XTX_shifted_T = XTX_shifted.T
            numexpr.evaluate('XTX_shifted + XTX_shifted_T', out=XTX_shifted)
        else:
            XTX_shifted += XTX_shifted.T

    # compute blocks in diagonal
    XTX_0 = X.T.dot(X)
    for d, shift in enumerate(shifts):
        index = slice(d * n, (d + 1) * n)
        destination = XTX_shifted[index, index]
        if shift == 0:
            destination[:] = XTX_0
        else:
            subtrehend = X[-shift:, :].T.dot(X[-shift:, :])
            if numexpr is not None:
                numexpr.evaluate('XTX_0 - subtrehend', out=destination)
            else:
                destination[:] = XTX_0 - subtrehend

    return XTX_shifted


def compute_XXT_shifted(X, shifts):
    """computes XXT with extra shifted copies along the first dimension of X

    Parameters
    ----------
    - X: array
    - shifts: iterable of int shifts
    """
    if not all(shift >= 0 for shift in shifts):
        raise NotImplementedError('negative shifts not implemented')

    m, n = X.shape
    XXT_0 = X.dot(X.T)
    XXT_shifted = np.zeros((m, m))
    for d, shift in enumerate(shifts):
        destination = slice(shift, None)
        source = slice(None, -shift)
        XXT_shifted[destination, destination] += XXT_0[source, source]
    return XXT_shifted
