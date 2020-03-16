
import multiprocessing
import types

import numpy as np
import scipy

from .. import aone


#
# # str utils
#

def fraction_str(*numbers):
    """normalized numbers list by sum and return as str

    Parameters
    ----------
    - numbers: numbers to print fractions of
    """
    total = sum(numbers)
    fractions = [number / float(total) for number in numbers]
    return '(' + ', '.join('{:0.03f}'.format(item) for item in fractions) + ')'


#
# # dict utils
#

def merge(*dicts, **kwargs):
    """merge dicts into one

    Parameters
    ----------
    - dicts: iterable of dicts
    - kwargs: dict of additional items for merged dict
    """
    dicts = list(dicts)
    dicts.append(kwargs)
    return {key: value for d in dicts for key, value in d.items()}


#
# # kernel utils
#

def linear_kernel(Xtrain, Xtest=None, n_chunks=None):
    """create linear kernel

    Parameters
    ----------
    - Xtrain: array of regressor training data
    - Xtest: array of regressor testing data
    - n_chunks: int of chunks to break matrix into, useful for large matrices
    """
    if Xtest is None:
        Xtest = Xtrain

    if n_chunks is None:
        product = np.dot(Xtrain, Xtest.T)
    else:
        x_chunks = np.array_split(Xtrain, n_chunks, axis=1)
        y_chunks = np.array_split(Xtest, n_chunks, axis=1)
        product = np.zeros((Xtrain.shape[0], Xtest.shape[0]))
        for c in range(n_chunks):
            product += np.dot(x_chunks[c], y_chunks[c].T)
    return product


def project_kernel_weights(Xtest, kernel_weights):
    """project kernel weights into ordinary weight space

    Parameters
    ----------
    - Xtest: array of regressor testing data
    - kernel_weights: weights in kernel space, somehting like [(XXT + aI)-1 Y]
    """
    B = np.dot(Xtest.T, kernel_weights)
    return B


#
# # cv utils
#


def generate_folds(n_samples, n_folds=10, train_fraction=.8, len_blocks=5,
                   overlapping=True):
    """generate indices of folds for leave-one-out cross validation

    Parameters
    ----------
    - n_samples: number of samples
    - n_folds: number of folds
    - train_pct: percentage of data to use for training
    - len_blocks: chunksize for random partitioning of data
        - if None, contiguous folds are generated
    """
    if overlapping:
        splits = np.array_split(np.arange(n_samples), n_folds)

        folds = []
        for s in range(n_folds):
            if s == 0:
                training = np.concatenate(splits[(s + 1):], axis=0)
            elif s == n_folds - 1:
                training = np.concatenate(splits[:s], axis=0)
            else:
                fold_splits = splits[:s] + splits[(s + 1):]
                training = np.concatenate(fold_splits, axis=0)
            validation = splits[s]
            folds.append((training, validation))
    else:
        folds = aone.utils.generate_trnval_folds(
            N=n_samples,
            sampler='cv',
            nfolds=n_folds,
            testpct=(1 - train_fraction),
            nchunks=len_blocks,
        )
        folds = [(training, validation) for training, validation in folds]
    return folds


def fold_data(fold, datas):
    """fold data into the specified training and validation indices

    Parameters
    ----------
    - fold: 2-tuple of (training, validation) indices
    - datas: dict of arrays as {str: array}
    """
    training, validation = fold
    folded = {}

    # fold Xtrain
    if datas.get('Xtrain') is not None:
        folded['Xtrain'] = datas['Xtrain'][training, :]
        folded['Xtest'] = datas['Xtrain'][validation, :]

    # fold Ktrain
    if datas.get('Ktrain') is not None:
        folded['Ktrain'] = datas['Ktrain'][training, :][:, training]
        folded['Ktest'] = datas['Ktrain'][validation, ][:, training]
        # if datas.get('Ktest') is not None:
        #     folded['Ktest'] = datas['Ktest'][:, validation]
        # else:
        #     folded['Ktest'] = folded['Xtest'].dot(folded['Xtrain'].T)

    # fold Ytrain
    if datas.get('Ytrain') is not None:
        folded['Ytrain'] = datas['Ytrain'][training, :]
        folded['Ytest'] = datas['Ytrain'][validation, :]

    return folded


#
# # cloud utils
#

def cloudload(path):
    raise NotImplementedError()


def load_data_specs(data_specs):
    """loads data according each data specification

    Data Specifications
    -------------------
    - str: data = cloudload(str)
    - function: data = function()
    - (function, args, kwargs): data = function(*args, **kwargs)
    - np.ndarray: data already loaded

    Parameters
    ----------
    - data_specs: dict of (key, value) where each value is a data specification
    """
    loaded = {}
    for key, value in data_specs.items():
        # case: data specification is path
        if isinstance(value, str):
            loaded[key] = cloudload(value)

        # case: data specification is function
        elif isinstance(value, types.FunctionType):
            loaded[key] = value()

        # case: data specification is function with arguments
        elif (
            isinstance(value, (tuple, list))
            and len(value) == 3
            and isinstance(value[0], types.FunctionType)
            and isinstance(value[1], (tuple, list))
            and isinstance(value[2], dict)
        ):
            function, args, kwargs = value
            loaded[key] = function(*args, **kwargs)

        # case: data is already loaded
        elif isinstance(value, np.ndarray):
            loaded[key] = value

        # case: no specification provided
        elif value is None:
            loaded[key] = None

        # case: data specification not understood
        else:
            raise Exception('value type not understood:', type(value))

    return loaded


#
# # parallel utils
#

def thread_map(f, args_list, n_threads=None):
    """map a function over args using multiple threads

    Parameters
    ----------
    - f: function to use for mapping
    - args_list: list of args to map over
    - n_threads: int number of threads to use
    """
    if n_threads is None:
        n_threads = int(multiprocessing.cpu_count() / 2)
    pool = multiprocessing.pool.ThreadPool(processes=n_threads)
    return pool.map(f, args_list)


#
# # linear algebra utils
#

def det_nth_root(X, method='lu'):
    """return Nth root of determinant of square N x N matrix X

    Parameters
    ----------
    - X: square (N, N) array
    - method: str of computation method to use ('lu', 'eig', or 'qr')
    """
    N = float(X.shape[0])
    if method == 'lu':
        P, L, U = scipy.linalg.lu(X)
        diags = (np.diag(L) ** (1 / N) * (np.diag(U) ** (1 / N)))
        determinant = np.product(diags)
    elif method == 'eig':
        L = np.linalg.eigvalsh(X)
        determinant = np.product(L ** (1 / float(L.size)))
    elif method == 'qr':
        (R,) = scipy.linalg.qr(X, mode='r')
        determinant = np.product(np.abs(np.diag(R)) ** (1 / N))
    else:
        raise Exception('method not understood')

    return np.nan_to_num(determinant)


def XTX_threshold(X, n_chunks=100, threshold=0, dtype=float, sparse=False):
    """compute sparse thresholded version of XTX

    Parameters
    ----------
    - X: array
    - n_chunks: int number of chunks for computation for memory safety
    - threshold: float threshold of values to keep
    - dtype: dtype to use for matrix
    - sparse: str of sparsity type
    """
    m, n = X.shape

    Xsplits = np.array_split(X, n_chunks, axis=1)
    split_indices = np.cumsum([0] + [Xsplit.shape[1] for Xsplit in Xsplits])

    if sparse == 'dok':
        XTX = scipy.sparse.dok_matrix((n, n))
    elif sparse == 'csc':
        XTX = scipy.sparse.csc_matrix((n, n))
    elif sparse == 'lil':
        XTX = scipy.sparse.csc_matrix((n, n))
    else:
        XTX = np.zeros((n, n), dtype=dtype)

    for r in range(n_chunks):
        row = slice(split_indices[r], split_indices[r + 1])
        for c in range(n_chunks):
            col = slice(split_indices[c], split_indices[c + 1])
            XTX_chunk = Xsplits[r].T.dot(Xsplits[c])
            if dtype != bool:
                XTX[row, col] = (np.abs(XTX_chunk) > threshold) * XTX_chunk
            if dtype == bool:
                XTX[row, col] = np.abs(XTX_chunk) > threshold

    return XTX
