"""functions for loading test datasets


Usage
-----
- use load_all() to load all datasets


Available Datasets
------------------
- 'qa_wide': question answering pilot data with 985 features (english1000)
- 'qa_thin': question answering pilot data with 41 features (question type)
- 'natural_movies_gabor_pyramid': natural movie data with full gabor pyramid
- 'natural_movies_mean_gabor': natural movie data with single mean gabor channel


Random Datasets
---------------
- create_random_single_matrices
- create_random_XY
- create_random_datasets
"""

import sys

import scipy.stats
import numpy as np

import glabtools

sys.path.append('/auto/k1/storm/python_path/datasets')
from datasets.qa import qa_initial_pilot


def preprocess_dataset(Xtrain, Ytrain, Xtest=None, Ytest=None, dtype=None,
                       zscore=True, denan=True, delays=[1, 2, 3, 4], order='C',
                       trim_random=False, trim_regressors=None,
                       trim_regressands=None):
    """preprocess a dataset

    Parameters
    ----------
    - Xtrain: array
    - Ytrain: array
    - Xtest: array
    - Ytest: array
    - dtype: numpy dtype to use
    - zscore: bool of whether to zscore data
    - denan: bool of whether to denan arrays
    - order: str of 'C' or 'F' for C-ordering or Fortran ordering
    """
    data = {
        'Xtrain': Xtrain,
        'Ytrain': Ytrain,
        'Xtest': Xtest,
        'Ytest': Ytest,
    }
    data = {key: value for key, value in data.items() if value is not None}

    if dtype is not None:
        data = {key: value.astype(dtype) for key, value in data.items()}
    if zscore:
        data = {key: scipy.stats.zscore(value) for key, value in data.items()}
    if denan:
        data = {key: np.nan_to_num(value) for key, value in data.items()}
    if delays:
        for key in list(data.keys()):
            if key.startswith('X'):
                data[key] = make_delayed(data[key], delays)
    if order == 'F':
        data = {key: np.asfortranarray(value) for key, value in data.items()}
    elif order == 'C':
        data = {key: np.ascontiguousarray(value) for key, value in data.items()}

    # trim dimensions
    if trim_random:
        f_keep = lambda before, after: np.sorted(np.random.choice(
            np.arange(n_regressors),
            new_n_regressors,
            replace=False,
        ))
    else:
        f_keep = lambda before, after: slice(None, after)
    if trim_regressors is not None:
        n_regressors = data['Xtrain'].shape[1]
        new_n_regressors = int(n_regressors * trim_regressors)
        keep = f_keep(n_regressors, new_n_regressors)
        data['Xtrain'] = data['Xtrain'][:, keep]
        data['Xtest'] = data['Xtest'][:, keep]

    if trim_regressands is not None:
        n_regressands = data['Ytrain'].shape[1]
        new_n_regressands = int(n_regressands * trim_regressands)
        keep = f_keep(n_regressands, new_n_regressands)
        data['Ytrain'] = data['Ytrain'][:, keep]
        data['Ytest'] = data['Ytest'][:, keep]

    return data


#
# # real datasets
#

def load_all(**preprocessing_kwargs):
    """load all available datasets"""
    datasets = {}

    loaders = [
        load_qa,
        load_natural_movies,
    ]

    for loader in loaders:
        datasets.update(loader())

    for key in list(datasets.keys()):
        dataset = datasets[key]
        dataset.update(preprocessing_kwargs)
        datasets[key] = preprocess_dataset(**dataset)

    return datasets


def load_qa():
    """load question answering dataset, with skinny and wide feature matrices"""
    Ytrain, Ytest = qa_initial_pilot.cloudload_responses()
    Xtrain_wide, Xtest_wide = qa_initial_pilot.load_stimuli(
        ['english1000'],
        delays=[0],
    )
    Xtrain_skinny, Xtest_skinny = qa_initial_pilot.load_stimuli(
        ['relation_onsets'],
        delays=[0],
    )

    datasets = {
        'qa_wide': {
            'Xtrain': Xtrain_wide,
            'Ytrain': Ytrain,
            'Xtest': Xtest_wide,
            'Ytest': Ytest,
        },
        'qa_skinny': {
            'Xtrain': Xtrain_skinny,
            'Ytrain': Ytrain,
            'Xtest': Xtest_skinny,
            'Ytest': Ytest,
        },
    }

    return datasets


def load_natural_movies(cpd=1.00):
    """load natural movies dataset

    Parameters
    ----------
    - cpd: float of cycles per degree, should be 1.00 or 1.33
    """
    if cpd not in {1.00, 1.33}:
        raise Exception('cpd must be in {1.00, 1.33}')
    if cpd == 1.00:
        cpd = '1.00'
    elif cpd == 1.33:
        cpd = '1.33'
    else:
        raise Exception('cpd must be in {1.00, 1.33}')

    # load X
    X_path = '/auto/k6/nbilenko/preproc_data/movie/dir{cpd}cpd_{dataset}stim.npy'
    Xtrain = np.load(X_path.format(cpd=cpd, dataset='t'))
    Xtest = np.load(X_path.format(cpd=cpd, dataset='v'))

    # load Y
    Y_path = 'auto/k8/anunez/proj/snmovies/datasets/snmovies_braindata_AH3T.hdf'
    cci = glabtools.io.get_cc_interface('anunez_raid', verbose=False)
    Y_data = cci.cloud2dict(Y_path, verbose=False)
    Ytrain = Y_data['Ytrain']
    Ytest = Y_data['Yval']

    return {
        'natural_movies_gabor_pyramid': {
            'Xtrain': Xtrain,
            'Ytrain': Ytrain,
            'Xtest': Xtest,
            'Ytest': Ytest,
        },
        'natural_movies_mean_gabor': {
            'Xtrain': Xtrain.mean(1, keepdims=True),
            'Ytrain': Ytrain,
            'Xtest': Xtest.mean(1, keepdims=True),
            'Ytest': Ytest,
        },
    }


#
# # random matrix datasets
#

def create_random_single_matrices():
    """create random matrices of gradually increasing size"""
    return {
        'small': [
            np.random.rand(100, 1000).astype(np.float32),
            np.random.rand(200, 1000).astype(np.float32),
            np.random.rand(400, 1000).astype(np.float32),
            np.random.rand(800, 1000).astype(np.float32),
        ],
        'medium': [
            np.random.rand(250, 10000).astype(np.float32),
            np.random.rand(500, 10000).astype(np.float32),
            np.random.rand(1000, 10000).astype(np.float32),
            np.random.rand(2000, 10000).astype(np.float32),
        ],
        'big': [
            np.random.rand(1000, 100000).astype(np.float32),
            np.random.rand(4000, 100000).astype(np.float32),
            np.random.rand(10000, 100000).astype(np.float32),
        ],
    }


def create_random_XY(name=None, m=None, n=None, v=None, x_rank=None, dist=None,
                     dtype=np.float32):
    """return random pair of X and Y matrices

    Parameters
    ----------
    - name: name of parameter set
    - m: int number of samples
    - n: int number of regressors
    - v: int number of regressands
    - x_rank: int rank of x matrix
    - dist: function to create random matrices
    """

    test_matrices = {
        'm = n': {'m': 1200, 'n': 1200, 'v': 100000},
        'm < n': {'m': 400, 'n': 4000, 'v': 100000},
        'm > n': {'m': 4000, 'n': 400, 'v': 100000},
        'm = n, low rank': {'m': 1200, 'n': 1200, 'v': 100000, 'x_rank': 20},
        'm < n, low rank': {'m': 400, 'n': 4000, 'v': 100000, 'x_rank': 20},
        'm > n, low rank': {'m': 4000, 'n': 400, 'v': 100000, 'x_rank': 20},
        'big': {'m': 10000, 'n': 4000, 'v': 300000},
        'medium': {'m': 3000, 'n': 4000, 'v': 50000},
        'small': {'m': 400, 'n': 400, 'v': 10000},
    }

    if name is not None:
        matrix_kwargs = test_matrices[name]
        m = matrix_kwargs.get('m')
        n = matrix_kwargs.get('n')
        v = matrix_kwargs.get('v')
        x_rank = matrix_kwargs.get('x_rank', None)
        dist = matrix_kwargs.get('dist', None)

    if dist is None:
        dist = np.random.rand

    X = dist(m, n).astype(dtype)
    Y = dist(m, v).astype(dtype)

    if x_rank is not None:
        U, S, VT = scipy.linalg.svd(X)
        Xhat = np.zeros((m, n))
        for i in range(x_rank):
            Xhat += S[i] * np.outer(U.T[i], VT[i])
        X = Xhat

    if dtype is not None:
        X = X.astype(dtype)
        Y = Y.astype(dtype)

    return X, Y


def create_random_datasets(name, n_datasets=1, test_data=True, **kwargs):
    """create random datasets

    Parameters
    ----------
    - name: str name of dataset type passed to create_random_XY()
    - n_datasets: int number of random datasets to generate
    - test_data: bool of whether to generate test data
    - kwargs: arguments passed to create_random_XY()

    Returns
    -------
    - datasets: dict of random matrices
        - Xtrain: random array
        - Ytrain: random array
        - Xtest: random array, returned if test_data == True
        - Ytest: random array, returned if test_data == True
    """
    datasets = {}

    for d in range(n_datasets):
        dataset_name = 'random_data_' + str(d)
        datasets[dataset_name] = {}

        Xtrain, Ytrain = create_random_XY(name, **kwargs)
        datasets[dataset_name]['Xtrain'] = Xtrain
        datasets[dataset_name]['Ytrain'] = Ytrain

        if test_data:
            Xtest, Ytest = create_random_XY(name, **kwargs)
            datasets[dataset_name]['Xtest'] = Xtest
            datasets[dataset_name]['Ytest'] = Ytest

    return datasets


#
# # delaying
#

def ndslice(ndim=None, axis=None, start=None, stop=None, step=None):
    """returns a list of slices for indexing an n-dimensional array

    - ndslice selects along a specific axis and leaves other axes unchanged
    - paramter combinations
        - must specify at least one of {ndim, array}

    Example Usage
    -------------
    slices = ndslice(ndim, axis, start, stop)
    subarray = ndarray[slices]
    """
    slices = [slice(None)] * ndim
    slices[axis] = slice(start, stop, step)
    return slices


def shift_slice(shift, axis, ndim):
    """makes slice objects that put a shifted copy of one array within another

    - is used by make_delayed()

    Example Usage
    -------------
    - to_slice, from_slice = shift_slice(shift, axis, ndim)
    - shifted_array[to_slice] = original_array[from_slice]
    """
    if shift > 0:
        bounds = {'from': {'stop': -shift}, 'to': {'start': shift}}
    elif shift < 0:
        bounds = {'from': {'start': -shift}, 'to': {'stop': shift}}
    else:
        bounds = {'from': {}, 'to': {}}
    from_slice = ndslice(ndim=ndim, axis=axis, **bounds['from'])
    to_slice = ndslice(ndim=ndim, axis=axis, **bounds['to'])
    return from_slice, to_slice


def make_delayed(array, delays, copy=True, memsafe=False):
    """inserts staggered replications of array along a particular dimension

    Delay Values
    ------------
    - negative values correspond to shifting backward along dimension
    - positive values correspond to shifting forward along dimension
    - zero values correspond to copies of original array
    """
    delayed = np.zeros((array.shape[0], array.shape[1] * len(delays)))

    for d, delay in enumerate(delays):
        delayed_array = np.zeros(array.shape)
        array_slice, delay_slice = shift_slice(delay, 0, array.ndim)
        delayed_array[delay_slice] = array[array_slice]
        delayed[:, (d * array.shape[1]):((d + 1) * array.shape[1])] = delayed_array
    return delayed
