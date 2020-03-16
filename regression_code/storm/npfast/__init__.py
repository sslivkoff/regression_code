"""faster versions of numpy functions


Main Functions
--------------
- np.sum() --> sum()
- np.mean() --> mean()
- np.nanmean() --> nanmean()
- np.std() --> std()
- scipy.stats.zscore() --> zscore(), zscore_inplace()
- np.isnan() --> isnan()
- np.nan_to_num() --> nan_to_num(), nan_to_num_inplace()
- np.copy() --> copy()


Other Functions
---------------
- [correlate] --> correlate()
- [R^2] --> R_squared()
- reduce(np.ndarray.__add__, arrays) -> list_sum()
- np.linalg.multi_dot() -> multi_dot()
    - is extended here to use np.dot's 'out' argument
    - useful if placing output of multi_dot in a preinitialized array


Tricks
------
1. use np.dot() and np.ones() for summations and means along axes
2. use numexpr for elementwise arithmetic (when available)
3. use inplace calculations when possible
4. these are all currently (numpy=1.11.1, scipy=0.16.0) single-threaded:
    - np.sum()
    - np.mean()
    - np.std()
    - scipy.stats.zscore()
    - np.isnan()
    - np.nan_to_num()
    - np.copy()
- other tricks not used here:
    - use np.dot's out argument to specify where to store output
    - use GPU


Usage
-----
- for inputs with mean 0, use demean=False to speed up computation
- X_inplace() versions of functions are faster but modify their inputs
- numexpr is faster but not required
    - use `sudo pip install numexpr` to install
    - the dot_X() functions are used when numexpr is not available
    - some extra dot_X() inplace functions are available
        - these are faster than normal dot functions but slower than numexpr
- precision can be an lost when summing along large axes
    - float64 will not have signficant issues
    - when dot_sum()ing float32's...
        - np.random.rand(1,000) -> mean absolute error is ~(0.000168)
        - np.random.rand(10,000) -> mean absolute error is ~(0.0617)
        - np.random.rand(100,000) -> mean absolute error is ~(0.188)
    - see npfast_precision_tests.ipynb for tests of absolute/relative precision


Check Future Package Updates
----------------------------
- check whether numpy parallelizes sum, mean, or einsum
- check whether numexpr parallelizes sum()
- run testing notebook []
    - ensure that the current ordering of functions reflects fastest speeds


Possible Todo
-------------
- improving dot_sum would expand functionality of most other functions
    - implement axis=None for whole array computations
    - implement functions for arrays with dimension > 2
"""

from .arithmetic import *
from .arrays import *
from .shift import *
