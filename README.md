
# Efficient multi-target ridge regression

**NOTE: This code was written by Storm Slivkoff in 2016. It is a subset of the Gallant lab analysis codebase. More recently, the lab has moved on to GPU multi-kernel methods. This repo is being shared for expository purposes.**

This package contains many algorithms for fitting cross-validated ridge regression models, particularly for the case where there are many regression targets that need to be solved in parallel.

## Files of Interest
- `notebooks/`
  - `ridge_benchmarks/` speed and precision benchmarks of regression code
  - `ridge_examples/` examples of how to use the code
  - `npfast/` speed and precision benchmarks of npfast functions
- `regression_code/storm/`
  - `cv.py` code for various types of cross-validation
  - `ridge.py` implements main `solve_ridge` function
  - `solvers.py` contains many optimized algorithms for solving the ridge problem
  - `npfast` reimplementation of numpy functions for speed (see below)

## Ridge Regression

#### Background
Ridge regression is a type of regularized linear regression. Whereas ordinary least squares finds a set of weights <img src="https://render.githubusercontent.com/render/math?math=\Beta"> that minimizes the quantity <img src="https://render.githubusercontent.com/render/math?math=(Y - X \Beta)^T (Y - X \Beta)">, ridge regression instead finds a  <img src="https://render.githubusercontent.com/render/math?math=\Beta"> that minimizes the quantity <img src="https://render.githubusercontent.com/render/math?math=(Y - X \Beta)^T (Y - X \Beta) %2B \lambda \left\lVert\Beta\right\rVert^2">. Here <img src="https://render.githubusercontent.com/render/math?math=\lambda"> is a regularization hyperparameter.

#### Solving Ridge regression

Ridge regression problems can be solved efficiently because they have a known analytical solution, <img src="https://render.githubusercontent.com/render/math?math=\Beta=(X^T X %2B \lambda)^{-1} X^T Y">. With the right numerical linear algebra methods, it is possible to efficiently obtain solutions for large numbers of <img src="https://render.githubusercontent.com/render/math?math=\lambda"> values and <img src="https://render.githubusercontent.com/render/math?math=Y"> regressands. The solution given above works well when there are more samples than features in <img src="https://render.githubusercontent.com/render/math?math=X">, but for cases where there are more features than samples, the equivalent kernel formulation of the solution (<img src="https://render.githubusercontent.com/render/math?math=\Beta=X^T(X X^T %2B \lambda)^{-1} Y">) can be used for efficiency.

There are many algorithms that can be used to obtain <img src="https://render.githubusercontent.com/render/math?math=\Beta">.
The optimal method depends on the sizes of the input matrices, the number of <img src="https://render.githubusercontent.com/render/math?math=\lambda"> values that need to be evaluated, and the required levels of precision and stability. For example, Cholesky decompositions are the fastest way to solve the problem for a single <img src="https://render.githubusercontent.com/render/math?math=\lambda"> value, but they can lead to significant instability. Using spectral decompositions and singular value decompositions are fastest when solving a large number of <img src="https://render.githubusercontent.com/render/math?math=\lambda"> values because they can express the <img src="https://render.githubusercontent.com/render/math?math=(X X^T %2B \lambda)^{-1}"> inversion as a simple matrix product. QR decompositions are a relatively fast and stable way to solve the problem for single <img src="https://render.githubusercontent.com/render/math?math=\lambda"> values. The `regression_code.storm.solvers` module documents these algorithms in extensive detail and provides heuristics for when each of them should be used.

#### Cross Validation

The value of <img src="https://render.githubusercontent.com/render/math?math=\lambda"> can significantly affect the quality of the model, especially in regimes of high noise and/or liited numbers of samples. In many cases, there will be an optimal value of <img src="https://render.githubusercontent.com/render/math?math=\lambda"> that produces the lowest error in the quantity <img src="https://render.githubusercontent.com/render/math?math=\left\lVert Y - X \Beta \right\rVert^2">. To find this optimal value, cross validation (or other related procedures) should be used. In particular there should be a separation between fit and test sets to avoid overfitting. The code in `regression_code.storm.cv` provides utilities for performing different types of cross validation, including nested cross-validation and temporally-aware data chunking to account for possible autocorrelation in the input data.

## npfast

**NOTE: These functions were developed as improvements to numpy 1.11.1. There have since been many improvements to numpy that render some of these npfast functions obsolete (including a [pull request](https://github.com/numpy/numpy/pull/15715) where I merged a small optimization from npfast upstream to numpy). Again, this repo is shared for expository purposes. You should perform your own benchmarks to determine whether any of these techniques will be useful for your particular use case.**

This package implements faster versions of common numpy functions. These improvements were possible because the native numpy versions were either single-threaded or created unnecessary matrix copies.

#### Techniques for speedup
1. **create multi-threaded versions of functions** Functions like `sum()` and `mean()` are single-threaded in numpy. However, they can be expressed in terms of vector operations, allowing them to take advantage of numpy's multi-threaded machinery. This can come with a precision penalty for certain matrix sizes (see notebook below).
2. **avoid creating unncessary matrix copies** Multi-step operations such as standard deviation, zscores, or Pearson's correlation are computed in multiple steps. Using in-place operations for the intermediate steps and/or the final step can save time when dealing with large matrices.
3. **use [numexpr](https://github.com/pydata/numexpr)** numexpr is a 3rd party library that uses its own op-codes to perform sequences of array operations. These op-codes are multi-threaded and can reduce unnecessary matrix copies.

#### Functions
- Array Functions: `astype()`, `copy()`, `nan_to_num()`, `isnan()`
- Functions implemented: `sum()`, `mean()`, `nanmean()`, `std()`, `zscore()`, `correlate()`, `multi_dot()`

#### Other numpy optimizations you should do
- avoid loops, or if you must loop, use `numba`
- use broadcasting
- use `np.linalg.multi_dot` when multiplying 3 or more matrices
- use in-place operators and `out` arguments when appropriate
- measure whether openBLAS / MKL / BLIS work best for your cpu

