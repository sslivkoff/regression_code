
import collections
import functools

from regression_code.storm import ridge
from regression_code.storm.ridge import solvers
from regression_code.storm.ridge import translate


ridge_functions = collections.OrderedDict([
    ['huth.ridge.ridge_corr', translate.ridge_corr_wrapper],
    ['aone.models.solve_l2_primal', translate.solve_l2_primal_wrapper],
    ['aone.models.solve_l2_dual', translate.solve_l2_dual_wrapper],
])
for solver in solvers.solvers:
    ridge_functions[solver] = functools.partial(ridge.solve_ridge, solver=solver)


cv_functions = collections.OrderedDict([
    ['cv_ridge', ridge.cv_ridge],
    ['aone.models.cvridge', translate.cvridge_wrapper],
    ['aone.models.kernel_cvridge', translate.kernel_cvridge_wrapper],
    ['huth.ridge.bootstrap_ridge', translate.bootstrap_ridge_wrapper],
])
