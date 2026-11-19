"""
parallelisation
===============

Description
-----------
This module is an adjusted version of pymoo.parallelization.joblib to address a
bug in pymo 0.6.2, in which the full joblib module is stored to self. This
causes a TypeError when running an optimisation with save_history=True, as a
module is not picklable. A temporary fix is implemented here, while awaiting a
bugfix for the associated issue on GitHub
(https://github.com/anyoptimization/pymoo/issues/815).

Notes
-----
This module will likely be removed from a future version of the UDC once the
bug is fixed in pymoo.

References
----------
For more details on pymoo and its capabilities, refer to the official
documentation:
https://pymoo.org/

Versioning
----------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@tudelft.nl
Version: 1.0

Changelog:
- V1.0: Initial implementation.
"""


class JoblibParallelisation:
    """
    Parallelisation using joblib.

    Args:
        n_jobs: Number of parallel jobs. -1 uses all available cores (default: -1).
        **kwargs: Additional arguments passed to joblib.Parallel.
    """

    def __init__(self, n_jobs: int = -1, **kwargs: object) -> None:
        try:
            from joblib import Parallel, delayed
        except ImportError:
            raise ImportError("joblib must be installed: pip install joblib")

        self.Parallel = Parallel
        self.delayed = delayed
        self.n_jobs = n_jobs
        self.kwargs = kwargs

    def __call__(self, f, X):
        with self.Parallel(n_jobs=self.n_jobs, **self.kwargs) as parallel:
            return parallel(self.delayed(f)(x) for x in X)

    def __getstate__(self):
        return self.__dict__.copy()
