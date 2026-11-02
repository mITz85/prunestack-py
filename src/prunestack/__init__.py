"""PruneStack: meta-feature guided performance-diversity pruning of heterogeneous ensembles.

A pool of heterogeneous base models is pruned with Elephant Herding
Optimization (EHO) under the fitness ``alpha * performance + (1 - alpha) *
diversity`` and the selected subsets are combined by a stacking meta-learner.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("prunestack")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "2.2.0"

__all__ = ["__version__"]
