"""
.. include:: ../README.md

.. include:: ../examples/regression_example.md
.. include:: ../examples/calibration_example.md
"""
REGRESSION = 'regression'
CLASSIFICATION = 'classification'

from laplace_bayeslib.baselaplace import BaseLaplace, ParametricLaplace, KronLaplace, DiagLaplace, LowRankLaplace
from laplace_bayeslib.laplace import Laplace
# from laplace_bayeslib.marglik_training import marglik_training

__all__ = ['Laplace',  # direct access to all Laplace classes via unified interface
           'BaseLaplace', 'ParametricLaplace',  # base-class and its (first-level) subclasses
           'KronLaplace', 'DiagLaplace', 'LowRankLaplace',  # all-weights
           ]  # methods
