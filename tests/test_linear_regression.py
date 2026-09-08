"""Regression checks to ensure _aqat_par reproduces full matrix operations.
Related to a possible bug on terminal nodes double counting.
"""

import numpy as np
import pandas as pd

from tx_fast_hydrology.muskingum import Muskingum
from tx_fast_hydrology.nutils import _aqat_par


def test_original_matrix_free_covariance_matches_dense_state_space():
    data = {
        'name': 'linear-confluence',
        'datetime': pd.Timestamp('2023-05-13T00:00:00Z'),
        'timedelta': pd.Timedelta(hours=1),
        'reach_ids': ['headwater-a', 'headwater-b', 'outlet'],
        'startnodes': np.arange(3, dtype=np.int64),
        'endnodes': np.array([2, 2, 2], dtype=np.int64),
        'K': np.full(3, 3600.0, dtype=np.float64),
        'X': np.full(3, 0.3, dtype=np.float64),
        'o_t': np.zeros(3, dtype=np.float64),
    }
    model = Muskingum(data, create_state_space=True)
    generator = np.random.default_rng(2401)
    factor = generator.normal(size=(model.n, model.n))
    covariance = factor @ factor.T

    actual = _aqat_par(
        covariance,
        np.empty_like(covariance),
        model.startnodes[model.indegree == 0],
        model.endnodes,
        model.alpha,
        model.beta,
        model.chi,
        model.indegree,
    )
    expected = model.A @ covariance @ model.A.T

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
