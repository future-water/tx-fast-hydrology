"""Routing numerical helpers checked against hand calculations."""

import numpy as np
import pytest

from tx_fast_hydrology import nutils
from tx_fast_hydrology.mc_kernel_troute import (
    _hydraulic_geometry,
    submuskingcunge,
    _mc_ax_bu,
)


@pytest.mark.parametrize(
    'method,x,expected',
    [
        (1, -1.0, 2.0),
        (1, 0.0, 2.0),
        (1, 2.0, 4.0),
        (1, 10.0, 12.0),
        (1, 20.0, 12.0),
        (0, 4.0, 2.0),
        (0, 6.0, 12.0),
        (0, 5.0, 2.0),
    ],
)
def test_interpolation_boundaries_and_nearest(method, x, expected):
    result = nutils.interpolate_sample(
        x, np.array([0.0, 10.0]), np.array([[2.0], [12.0]]), method
    )
    np.testing.assert_allclose(result, [expected])


def test_vector_interpolation_and_index_matching():
    xp = np.array([0.0, 10.0])
    actual = nutils.interpolate_samples(
        np.array([-1.0, 5.0, 11.0]), xp, np.array([[2.0, 4.0], [12.0, 24.0]])
    )
    np.testing.assert_allclose(actual, [[2.0, 4.0], [7.0, 14.0], [12.0, 24.0]])
    assert nutils.has_same_index(xp, xp.copy())
    assert not nutils.has_same_index(xp, xp + 1)
    assert not nutils.has_same_index(xp, xp[:1])


def test_serial_covariance_kernels_match_dense(linear):
    covariance = np.array([[2.0, 0.1, 0.2], [0.1, 1.0, 0.3], [0.2, 0.3, 3.0]])
    # Compare serial and parallel kernels with the same dense reference.
    args = (
        linear.startnodes[linear.indegree == 0],
        linear.endnodes,
        linear.alpha,
        linear.beta,
        linear.chi,
        linear.indegree,
    )
    for kernel in (nutils._ap, nutils._ap_par):
        np.testing.assert_allclose(
            kernel(covariance, np.empty((3, 3)), *args), linear.A @ covariance
        )
    np.testing.assert_allclose(
        nutils._aqat(covariance, np.empty((3, 3)), *args),
        linear.A @ covariance @ linear.A.T,
    )


@pytest.mark.parametrize(
    'h,compound', [(0.0, 30.0), (1.0, 30.0), (8.0, 30.0), (8.0, 0.0)]
)
def test_hydraulic_geometry_area_and_radius(h, compound):
    result = _hydraulic_geometry(h, 5.0, 5.0, compound, 1.0)
    _, radius, area, flood_area, perimeter, flood_perimeter, _, _ = result
    # Above bankfull, the compound section contributes separate flow area.
    channel_depth = min(h, 5.0) if compound else h
    assert area == pytest.approx((5.0 + channel_depth) * channel_depth)
    assert flood_area == pytest.approx(compound * max(h - 5.0, 0.0))
    assert radius == pytest.approx(
        (area + flood_area) / (perimeter + flood_perimeter)
    )


@pytest.mark.parametrize(
    'bw,tw,cs,ql',
    [
        (5.0, 15.0, 1.0, 1.0),
        (5.0, 5.0, 0.0, 1.0),
        (15.0, 5.0, 1.0, 1.0),
        (5.0, 15.0, 1.0, 0.0),
    ],
)
def test_mc_dry_wet_and_channel_shapes(bw, tw, cs, ql):
    initial = 0.0 if ql == 0.0 else 1.0
    flow, velocity, depth, travel, weight = submuskingcunge(
        initial,
        initial,
        initial,
        ql,
        300.0,
        0.001,
        1000.0,
        0.035,
        cs,
        bw,
        tw,
        30.0,
        0.06,
        0.1,
    )
    assert np.isfinite([flow, velocity, depth, travel, weight]).all()
    assert min(flow, velocity, depth) >= 0.0
    assert travel >= 300.0
    assert 0.0 <= weight <= 0.5
    if initial == 0.0:
        assert (flow, velocity, depth, travel, weight) == (
            0.0,
            0.0,
            0.0,
            300.0,
            0.5,
        )


def test_mc_cycle_rejected(geometry):
    with pytest.raises(ValueError, match='cyclic'):
        _mc_ax_bu(
            np.array([], dtype=np.int64),
            np.array([1, 2, 0]),
            np.ones(3, dtype=np.int64),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            300.0,
            geometry['So'],
            geometry['dx'],
            geometry['n'],
            geometry['Cs'],
            geometry['Bw'],
            geometry['Tw'],
            geometry['TwCC'],
            geometry['nCC'],
            np.zeros(3),
            False,
        )
