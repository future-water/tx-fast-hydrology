# tx-fast-hydrology

Routing and data-assimilation tools developed for the TxDOT FAST project.

This branch preserves the original linear Muskingum implementation and adds:

- nonlinear Muskingum-Cunge routing based on the t-route kernel;
- a frozen-coefficient extended Kalman filter (EKF);
- matrix-free covariance propagation for both the original filter and EKF;
- configurable t-route current-flow and short-timestep traversal modes;
- NWM/WRF-Hydro streamflow nudging with forecast persistence;
- reservoir submodels inside mixed routing collections; and
- direct loading of NextGen Hydrofabric GeoPackages.

## Installation

Python 3.11 or 3.12 is recommended.

```console
python -m venv .venv
.venv/Scripts/python -m pip install --upgrade pip
.venv/Scripts/python -m pip install -e ".[dev]"
```

On Linux or macOS, replace `.venv/Scripts/python` with
`.venv/bin/python`. The NumPy and Numba bounds in `setup.py` are intentional:
Numba 0.59 supports NumPy 1.22 through 1.26.

For an exactly reproducible development environment:

```console
python -m pip install -r requirements.txt
```

## Muskingum-Cunge and the EKF

`MuskingumCunge` is an additive subclass of the original `Muskingum` model.
It updates hydraulic depth, velocity, travel time `K`, and weighting factor `X`
at every routing step.

```python
from tx_fast_hydrology.muskingum_cunge import MuskingumCunge

model = MuskingumCunge(data, geometry=geometry)
model.step(lateral_inflow)
```

The default `assume_short_ts=True` uses saved upstream discharge for both MC
upstream-flow terms and reproduces the bundled NWM validation behavior. Pass
`assume_short_ts=False` for t-route's upstream-to-downstream current-flow mode.
The EKF selects the corresponding matrix-free covariance operator automatically.

## Streamflow nudging

`StreamflowNudging` defaults to t-route's `simple_da` behavior: a valid
observation replaces modeled discharge at its collocated reach, then the last
observation is blended with the current model value using a configurable
exponential decay when observations are missing or have ended. Input time
slices are linearly interpolated to routing times, as in t-route.
When supplied, observation quality is used only to reject values below
`quality_threshold` (default `1.0`); it does not scale accepted observations.
The default callback works with both `Muskingum` and `MuskingumCunge`; the
corrected discharge becomes the routing state used by the next model step.

```python
from tx_fast_hydrology import StreamflowNudging

nudging = StreamflowNudging(
    model, measurements,
    reach_ids=gage_reach_ids,
    decay_coefficient=120.0,
    assimilation_end=forecast_issue_time,
)
model.bind_callback(nudging, key="streamflow_nudging")
nudging.update()
```

The earlier WRF-Hydro implementation remains available explicitly. It supports
observation quality, weighted observation windows, flow-dependent persistence,
and loading `G`, `tau`, `qThresh`, and `expCoeff` from `nudgingParams.nc`. It
also works with both linear `Muskingum` and `MuskingumCunge` routing:

```python
from tx_fast_hydrology import WRFHydroStreamflowNudging

nudging = WRFHydroStreamflowNudging.from_nwm_parameters(
    model, measurements, "nudgingParams_CONUS.nc",
    reach_ids=gage_reach_ids,
    station_ids=gage_station_ids,
    assimilation_end=forecast_issue_time,
)
```

Nudging and the EKF are separate assimilation methods; bind the one intended
for a given experiment rather than applying both to the same observations.

`ExtendedKalmanFilter` performs its covariance prediction after the nonlinear
routing step. The newest `K` and `X` values are frozen for that prediction.
Production code computes `F @ P @ F.T` by applying the frozen routing operator
twice; it never constructs the dense transition matrix `F`.

```python
from tx_fast_hydrology.da import ExtendedKalmanFilter

ekf = ExtendedKalmanFilter(model, measurements, Q_cov, R_cov, P_t_init)
model.bind_callback(ekf, key="ekf")
```

The explicit `state_transition_jacobian()` method exists for diagnostics and
small test oracles only.

## NextGen Hydrofabric input

The loader reads `flowpaths`, `flowpath-attributes`, and `nexus` directly from
a NextGen GeoPackage using SQLite, so GeoPandas is not required.

```python
model = MuskingumCunge.from_hydrofabric(
    "path/to/hydrofabric.gpkg",
    timedelta="5min",
)
```

The existing RouteLink-based geometry loader remains available through
`MuskingumCunge.load_routelink()`.

## Reservoir behavior

Reservoirs remain separate deterministic routing components, matching the
original repository's design. They can be loaded, initialized, serialized,
and connected between linear or Muskingum-Cunge channel submodels. Reservoir
stage and release are not part of `ExtendedKalmanFilter`'s state vector.
Observations associated with a reservoir should therefore be mapped to an
appropriate downstream channel reach unless a future coupling supplies a
reservoir-aware state transition.

The target branch also provides RFC reservoir nudging. This callback replaces
the modeled outlet release with the time-interpolated RFC streamflow forecast
through the final forecast timestamp:

```python
from tx_fast_hydrology.da import ReservoirNudging

rfc = ReservoirNudging(reservoir, forecast_streamflow)
reservoir.bind_callback(rfc, key="rfc")
```

This is an outlet-flow boundary update, not an EKF update: it does not adjust
reservoir stage or covariance. Its callback state participates in the existing
model checkpoint save/load mechanism.

## Tests

The included tests are data-free and cover:

- dynamic Muskingum-Cunge coefficients and state restoration;
- equivalence of matrix-free covariance propagation to a dense test oracle;
- EKF updates, missing observations, and covariance symmetry;
- both MC traversal modes and their matrix-free covariance operators;
- streamflow nudging weights, persistence, routing feedback, and state restore;
- NextGen topology and geometry alignment;
- model-collection serialization for Muskingum-Cunge and reservoirs;
- reservoir rating curves and one-step mass conservation;
- RFC reservoir-outflow interpolation and checkpoint restoration; and
- asynchronous channel-reservoir-channel scheduling.

Run them with:

```console
python -m pytest -q
```

See `MERGE_NOTES.md` for the change inventory and validation boundary.
