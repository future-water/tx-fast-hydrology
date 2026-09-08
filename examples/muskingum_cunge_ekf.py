"""A very small Muskingum Cunge EKF example which doesn't require any input data"""

import numpy as np
import pandas as pd

from tx_fast_hydrology import ExtendedKalmanFilter, MuskingumCunge


def build_model() -> MuskingumCunge:
    start = pd.Timestamp("2023-05-13T00:00:00Z")
    data = {
        "name": "two-tributary-example",
        "datetime": start,
        "timedelta": pd.Timedelta(minutes=5),
        "reach_ids": ["tributary-a", "tributary-b", "outlet"],
        "startnodes": np.arange(3, dtype=np.int64),
        "endnodes": np.array([2, 2, 2], dtype=np.int64),
        # Initial K and X are replaced by the first hydraulic solve.
        "K": np.full(3, 300.0, dtype=np.float64),
        "X": np.full(3, 0.3, dtype=np.float64),
        "o_t": np.array([1.0, 0.5, 1.5], dtype=np.float64),
        "dx": np.full(3, 1000.0, dtype=np.float64),
    }
    geometry = {
        "So": np.full(3, 0.001, dtype=np.float64),
        "dx": np.full(3, 1000.0, dtype=np.float64),
        "n": np.full(3, 0.035, dtype=np.float64),
        "Cs": np.full(3, 1.0, dtype=np.float64),
        "Bw": np.full(3, 5.0, dtype=np.float64),
        "Tw": np.full(3, 15.0, dtype=np.float64),
        "TwCC": np.full(3, 30.0, dtype=np.float64),
        "nCC": np.full(3, 0.06, dtype=np.float64),
    }
    return MuskingumCunge(data, geometry=geometry)


def main() -> None:
    model = build_model()
    observations = pd.DataFrame(
        {"outlet": [1.6, 1.8, 2.0]},
        index=pd.date_range(model.datetime, periods=3, freq="5min"),
    )
    filter_ = ExtendedKalmanFilter(
        model,
        observations,
        Q_cov=np.eye(model.n) * 0.01,
        R_cov=np.eye(1) * 0.04,
        P_t_init=np.eye(model.n),
    )
    model.bind_callback(filter_, key="ekf")

    lateral_inflow = np.array([0.1, 0.1, 0.0], dtype=np.float64)
    for _ in range(2):
        model.step(lateral_inflow)
        print(model.datetime, dict(zip(model.reach_ids, model.o_t_next)))


if __name__ == "__main__":
    main()
