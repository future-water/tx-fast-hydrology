"""Dataset parity gate for submodel 754 against NWM no-DA streamflow.

Run ``python tests/test_nwm_mc_parity.py`` from the repository root to run
the tests and write diagnostics. Discovery with unittest/pytest does not
write files. Input fixtures live in ``test_data``; install dependencies
from ``requirements.txt``.
Acceptance requires pooled NSE >= 0.99 across all scored reaches and hours.
Only short-timestep routing is evaluated. Per-reach metrics and hourly
errors are retained as diagnostics.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# Resolve imports and fixtures from this checkout, independently of the cwd.
ROOT = Path(__file__).resolve().parent
BUNDLE_ROOT = ROOT.parent
DATA_ROOT = BUNDLE_ROOT / "test_data"
SOURCE_ROOT = BUNDLE_ROOT
sys.path.insert(0, str(SOURCE_ROOT))

from tx_fast_hydrology.io import ModelDecoder
from tx_fast_hydrology.muskingum_cunge import MuskingumCunge


START = pd.Timestamp("2023-05-13T00:00:00Z")
END = pd.Timestamp("2023-05-14T23:00:00Z")
SCORE_START = START + pd.Timedelta(hours=24)
STEP = pd.Timedelta(minutes=5)
REACH_COUNT = 479
MINIMUM_POOLED_NSE = 0.99
REFERENCE_KEY = "no_data_assimilation__streamflow"
FORCING_KEYS = (
    "no_data_assimilation__qSfcLatRunoff",
    "no_data_assimilation__qBucket",
)

# Fixture
@dataclass
class data:
    model_data: dict
    geometry: pd.DataFrame
    routelink: pd.DataFrame
    lateral: pd.DataFrame
    reference: pd.DataFrame
    boundaries: list[dict]


def load_data() -> data:
    """Load data"""
    # Load from json
    collection = json.loads(
        (DATA_ROOT / "texas_with_reservoirs_calibrated.json").read_text(),
        cls=ModelDecoder,
    )

    # This should be the only model here
    selected = collection["models"]["754"]
    model_data = selected["model"]
    reach_ids = list(map(str, model_data["reach_ids"]))
    boundaries = []
    for cid in selected["sources"]:
        connection = collection["connections"][cid]
        upstream = collection["models"][str(connection["upstream_model"])]["model"]
        boundaries.append({
            "upstream_reach_id": str(upstream["reach_ids"][connection["upstream_index"]]),
            "downstream_reach_id": reach_ids[connection["downstream_index"]],
        })

    # Make sure size and format match
    all_ids = reach_ids + [row["upstream_reach_id"] for row in boundaries]
    if len(set(all_ids)) != REACH_COUNT + 2:
        raise ValueError("Expected 479 channel COMIDs and two distinct boundary COMIDs")

    # Generate forcings data format
    tables = {key: [] for key in (*FORCING_KEYS, REFERENCE_KEY)}
    for day in pd.date_range(START.normalize(), END.normalize(), freq="1D"):
        path = DATA_ROOT / "nwm_h5" / f"{day:%Y%m%d}00.h5"
        with pd.HDFStore(path, mode="r") as store:
            for key in tables:
                frame = store[key]
                frame.columns = frame.columns.astype(str)
                frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True)).as_unit("ns")
                if frame.index.has_duplicates or frame.columns.has_duplicates:
                    raise ValueError(f"Duplicate timestamps or COMIDs in {path.name}/{key}")
                expected = pd.date_range(day, periods=24, freq="1h")
                pd.testing.assert_index_equal(frame.index, expected, check_names=False)
                tables[key].append(frame.loc[:, all_ids])
    tables = {key: pd.concat(frames) for key, frames in tables.items()}
    reference = tables[REFERENCE_KEY]
    lateral = (tables[FORCING_KEYS[0]] + tables[FORCING_KEYS[1]])[reach_ids].copy()

    # Boundary condition
    for boundary in boundaries:
        lateral[boundary["downstream_reach_id"]] += reference[boundary["upstream_reach_id"]]

    # Load NWM MC params
    with xr.open_dataset(DATA_ROOT / "RouteLink_CONUS.nc", decode_times=False) as ds:
        fields = ("link", "to", "So", "Length", "n", "ChSlp", "BtmWdth", "TopWdth", "TopWdthCC", "nCC", "NHDWaterbodyComID")
        routelink = pd.DataFrame({field: ds[field].values for field in fields})
    routelink["link"] = routelink["link"].astype(np.int64).astype(str)
    routelink["to"] = routelink["to"].astype(np.int64).astype(str)
    geometry = routelink.set_index("link").loc[reach_ids].reset_index()
    return data(model_data, geometry, routelink, lateral, reference[reach_ids], boundaries)


def route() -> pd.DataFrame:
    """Initialize, warm up May 13, and score through May 14."""
    data = load_data()
    # This NWM retrospective seems to use short_ts
    model = MuskingumCunge(copy.deepcopy(data.model_data), assume_short_ts=True)
    if not Path(sys.modules[type(model).__module__].__file__).resolve().is_relative_to(SOURCE_ROOT):
        raise RuntimeError("Parity must exercise this bundle's routing code")
    model.load_routelink(data.geometry)
    model.datetime = START
    model.timedelta = STEP
    model.init_states(o_t_next=data.reference.loc[START].to_numpy(dtype=np.float64))
    model.depth[:] = 0.0
    model.K_history = {START: model.K.copy()}
    model.X_history = {START: model.X.copy()}

    # Make sure it isn't assimilating
    if model.callbacks:
        raise RuntimeError("This run should not be assiilating")

    # Ensure it's timestepping correctly
    output = [model.o_t_next.copy()]
    for timestamp in data.lateral.index[1:]:
        forcing = data.lateral.loc[timestamp].to_numpy(dtype=np.float64)
        for _ in range(12):
            model.step(forcing, timedelta=STEP)
        if model.datetime != timestamp:
            raise RuntimeError(f"Routing clock {model.datetime} != reference time {timestamp}")
        output.append(model.o_t_next.copy())
    return pd.DataFrame(output, index=data.reference.index, columns=data.reference.columns)


def scored_arrays(simulation: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    reference = load_data().reference.loc[SCORE_START:END]
    # Assert labels before comparing values; never silently intersect or drop.
    pd.testing.assert_index_equal(simulation.index, load_data().reference.index)
    pd.testing.assert_index_equal(simulation.columns, reference.columns)
    predicted = simulation.loc[SCORE_START:END].to_numpy(dtype=np.float64)
    expected = reference.to_numpy(dtype=np.float64)
    return predicted, expected


def diagnostics() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    predicted, expected = scored_arrays(route())
    error = predicted - expected
    absolute = np.abs(error)
    pooled_variance = float(np.sum((expected - expected.mean()) ** 2))
    if pooled_variance <= 0:
        raise ValueError("Pooled NSE requires nonconstant reference streamflow")
    pooled_nse = float(1 - np.sum(error ** 2) / pooled_variance)
    variance = np.sum((expected - expected.mean(axis=0)) ** 2, axis=0)
    nse = np.full(REACH_COUNT, np.nan)
    np.divide(np.sum(error ** 2, axis=0), variance, out=nse, where=variance > 0)
    nse = 1 - nse
    data = load_data()
    reach_ids = data.reference.columns
    per_reach = pd.DataFrame({
        "reach_id": reach_ids, "samples": len(expected),
        "RMSE_cms": np.sqrt(np.mean(error ** 2, axis=0)),
        "MAE_cms": absolute.mean(axis=0), "max_abs_error_cms": absolute.max(axis=0),
        "bias_cms": error.mean(axis=0), "NSE": nse,
        "meets_nse_threshold": nse >= MINIMUM_POOLED_NSE,
    })
    times = data.reference.loc[SCORE_START:END].index
    detail = pd.DataFrame({
        "time_utc": np.repeat(times, REACH_COUNT),
        "reach_id": np.tile(reach_ids, len(times)),
        "nwm_streamflow_cms": expected.ravel(), "mc_streamflow_cms": predicted.ravel(),
        "error_cms": error.ravel(), "absolute_error_cms": absolute.ravel(),
    })
    worst = np.unravel_index(np.argmax(absolute), absolute.shape)
    summary = {
        "status": "passed" if pooled_nse >= MINIMUM_POOLED_NSE else "failed",
        "submodel": "754", "channel_reaches": REACH_COUNT,
        "reference_key": REFERENCE_KEY, "forcing_keys": list(FORCING_KEYS),
        "initialization_time": START.isoformat(), "initial_depth": "zero; 24-hour warmup",
        "score_start": SCORE_START.isoformat(), "score_end": END.isoformat(),
        "routing_dt_seconds": int(STEP.total_seconds()),
        "assume_short_ts": True, "assimilation": "none",
        "boundary_handling": "two prescribed hourly NWM no-DA reservoir outlets, held at right-hour value",
        "acceptance_criterion": "pooled_NSE >= minimum_pooled_NSE",
        "minimum_pooled_NSE": MINIMUM_POOLED_NSE,
        "samples": int(expected.size),
        "reaches_at_or_above_nse_threshold": int((nse >= MINIMUM_POOLED_NSE).sum()),
        "RMSE_cms": float(np.sqrt(np.mean(error ** 2))),
        "pooled_NSE": pooled_nse,
        "max_abs_error_cms": float(absolute.max()),
        "worst_reach_id": str(reach_ids[worst[1]]), "worst_time_utc": times[worst[0]].isoformat(),
        "worst_nwm_streamflow_cms": float(expected[worst]),
        "worst_mc_streamflow_cms": float(predicted[worst]),
        "limitations": "Hourly forcing/boundary values and no NWM hydraulic restart depths; this test measures agreement and does not establish the source of a mismatch.",
    }
    return summary, per_reach, detail


class NwmNoDaParityTests(unittest.TestCase):
    def test_dataset_has_complete_reach_and_time_coverage(self):
        data = load_data()
        self.assertEqual(data.reference.shape, (48, REACH_COUNT))
        pd.testing.assert_index_equal(
            data.reference.index, pd.date_range(START, END, freq="1h"),
            check_names=False,
        )
        csv_ids = pd.read_csv(DATA_ROOT / "reach_ids_754.csv", dtype=str)["reach_id"].tolist()
        self.assertEqual(data.reference.columns.tolist(), csv_ids)
        for frame in (data.reference, data.lateral):
            self.assertTrue(np.isfinite(frame.to_numpy()).all())
        self.assertTrue((data.reference.to_numpy() >= 0).all())

    def test_topology_and_boundaries_match_routelink(self):
        data = load_data()
        ids = data.reference.columns.tolist()
        positions = {rid: i for i, rid in enumerate(ids)}
        keyed = data.routelink.set_index("link")
        expected_endnodes = [positions.get(keyed.loc[rid, "to"], i) for i, rid in enumerate(ids)]
        np.testing.assert_array_equal(data.model_data["endnodes"], expected_endnodes)
        self.assertEqual(len(data.boundaries), 2)
        for boundary in data.boundaries:
            self.assertEqual(keyed.loc[boundary["upstream_reach_id"], "to"], boundary["downstream_reach_id"])
        self.assertTrue((data.geometry.NHDWaterbodyComID == -9999).all())

    def test_open_loop_routing_is_finite_nonnegative_and_complete(self):
        simulation = route()
        self.assertEqual(simulation.shape, (48, REACH_COUNT))
        self.assertTrue(np.isfinite(simulation.to_numpy()).all())
        self.assertTrue((simulation.to_numpy() >= 0).all())

    def test_mc_no_da_nwm_pooled_nse_is_at_least_099(self):
        summary, _, _ = diagnostics()
        self.assertGreaterEqual(
            summary["pooled_NSE"], MINIMUM_POOLED_NSE,
            msg=(
                f"MC must achieve pooled NSE >= {MINIMUM_POOLED_NSE:.2f} after warmup. "
                f"RMSE={summary['RMSE_cms']:.6f} m3/s, pooled NSE={summary['pooled_NSE']:.6f}, "
                f"max error={summary['max_abs_error_cms']:.6f} m3/s. "
                f"Worst reach/time: {summary['worst_reach_id']} / {summary['worst_time_utc']}. "
                "Run this file directly to export per-reach and per-hour diagnostics."
            ),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "nwm_parity_results")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(NwmNoDaParityTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    summary, per_reach, detail = diagnostics()
    (args.report_dir / "short_timestep_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    per_reach.to_csv(args.report_dir / "short_timestep_per_reach.csv", index=False)
    detail.to_csv(args.report_dir / "short_timestep_hourly_comparison.csv", index=False)
    print(f"short_timestep: {summary['status']}; RMSE={summary['RMSE_cms']:.6f} m3/s; NSE={summary['pooled_NSE']:.6f}; minimum NSE={MINIMUM_POOLED_NSE:.2f}; {summary['samples']} scored samples", flush=True)
    print(f"Reports: {args.report_dir.resolve()}", flush=True)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
