import os
import random
import tracemalloc
from pathlib import Path
from importlib import metadata
import json as jsonlib
from datetime import datetime, timezone, timedelta
import asyncio
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, APIRouter
from fastapi.responses import RedirectResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from loguru import logger
from tx_fast_hydrology.muskingum import ModelCollection
from tx_fast_hydrology.da import KalmanFilter
from tx_fast_hydrology.simulation import AsyncSimulation, CheckPoint
from tx_fast_hydrology.download import (
    download_gage_data,
    get_forcing_directories,
    get_forecast_path,
    download_nwm_forcings,
    download_nwm_forcings2,
    download_nwm_streamflow,
)
from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field
import xarray as xr
from typing import Optional, cast

from tx_fast_hydrology.s3 import (
    upload_file_to_s3,
    save_file_from_s3,
    S3Settings,
)


class RetryableError(Exception):
    """Errors that should just backoff and retry the next tick."""

    pass


def _is_no_concat_error(err: Exception) -> bool:
    return isinstance(err, ValueError) and "No objects to concatenate" in str(err)


def _get_historic_path(output_path: Path, ref_datetime: datetime) -> str:
    dt_str = ref_datetime.strftime("%Y%m%d%H")  # e.g. 2025080123

    historic_path = str(output_path.parent / "hist" / f"{dt_str}_{output_path.name}")

    return str(historic_path)


def _write_gage_geojson(
    app: FastAPI,
    outputs: pd.DataFrame,
    streamflow_nwm: pd.DataFrame,
    measurements: pd.DataFrame,
    out_local: Path,
    s3_live_key: str,
    s3_hist_key_dt: datetime,
) -> None:
    """
    Build a GeoJSON FeatureCollection of gage points using in-memory metadata from app.state.all_ids.

    Requirements in app.state.all_ids (column names are flexible; the function will search for them):
      - key:   'comid' (string), optional: 'gage_id', 'timeseries_id', 'locationName'
      - coords: one of ['locationLatitude','lat','Latitude','LAT'] and
                one of ['locationLongitude','lon','Longitude','LON','long']
    Joins on COMID (string) to columns of outputs/streamflow/measurements.
    """
    # ---- Gather metadata from state (no file I/O) ----
    if not hasattr(app.state, "all_ids"):
        logger.error("app.state.all_ids is not set; cannot build gage GeoJSON.")
        return

    df_loc = app.state.all_ids.copy()

    # Ensure types (preserve leading zeros for gage_id; comid as string to match frames)
    if "gage_id" in df_loc.columns:
        df_loc["gage_id"] = df_loc["gage_id"].astype("string")
    if "comid" not in df_loc.columns:
        logger.error("app.state.all_ids has no 'comid' column; cannot build gage GeoJSON.")
        return
    df_loc["comid"] = df_loc["comid"].astype("string")

    # Find coordinate columns
    lat_col = next(
        (c for c in ["locationLatitude", "lat", "Latitude", "LAT"] if c in df_loc.columns), None
    )
    lon_col = next(
        (c for c in ["locationLongitude", "lon", "Longitude", "LON", "long"] if c in df_loc.columns),
        None,
    )
    if not lat_col or not lon_col:
        logger.error("app.state.all_ids is missing recognizable lat/lon columns.")
        return

    # Clean coords
    df_loc[lat_col] = pd.to_numeric(df_loc[lat_col], errors="coerce")
    df_loc[lon_col] = pd.to_numeric(df_loc[lon_col], errors="coerce")
    df_loc = df_loc.dropna(subset=[lat_col, lon_col])

    # Join domain: COMIDs that exist in outputs (and friends)
    comids = sorted(set(outputs.columns.astype(str)).intersection(set(df_loc["comid"])))
    if not comids:
        logger.error("No overlapping COMIDs between outputs and app.state.all_ids.")
        return

    # Align views and time axis
    time_axis = [ts.isoformat() for ts in outputs.index]
    outputs_view = outputs.loc[:, comids]
    streamflow_nwm_view = streamflow_nwm.loc[:, comids].reindex(index=outputs.index, columns=comids)
    measurements_view = measurements.loc[:, comids].reindex(index=outputs.index, columns=comids)

    latest_t = outputs.index[-1]

    # Index metadata by comid for O(1) row access
    meta = df_loc.set_index("comid", drop=False)

    features = []
    for cid in comids:
        m = meta.loc[cid]
        lat = float(m[lat_col])
        lon = float(m[lon_col])
        gage_id = m["gage_id"] if "gage_id" in m and pd.notna(m["gage_id"]) else None
        loc_name = m["locationName"] if "locationName" in m and pd.notna(m["locationName"]) else None
        ts_id = m["timeseries_id"] if "timeseries_id" in m and pd.notna(m["timeseries_id"]) else None

        da_series = outputs_view[cid].tolist()
        nwm_series = streamflow_nwm_view[cid].tolist()
        meas_series = measurements_view[cid].tolist()

        latest_da = (
            float(outputs_view.loc[latest_t, cid])
            if pd.notna(outputs_view.loc[latest_t, cid])
            else None
        )
        latest_nwm = (
            float(streamflow_nwm_view.loc[latest_t, cid])
            if pd.notna(streamflow_nwm_view.loc[latest_t, cid])
            else None
        )
        latest_meas = (
            float(measurements_view.loc[latest_t, cid])
            if pd.notna(measurements_view.loc[latest_t, cid])
            else None
        )

        props = {
            "comid": cid,
            "gage_id": gage_id,
            "timeseries_id": ts_id,
            "locationName": loc_name,
            "time_utc": time_axis,
            "streamflow_da": da_series,
            "streamflow_nwm": nwm_series,
            "measurements": meas_series,
            "latest_da": latest_da,
            "latest_nwm": latest_nwm,
            "latest_measurement": latest_meas,
        }
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": props,
            }
        )

    fc = {"type": "FeatureCollection", "features": features}

    # Save locally and upload (live + historic)
    try:
        out_local.parent.mkdir(parents=True, exist_ok=True)
        with out_local.open("w", encoding="utf-8") as f:
            jsonlib.dump(fc, f)
        logger.info(f"Gage GeoJSON written: {out_local}")
    except Exception as e:
        logger.error(f"Failed to write gage GeoJSON locally: {e}")
        return

    try:
        upload_file_to_s3(
            bucket_name=S3Settings().bucket_name,
            s3_key=s3_live_key,
            filename=out_local,
        )
        # hist_key = _get_historic_path(Path(s3_live_key), s3_hist_key_dt)
        # upload_file_to_s3(
        #     bucket_name=S3Settings().bucket_name,
        #     s3_key=hist_key,
        #     filename=out_local,
        # )
        logger.info(f"Gage GeoJSON uploaded to s3://{S3Settings().bucket_name}/{s3_live_key}")
    except Exception as e:
        logger.error(f"Failed to upload gage GeoJSON to S3: {e}")


class S3Asset(BaseModel):
    location: str
    target: Optional[str] = None

    def resolve(self, cache_dir: str) -> None:
        save_file_from_s3(
            bucket_name=S3Settings().bucket_name,
            object_key=self.location,
            local_dir=Path(cache_dir),
            target=self.target,
        )

    def get_target_path(self) -> Path:
        if self.target:
            return self.target
        return self.location


class TxFastHydrologySettings(BaseSettings):
    cache_dir: str = "./cache"
    gage_lookback_hours: int = 3
    s3_assets: list[S3Asset] = [
        S3Asset(
            location="nwm_txdot_config/tx_fast_hydrology_assets/KF_v001/comids_mod.csv",
            target="comids_mod.csv",
        ),
        S3Asset(
            location="nwm_txdot_config/tx_fast_hydrology_assets/KF_v001/usgs_subset_attila_extended.csv",
            target="usgs_subset_attila_extended.csv",
        ),
        S3Asset(
            location="nwm_txdot_config/tx_fast_hydrology_assets/KF_v001/huc8_no_lake.json",
            target="huc8_no_lake.json",
        ),
        S3Asset(
            location="nwm_txdot_config/tx_fast_hydrology_assets/KF_v001/huc2_12_nhd_min.json",
            target="huc2_12_nhd_min.json",
        ),
    ]
    tick_dt: float = 600
    # assets
    comids_path: str = "./cache/comids_mod.csv"
    gage_ids_path: str = "./cache/usgs_subset_attila_extended.csv"
    model_path: str = "./cache/huc8_no_lake.json"
    stream_network_path: str = "./cache/huc2_12_nhd_min.json"  # this is for the /map endpoint
    # outputs
    streamflow_output_path: str = "nwm_txdot_output/short_range_da_kf/streamflow_kf_sr.nc"
    gage_geojson_key: str = "nwm_txdot_output/short_range_da_kf/streamflow_gages.geojson"
    #
    trigger_workflow: Optional[str] = Field(
        None,
        description="Workflow definition to trigget with t0 after a successful DA run",
    )


# Constants


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Initialize the app state and start periodic updates."""
        tracemalloc.start()  # Start tracing memory allocations

        # get settings from env vars
        app.state.settings = TxFastHydrologySettings()
        app.state.settings = cast(TxFastHydrologySettings, app.state.settings)  # help the typecheck
        # apply settings
        app.state.tick_dt = app.state.settings.tick_dt  # 10 minutes

        app.state.health = {
            "last_attempt_at": None,  # UTC ISO string
            "last_success_at": None,  # UTC ISO string
            "last_model_run_time": None,
            "last_gage_sampling_time": None,
            "last_error": None,  # str or None
            "last_skip_at": None,  # UTC ISO time a tick was skipped
            "last_skip_reason": None,  # reason string for the last skip
            "consecutive_failures": 0,
        }

        Path(app.state.settings.cache_dir).mkdir(parents=True, exist_ok=True)

        # Pull the necessary assets from S3
        for s3_asset in app.state.settings.s3_assets:
            local_cache = Path(app.state.settings.cache_dir) / s3_asset.get_target_path()
            if local_cache.exists():
                continue
            s3_asset.resolve(cache_dir=Path(app.state.settings.cache_dir))

        # Check for required assets
        app.state.all_ids = pd.read_csv(app.state.settings.gage_ids_path).drop_duplicates(
            subset="comid"
        )
        app.state.comids = pd.read_csv(app.state.settings.comids_path, index_col=0)["0"].values

        # Initialization logic
        logger.info("Initializing the simulation...")
        input_path = app.state.settings.model_path  # "./data/huc8_no_lake.json"
        gage_end_time = pd.to_datetime(datetime.now(timezone.utc))
        gage_start_time = pd.to_datetime(
            datetime.now(timezone.utc) - timedelta(hours=app.state.settings.gage_lookback_hours)
        )

        # app.state.all_ids = app.state.all_ids
        app.state.all_ids["to"] = gage_end_time.isoformat()
        app.state.all_ids["from"] = gage_start_time.isoformat()

        logger.info("Downloading initial gage data...")
        measurements = await download_gage_data(app.state.all_ids.to_dict(orient="records"))
        measurements = measurements.reindex(
            app.state.all_ids["comid"].values.astype(str), axis=1
        ).fillna(0.0)

        measurements = measurements * 0.02831683199881  # CFS to M3PS

        logger.info("Loading model collection...")
        model_collection = ModelCollection.from_file(input_path)
        for model in model_collection.models.values():
            checkpoint = CheckPoint(model, timedelta=3600)
            model.bind_callback(checkpoint, key="checkpoint")
            model_sites = [
                reach_id for reach_id in model.reach_ids if reach_id in measurements.columns
            ]
            if model_sites:
                basin_measurements = measurements[model_sites]
                Q_cov = 2 * np.eye(model.n)
                R_cov = 1e-2 * np.eye(basin_measurements.shape[1])
                P_t_init = Q_cov.copy()
                kf = KalmanFilter(model, basin_measurements, Q_cov, R_cov, P_t_init)
                model.bind_callback(kf, key="kf")

        logger.info("Downloading initial NWM forcings and streamflows...")
        urls = get_forcing_directories()
        nwm_dir, forecast_hour, timestamp = get_forecast_path(urls)
        streamflow = download_nwm_streamflow(
            nwm_dir, forecast_hour=forecast_hour, comids=app.state.comids
        )
        inputs, streamflow_nwm = download_nwm_forcings2(
            nwm_dir, forecast_hour=forecast_hour, comids=app.state.comids
        )

        simulation = AsyncSimulation(model_collection, inputs)
        timestamp = streamflow.index.item()
        streamflow_values = streamflow.loc[timestamp]
        simulation.set_datetime(timestamp)
        simulation.init_states(streamflow_values)
        simulation.save_states()
        outputs, gains = await simulation.simulate()
        outputs = pd.concat([series for series in outputs.values()], axis=1)
        gains2 = pd.concat([series for series in gains.values()], axis=1).iloc[[0]]

        # Add initial objects to app state
        app.state.simulation = simulation
        app.state.outputs = outputs
        app.state.current_timestamp = timestamp
        app.state.streamflow = streamflow

        with open(app.state.settings.stream_network_path) as basin:
            app.state.stream_network = jsonlib.load(basin)

        logger.info("Initialization complete. Starting periodic updates...")

        # all_o_t_gain = np.concatenate(
        #     [model.o_t_gain for model in model_collection.models.values()]
        # )
        # all_reach_ids = np.concatenate(
        #     [model.reach_ids for model in model_collection.models.values()]
        # )

        # # Create the DataFrame from the collected data
        # gain = pd.DataFrame(
        #     data=[all_o_t_gain],  # Single row of o_t_gain values
        #     columns=all_reach_ids,  # Reach IDs as column names
        #     index=streamflow.index,  # Use the datetime index from streamflow
        # )

        # Create a list of column sets
        column_sets = [
            set(streamflow.columns),
            set(inputs.columns),
            set(outputs.columns),
            set(gains2.columns),
        ]

        # Check if all column sets are identical
        if all(column_sets[0] == col_set for col_set in column_sets[1:]):
            logger.info("All column sets are consistent.")
        else:
            logger.error("Mismatch detected among column sets.")
            logger.error("Output is skipped")

        outputs = outputs.reindex(columns=streamflow.columns)
        gain = gains2.reindex(columns=streamflow.columns)
        streamflow_nwm = streamflow_nwm.reindex(columns=streamflow.columns)

        # Add NaN row to inputs at t0
        inputs = pd.concat(
            [
                pd.DataFrame([np.nan] * len(inputs.columns), index=inputs.columns).T,
                inputs,
            ]
        )
        inputs.index = outputs.index

        # Add the AA streamflows to the nwm streamflows before the forecasts
        streamflow_nwm = pd.concat([streamflow, streamflow_nwm])

        # reformat the measurements too based on "outputs"
        df_measurements = measurements.reindex(
            index=outputs.index,  # keep all timestamps from outputs
            columns=outputs.columns,  # keep all IDs from outputs
        )
        ds = xr.Dataset(
            {
                "streamflow": (["time", "feature_id"], outputs.values),
                "streamflow_nwm": (["time", "feature_id"], streamflow_nwm.values),
                "inputs": (["time", "feature_id"], inputs.values),
                "diff": (
                    ["time", "feature_id"],
                    outputs.values - streamflow_nwm.values,
                ),
                "measurements": (["time", "feature_id"], df_measurements.values),
                "gain": (["reference_time", "feature_id"], gain.values),
                "streamflow_nwm_aa": (
                    ["reference_time", "feature_id"],
                    streamflow.values,
                ),
            },
            coords={
                "time": outputs.index.tz_localize(None).astype(
                    "datetime64[ns]"
                ),  # Ensure nanosecond precision
                "reference_time": (
                    "reference_time",
                    [np.datetime64(timestamp).astype("datetime64[ns]")],
                ),  # noqa: E501
                "feature_id": [int(f_id) for f_id in outputs.columns],
            },
            attrs={
                "TITLE": "OUTPUT FROM Kalman-Filter by MDB",
                "version": metadata.version("kisters.model_integration.adapters.fast_hydrology"),
                "featureType": "timeSeries",
                "proj4": "+proj=lcc +units=m +a=6370000.0 +b=6370000.0 +lat_1=30.0 +lat_2=60.0 +lat_0=40.0 +lon_0=-97.0",  # noqa: E501
                "model_initialization_time": str(timestamp),
                "station_dimension": "feature_id",
                "model_output_valid_time": str(outputs.index[0]),
                "model_configuration": "short_range",
                "dev_OVRTSWCRT": 1,
                "dev_NOAH_TIMESTEP": 3600,
                "dev_channel_only": 0,
                "dev_channelBucket_only": 0,
                "dev": "dev_ prefix indicates development/internal metrics",
                "kf_version": "0.1.0",
                "kf_gage_sampling_time": str(gage_end_time),
                "kf_gages_USGS": 1,
                "kf_gages_LCRA": 0,
                "kf_gages_RQ30": 0,
                "kf_probabilistic_output": 0,
                "model_coverage": "Texas statewide",
            },
        )
        # Define attributes for each variable
        variable_attrs = {
            "streamflow": {
                "units": "m3 s-1",
                "long_name": "River Flow with Kalman-Filter DA",
            },
            "streamflow_nwm": {
                "units": "m3 s-1",
                "long_name": "River Flow from the NWM",
            },
            "inputs": {
                "units": "m3 s-1",
                "long_name": "NWM inflow forcings= qBucket+qLatRunoff",
            },
            "diff": {
                "units": "m3 s-1",
                "long_name": "Difference between NWM (streamflow_nwm) and DA (streamflow)",
            },
            "measurements": {
                "units": "m3 s-1",
                "long_name": "Streamflow observations by USGS from Datasphere",
            },
            "gain": {
                "units": "m3 s-1",
                "long_name": "Lateral inflow correction from Kalman-Filter",
            },
            "streamflow_nwm_aa": {
                "units": "m3 s-1",
                "long_name": "River Flow from the NWM",
            },
        }

        # Apply attributes to each variable
        for var_name, attrs in variable_attrs.items():
            for attr_name, attr_value in attrs.items():
                ds[var_name].attrs[attr_name] = attr_value

        ds.to_netcdf(
            Path(app.state.settings.cache_dir) / "streamflow_output.nc",
            engine="netcdf4",
        )

        upload_file_to_s3(
            bucket_name=S3Settings().bucket_name,
            s3_key=app.state.settings.streamflow_output_path,
            filename=Path(app.state.settings.cache_dir) / "streamflow_output.nc",
        )

        hist_key = _get_historic_path(
            Path(app.state.settings.streamflow_output_path), outputs.index[0]
        )
        upload_file_to_s3(
            bucket_name=S3Settings().bucket_name,
            s3_key=hist_key,
            filename=Path(app.state.settings.cache_dir) / "streamflow_output.nc",
        )

        # create gage geojson layer
        # join it with the corresponding measurement values
        # join it with a list of streamflow_da, streamflow_nwm
        try:
            gage_geojson_local = Path(app.state.settings.cache_dir) / "gage_points.geojson"
            gage_geojson_live_key = app.state.settings.gage_geojson_key
            _write_gage_geojson(
                app=app,
                outputs=outputs,
                streamflow_nwm=streamflow_nwm,
                measurements=df_measurements,  # aligned measurements created above
                out_local=gage_geojson_local,
                s3_live_key=gage_geojson_live_key,
                s3_hist_key_dt=outputs.index[0],
            )
        except Exception as e:
            logger.error(f"Init gage GeoJSON failed: {e}")

        current, peak = tracemalloc.get_traced_memory()  # Get current and peak memory usage
        logger.info(f"Current memory usage: {current / 1024**2:.2f} MB")
        logger.info(f"Peak memory usage: {peak / 1024**2:.2f} MB")

        tracemalloc.stop()  # Stop tracing memory allocations
        # Start the periodic background task
        now_iso = datetime.now(timezone.utc).isoformat()
        app.state.health.update(
            {
                "last_attempt_at": now_iso,
                "last_success_at": now_iso,
                "last_model_run_time": str(outputs.index[0]),
                "last_gage_sampling_time": str(gage_end_time),
                "last_error": None,
                "consecutive_failures": 0,
            }
        )
        asyncio.create_task(tick(app))

        yield

        logger.info("Application shutdown.")

    app = FastAPI(
        title="Muskingum Forecast API", lifespan=lifespan, root_path="/kf"
    )  # Set the root path to "/kf")

    # Set up static files and templates with absolute paths
    base_dir = os.path.dirname(__file__)
    static_dir = os.path.join(base_dir, "static")
    templates_dir = os.path.join(base_dir, "templates")

    # app.mount("/static", StaticFiles(directory=static_dir), name="static")
    templates = Jinja2Templates(directory=templates_dir)
    templates.env.globals["static_url"] = f"{app.root_path}/static"

    router = APIRouter()

    @router.get("/forecast/{reach_id}")
    async def reach_forecast(reach_id: str):
        outputs = app.state.outputs
        timestamp_utc = [index.isoformat() for index in outputs.index]
        streamflow_cms = [value for value in outputs[reach_id].values]
        json_output = {
            "timestamp__utc": timestamp_utc,
            "streamflow__cms": streamflow_cms,
        }
        return JSONResponse(content=json_output)

    @router.get("/diff")
    async def reach_diff():
        outputs = app.state.outputs
        streamflow = app.state.streamflow
        time_index = streamflow.index.item()
        diff = outputs.loc[time_index, streamflow.columns] - streamflow
        pct_diff = diff / streamflow
        diff = diff.loc[time_index].fillna(0.0).replace([np.inf, -np.inf], 0.0).to_dict()
        pct_diff = pct_diff.loc[time_index].fillna(0.0).replace([np.inf, -np.inf], 0.0).to_dict()
        json_output = {"diff__cms": diff, "pct_diff__pct": pct_diff}
        return JSONResponse(content=json_output)

    @router.get("/map")
    async def map_handler(request: Request):
        static_url = f"{request.base_url}static/style.css"  # noqa: F841
        stream_network = app.state.stream_network
        outputs = app.state.outputs
        streamflow = app.state.streamflow
        time_index = streamflow.index.item()
        diff = outputs.loc[time_index, streamflow.columns] - streamflow
        pct_diff = diff / streamflow
        diff = diff.loc[time_index].fillna(0.0).replace([np.inf, -np.inf], 0.0)
        pct_diff = pct_diff.loc[time_index].fillna(0.0).replace([np.inf, -np.inf], 0.0)
        hi = float(diff.quantile(0.90))
        lo = float(diff.quantile(0.10))
        for feature in stream_network["features"]:
            if str(feature["properties"]["COMID"]) in streamflow.columns:
                comid = str(feature["properties"]["COMID"])
                path_diff = diff[comid]
                if path_diff > hi:
                    c = "positive"
                elif path_diff < lo:
                    c = "negative"
                else:
                    c = "zero"
                feature["properties"]["change"] = c
            else:
                feature["properties"]["change"] = 0

        logger.info(f"URL for static style.css: {request.url_for('static', filename='style.css')}")

        return templates.TemplateResponse(
            "show_page.html", {"request": request, "streams_json": stream_network}
        )

    @router.get("/static/{filename:path}", include_in_schema=False)
    async def static(filename: str):
        file_path = os.path.join(static_dir, filename)
        if os.path.exists(file_path):
            return FileResponse(file_path)
        else:
            return JSONResponse({"error": "File not found"}, status_code=404)

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url="/kf/docs")

    @app.get("/health", include_in_schema=False)
    async def health_check():
        """
        A simple health check endpoint to confirm the service is running.
        """
        h = app.state.health
        return JSONResponse(
            {
                "status": "OK" if h["last_success_at"] else "INIT",
                "last_attempt_at": h["last_attempt_at"],
                "last_success_at": h["last_success_at"],  # <= you asked for this explicitly
                "last_error": h["last_error"],
                "consecutive_failures": h["consecutive_failures"],
            },
            status_code=200,
        )

    # app.include_router(router)
    # for route in app.routes:
    #     logger.info(f"Path: {route.path}, Name: {route.name}")
    return app


async def tick(app: FastAPI):
    """Periodic background task to update simulation and state."""
    base_backoff = 5.0  # seconds
    max_backoff = 300.0  # cap at 5 minutes

    consecutive_failures = 0

    while True:
        # Always sleep first so we keep the cadence even after a failure
        tick_dt = float(getattr(app.state, "tick_dt", 600))
        logger.info(f"Sleeping for {tick_dt / 60} minutes ...")
        await asyncio.sleep(tick_dt)

        # record attempt
        now_iso = datetime.now(timezone.utc).isoformat()
        app.state.health["last_attempt_at"] = now_iso

        try:
            tracemalloc.start()

            simulation = cast(AsyncSimulation, app.state.simulation)
            last_timestamp = app.state.current_timestamp

            # --- Forcings discover ---
            urls = get_forcing_directories()
            nwm_dir, forecast_hour, timestamp = get_forecast_path(urls)

            if timestamp <= last_timestamp:
                logger.info("No new forcings available; skipping update")
                # successful “no-op” still clears failures
                app.state.health.update(
                    {
                        "last_success_at": now_iso,
                        "last_error": None,
                        "consecutive_failures": 0,
                    }
                )
                consecutive_failures = 0
                continue

            logger.info(f"New forcings at {timestamp}")

            # --- Streamflow + inputs ---
            streamflow = download_nwm_streamflow(
                nwm_dir, forecast_hour=forecast_hour, comids=app.state.comids
            )
            inputs, streamflow_nwm = download_nwm_forcings(
                nwm_dir, forecast_hour=forecast_hour, comids=app.state.comids
            )

            # Load previous states and inputs
            simulation.load_states()
            simulation.inputs = simulation.load_inputs(inputs)

            # --- Gage data window (robust) ---
            gage_end_time = max(pd.to_datetime(datetime.now(timezone.utc)), timestamp)
            gage_start_time = min(
                pd.to_datetime(datetime.now(timezone.utc))
                - timedelta(hours=app.state.settings.gage_lookback_hours),
                last_timestamp,
            )
            app.state.all_ids["to"] = gage_end_time.isoformat()
            app.state.all_ids["from"] = gage_start_time.isoformat()

            logger.info("Downloading gage data with safeguard retries...")
            MAX_GAGE_RETRIES = 5
            gage_attempt = 0
            last_gage_error: str | None = None

            while gage_attempt < MAX_GAGE_RETRIES:
                try:
                    measurements = await download_gage_data(
                        app.state.all_ids.to_dict(orient="records")
                    )

                    # If function returned an empty frame, treat as retryable “no data”
                    if measurements is None or getattr(measurements, "empty", False):
                        raise ValueError("No objects to concatenate (empty measurements)")

                    # Reindex & sanitize
                    measurements = measurements.reindex(
                        app.state.all_ids["comid"].values.astype(str), axis=1
                    ).fillna(0.0)

                    measurements = measurements * 0.02831683199881  # CFS to M3PS
                    logger.info("Gage data downloaded")
                    break  # success

                except Exception as e:
                    gage_attempt += 1
                    last_gage_error = f"{type(e).__name__}: {e}"
                    if _is_no_concat_error(e):
                        # brief backoff so the data source has a moment to catch up
                        backoff = min(10.0, 2.0 * gage_attempt)  # 2s, 4s, 6s, 8s, 10s
                        logger.warning(
                            f"Gage data unavailable (attempt {gage_attempt}/{MAX_GAGE_RETRIES}): "
                            f"{e}. Retrying in {backoff:.1f}s..."
                        )
                        await asyncio.sleep(backoff)
                        continue
                    else:
                        # different error -> propagate (you may choose to treat as retryable too)
                        raise

            # If exhausted retries, SKIP this tick cleanly
            if gage_attempt >= MAX_GAGE_RETRIES:
                msg = (
                    "Skipping tick: reached max gage retries due to no gage data. "
                    f"Last error: {last_gage_error}"
                )
                logger.error(msg)

                now_iso = datetime.now(timezone.utc).isoformat()
                app.state.health.update(
                    {
                        "last_attempt_at": now_iso,
                        "last_skip_at": now_iso,
                        "last_skip_reason": "no_gage_data",
                        "last_error": last_gage_error,
                        "consecutive_failures": app.state.health.get("consecutive_failures", 0) + 1,
                    }
                )
                # go wait for the next loop interval without failing the task
                continue

            # Assign measurements into model callbacks if present
            logger.info("Assigning gage data to subbasin models...")
            for model in simulation.model_collection.models.values():
                if hasattr(model, "callbacks") and "kf" in model.callbacks:
                    mcols = model.callbacks["kf"].measurements.columns
                    basin_meas = measurements[mcols]
                    model.callbacks["kf"].measurements = basin_meas

            logger.info(
                "Windows:\n"
                f"  Gage: {measurements.index.min().isoformat()} -> {measurements.index.max().isoformat()}\n"  # noqa
                f"  Inputs: {inputs.index.min().isoformat()} -> {inputs.index.max().isoformat()}\n"
                f"  Model:  {simulation.datetime.isoformat()}"
            )

            # --- Run simulation ---
            logger.info("Beginning simulation...")
            outputs_dict, gains_dict = await simulation.simulate()
            outputs = pd.concat([s for s in outputs_dict.values()], axis=1)
            gains = pd.concat([s for s in gains_dict.values()], axis=1).iloc[[0]]
            logger.info("Simulation finished")

            # Update app state
            app.state.simulation = simulation
            app.state.outputs = outputs
            app.state.current_timestamp = timestamp
            app.state.streamflow = streamflow

            # --- Gains marshalling ---
            # all_o_t_gain = np.concatenate(
            #     [m.o_t_gain for m in simulation.model_collection.models.values()]
            # )
            # all_reach_ids = np.concatenate(
            #     [m.reach_ids for m in simulation.model_collection.models.values()]
            # )
            # gain = pd.DataFrame(
            #     data=[all_o_t_gain],
            #     columns=all_reach_ids,
            #     index=streamflow.index,
            # )

            # --- Column consistency check ---
            column_sets = [
                set(streamflow.columns),
                set(inputs.columns),
                set(outputs.columns),
                set(gains.columns),
            ]
            if all(column_sets[0] == cs for cs in column_sets[1:]):
                logger.info("All column sets are consistent.")
            else:
                logger.error("Mismatch detected among column sets. Skipping export for this tick.")
                # Consider a soft failure that still allows next tick
                raise RetryableError("Column mismatch; will retry next tick.")

            # Align and build NetCDF payload
            logger.info("Start reindexing")
            outputs = outputs.reindex(columns=streamflow.columns)
            gain = gains.reindex(columns=streamflow.columns)
            streamflow_nwm = streamflow_nwm.reindex(columns=streamflow.columns)

            logger.info("Inputs concat")
            inputs = pd.concat(
                [
                    pd.DataFrame([np.nan] * len(inputs.columns), index=inputs.columns).T,
                    inputs,
                ]
            )
            inputs.index = outputs.index

            logger.info("Streamflow concat")
            # Prepend AA streamflows to nwm streamflows
            streamflow_nwm = pd.concat([streamflow, streamflow_nwm])

            # reformat the measurements too based on "outputs"
            df_measurements = measurements.reindex(
                index=outputs.index,  # keep all timestamps from outputs
                columns=outputs.columns,  # keep all IDs from outputs
            )
            logger.info("Compose output dataset")
            ds = xr.Dataset(
                {
                    "streamflow": (["time", "feature_id"], outputs.values),
                    "streamflow_nwm": (["time", "feature_id"], streamflow_nwm.values),
                    "inputs": (["time", "feature_id"], inputs.values),
                    "diff": (
                        ["time", "feature_id"],
                        outputs.values - streamflow_nwm.values,
                    ),
                    "measurements": (["time", "feature_id"], df_measurements.values),
                    "gain": (["reference_time", "feature_id"], gain.values),
                    "streamflow_nwm_aa": (
                        ["reference_time", "feature_id"],
                        streamflow.values,
                    ),
                },
                coords={
                    "time": outputs.index.tz_localize(None).astype("datetime64[ns]"),
                    "reference_time": (
                        "reference_time",
                        [np.datetime64(timestamp).astype("datetime64[ns]")],
                    ),
                    "feature_id": [int(fid) for fid in outputs.columns],
                },
                attrs={
                    "TITLE": "OUTPUT FROM Kalman-Filter by MDB",
                    "version": metadata.version("kisters.model_integration.adapters.fast_hydrology"),
                    "featureType": "timeSeries",
                    "proj4": "+proj=lcc +units=m +a=6370000.0 +b=6370000.0 +lat_1=30.0 +lat_2=60.0 +lat_0=40.0 +lon_0=-97.0",  # noqa
                    "model_initialization_time": str(timestamp),
                    "station_dimension": "feature_id",
                    "model_output_valid_time": str(outputs.index[0]),
                    "model_configuration": "short_range",
                    "dev_OVRTSWCRT": 1,
                    "dev_NOAH_TIMESTEP": 3600,
                    "dev_channel_only": 0,
                    "dev_channelBucket_only": 0,
                    "dev": "dev_ prefix indicates development/internal metrics",
                    "kf_version": "0.1.0",
                    "kf_gage_sampling_time": str(gage_end_time),
                    "kf_gages_USGS": 1,
                    "kf_gages_LCRA": 0,
                    "kf_gages_RQ30": 0,
                    "kf_probabilistic_output": 0,
                    "model_coverage": "Texas statewide",
                },
            )
            variable_attrs = {
                "streamflow": {
                    "units": "m3 s-1",
                    "long_name": "River Flow with Kalman-Filter DA",
                },
                "streamflow_nwm": {
                    "units": "m3 s-1",
                    "long_name": "River Flow from the NWM",
                },
                "inputs": {
                    "units": "m3 s-1",
                    "long_name": "NWM inflow forcings= qBucket+qLatRunoff",
                },
                "diff": {
                    "units": "m3 s-1",
                    "long_name": "Difference between NWM and DA",
                },
                "measurements": {
                    "units": "m3 s-1",
                    "long_name": "Streamflow observations by USGS from Datasphere",
                },
                "gain": {
                    "units": "m3 s-1",
                    "long_name": "Lateral inflow correction from Kalman-Filter",
                },
                "streamflow_nwm_aa": {
                    "units": "m3 s-1",
                    "long_name": "River Flow from the NWM",
                },
            }
            for v, attrs in variable_attrs.items():
                for k, val in attrs.items():
                    ds[v].attrs[k] = val

            logger.info("Save temporary output file")
            out_nc = Path(app.state.settings.cache_dir) / "streamflow_output.nc"
            ds.to_netcdf(out_nc, engine="netcdf4")

            logger.info("Upload results")
            # upload live
            upload_file_to_s3(
                bucket_name=S3Settings().bucket_name,
                s3_key=app.state.settings.streamflow_output_path,
                filename=out_nc,
            )

            # upload historic
            hist_key = _get_historic_path(
                Path(app.state.settings.streamflow_output_path), outputs.index[0]
            )
            upload_file_to_s3(
                bucket_name=S3Settings().bucket_name,
                s3_key=hist_key,
                filename=out_nc,
            )
            # create geojsons
            try:
                gage_geojson_local = Path(app.state.settings.cache_dir) / "gage_points.geojson"
                gage_geojson_live_key = app.state.settings.gage_geojson_key
                _write_gage_geojson(
                    app=app,
                    outputs=outputs,
                    streamflow_nwm=streamflow_nwm,
                    measurements=df_measurements,  # aligned to outputs above
                    out_local=gage_geojson_local,
                    s3_live_key=gage_geojson_live_key,
                    s3_hist_key_dt=outputs.index[0],
                )
            except Exception as e:
                logger.error(f"Tick gage GeoJSON failed: {e}")

            # success -> reset failure counters & update health
            current, peak = tracemalloc.get_traced_memory()
            logger.info(f"Current memory usage: {current / 1024**2:.2f} MB")
            logger.info(f"Peak memory usage: {peak / 1024**2:.2f} MB")
            logger.info("Data Assimilation run was successful")

            tracemalloc.stop()

            consecutive_failures = 0
            app.state.health.update(
                {
                    "last_success_at": now_iso,
                    "last_model_run_time": str(outputs.index[0]),
                    "last_gage_sampling_time": str(gage_end_time),
                    "last_error": None,
                    "consecutive_failures": 0,
                }
            )

            # Optional: trigger workflow (kept as-is)
            if app.state.settings.trigger_workflow:
                try:
                    from kisters.analytics.run_manager.manager import get_run_manager
                    from kisters.analytics.run_manager.schema import WorkflowTrigger

                    rm = get_run_manager()
                    trigger = WorkflowTrigger(
                        t0=timestamp, definition_id=app.state.settings.trigger_workflow
                    )
                    r = await rm.trigger_workflow(trigger)
                    logger.info(f"Workflow triggered:{trigger}, Result: {r}")
                except Exception as e:
                    logger.error(f"Workflow trigger failed: {e}")
                    # do not fail the tick for trigger issues

        except RetryableError as e:
            # backoff but keep loop alive
            tracemalloc.stop()
            consecutive_failures += 1
            app.state.health.update(
                {
                    "last_error": str(e),
                    "consecutive_failures": consecutive_failures,
                }
            )
            backoff = min(max_backoff, base_backoff * (2 ** min(consecutive_failures - 1, 6)))
            delay = backoff + random.uniform(0, 1)
            logger.warning(f"{e} Retrying in {delay:.2f}s.")
            await asyncio.sleep(delay)
            continue

        except Exception:
            tracemalloc.stop()
            consecutive_failures += 1
            backoff = min(max_backoff, base_backoff * (2 ** min(consecutive_failures - 1, 6)))
            delay = backoff + random.uniform(0, 1)

            logger.exception("Tick error. Retrying in %.2fs", delay)  # prints traceback

            await asyncio.sleep(delay)
            continue
