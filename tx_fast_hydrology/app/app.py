import os
import time
import tracemalloc
from memory_profiler import profile
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
    download_nwm_streamflow,
)
from pydantic_settings import BaseSettings
from pydantic import BaseModel
import xarray as xr
from typing import Optional, cast

from tx_fast_hydrology.s3 import upload_file_to_s3, save_file_from_s3, S3Settings


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
    gage_lookback_hours: int = 2
    s3_assets: list[S3Asset] = [
        S3Asset(location="nwm_txdot_config/tx_fast_hydrology_assets/KF_v001/comids_mod.csv", target="comids_mod.csv"),
        S3Asset(location="nwm_txdot_config/tx_fast_hydrology_assets/KF_v001/usgs_subset_attila.csv", target="usgs_subset_attila.csv"),
        S3Asset(location="nwm_txdot_config/tx_fast_hydrology_assets/KF_v001/huc8_no_lake.json", target="huc8_no_lake.json"),
        S3Asset(location="nwm_txdot_config/tx_fast_hydrology_assets/KF_v001/huc2_12_nhd_min.json", target="huc2_12_nhd_min.json"),
    ]
    tick_dt: float = 600
    # assets
    comids_path: str = "./cache/comids_mod.csv"
    gage_ids_path: str = "./cache/usgs_subset_attila.csv"
    model_path: str = "./cache/huc8_no_lake.json"
    stream_network_path: str = (
        "./cache/huc2_12_nhd_min.json"  # this is for the /map endpoint
    )
    # outputs
    streamflow_output_path: str = "nwm_txdot_output/short_range_da_kf/streamflow_kf_sr.nc"


# Constants


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Initialize the app state and start periodic updates."""
        tracemalloc.start()  # Start tracing memory allocations

        # get settings from env vars
        app.state.settings = TxFastHydrologySettings()
        app.state.settings = cast(
            TxFastHydrologySettings, app.state.settings
        )  # help the typecheck
        # apply settings
        app.state.tick_dt = app.state.settings.tick_dt  # 10 minutes

        Path(app.state.settings.cache_dir).mkdir(parents=True,exist_ok=True)

        # Pull the necessary assets from S3
        for s3_asset in app.state.settings.s3_assets:
            local_cache = (
                Path(app.state.settings.cache_dir) / s3_asset.get_target_path()
            )
            if local_cache.exists():
                continue
            s3_asset.resolve(cache_dir=Path(app.state.settings.cache_dir))

        # Check for required assets
        app.state.all_ids = pd.read_csv(
            app.state.settings.gage_ids_path
        ).drop_duplicates(subset="comid")
        app.state.comids = pd.read_csv(app.state.settings.comids_path, index_col=0)[
            "0"
        ].values

        # Initialization logic
        logger.info("Initializing the simulation...")
        input_path = app.state.settings.model_path  # "./data/huc8_no_lake.json"
        gage_end_time = pd.to_datetime(datetime.now(timezone.utc))
        gage_start_time = pd.to_datetime(
            datetime.now(timezone.utc)
            - timedelta(hours=app.state.settings.gage_lookback_hours)
        )

        # app.state.all_ids = app.state.all_ids
        app.state.all_ids["to"] = gage_end_time.isoformat()
        app.state.all_ids["from"] = gage_start_time.isoformat()

        logger.info("Downloading initial gage data...")
        measurements = await download_gage_data(
            app.state.all_ids.to_dict(orient="records")
        )
        measurements = measurements.reindex(
            app.state.all_ids["comid"].values.astype(str), axis=1
        ).fillna(0.0)

        logger.info("Loading model collection...")
        model_collection = ModelCollection.from_file(input_path)
        for model in model_collection.models.values():
            checkpoint = CheckPoint(model, timedelta=3600)
            model.bind_callback(checkpoint, key="checkpoint")
            model_sites = [
                reach_id
                for reach_id in model.reach_ids
                if reach_id in measurements.columns
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
        inputs = download_nwm_forcings(
            nwm_dir, forecast_hour=forecast_hour, comids=app.state.comids
        )

        simulation = AsyncSimulation(model_collection, inputs)
        timestamp = streamflow.index.item()
        streamflow_values = streamflow.loc[timestamp]
        simulation.set_datetime(timestamp)
        simulation.init_states(streamflow_values)
        simulation.save_states()
        outputs = await simulation.simulate()
        outputs = pd.concat([series for series in outputs.values()], axis=1)

        # Add initial objects to app state
        app.state.simulation = simulation
        app.state.outputs = outputs
        app.state.current_timestamp = timestamp
        app.state.streamflow = streamflow

        with open(app.state.settings.stream_network_path) as basin:
            app.state.stream_network = jsonlib.load(basin)

        logger.info("Initialization complete. Starting periodic updates...")


        ds = xr.Dataset(
            {"streamflow": (["time", "feature_id"], outputs.values)},
            coords={
                "time": outputs.index.tz_localize(None).astype("datetime64[ns]"),
                "reference_time": ("reference_time", [np.datetime64(timestamp)]),
                "feature_id": [int(f_id) for f_id in outputs.columns],
            },
            attrs={
                "TITLE": "OUTPUT FROM Kalman-Filter by MDB",
                "version": metadata.version("tx-fast-hydrology"),
                "featureType": "timeSeries",
                "proj4": "+proj=lcc +units=m +a=6370000.0 +b=6370000.0 +lat_1=30.0 +lat_2=60.0 +lat_0=40.0 +lon_0=-97.0",
                "model_initialization_time": str(timestamp),
                "station_dimension": "feature_id",
                "model_output_valid_time": str(outputs.index[0]),
                "model_configuration": "short_range",
                "dev_OVRTSWCRT": 1,
                "dev_NOAH_TIMESTEP": 3600,
                "dev_channel_only": 0,
                "dev_channelBucket_only": 0,
                "dev": "dev_ prefix indicates development/internal metrics",
                "units": "m3 s-1",
            },
        )
        ds["streamflow"].attrs["units"] = "m3 s-1"
        ds.to_netcdf(Path(app.state.settings.cache_dir) / "streamflow_output.nc", engine="netcdf4")

        upload_file_to_s3(
            bucket_name=S3Settings().bucket_name,
            s3_key=app.state.settings.streamflow_output_path,
            filename=Path(app.state.settings.cache_dir) / "streamflow_output.nc",
        )


        current, peak = (
            tracemalloc.get_traced_memory()
        )  # Get current and peak memory usage
        logger.info(f"Current memory usage: {current / 1024**2:.2f} MB")
        logger.info(f"Peak memory usage: {peak / 1024**2:.2f} MB")

        tracemalloc.stop()  # Stop tracing memory allocations
        # Start the periodic background task
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
        diff = (
            diff.loc[time_index].fillna(0.0).replace([np.inf, -np.inf], 0.0).to_dict()
        )
        pct_diff = (
            pct_diff.loc[time_index]
            .fillna(0.0)
            .replace([np.inf, -np.inf], 0.0)
            .to_dict()
        )
        json_output = {"diff__cms": diff, "pct_diff__pct": pct_diff}
        return JSONResponse(content=json_output)

    @router.get("/map")
    async def map_handler(request: Request):
        static_url = f"{request.base_url}static/style.css"
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

        logger.info(
            f"URL for static style.css: {request.url_for('static', filename='style.css')}"
        )

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

    # app.include_router(router)
    # for route in app.routes:
    #     logger.info(f"Path: {route.path}, Name: {route.name}")
    return app


async def tick(app: FastAPI):
    """Periodic background task to update simulation and state."""
    while True:
        tracemalloc.start()  # Start tracing memory allocations
        tick_dt = app.state.tick_dt
        simulation = app.state.simulation
        last_timestamp = app.state.current_timestamp

        logger.info(f"Sleeping for {tick_dt} seconds...")
        await asyncio.sleep(tick_dt)

        # Download new NWM forcings and streamflows
        urls = get_forcing_directories()
        nwm_dir, forecast_hour, timestamp = get_forecast_path(urls)

        if timestamp > last_timestamp:
            logger.info(f"New forcings available at timestamp {timestamp}")
            streamflow = download_nwm_streamflow(
                nwm_dir, forecast_hour=forecast_hour, comids=app.state.comids
            )
            logger.info("Streamflow downloaded")
            inputs = download_nwm_forcings(
                nwm_dir, forecast_hour=forecast_hour, comids=app.state.comids
            )
            logger.info("Forcings downloaded")
            simulation.load_states()
            simulation.inputs = simulation.load_inputs(inputs)
            # Download gage data
            gage_end_time = pd.to_datetime(datetime.now(timezone.utc))
            gage_start_time = pd.to_datetime(
                datetime.now(timezone.utc)
                - timedelta(hours=app.state.settings.gage_lookback_hours)
            )
            # TODO: Check these start and end times
            gage_end_time = max(gage_end_time, timestamp)
            gage_start_time = min(gage_start_time, last_timestamp)
            app.state.all_ids["to"] = gage_end_time.isoformat()
            app.state.all_ids["from"] = gage_start_time.isoformat()
            logger.info("Downloading gage data...")
            measurements = await download_gage_data(
                app.state.all_ids.to_dict(orient="records")
            )
            # Ensure that we always use the same ordering and columns
            measurements = measurements.reindex(
                app.state.all_ids["comid"].values.astype(str), axis=1
            )
            # TODO: This needs to be fixed eventually, filter out totally empty timeseries!!!!
            measurements = measurements.fillna(0.0)
            logger.info("Gage data downloaded")
            logger.info("Assigning gage data to subbasin models...")
            for model in simulation.model_collection.models.values():
                if hasattr(model, "callbacks") and "kf" in model.callbacks:
                    measurements_columns = model.callbacks["kf"].measurements.columns
                    basin_measurements = measurements[measurements_columns]
                    # print(f'Removing duplicate columns')
                    # basin_measurements = basin_measurements.loc[:, ~basin_measurements.columns.duplicated()].copy()
                    model.callbacks["kf"].measurements = basin_measurements
            # Print current times
            logger.info(
                f"Gage start time: {measurements.index.min().isoformat()}\n"
                f"Gage end time: {measurements.index.max().isoformat()}\n"
                f"Input start time: {inputs.index.min().isoformat()}\n"
                f"Input end time: {inputs.index.max().isoformat()}\n"
                f"Model timestamp: {simulation.datetime.isoformat()}"
            )
            # Step model forward in time
            logger.info("Beginning simulation...")
            outputs = await simulation.simulate()
            outputs = pd.concat([series for series in outputs.values()], axis=1)
            logger.info("Simulation finished")
            app.state.simulation = simulation
            app.state.outputs = outputs
            app.state.current_timestamp = timestamp
            app.state.streamflow = streamflow

            # Export to S3
            ds = xr.Dataset(
                {"streamflow": (["time", "feature_id"], outputs.values)},
                coords={
                    "time": outputs.index,
                    "reference_time": ("reference_time", [np.datetime64(timestamp)]),
                    "feature_id": [int(f_id) for f_id in outputs.columns],
                },
                attrs={
                    "TITLE": "OUTPUT FROM Kalman-Filter by MDB",
                    "version": metadata.version("tx-fast-hydrology"),
                    "featureType": "timeSeries",
                    "proj4": "+proj=lcc +units=m +a=6370000.0 +b=6370000.0 +lat_1=30.0 +lat_2=60.0 +lat_0=40.0 +lon_0=-97.0",
                    "model_initialization_time": str(timestamp),
                    "station_dimension": "feature_id",
                    "model_output_valid_time": str(outputs.index[0]),
                    "model_configuration": "short_range",
                    "dev_OVRTSWCRT": 1,
                    "dev_NOAH_TIMESTEP": 3600,
                    "dev_channel_only": 0,
                    "dev_channelBucket_only": 0,
                    "dev": "dev_ prefix indicates development/internal metrics",
                    "units": "m3 s-1",
                },
            )
            ds["streamflow"].attrs["units"] = "m3 s-1"
            ds.to_netcdf(Path(app.state.settings.cache_dir) / "streamflow_output.nc", engine="netcdf4")

            upload_file_to_s3(
                bucket_name=S3Settings().bucket_name,
                s3_key=app.state.settings.streamflow_output_path,
                filename=Path(app.state.settings.cache_dir) / "streamflow_output.nc",
            )
        else:
            logger.info("No new forcings available; skipping update")
        current, peak = (
            tracemalloc.get_traced_memory()
        )  # Get current and peak memory usage
        logger.info(f"Current memory usage: {current / 1024**2:.2f} MB")
        logger.info(f"Peak memory usage: {peak / 1024**2:.2f} MB")

        tracemalloc.stop()  # Stop tracing memory allocations
