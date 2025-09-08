import os
import asyncio
import random
import time
import pathlib
import io
import re
import requests
import xarray as xr
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import httpx
from httpx import Limits, Timeout
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
from typing import Any, Tuple
from tqdm.asyncio import tqdm as tqdm_asyncio
from tqdm import tqdm

# Define the macros as environment variables or constants
KISTERS_USER = os.getenv("KISTERS_USER", "txdot-analytics@kisters.net")
KISTERS_PASS = os.getenv("KISTERS_PASS", "dsKNahM3t!2")
KISTERS_BASE_URL = os.getenv("KISTERS_BASE_URL", "https://na.datasphere.online/external")
# MAX_CONCURRENT_REQUESTS = 20
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENCY", "64"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
RETRY_DELAY_BASE = float(os.getenv("RETRY_DELAY_BASE", "0.5"))  # seconds
CONNECT_TIMEOUT = float(os.getenv("CONNECT_TIMEOUT", "5"))
READ_TIMEOUT = float(os.getenv("READ_TIMEOUT", "20"))
WRITE_TIMEOUT = float(os.getenv("WRITE_TIMEOUT", "5"))
POOL_TIMEOUT = float(os.getenv("POOL_TIMEOUT", "5"))


def results_to_df(results: list[dict[str, Any]]) -> pd.DataFrame:
    dataframes = []
    for result in tqdm(results, desc="Combining DataFrames", ncols=100):
        comid = result["comid"]
        data_entries = result.get("data", [])

        # Create a DataFrame with datetime index and comid as the column
        df = pd.DataFrame(data_entries, columns=["timestamp", "value", "quality", "remark"])
        if df.empty:  # Skip if the DataFrame is empty
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df = df[["value"]].rename(columns={"value": comid})

        dataframes.append(df)
    # TODO: Sometimes this fails with no data obtained from the server
    combined_df = pd.concat(dataframes, axis=1)
    return combined_df


async def fetch_last_values(client: httpx.AsyncClient, sem: asyncio.Semaphore, params: dict) -> dict:
    pass
    # url = f"{KISTERS_BASE_URL}/channels/{params['channel_id']}/timeSeries/lastValues"
    # request_params = {
    #     # "to": params["to"],
    #     # "from": params["from"],
    #     "channel_id": params["channel_id"],
    #     # "tsId": params["timeseries_id"],
    #     # "format": "JSON",
    # }


MAX_RETRIES = 5
RETRY_DELAY = 5  # seconds


def _retry_backoff(attempt: int) -> float:
    # exponential backoff with jitter
    return (RETRY_DELAY_BASE * (2 ** (attempt - 1))) + random.uniform(0, 0.2)


def make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=KISTERS_BASE_URL,  # e.g. "https://api.kisters.cloud"
        auth=(KISTERS_USER, KISTERS_PASS),
        http2=True,  # huge win if server supports it
        limits=Limits(
            max_connections=MAX_CONCURRENT_REQUESTS,  # cap total sockets
            max_keepalive_connections=max(8, MAX_CONCURRENT_REQUESTS // 2),
            keepalive_expiry=30.0,
        ),
        timeout=Timeout(
            connect=CONNECT_TIMEOUT,
            read=READ_TIMEOUT,
            write=WRITE_TIMEOUT,
            pool=POOL_TIMEOUT,
        ),
        headers={
            "Connection": "keep-alive",
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate, br",
            "User-Agent": "FAST/1.0 (+async httpx)",
        },
        trust_env=False,  # ignore HTTP(S)_PROXY in cloud images
        verify=True,
    )


async def fetch_data(client: httpx.AsyncClient, sem: asyncio.Semaphore, params: dict) -> dict:
    url = f"/channels/{params['channel_id']}/timeSeries/data"
    request_params = {
        "to": params["to"],
        "from": params["from"],
        "channel_id": params["channel_id"],
        "tsId": params["timeseries_id"],
        "format": "JSON",
    }
    async with sem:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = await client.get(url, params=request_params)
                # Avoid retrying for certain client errors
                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    resp.raise_for_status()
                resp.raise_for_status()
                result = resp.json()
                return {
                    "success": True,
                    "data": {
                        "comid": params["comid"],
                        "channel_id": params["channel_id"],
                        "timeseries_id": params["timeseries_id"],
                        **result[0]["locations"][0]["timeseries"][0],
                    },
                }
            except httpx.HTTPStatusError as e:
                error_message = f"Attempt {attempt}/{MAX_RETRIES} - HTTP {e.response.status_code}: {e.response.text[:200]}"  # noqa
            except httpx.RequestError as e:
                error_message = f"Attempt {attempt}/{MAX_RETRIES} - RequestError: {e!s}"
            except Exception as e:
                error_message = f"Attempt {attempt}/{MAX_RETRIES} - UnexpectedError: {e!s}"

            if attempt == MAX_RETRIES:
                return {"success": False, "error": error_message, "payload": params}

            await asyncio.sleep(_retry_backoff(attempt))


async def download_gage_data(data_list: list[dict]):
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    async with make_client() as client:
        tasks = [fetch_data(client, sem, p) for p in data_list]
        results = await tqdm_asyncio.gather(
            *tasks, desc="Fetching Data", total=len(tasks), ncols=100
        )

    successful_requests, unsuccessful_requests, successful_data_list = [], [], []
    for idx, result in enumerate(results):
        if result["success"]:
            successful_requests.append(result["data"])
            successful_data_list.append(data_list[idx])
        else:
            unsuccessful_requests.append({"error": result["error"], "payload": result["payload"]})

    if not successful_requests:
        logger.error("No successful requests were made.")
        return None
    logger.info(
        f"Datasphere requests - Successful: {len(successful_requests)} | Failed: {len(unsuccessful_requests)}| Ratio: {len(successful_requests) / (len(successful_requests + unsuccessful_requests))}"  # noqa
    )

    df = results_to_df(successful_requests)
    # Consider doing interpolation per-series to avoid big global passes if huge
    df = df.interpolate().bfill().ffill()

    df.columns = df.columns.astype(str)
    return df


def get_forcing_directories():
    baseurl = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/nwm/prod"
    regex = re.compile(r"^nwm\.\d{8}/$")
    response = requests.get(baseurl)
    if response.status_code == 200:
        doc = BeautifulSoup(response.text, "lxml")
        body = doc.find(name="body")
        links = body.find_all(name="a")
        urls = []
        for link in links:
            if "href" in link.attrs:
                url = link.attrs["href"]
                match = regex.match(url)
                if match:
                    urls.append(url)
        return [f"{baseurl}/{url}" for url in urls]
    else:
        return []


def get_forcing_directory_for_date(target_date: datetime, product="short_range"):
    base_url = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/nwm/prod"
    date_str = target_date.strftime("%Y%m%d")
    time_str = "00"  # Default cycle time; can also try others like 06, 12, 18 if needed
    sample_filename = f"nwm.t{time_str}z.{product}.channel_rt.f001.conus.nc"

    test_url = f"{base_url}/nwm.{date_str}/{product}/{sample_filename}"

    response = requests.head(test_url)
    if response.status_code == 200:
        return f"{base_url}/nwm.{date_str}/"
    elif response.status_code == 403:
        raise PermissionError(f"Access denied when checking {test_url}")
    elif response.status_code == 404:
        return None
    else:
        raise RuntimeError(f"Unexpected status {response.status_code} checking {test_url}")


def get_latest_forecast_hour(nwm_dir):
    regex = re.compile(r"^nwm\.t(\d{2})z\.short_range.channel_rt")
    response = requests.get(f"{nwm_dir}/short_range/")
    if response.status_code == 200:
        doc = BeautifulSoup(response.content, "lxml")
        body = doc.find(name="body")
        links = body.find_all(name="a")
        forecast_start_hours = set()
        for link in links:
            if "href" in link.attrs:
                url = link.attrs["href"]
                match = regex.match(url)
                if match:
                    next_hour = int(match.group(1))
                    forecast_start_hours.add(next_hour)
        return max(forecast_start_hours)
    else:
        return None


def get_forecast_path(nwm_dirs) -> tuple[str, int, datetime]:
    for url in reversed(sorted(nwm_dirs)):
        forecast_hour = get_latest_forecast_hour(url)
        if forecast_hour is None:
            continue
        else:
            date = pathlib.Path(url).name.split(".")[1]
            hour = pd.to_timedelta(forecast_hour, unit="h")
            timestamp = pd.to_datetime(date, utc=True) + hour
            return url, forecast_hour, timestamp
    raise LookupError("No files found.")


def get_forecast_path_for_timestamp(
    nwm_dirs: list[str], target_time: datetime
) -> tuple[str, int, datetime]:
    """
    For a given timestamp, find the available NWM forecast directory from the list.
    Returns (base_url, forecast_hour, timestamp) if a forecast file exists.
    """
    target_date_str = target_time.strftime("%Y%m%d")
    target_hour = target_time.hour
    forecast_filename = f"nwm.t{target_hour:02d}z.short_range.channel_rt.f001.conus.nc"

    for url in sorted(nwm_dirs, reverse=True):
        date_str = pathlib.Path(url).name.split(".")[1]
        if date_str != target_date_str:
            continue

        file_url = f"{url}/short_range/{forecast_filename}"
        response = requests.head(file_url)

        if response.status_code == 200:
            timestamp = pd.to_datetime(date_str, utc=True) + pd.to_timedelta(target_hour, unit="h")
            return url, target_hour, timestamp

    raise LookupError(f"No available forecast found for {target_time.isoformat()}")


def download_nwm_streamflow(nwm_dir, forecast_hour, comids, sleeptime=0.0):
    # TODO: Should this include tm01 and tm02 as well?
    nc_url = f"analysis_assim/nwm.t{forecast_hour:02}z.analysis_assim.channel_rt.tm00.conus.nc"
    url = os.path.join(nwm_dir, nc_url)
    response = requests.get(url)
    if response.status_code == 200:
        dataset = xr.load_dataset(io.BytesIO(response.content), engine="h5netcdf")
    else:
        raise PermissionError(response.status_code)
    datetime = pd.to_datetime(dataset["time"].values.item(), utc=True)
    streamflow = dataset["streamflow"].sel(feature_id=comids).values
    streamflow = pd.DataFrame(pd.Series(streamflow, index=comids), columns=[datetime]).T
    streamflow.columns = streamflow.columns.astype(str)
    return streamflow


def download_nwm_forcings2(
    nwm_dir: str, forecast_hour: int, comids: list[int], max_workers: int = 6
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download NWM short-range forcings for a set of COMIDs and return
    inputs (qSfcLatRunoff + qBucket) and streamflow_nwm DataFrames.
    """
    # Build URLs
    urls = {
        lookahead_hour: os.path.join(
            nwm_dir,
            f"short_range/nwm.t{forecast_hour:02}z.short_range.channel_rt.f{lookahead_hour:03}.conus.nc",
        )
        for lookahead_hour in range(1, 19)
    }

    def fetch_nc(hour_url_tuple):
        hour, url = hour_url_tuple
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        ds = xr.load_dataset(io.BytesIO(r.content), engine="h5netcdf")
        return hour, ds

    datasets = {}
    # Parallel download
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_hour = {executor.submit(fetch_nc, item): item[0] for item in urls.items()}
        for future in as_completed(future_to_hour):
            hour = future_to_hour[future]
            try:
                _, ds = future.result()
                datasets[hour] = ds
            except Exception as e:
                raise RuntimeError(f"Failed to download or parse hour {hour}: {e}")

    # Sort datasets by lookahead_hour
    datasets = dict(sorted(datasets.items()))

    # Parse NetCDF and combine into DataFrames
    times, qSfcLatRunoff_list, qBucket_list, streamflow_list = [], [], [], []

    for ds in datasets.values():
        times.append(pd.to_datetime(ds["time"].values.item(), utc=True))
        qSfcLatRunoff_list.append(ds["qSfcLatRunoff"].sel(feature_id=comids).values)
        qBucket_list.append(ds["qBucket"].sel(feature_id=comids).values)
        streamflow_list.append(ds["streamflow"].sel(feature_id=comids).values)

    qSfcLatRunoff_df = pd.DataFrame(qSfcLatRunoff_list, index=times, columns=comids)
    qBucket_df = pd.DataFrame(qBucket_list, index=times, columns=comids)
    streamflow_df = pd.DataFrame(streamflow_list, index=times, columns=[str(c) for c in comids])

    inputs_df = qSfcLatRunoff_df + qBucket_df
    inputs_df.columns = inputs_df.columns.astype(str)

    return inputs_df, streamflow_df


def download_nwm_forcings(
    nwm_dir: str, forecast_hour: int, comids: list[int], sleeptime: float = 0.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Download NetCDF forcings
    datasets = {}
    for lookahead_hour in range(1, 19):
        nc_url = f"short_range/nwm.t{forecast_hour:02}z.short_range.channel_rt.f{lookahead_hour:03}.conus.nc"  # noqa
        url = os.path.join(nwm_dir, nc_url)
        response = requests.get(url)
        if response.status_code == 200:
            dataset = xr.load_dataset(io.BytesIO(response.content), engine="h5netcdf")
        else:
            raise PermissionError(response.status_code)
        datasets[lookahead_hour] = dataset
        time.sleep(sleeptime)
    # Parse NetCDF forcings
    qSfcLatRunoff = {}
    qBucket = {}
    streamflow_nwm = {}
    for key, dataset in datasets.items():
        datetime = pd.to_datetime(dataset["time"].values.item(), utc=True)
        runoff = dataset["qSfcLatRunoff"].sel(feature_id=comids).values
        bucket = dataset["qBucket"].sel(feature_id=comids).values
        streamflow = dataset["streamflow"].sel(feature_id=comids).values
        qSfcLatRunoff[datetime] = runoff
        qBucket[datetime] = bucket
        streamflow_nwm[datetime] = streamflow
    qSfcLatRunoff = pd.DataFrame.from_dict(qSfcLatRunoff, orient="index", columns=comids)
    qBucket = pd.DataFrame.from_dict(qBucket, orient="index", columns=comids)
    streamflow_nwm = pd.DataFrame.from_dict(streamflow_nwm, orient="index", columns=comids)
    streamflow_nwm.columns = streamflow_nwm.columns.astype(str)
    inputs = qSfcLatRunoff + qBucket
    inputs.columns = inputs.columns.astype(str)
    return inputs, streamflow_nwm
