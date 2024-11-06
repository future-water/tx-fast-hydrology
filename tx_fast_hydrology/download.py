import os
import time
import pathlib
import io
import re
import requests
import xarray as xr
import pandas as pd
from bs4 import BeautifulSoup

def get_forcing_directories():
    baseurl = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/nwm/prod'
    regex = re.compile('^nwm\.\d{8}/$')
    response = requests.get(baseurl)
    if response.status_code == 200:
        doc = BeautifulSoup(response.text, 'lxml')
        body = doc.find(name='body')
        links = body.find_all(name='a')
        urls = []
        for link in links:
            if 'href' in link.attrs:
                url = link.attrs['href']
                match = regex.match(url)
                if match:
                    urls.append(url)
        return [f'{baseurl}/{url}' for url in urls]
    else:
        return []

def get_latest_forecast_hour(nwm_dir):
    regex = re.compile('^nwm\.t(\d{2})z\.short_range.channel_rt')
    response = requests.get(f'{nwm_dir}/short_range/')
    if response.status_code == 200:
        doc = BeautifulSoup(response.content, 'lxml')
        body = doc.find(name='body')
        links = body.find_all(name='a')
        forecast_start_hours = set()
        for link in links:
            if 'href' in link.attrs:
                url = link.attrs['href']
                match = regex.match(url)
                if match:
                    next_hour = int(match.group(1))
                    forecast_start_hours.add(next_hour)
        return max(forecast_start_hours)
    else:
        return None

def get_forecast_path(nwm_dirs):
    for url in reversed(sorted(nwm_dirs)):
        forecast_hour = get_latest_forecast_hour(url)
        if forecast_hour is None:
            continue
        else:
            date = pathlib.Path(url).name.split('.')[1]
            hour = pd.to_timedelta(forecast_hour, unit='h')
            timestamp = pd.to_datetime(date, utc=True) + hour
            return url, forecast_hour, timestamp
    raise LookupError('No files found.')

def download_nwm_streamflow(nwm_dir, forecast_hour, comids, sleeptime=0.):
        # TODO: Should this include tm01 and tm02 as well?
        nc_url = f'analysis_assim/nwm.t{forecast_hour:02}z.analysis_assim.channel_rt.tm00.conus.nc'
        url = os.path.join(nwm_dir, nc_url)
        response = requests.get(url)
        if response.status_code == 200:
            dataset = xr.load_dataset(io.BytesIO(response.content), engine='h5netcdf')
        else:
            raise PermissionError(response.status_code)
        datetime = pd.to_datetime(dataset['time'].values.item(), utc=True)
        streamflow = dataset['streamflow'].sel(feature_id=comids).values
        streamflow = pd.DataFrame(pd.Series(streamflow, index=comids), columns=[datetime]).T
        streamflow.columns = streamflow.columns.astype(str)
        return streamflow

def download_nwm_forcings(nwm_dir, forecast_hour, comids, sleeptime=0.):
    # Download NetCDF forcings
    datasets = {}
    for lookahead_hour in range(1, 19):
        nc_url = f'short_range/nwm.t{forecast_hour:02}z.short_range.channel_rt.f{lookahead_hour:03}.conus.nc'
        url = os.path.join(nwm_dir, nc_url)
        response = requests.get(url)
        if response.status_code == 200:
            dataset = xr.load_dataset(io.BytesIO(response.content), engine='h5netcdf')
        else:
            raise PermissionError(response.status_code)
        datasets[lookahead_hour] = dataset
        time.sleep(sleeptime)
    # Parse NetCDF forcings
    qSfcLatRunoff = {}
    qBucket = {}
    for key, dataset in datasets.items():
        datetime = pd.to_datetime(dataset['time'].values.item(), utc=True)
        runoff = dataset['qSfcLatRunoff'].sel(feature_id=comids).values
        bucket = dataset['qBucket'].sel(feature_id=comids).values
        qSfcLatRunoff[datetime] = runoff
        qBucket[datetime] = bucket
    qSfcLatRunoff = pd.DataFrame.from_dict(qSfcLatRunoff, orient='index', columns=comids)
    qBucket = pd.DataFrame.from_dict(qBucket, orient='index', columns=comids)
    inputs = qSfcLatRunoff + qBucket
    inputs.columns = inputs.columns.astype(str)
    return inputs    