import os
import time
import datetime
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
    for url in nwm_dirs:
        forecast_hour = get_latest_forecast_hour(url)
        if forecast_hour is None:
            continue
    return url, forecast_hour


def download_forcings(nwm_dir, forecast_hour, comids, sleeptime=0.1):
    # Download NetCDF forcings
    datasets = {}
    for lookahead_hour in range(1, 19):
        nc_url = f'short_range/nwm.t{forecast_hour:02}z.short_range.channel_rt.f{lookahead_hour:03}.conus.nc'
        url = os.path.join(nwm_dir, nc_url)
        print(url)
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
    return inputs    

