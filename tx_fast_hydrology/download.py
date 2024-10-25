import time
import io
import re
import requests
import xarray as xr
import pandas as pd
from bs4 import BeautifulSoup

def download_forcings(nwm_dir, forecast_start_hour, comids, sleeptime=0.1):
    # Download NetCDF forcings
    datasets = {}
    for lookahead_hour in range(1, 19):
        url = f'{nwm_dir}/short_range/nwm.t{forecast_start_hour:02}z.short_range.channel_rt.f{lookahead_hour:03}.conus.nc'
        response = requests.get(url)
        if response.status_code == 200:
            dataset = xr.load_dataset(io.BytesIO(response.content), engine='h5netcdf')
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

def get_forcing_directory():
    baseurl = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/nwm/prod'
    regex = re.compile('^nwm\.\d{8}/$')
    response = requests.get(baseurl)
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
    url = max(urls)
    return f'{baseurl}/{url}'